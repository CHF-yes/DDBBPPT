"""Independent COCO encoders, high-resolution P2 PAN, spatial memory v3.

Legacy models remain in model.py for checkpoint compatibility. No hooks, shared
parameter aliases, detector voting, or persistent inter-image state are used here.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C3k2, Detect
from ultralytics.cfg import DEFAULT_CFG
from independent_fusion import EvidenceEmbedding, LocalCorrespondence, ComplementaryFusion, warp, resize_flow
from memory_fusion import CrossScaleMemory, NeckMemoryRead, masked_pool

SCALES = ("p2", "p3", "p4", "p5")
INDICES = (2, 4, 6, 10)
MODES = ("rgb", "ir", "dep")


@contextmanager
def preserve_bn_buffers(module):
    saved = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
             for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    try:
        yield
    finally:
        with torch.no_grad():
            for m, mean, var, count in saved:
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(count)


class SpatialDetect(Detect):
    """One standard detector; geometry bypass influences regression, not class logits."""
    def forward(self, x, localization=None):
        if localization is None:
            return super().forward(x)
        b = x[0].shape[0]
        preds = {"boxes": torch.cat([self.cv2[i](localization[i]).view(b, self.reg_max*4, -1) for i in range(self.nl)], -1),
                 "scores": torch.cat([self.cv3[i](x[i]).view(b, self.nc, -1) for i in range(self.nl)], -1), "feats": x}
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)


class IndependentMMYOLO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.encoder.depth_input_channels != 4 or cfg.encoder.share_tier != "a":
            raise ValueError("v3 requires independent encoders (share-tier=a) and four-channel Depth")
        if not cfg.encoder.metric_branch or not cfg.fusion.use_bus:
            raise ValueError("v3 requires metric branch and cross-scale memory")
        self.cfg = cfg
        self.nc = cfg.class_num()
        self.class_names = list(cfg.class_names) or [str(i) for i in range(self.nc)]
        self.backbone = YOLO(cfg.resolve_weights()).model.float()
        # Reuse the audited same-name class remapping, retaining original head widths.
        from model import MMYOLO
        MMYOLO._adapt_detect_cls(self, self.backbone.model[-1], self.nc, cfg.cls_remap)
        self.backbone.names = dict(enumerate(self.class_names))
        self.backbone.nc = self.nc
        self.channels = {}
        was = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 96)
            for i, layer in enumerate(self.backbone.model[:11]):
                x = layer(x)
                if i in INDICES:
                    self.channels[SCALES[INDICES.index(i)]] = x.shape[1]
        self.backbone.train(was)
        self.aux_encoders = nn.ModuleDict({m: copy.deepcopy(self.backbone.model[:11]) for m in ("ir", "dep")})
        for encoder in self.aux_encoders.values():
            old = encoder[0].conv
            conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                conv.weight.copy_(old.weight.sum(1, keepdim=True))
            encoder[0].conv = conv
        self.metric_encoder = nn.ModuleDict({s: nn.Sequential(nn.Conv2d(4, 64, 1), nn.SiLU(),
                                                              nn.Conv2d(64, c, 1, bias=False)) for s, c in self.channels.items()})
        dim, md = cfg.fusion.spatial_dim, cfg.fusion.bus_dim
        self.embeddings = nn.ModuleDict({s: nn.ModuleList([EvidenceEmbedding(c, dim) for _ in MODES]) for s,c in self.channels.items()})
        for blocks in self.embeddings.values():
            for block in blocks[1:]:
                block.common.load_state_dict(blocks[0].common.state_dict())
        matcher_scales = SCALES if cfg.fusion.p2_match_refine else ("p3", "p4", "p5")
        self.matchers = nn.ModuleDict({s: nn.ModuleList([LocalCorrespondence() for _ in range(2)]) for s in matcher_scales})
        self.fusion = nn.ModuleDict({s: ComplementaryFusion(c, dim, md) for s,c in self.channels.items()})
        self.register_bus = CrossScaleMemory(self.channels, dim=md, heads=cfg.fusion.heads,
                                             tokens_per_modality=cfg.fusion.memory_tokens_per_modality)
        old = self.backbone.model[-1]
        neck_ch = [int(m[0].conv.in_channels) for m in old.cv2]
        p2_ch = max(64, neck_ch[0]//2)
        self.neck_channels = [p2_ch, *neck_ch]
        self.p2_lateral = Conv(self.channels["p2"], p2_ch, 1)
        self.p2_neck = C3k2(neck_ch[0]+p2_ch, p2_ch, n=2)
        self.p2_down = Conv(p2_ch, neck_ch[0], 3, 2)
        self.p3_refine = C3k2(neck_ch[0]*2, neck_ch[0], n=2)
        self.neck_gain = nn.Parameter(torch.tensor(.05))
        det = SpatialDetect(nc=self.nc, reg_max=old.reg_max, ch=self.neck_channels)
        # P2's new default width must NOT shrink pretrained P3/P4/P5 classifiers.
        det.cv2 = nn.ModuleList([det.cv2[0], *old.cv2])
        det.cv3 = nn.ModuleList([det.cv3[0], *old.cv3])
        det.dfl = old.dfl
        det.dfl.requires_grad_(False)
        det.stride = torch.tensor([4., 8., 16., 32.])
        det.cv2[0][-1].bias.data.fill_(2.)
        det.cv3[0][-1].bias.data.fill_(math.log(5/self.nc/(640/4)**2))
        self.backbone.model[-1] = det
        self.backbone.stride = det.stride
        self.semantic_adapters = None
        self.semantic_detect = None
        if cfg.fusion.branch_aux_weight > 0:
            # One shared training-only detector sees the common representation
            # from each modality. Sharing the head makes semantic compatibility
            # operational rather than merely encouraging similar magnitudes.
            self.semantic_adapters = nn.ModuleDict({
                s: Conv(dim, c, 1) for s, c in zip(SCALES, self.neck_channels)
            })
            self.semantic_detect = copy.deepcopy(det)
        self.neck_memory = NeckMemoryRead(p2_ch, md, cfg.fusion.heads)
        self.localization = nn.ModuleList([Conv(self.channels[s], c, 1) for s,c in zip(SCALES,self.neck_channels)])
        self.loc_gain = nn.Parameter(torch.full((4,), math.log(.05/.95)))
        self.modality_off = set()
        self.spatial_memory = True
        self.share_tier = "a"
        self.args = copy.copy(DEFAULT_CFG)
        self._struct = cfg.structure()
        self.aux_loss = torch.tensor(0.)
        self._last_register_state = None
        self.semantic_branch_present = None
        self._semantic_common = None
        self._semantic_masks = None
        self._semantic_flows = None
        self.last_semantic_losses = {}
        self.train()

    @property
    def model(self):
        return self.backbone.model

    @property
    def stride(self):
        return self.model[-1].stride

    def encoder_modules(self):
        return [self.backbone.model[:11], *self.aux_encoders.values()]

    def structure_kwargs(self):
        return copy.deepcopy(self._struct)

    def param_report(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        pretrained = n(self.backbone) - n(self.model[-1].cv2[0]) - n(self.model[-1].cv3[0]) + n(self.aux_encoders)
        return {"total": n(self), "pretrained": pretrained, "new": n(self)-pretrained,
                "rgb_encoder": n(self.backbone.model[:11]), "ir_encoder": n(self.aux_encoders["ir"]),
                "depth_encoder": n(self.aux_encoders["dep"]), "fusion": n(self.fusion)+n(self.embeddings),
                "register_bus": n(self.register_bus)}

    def _encode(self, x, modality, present):
        b, _, h, w = x.shape
        active = present.nonzero(as_tuple=True)[0]
        if not active.numel():
            return {s: x.new_zeros(b,c,h//(2**(i+2)),w//(2**(i+2))) for i,(s,c) in enumerate(self.channels.items())}
        y = x.index_select(0, active)
        encoder = self.backbone.model[:11] if modality == "rgb" else self.aux_encoders[modality]
        result = {}
        for i, layer in enumerate(encoder):
            if self.training and self.cfg.encoder.checkpoint_encoder and any(p.requires_grad for p in layer.parameters()):
                y = checkpoint(layer, y, use_reentrant=False,
                               context_fn=lambda layer=layer: (nullcontext(), preserve_bn_buffers(layer)))
            else:
                y = layer(y)
            if i in INDICES:
                result[SCALES[INDICES.index(i)]] = y.new_zeros(b,*y.shape[1:]).index_copy(0,active,y)
        return result

    def forward(self, rgb, ir=None, depth=None, quality=None, prior=None, keep=None):
        b, _, h, w = rgb.shape
        if h % 32 or w % 32:
            raise ValueError("v3 rectangular canvas height/width must both be multiples of 32")
        available = {"rgb": True, "ir": ir is not None, "dep": depth is not None}
        present = torch.stack([torch.as_tensor((keep or {}).get(m, torch.ones(b,device=rgb.device)),device=rgb.device).reshape(b).bool()
                               & available[m] & (m not in self.modality_off) for m in MODES], 1)
        if depth is not None:
            present[:,2] &= depth[:,2].flatten(1).any(1)
        if not present.any(1).all():
            raise ValueError("each sample must retain an observed modality")
        ir = torch.zeros_like(rgb[:,:1]) if ir is None else ir
        depth = rgb.new_zeros(b,4,h,w) if depth is None else depth
        raw = [self._encode(rgb,"rgb",present[:,0]), self._encode(ir,"ir",present[:,1]),
               self._encode(depth[:,:1],"dep",present[:,2])]
        metric_valid = depth[:,2:3] * depth[:,3:4] * present[:,2,None,None,None]
        absolute = depth[:,1:2] * metric_valid
        metric = torch.cat((absolute,torch.log1p(20*absolute.float())/math.log(21), depth[:,2:3],metric_valid),1)
        for s in SCALES:
            values, fraction = masked_pool(metric,metric_valid,raw[2][s].shape[-2:])
            raw[2][s] = raw[2][s] + .1*self.metric_encoder[s](values) * (fraction>0)
        masks, commons, privates, reliabilities = {}, {}, {}, {}
        auxiliary = rgb.new_zeros((), dtype=torch.float32)
        # Depth continuity is only a SOFT uncertainty cue: real edges remain usable.
        valid_d = depth[:,2:3].float()
        local = F.avg_pool2d(absolute.float(),3,1,1) / F.avg_pool2d(metric_valid.float(),3,1,1).clamp_min(1e-6)
        consistency = torch.exp(-10*(absolute.float()-local).abs())
        reliability_d = valid_d * torch.where(metric_valid>0, .35+.65*consistency, torch.ones_like(consistency))
        for s in SCALES:
            shape = raw[0][s].shape[-2:]
            masks[s] = [present[:,i,None,None,None].to(rgb.dtype).expand(-1,1,*shape) for i in range(3)]
            if quality and "availability" in quality:
                spatial = F.interpolate(quality["availability"].float(),shape,mode="nearest")
                masks[s] = [masks[s][i]*spatial[:,i:i+1] for i in range(3)]
            masks[s][2] = masks[s][2] * F.interpolate(valid_d, shape, mode="nearest")
            reliabilities[s] = [masks[s][0], masks[s][1], F.interpolate(reliability_d,shape,mode="area") * masks[s][2]]
            commons[s], privates[s] = [], []
            for m in range(3):
                c,u,loss = self.embeddings[s][m](raw[m][s],masks[s][m])
                commons[s].append(c)
                privates[s].append(u)
                auxiliary = auxiliary + loss / 12
        aligned, flows, confidence = {}, {}, {}
        previous = [None,None]
        for s in reversed(SCALES):
            shape = raw[0][s].shape[-2:]
            flows[s], confidence[s] = [], [masks[s][0]]
            for m in range(1,3):
                if s == "p2" and not self.cfg.fusion.p2_match_refine:
                    flow = resize_flow(previous[m-1],shape)
                    conf = F.interpolate(confidence["p3"][m],shape,mode="bilinear",align_corners=False)
                else:
                    scene = None if not quality or "scene_id" not in quality else F.interpolate(quality["scene_id"].float(),shape,mode="nearest")
                    flow,conf = self.matchers[s][m-1](commons[s][0],commons[s][m],masks[s][0],masks[s][m],previous[m-1],scene)
                # Without RGB observations use nominal geometric coordinates, not hallucinated matches.
                flow = flow * masks[s][0]
                previous[m-1] = flow
                flows[s].append(flow)
                confidence[s].append(conf)
        state = None
        fused, geometry, aligned_common, aligned_masks = {}, {}, {}, {}
        alignment_loss = auxiliary.new_zeros(())
        for s in SCALES:
            own = [raw[m][s] for m in range(3)]
            state = self.register_bus(s,own,masks[s],state)
            qual = [None if not quality or m not in quality else F.interpolate(quality[m].float(),own[0].shape[-2:],mode="bilinear",align_corners=False) for m in MODES]
            scene = None if not quality or "scene_id" not in quality else F.interpolate(quality["scene_id"].float(),own[0].shape[-2:],mode="nearest")
            c,u,mask,rel,values = [commons[s][0]], [privates[s][0]], [masks[s][0]], [reliabilities[s][0]], [own[0]]
            for m in range(1,3):
                flow = flows[s][m-1]
                vm = warp(masks[s][m],flow).clamp(0,1)
                if scene is not None:
                    vm = vm * ((warp(scene,flow)-scene).abs()<.01)
                if qual[m] is not None:
                    qual[m] = warp(qual[m],flow)*vm
                c.append(warp(commons[s][m],flow)*vm)
                u.append(warp(privates[s][m],flow)*vm)
                mask.append(vm)
                rel.append(warp(reliabilities[s][m],flow)*vm)
                values.append(warp(own[m],flow)*vm)
                weight = (confidence[s][m]*rel[m]).detach()
                similarity = (F.normalize(c[0].detach().float(),dim=1)*F.normalize(c[m].float(),dim=1)).sum(1,keepdim=True)
                alignment_loss = alignment_loss + ((1-similarity)*weight).sum()/weight.sum().clamp_min(1)/8
            aligned_common[s], aligned_masks[s] = c, mask
            fused[s] = self.fusion[s](values,c,u,mask,confidence[s],rel,state,qual)
            # A separate dense, confidence-gated boundary bypass supports box regression.
            geometry[s] = values[0]*mask[0] + sum(values[m]*confidence[s][m]*rel[m] for m in (1,2))*.25
        self.aux_loss = .01*auxiliary + .005*alignment_loss if self.training else auxiliary.detach()*0
        self.semantic_branch_present = present
        self._semantic_common = aligned_common
        self._semantic_masks = aligned_masks
        self._semantic_flows = flows
        self._last_register_state = state.detach()
        layers = self.backbone.model
        p5 = fused["p5"]
        p4_td = layers[13](torch.cat((F.interpolate(p5,scale_factor=2,mode="nearest"),fused["p4"]),1))
        p3_td = layers[16](torch.cat((F.interpolate(p4_td,scale_factor=2,mode="nearest"),fused["p3"]),1))
        p2 = self.p2_neck(torch.cat((F.interpolate(p3_td,scale_factor=2,mode="nearest"),self.p2_lateral(fused["p2"])),1))
        p2 = self.neck_memory(p2,state,present)
        p3 = p3_td + self.neck_gain.tanh()*self.p3_refine(torch.cat((self.p2_down(p2),p3_td),1))
        p4 = layers[19](torch.cat((layers[17](p3),p4_td),1))
        p5 = layers[22](torch.cat((layers[20](p4),fused["p5"]),1))
        features = [p2,p3,p4,p5]
        loc = [x+self.loc_gain[i].sigmoid()*self.localization[i](geometry[s]) for i,(s,x) in enumerate(zip(SCALES,features))]
        return self.model[-1](features,loc)

    def semantic_branch_prediction(self, name):
        """Run the shared training-only detector on ONE modality's common evidence.

        Three simultaneous auxiliary assigners plus the main detector exceeded
        23.5 GB at batch=4 / 736x1280 on a 4090, so the caller rotates modalities
        instead: every branch still receives identical long-run supervision.
        """
        if not self.training or self.semantic_detect is None or self._semantic_common is None:
            return None
        m = MODES.index(name)
        feats = [self.semantic_adapters[s](self._semantic_common[s][m]) for s in SCALES]
        return self.semantic_detect(feats, feats)
    @staticmethod
    def _object_vectors(feature, valid, targets, grid_size=3):
        """Differentiable object-region pooling in normalized canvas coordinates."""
        batch_idx = targets["batch_idx"].long()
        boxes = targets["bboxes"].float()
        if not batch_idx.numel():
            return feature.new_zeros((0, feature.shape[1])), feature.new_zeros(0, dtype=torch.bool)
        axis = torch.linspace(-.3, .3, grid_size, device=feature.device, dtype=boxes.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        cx, cy, bw, bh = boxes.unbind(1)
        gx = cx[:, None, None] + xx[None] * bw[:, None, None]
        gy = cy[:, None, None] + yy[None] * bh[:, None, None]
        grid = torch.stack((gx * 2 - 1, gy * 2 - 1), -1)
        selected = feature.index_select(0, batch_idx)
        selected_valid = valid.index_select(0, batch_idx)
        with torch.autocast(feature.device.type, enabled=False):
            values = F.grid_sample(selected.float(), grid.float(), align_corners=False)
            observed = F.grid_sample(selected_valid.float(), grid.float(), align_corners=False)
        weights = observed.clamp(0, 1)
        vector = (values * weights).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1e-4)
        return F.normalize(vector, dim=1), weights.mean((1, 2, 3)) > .5

    def semantic_regularization(self, targets, alignment_shift, alignment_supervised):
        """Return unweighted flow/NCE losses for the current forward pass."""
        zero = next(self.parameters()).new_zeros((), dtype=torch.float32)
        result = {"flow": zero, "nce": zero}
        if not self.training or self._semantic_common is None:
            self.last_semantic_losses = {k: v.detach() for k, v in result.items()}
            return result

        if self.cfg.fusion.flow_supervision_weight > 0:
            terms = []
            shift = alignment_shift.float()
            supervised = alignment_supervised.float()[:, None, None, None]
            for s in SCALES:
                pred = self._semantic_flows[s][1].float()  # Depth -> RGB source-sampling flow
                fh, fw = pred.shape[-2:]
                sx = float(self.infer_canvas[1] if hasattr(self, "infer_canvas") else 0) / fw
                sy = float(self.infer_canvas[0] if hasattr(self, "infer_canvas") else 0) / fh
                if sx <= 0 or sy <= 0:
                    # Training sets infer_canvas from the real canvas before the first forward.
                    raise RuntimeError("semantic flow supervision requires model.infer_canvas")
                target = torch.stack((shift[:, 0] / sx, shift[:, 1] / sy), 1)[:, :, None, None]
                target = target.expand_as(pred)
                mask = self._semantic_masks[s][0].float() * supervised
                # The shifted Depth must be observable at the target sampling position.
                mask = mask * warp(self._semantic_masks[s][2].float(), target).clamp(0, 1)
                error = F.smooth_l1_loss(pred, target, reduction="none", beta=.25).mean(1, keepdim=True)
                terms.append((error * mask).sum() / mask.sum().clamp_min(1))
            result["flow"] = torch.stack(terms).mean() if terms else zero

        if self.cfg.fusion.cross_modal_nce_weight > 0:
            terms, classes = [], targets["cls"].long()
            temperature = float(self.cfg.fusion.nce_temperature)
            for s in ("p3", "p4", "p5"):
                vectors, observed = [], []
                for m in range(3):
                    z, ok = self._object_vectors(self._semantic_common[s][m], self._semantic_masks[s][m], targets)
                    vectors.append(z)
                    observed.append(ok)
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    keep = observed[a] & observed[b]
                    if not keep.any():
                        continue
                    qa, kb, cls = vectors[a][keep], vectors[b][keep], classes[keep]
                    if len(qa) == 1:
                        terms.append(1 - (qa * kb).sum(1).mean())
                        continue
                    logits = qa @ kb.t() / temperature
                    # Other instances of the same class are neither negatives nor
                    # forced positives; the diagonal is the same physical object.
                    same_class = cls[:, None].eq(cls[None, :])
                    diagonal = torch.eye(len(cls), dtype=torch.bool, device=cls.device)
                    logits = logits.masked_fill(same_class & ~diagonal, -1e4)
                    labels = torch.arange(len(cls), device=cls.device)
                    terms.append((F.cross_entropy(logits, labels) +
                                  F.cross_entropy(logits.t(), labels)) * .5)
            result["nce"] = torch.stack(terms).mean() if terms else zero

        self.last_semantic_losses = {k: v.detach() for k, v in result.items()}
        return result
