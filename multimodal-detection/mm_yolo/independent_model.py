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
from independent_fusion import (EvidenceEmbedding, LocalCorrespondence,
                                ComplementaryFusion, SpatialEvidenceRouter,
                                TrustedEvidenceRouter, EmbeddingComplementPlugin,
                                V5IRQualityFusion, V511TrustedIRFusion,
                                CoarseAffineAligner, ConditionalIRAligner, affine_flow,
                                identity_residual_align, warp, resize_flow)
from memory_fusion import CrossScaleMemory, NeckMemoryRead, masked_pool

SCALES = ("p2", "p3", "p4", "p5")
INDICES = (2, 4, 6, 10)
MODES = ("rgb", "ir", "dep")


def depth_reliability_map(valid, absolute, metric_valid, mode):
    """Return sampling reliability without confusing geometry with corruption."""
    valid = valid.float()
    if mode == "valid_support_v2":
        support = F.avg_pool2d(valid, 5, 1, 2)
        return valid * (.60 + .40 * support)
    if mode == "legacy_edge_v1":
        local = (F.avg_pool2d(absolute.float(), 3, 1, 1) /
                 F.avg_pool2d(metric_valid.float(), 3, 1, 1).clamp_min(1e-6))
        consistency = torch.exp(-10 * (absolute.float() - local).abs())
        return valid * torch.where(
            metric_valid > 0, .35 + .65 * consistency, torch.ones_like(consistency))
    raise ValueError(f"unknown depth reliability: {mode}")


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


class IndependentBranchDetector(nn.Module):
    """Training-only full P2--P5 detector for one auxiliary modality.

    A shallow projection into the fused head can report a loss without proving
    that the auxiliary encoder can actually detect objects.  Each auxiliary
    branch therefore receives its own COCO-initialized neck and detector.  The
    two branches are executed and backpropagated sequentially, so the additional
    parameters do not duplicate high-resolution activations on a 24 GB GPU.  These modules are
    omitted from the deployment path; they are supervision, not detector voting.
    """
    def __init__(self, source_layers, channels, neck_channels, detector):
        super().__init__()
        p2_ch = neck_channels[0]
        self.p4_top_down = copy.deepcopy(source_layers[13])
        self.p3_top_down = copy.deepcopy(source_layers[16])
        self.p3_down = copy.deepcopy(source_layers[17])
        self.p4_bottom_up = copy.deepcopy(source_layers[19])
        self.p4_down = copy.deepcopy(source_layers[20])
        self.p5_bottom_up = copy.deepcopy(source_layers[22])
        self.p2_lateral = Conv(channels["p2"], p2_ch, 1)
        self.p2_neck = C3k2(neck_channels[1] + p2_ch, p2_ch, n=2)
        self.p2_down = Conv(p2_ch, neck_channels[1], 3, 2)
        self.p3_refine = C3k2(neck_channels[1] * 2, neck_channels[1], n=2)
        self.neck_gain = nn.Parameter(torch.tensor(.05))
        self.detector = copy.deepcopy(detector)

    def forward(self, evidence):
        p5_raw = evidence["p5"]
        p4_td = self.p4_top_down(torch.cat((
            F.interpolate(p5_raw, scale_factor=2, mode="nearest"), evidence["p4"]), 1))
        p3_td = self.p3_top_down(torch.cat((
            F.interpolate(p4_td, scale_factor=2, mode="nearest"), evidence["p3"]), 1))
        p2 = self.p2_neck(torch.cat((
            F.interpolate(p3_td, scale_factor=2, mode="nearest"),
            self.p2_lateral(evidence["p2"])), 1))
        p3 = p3_td + self.neck_gain.tanh() * self.p3_refine(
            torch.cat((self.p2_down(p2), p3_td), 1))
        p4 = self.p4_bottom_up(torch.cat((self.p3_down(p3), p4_td), 1))
        p5 = self.p5_bottom_up(torch.cat((self.p4_down(p4), p5_raw), 1))
        features = [p2, p3, p4, p5]
        return self.detector(features, features)


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
        # V5.1.1 must improve standalone thermal recognition without mutating
        # the IR encoder used by the protected V4.4 anchor.  This branch starts
        # from the same COCO-derived IR weights, then learns independently.
        self.v511_ir_encoder = (copy.deepcopy(self.aux_encoders["ir"])
                                if cfg.fusion.fusion_strategy ==
                                "v511_conditional_ir_v1" else None)
        self.metric_encoder = nn.ModuleDict({s: nn.Sequential(nn.Conv2d(4, 64, 1), nn.SiLU(),
                                                              nn.Conv2d(64, c, 1, bias=False)) for s, c in self.channels.items()})
        # V4.4 used a fixed 0.1 metric residual.  A bounded learnable gain starts
        # at exactly the same value, so V4.4 checkpoints retain their function
        # while valid absolute distance can become more useful during fine-tune.
        if cfg.fusion.fusion_strategy == "v44_incremental_router_v1":
            metric_initial = .1 / .5
            self.metric_gain_logit = nn.Parameter(torch.full(
                (len(SCALES),), math.log(metric_initial / (1 - metric_initial))))
        else:
            # Do not change the state_dict of legacy V4.4 checkpoints.
            self.metric_gain_logit = None
        dim, md = cfg.fusion.spatial_dim, cfg.fusion.bus_dim
        self.embeddings = nn.ModuleDict({s: nn.ModuleList([EvidenceEmbedding(c, dim) for _ in MODES]) for s,c in self.channels.items()})
        for blocks in self.embeddings.values():
            for block in blocks[1:]:
                block.common.load_state_dict(blocks[0].common.state_dict())
        self.v511_ir_embeddings = (
            nn.ModuleDict({s: copy.deepcopy(self.embeddings[s][1]) for s in SCALES})
            if cfg.fusion.fusion_strategy == "v511_conditional_ir_v1" else None)
        matcher_scales = SCALES if cfg.fusion.p2_match_refine else ("p3", "p4", "p5")
        self.matchers = nn.ModuleDict({s: nn.ModuleList([LocalCorrespondence() for _ in range(2)]) for s in matcher_scales})
        self.fusion = nn.ModuleDict({s: ComplementaryFusion(c, dim, md) for s,c in self.channels.items()})
        if cfg.fusion.fusion_strategy not in (
                "legacy_residual_v2", "evidence_router_v3", "v44_incremental_router_v1",
                "v47_trusted_evidence_v1", "v48_embedding_complement_v1",
                "v5_ir_quality_evidence", "v511_conditional_ir_v1"):
            raise ValueError(f"unknown fusion strategy: {cfg.fusion.fusion_strategy}")
        self.evidence_router = nn.ModuleDict()
        if cfg.fusion.fusion_strategy in ("evidence_router_v3", "v44_incremental_router_v1"):
            self.evidence_router = nn.ModuleDict({
                s: SpatialEvidenceRouter(c, dim, md, context_kernel=5 if s in ("p2", "p3") else 3)
                for s, c in self.channels.items()
            })
        elif cfg.fusion.fusion_strategy == "v47_trusted_evidence_v1":
            self.evidence_router = nn.ModuleDict({
                s: TrustedEvidenceRouter(c, dim, md) for s, c in self.channels.items()
            })
        elif cfg.fusion.fusion_strategy == "v48_embedding_complement_v1":
            self.evidence_router = nn.ModuleDict({
                s: EmbeddingComplementPlugin(c, dim, md) for s, c in self.channels.items()
            })
        elif cfg.fusion.fusion_strategy == "v5_ir_quality_evidence":
            self.evidence_router = nn.ModuleDict({
                s: V5IRQualityFusion(c, dim, md,
                                     ir_quality_channels=cfg.fusion.quality_channels,
                                     depth_quality_channels=3)
                for s, c in self.channels.items()
            })
        elif cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
            self.evidence_router = nn.ModuleDict({
                s: V511TrustedIRFusion(c, dim, md,
                                       ir_quality_channels=cfg.fusion.quality_channels,
                                       depth_quality_channels=3)
                for s, c in self.channels.items()
            })
        self.ir_coarse_aligner = (
            ConditionalIRAligner(dim, cfg.fusion.quality_channels)
            if cfg.fusion.ir_coarse_align and
               cfg.fusion.fusion_strategy == "v511_conditional_ir_v1"
            else (CoarseAffineAligner(dim) if cfg.fusion.ir_coarse_align else None))
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
        if (cfg.fusion.branch_aux_weight > 0 or
                any(float(v) > 0 for v in cfg.fusion.branch_aux_weights) or
                cfg.fusion.fusion_strategy in ("v47_trusted_evidence_v1",
                                               "v5_ir_quality_evidence",
                                               "v511_conditional_ir_v1")):
            # One shared training-only detector sees the common representation
            # from each modality. Sharing the head makes semantic compatibility
            # operational rather than merely encouraging similar magnitudes.
            self.semantic_adapters = nn.ModuleDict({
                s: Conv(dim, c, 1) for s, c in zip(SCALES, self.neck_channels)
            })
            self.semantic_detect = copy.deepcopy(det)
        # Stage A uses two genuinely independent training-only detectors.  They
        # consume each encoder's raw pyramid, not fused/common features.
        self.independent_aux = nn.ModuleDict()
        if (cfg.fusion.branch_aux_weight > 0 or
                any(float(v) > 0 for v in cfg.fusion.branch_aux_weights) or
                cfg.fusion.fusion_strategy in ("v47_trusted_evidence_v1",
                                               "v5_ir_quality_evidence",
                                               "v511_conditional_ir_v1")):
            if cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
                aux_names = ("rgb", "ir")
            elif cfg.fusion.fusion_strategy == "v5_ir_quality_evidence":
                aux_names = MODES
            else:
                aux_names = ("ir", "dep")
            self.independent_aux = nn.ModuleDict({
                m: IndependentBranchDetector(self.backbone.model, self.channels,
                                             self.neck_channels, det)
                for m in aux_names
            })
        self.neck_memory = NeckMemoryRead(p2_ch, md, cfg.fusion.heads)
        self.localization = nn.ModuleList([Conv(self.channels[s], c, 1) for s,c in zip(SCALES,self.neck_channels)])
        self.occlusion_context = nn.ModuleList()
        if cfg.fusion.fusion_strategy in ("evidence_router_v3", "v44_incremental_router_v1"):
            for c in self.neck_channels[:2]:
                block = nn.Sequential(
                    nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False),
                    nn.GroupNorm(max(1, min(16, c // 8)), c), nn.SiLU(),
                    nn.Conv2d(c, c, 1, bias=False))
                nn.init.zeros_(block[-1].weight)
                self.occlusion_context.append(block)
        self.loc_gain = nn.Parameter(torch.full((4,), math.log(.05/.95)))
        self.localization_scale = nn.Parameter(torch.zeros(4))
        # V5.1.1 starts from the complete learned V4.4 localization feature.
        # A zero transition is therefore function preserving.  Detection loss
        # may then move each scale towards the trusted-only geometry route.
        self.v511_localization_transition = (
            nn.Parameter(torch.zeros(4))
            if cfg.fusion.fusion_strategy == "v511_conditional_ir_v1" else None)
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
        self._ir_affine_prediction = None
        self._ir_affine_confidence = None
        self.auxiliary_eval_branch = None
        self.embedding_aux_losses = {"reconstruction": torch.tensor(0.),
                                     "alignment": torch.tensor(0.)}
        self._evidence_logits = {}
        self.train()

    @property
    def model(self):
        return self.backbone.model

    @property
    def stride(self):
        return self.model[-1].stride

    def encoder_modules(self):
        modules = [self.backbone.model[:11], *self.aux_encoders.values()]
        if self.v511_ir_encoder is not None:
            modules.append(self.v511_ir_encoder)
        return modules

    def structure_kwargs(self):
        return copy.deepcopy(self._struct)

    def param_report(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        pretrained = n(self.backbone) - n(self.model[-1].cv2[0]) - n(self.model[-1].cv3[0]) + n(self.aux_encoders)
        return {"total": n(self), "pretrained": pretrained, "new": n(self)-pretrained,
                "rgb_encoder": n(self.backbone.model[:11]), "ir_encoder": n(self.aux_encoders["ir"]),
                "depth_encoder": n(self.aux_encoders["dep"]),
                "fusion": n(self.fusion)+n(self.embeddings)+n(self.evidence_router) +
                          (n(self.ir_coarse_aligner) if self.ir_coarse_aligner is not None else 0),
                "register_bus": n(self.register_bus)}

    def _encode(self, x, modality, present):
        b, _, h, w = x.shape
        active = present.nonzero(as_tuple=True)[0]
        if not active.numel():
            return {s: x.new_zeros(b,c,h//(2**(i+2)),w//(2**(i+2))) for i,(s,c) in enumerate(self.channels.items())}
        y = x.index_select(0, active)
        if modality == "rgb":
            encoder = self.backbone.model[:11]
        elif modality == "v511_ir":
            if self.v511_ir_encoder is None:
                raise RuntimeError("V5.1.1 thermal encoder is unavailable")
            encoder = self.v511_ir_encoder
        else:
            encoder = self.aux_encoders[modality]
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

    def _add_depth_metric(self, raw, depth, present):
        """Inject valid metric distance without interpolating invalid pixels."""
        metric_valid = depth[:,2:3] * depth[:,3:4] * present[:,None,None,None]
        absolute = depth[:,1:2] * metric_valid
        metric = torch.cat((absolute,
                            torch.log1p(20*absolute.float())/math.log(21),
                            depth[:,2:3], metric_valid), 1)
        for i, s in enumerate(SCALES):
            values, fraction = masked_pool(metric, metric_valid, raw[s].shape[-2:])
            gain = (.5 * self.metric_gain_logit[i].sigmoid()
                    if self.metric_gain_logit is not None else raw[s].new_tensor(.1))
            raw[s] = raw[s] + gain*self.metric_encoder[s](values) * (fraction > 0)
        return raw, absolute, metric_valid

    def independent_branch_prediction(self, name, rgb=None, ir=None, depth=None, keep=None):
        """Predict from one auxiliary sensor with no RGB/fusion information."""
        if name not in self.independent_aux:
            raise ValueError(f"independent auxiliary detector is unavailable: {name}")
        source = rgb if name == "rgb" else (ir if name == "ir" else depth)
        if source is None:
            raise ValueError(f"{name} input is required")
        b = source.shape[0]
        present = torch.as_tensor(
            (keep or {}).get(name, torch.ones(b, device=source.device)),
            device=source.device).reshape(b).bool()
        if name == "rgb":
            present &= source[:, :3].flatten(1).any(1)
            raw = self._encode(source, "rgb", present)
        elif name == "dep":
            present &= depth[:,2].flatten(1).any(1)
            raw = self._encode(depth[:,:1], "dep", present)
            raw, _, _ = self._add_depth_metric(raw, depth, present)
        else:
            raw = self._encode(
                ir, "v511_ir" if self.v511_ir_encoder is not None else "ir", present)
        return self.independent_aux[name](raw), present

    def alignment_prediction(self, rgb, ir, quality=None, keep=None):
        """Stage-A A1 forward without entering the fused detector route."""
        if self.ir_coarse_aligner is None or ir is None:
            self._ir_affine_prediction = self._ir_affine_confidence = None
            return None, None
        b = rgb.shape[0]
        rgb_present = torch.as_tensor(
            (keep or {}).get("rgb", torch.ones(b, device=rgb.device)),
            device=rgb.device).reshape(b).bool()
        ir_present = torch.as_tensor(
            (keep or {}).get("ir", torch.ones(b, device=ir.device)),
            device=ir.device).reshape(b).bool()
        rgb_raw = self._encode(rgb, "rgb", rgb_present)
        ir_raw = self._encode(
            ir, "v511_ir" if self.v511_ir_encoder is not None else "ir", ir_present)
        shape = rgb_raw["p4"].shape[-2:]
        rgb_mask = rgb_present[:, None, None, None].to(rgb.dtype).expand(-1, 1, *shape)
        ir_mask = ir_present[:, None, None, None].to(ir.dtype).expand(-1, 1, *shape)
        if quality and "availability" in quality:
            availability = F.interpolate(quality["availability"].float(), shape, mode="nearest")
            rgb_mask = rgb_mask * availability[:, 0:1]
            ir_mask = ir_mask * availability[:, 1:2]
        rgb_common = self.embeddings["p4"][0](rgb_raw["p4"], rgb_mask)[0]
        ir_embedding = (self.v511_ir_embeddings["p4"]
                        if self.v511_ir_embeddings is not None
                        else self.embeddings["p4"][1])
        ir_common = ir_embedding(ir_raw["p4"], ir_mask)[0]
        if isinstance(self.ir_coarse_aligner, ConditionalIRAligner):
            prediction, confidence = self.ir_coarse_aligner(
                rgb_common, ir_common, rgb_mask, ir_mask,
                None if not quality else quality.get("ir"))
        else:
            prediction, confidence = self.ir_coarse_aligner(
                rgb_common, ir_common, rgb_mask, ir_mask)
        self._ir_affine_prediction, self._ir_affine_confidence = prediction, confidence
        return prediction, confidence

    def affine_supervision_loss(self, target, supervised, target_confidence=None):
        zero = next(self.parameters()).new_zeros((), dtype=torch.float32)
        if self._ir_affine_prediction is None:
            return zero
        target = target.float()
        supervised = supervised.float().reshape(-1).clamp(0, 1)
        target_confidence = (supervised if target_confidence is None else
                             target_confidence.float().reshape(-1).clamp(0, 1))
        error = F.smooth_l1_loss(
            self._ir_affine_prediction.float(), target, reduction="none", beta=.10).mean(1)
        weight = supervised * target_confidence.clamp_min(.05)
        geometry = (error * weight).sum() / weight.sum().clamp_min(1)
        predicted = self._ir_affine_confidence.float().reshape(-1).clamp(1e-5, 1-1e-5)
        return geometry + .05 * F.binary_cross_entropy(predicted, target_confidence)

    def forward(self, rgb, ir=None, depth=None, quality=None, prior=None, keep=None):
        if self.auxiliary_eval_branch is not None:
            return self.independent_branch_prediction(
                self.auxiliary_eval_branch, rgb=rgb, ir=ir, depth=depth, keep=keep)[0]
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
        raw[2], absolute, metric_valid = self._add_depth_metric(raw[2], depth, present[:,2])
        thermal_raw = (self._encode(ir, "v511_ir", present[:, 1])
                       if self.v511_ir_encoder is not None else raw[1])
        masks, commons, privates, reliabilities = {}, {}, {}, {}
        auxiliary = rgb.new_zeros((), dtype=torch.float32)
        # Depth validity and reliability are different concepts.  The hard mask
        # removes invalid distance samples.  V4.2 reliability only reflects how
        # much valid support exists nearby; a real object-boundary depth jump is
        # useful geometry and must not be treated as sensor failure.
        valid_d = depth[:,2:3].float()
        reliability_d = depth_reliability_map(
            valid_d, absolute, metric_valid, self.cfg.fusion.depth_reliability)
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
        thermal_common, thermal_private = {}, {}
        if self.v511_ir_embeddings is not None:
            for s in SCALES:
                thermal_common[s], thermal_private[s], loss = self.v511_ir_embeddings[s](
                    thermal_raw[s], masks[s][1])
                auxiliary = auxiliary + loss / 16
        else:
            thermal_common = {s: commons[s][1] for s in SCALES}
            thermal_private = {s: privates[s][1] for s in SCALES}
        ir_affine_flow, ir_affine_conf = {}, None
        if self.ir_coarse_aligner is not None:
            if isinstance(self.ir_coarse_aligner, ConditionalIRAligner):
                raw_affine, ir_affine_prediction_conf = self.ir_coarse_aligner(
                    commons["p4"][0], thermal_common["p4"], masks["p4"][0], masks["p4"][1],
                    None if not quality else quality.get("ir"))
            else:
                raw_affine, ir_affine_prediction_conf = self.ir_coarse_aligner(
                    commons["p4"][0], commons["p4"][1], masks["p4"][0], masks["p4"][1])
            # V4.7 must be an exact V4.4 function at migration.  The affine
            # regressor starts at zero motion, so no local IR warp is accepted
            # until supervised artificial transforms teach non-zero geometry.
            if self.cfg.fusion.fusion_strategy == "v47_trusted_evidence_v1":
                motion = torch.tanh(4 * raw_affine.float().abs().mean(1, keepdim=True))
                ir_affine_conf = ir_affine_prediction_conf * motion.to(ir_affine_prediction_conf.dtype)
            else:
                ir_affine_conf = ir_affine_prediction_conf
            max_angle = math.radians(float(self.cfg.fusion.ir_affine_max_degrees))
            canvas_h, canvas_w = getattr(self, "infer_canvas", (h, w))
            if self.cfg.fusion.fusion_strategy == "v48_embedding_complement_v1":
                # The observed physical defect is irregular IR tilt.  V4.8 does
                # not invent unseen translation/scale and learns rotation only.
                physical = torch.stack((
                    raw_affine[:, 0] * max_angle,
                    raw_affine[:, 1] * 0,
                    raw_affine[:, 2] * 0,
                    raw_affine[:, 3] * 0), 1)
            else:
                physical = torch.stack((
                    raw_affine[:, 0] * max_angle,
                    raw_affine[:, 1] * float(self.cfg.fusion.ir_affine_max_shift) / float(canvas_w),
                    raw_affine[:, 2] * float(self.cfg.fusion.ir_affine_max_shift) / float(canvas_h),
                    raw_affine[:, 3] * float(self.cfg.fusion.ir_affine_max_scale)), 1)
            for s in SCALES:
                ir_affine_flow[s] = affine_flow(physical, raw[0][s].shape[-2:])
            self._ir_affine_prediction = raw_affine
            self._ir_affine_confidence = ir_affine_prediction_conf
        else:
            self._ir_affine_prediction = None
            self._ir_affine_confidence = None
        aligned, flows, confidence = {}, {}, {}
        previous = [None,None]
        for s in reversed(SCALES):
            shape = raw[0][s].shape[-2:]
            flows[s], confidence[s] = [], [masks[s][0]]
            for m in range(1,3):
                # RGB/IR are captured on the same image grid.  Their learned
                # descriptor similarity is low because IR is not RGB texture,
                # not because the pixels are geometrically unmatched.  Keep IR
                # on the nominal identity grid; Depth retains residual matching
                # because its invalid boundaries can shift local support.
                if (m == 1 and self.ir_coarse_aligner is not None):
                    coarse = ir_affine_flow[s]
                    coarse_conf = ir_affine_conf[:, :, None, None] * masks[s][m].float()
                    # Only P2/P3 need local correction after the image-level
                    # affine.  At coarse scales, a learned local warp is more
                    # likely to confuse cross-modal appearance with geometry.
                    if (s in (("p3", "p4") if
                              self.cfg.fusion.fusion_strategy in (
                                  "v5_ir_quality_evidence", "v511_conditional_ir_v1")
                              else ("p2", "p3")) and
                            self.cfg.fusion.fusion_strategy != "v48_embedding_complement_v1"):
                        scene = (None if not quality or "scene_id" not in quality else
                                 F.interpolate(quality["scene_id"].float(), shape, mode="nearest"))
                        match_key = (thermal_common[s]
                                     if m == 1 and self.cfg.fusion.fusion_strategy ==
                                     "v511_conditional_ir_v1" else commons[s][m])
                        flow, conf = self.matchers[s][m-1](
                            commons[s][0], match_key, masks[s][0], masks[s][m],
                            coarse, scene)
                        if self.cfg.fusion.fusion_strategy in (
                                "v47_trusted_evidence_v1", "v48_embedding_complement_v1",
                                "v5_ir_quality_evidence", "v511_conditional_ir_v1"):
                            # Local correction is conditional on the image-level
                            # detector accepting that this sample is misaligned.
                            conf = conf * coarse_conf
                        else:
                            conf = torch.maximum(conf, .25 * coarse_conf)
                    else:
                        flow, conf = coarse, coarse_conf
                elif (m == 1 and
                      self.cfg.fusion.alignment_mode == "identity_residual_v2"):
                    flow = raw[0][s].new_zeros(raw[0][s].shape[0], 2, *shape,
                                                dtype=torch.float32)
                    conf = masks[s][m].float()
                elif s == "p2" and not self.cfg.fusion.p2_match_refine:
                    flow = resize_flow(previous[m-1],shape)
                    conf = F.interpolate(confidence["p3"][m],shape,mode="bilinear",align_corners=False)
                else:
                    scene = None if not quality or "scene_id" not in quality else F.interpolate(quality["scene_id"].float(),shape,mode="nearest")
                    flow,conf = self.matchers[s][m-1](commons[s][0],commons[s][m],masks[s][0],masks[s][m],previous[m-1],scene)
                if (self.cfg.fusion.alignment_mode == "legacy_gate_v1" and
                        self.cfg.fusion.match_floor > 0):
                    # 只抬下限，不改变匹配置信度的排序：真实高置信对应仍保留原值。
                    conf = conf.clamp_min(float(self.cfg.fusion.match_floor))
                # Without RGB observations use nominal geometric coordinates, not hallucinated matches.
                flow = flow * masks[s][0]
                previous[m-1] = flow
                flows[s].append(flow)
                confidence[s].append(conf)
        state = None
        v511_state = None
        fused, geometry, aligned_common, aligned_masks = {}, {}, {}, {}
        alignment_loss = auxiliary.new_zeros(())
        for s in SCALES:
            own = [raw[m][s] for m in range(3)]
            state = self.register_bus(s,own,masks[s],state)
            qual = [None if not quality or m not in quality else F.interpolate(quality[m].float(),own[0].shape[-2:],mode="bilinear",align_corners=False) for m in MODES]
            raw_qual = list(qual)
            plugin_raw = list(own)
            plugin_raw[1] = thermal_raw[s]
            plugin_raw_common = list(commons[s])
            plugin_raw_common[1] = thermal_common[s]
            plugin_raw_private = list(privates[s])
            plugin_raw_private[1] = thermal_private[s]
            # V4.4's legacy gates were built for three quality channels.  V5
            # keeps that protected base contract and exposes the full nine IR
            # channels only to the new quality-aware plugin.
            base_qual = [None if value is None else value[:, :3] for value in qual]
            scene = None if not quality or "scene_id" not in quality else F.interpolate(quality["scene_id"].float(),own[0].shape[-2:],mode="nearest")
            c,u,mask,rel,values = [commons[s][0]], [privates[s][0]], [masks[s][0]], [reliabilities[s][0]], [own[0]]
            plugin_c, plugin_u = [commons[s][0]], [privates[s][0]]
            plugin_mask, plugin_rel = [masks[s][0]], [reliabilities[s][0]]
            for m in range(1,3):
                flow = flows[s][m-1]
                vm = warp(masks[s][m],flow).clamp(0,1)
                if scene is not None:
                    vm = vm * ((warp(scene,flow)-scene).abs()<.01)
                conf = confidence[s][m]
                if self.cfg.fusion.alignment_mode == "identity_residual_v2":
                    # The data are nominally registered.  Confidence determines
                    # how much residual motion to accept, never whether the
                    # modality exists.  All branches use the same interpolation.
                    am = (masks[s][m].float() + conf.float() *
                          (vm.float() - masks[s][m].float())).clamp(0, 1)
                    ac = identity_residual_align(commons[s][m], flow, conf, vm) * am
                    au = identity_residual_align(privates[s][m], flow, conf, vm) * am
                    ar = identity_residual_align(reliabilities[s][m], flow, conf, vm).clamp(0, 1) * am
                    av = identity_residual_align(own[m] * masks[s][m], flow, conf, vm) * am
                    if qual[m] is not None:
                        qual[m] = identity_residual_align(
                            qual[m] * masks[s][m], flow, conf, vm) * am
                elif self.cfg.fusion.alignment_mode == "legacy_gate_v1":
                    am = vm
                    ac = warp(commons[s][m], flow) * vm
                    au = warp(privates[s][m], flow) * vm
                    ar = warp(reliabilities[s][m], flow) * vm
                    av = warp(own[m], flow) * vm
                    if qual[m] is not None:
                        qual[m] = warp(qual[m], flow) * vm
                else:
                    raise ValueError(f"unknown alignment mode: {self.cfg.fusion.alignment_mode}")
                # V4.8 keeps the complete V4.4 route on the nominal IR grid.
                # The corrected IR is visible only to the zero-initialized
                # additive plugin, so a wrong angle cannot corrupt the base.
                if (self.cfg.fusion.fusion_strategy in ("v48_embedding_complement_v1",
                                                        "v5_ir_quality_evidence",
                                                        "v511_conditional_ir_v1")
                        and m == 1):
                    if self.cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
                        if self.cfg.fusion.alignment_mode == "identity_residual_v2":
                            plugin_c.append(identity_residual_align(
                                thermal_common[s], flow, conf, vm) * am)
                            plugin_u.append(identity_residual_align(
                                thermal_private[s], flow, conf, vm) * am)
                        else:
                            plugin_c.append(warp(thermal_common[s], flow) * vm)
                            plugin_u.append(warp(thermal_private[s], flow) * vm)
                    else:
                        plugin_c.append(ac)
                        plugin_u.append(au)
                    plugin_mask.append(am)
                    plugin_rel.append(ar)
                    base_mask = masks[s][m]
                    c.append(commons[s][m] * base_mask)
                    u.append(privates[s][m] * base_mask)
                    mask.append(base_mask)
                    rel.append(reliabilities[s][m] * base_mask)
                    values.append(own[m] * base_mask)
                else:
                    c.append(ac)
                    u.append(au)
                    mask.append(am)
                    rel.append(ar)
                    values.append(av)
                    plugin_c.append(ac)
                    plugin_u.append(au)
                    plugin_mask.append(am)
                    plugin_rel.append(ar)
                weight = (rel[m] if self.cfg.fusion.alignment_mode == "identity_residual_v2"
                          else confidence[s][m] * rel[m]).detach()
                aligned_for_loss = (plugin_c[m]
                                    if self.cfg.fusion.fusion_strategy in (
                                        "v48_embedding_complement_v1",
                                        "v5_ir_quality_evidence",
                                        "v511_conditional_ir_v1")
                                    else c[m])
                similarity = (F.normalize(c[0].detach().float(),dim=1)*
                              F.normalize(aligned_for_loss.float(),dim=1)).sum(1,keepdim=True)
                alignment_loss = alignment_loss + ((1-similarity)*weight).sum()/weight.sum().clamp_min(1)/8
            if self.cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
                semantic_c, semantic_mask = list(c), list(mask)
                semantic_c[1], semantic_mask[1] = plugin_c[1], plugin_mask[1]
                aligned_common[s], aligned_masks[s] = semantic_c, semantic_mask
            else:
                aligned_common[s], aligned_masks[s] = c, mask
            base_match = confidence[s]
            if self.cfg.fusion.fusion_strategy in ("v48_embedding_complement_v1",
                                                  "v5_ir_quality_evidence",
                                                  "v511_conditional_ir_v1"):
                # The learned V4.4 route treated IR as nominally registered.
                # Rotation confidence belongs only to the new additive plugin.
                base_match = list(confidence[s])
                base_match[1] = masks[s][1].float()
            if self.cfg.fusion.fusion_strategy == "evidence_router_v3":
                fusion_block = self.evidence_router[s]
                fused[s] = fusion_block(values,c,u,mask,confidence[s],rel,state,qual)
            elif self.cfg.fusion.fusion_strategy == "v44_incremental_router_v1":
                # Preserve the complete learned V4.4 route.  The new router is
                # evaluated around that base and its zero-initialized output and
                # context projections make the migration function-preserving.
                base = self.fusion[s](values,c,u,mask,confidence[s],rel,state,qual)
                fusion_block = self.evidence_router[s]
                fused[s] = fusion_block(
                    values,c,u,mask,confidence[s],rel,state,qual,anchor=base)
            elif self.cfg.fusion.fusion_strategy == "v47_trusted_evidence_v1":
                # V4.4 remains the immutable detector route.  The plugin has no
                # unconditional context branch and can write only target-evidence
                # gated, bounded IR/Depth residuals.
                base = self.fusion[s](values,c,u,mask,base_match,rel,state,qual)
                fusion_block = self.evidence_router[s]
                fused[s] = fusion_block(
                    values,c,u,mask,confidence[s],rel,state,qual,anchor=base)
            elif self.cfg.fusion.fusion_strategy == "v48_embedding_complement_v1":
                # V4.4 is the protected detector feature.  Corrected IR common/
                # private embeddings can add a complement; Depth returns a
                # localization-only support tensor used below.
                base = self.fusion[s](values,c,u,mask,base_match,rel,state,qual)
                fusion_block = self.evidence_router[s]
                fused[s], depth_support = fusion_block(
                    plugin_c, plugin_u, plugin_mask, plugin_rel, state, base)
            elif self.cfg.fusion.fusion_strategy == "v5_ir_quality_evidence":
                # V5 keeps the complete V4.4 route on its nominal grid and
                # exposes corrected IR only to the additive quality-gated plugin.
                base = self.fusion[s](values, c, u, mask, base_match, rel, state, base_qual)
                fusion_block = self.evidence_router[s]
                fused[s] = fusion_block(
                    values, plugin_c, plugin_u, plugin_mask, confidence[s],
                    plugin_rel, state, qual, anchor=base)
            elif self.cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
                # The V4.4 fusion remains the exact anchor.  Raw IR stays on
                # its nominal grid in that path; only the new zero-initialized
                # plugin sees the conditional aligned candidate.
                base = self.fusion[s](values, c, u, mask, base_match, rel, state, base_qual)
                fusion_block = self.evidence_router[s]
                fused[s], shared_support, depth_support, next_v511_state = fusion_block(
                    plugin_raw, plugin_raw_common, plugin_c, plugin_raw_private, masks[s],
                    confidence[s], reliabilities[s], state, qual,
                    raw_quality=raw_qual, cross_scale_state=v511_state, anchor=base)
                if s != "p2":
                    v511_state = next_v511_state
            else:
                fusion_block = self.fusion[s]
                fused[s] = fusion_block(values,c,u,mask,confidence[s],rel,state,qual)
            fusion_block.last_health["ir_flow_rms"] = flows[s][0].detach().float().square().mean().sqrt()
            fusion_block.last_health["dep_flow_rms"] = flows[s][1].detach().float().square().mean().sqrt()
            # values are already identity/residual aligned in V4.2, so match must
            # not suppress the high-resolution localization path a second time.
            if self.cfg.fusion.alignment_mode == "identity_residual_v2":
                geometry[s] = values[0]*mask[0] + sum(values[m]*rel[m] for m in (1,2))*.25
            else:
                geometry[s] = values[0]*mask[0] + sum(
                    values[m]*confidence[s][m]*rel[m] for m in (1,2))*.25
            if self.cfg.fusion.fusion_strategy == "v48_embedding_complement_v1":
                geometry[s] = geometry[s] + depth_support
            elif self.cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
                # Start exactly at V4.4, then learn whether replacing its broad
                # auxiliary geometry with trusted support improves regression.
                trusted = values[0] * mask[0]
                trusted = trusted + .25 * shared_support * confidence[s][1]
                if s in ("p4", "p5"):
                    trusted = trusted + .10 * depth_support
                transition = self.v511_localization_transition[
                    SCALES.index(s)].tanh().to(geometry[s].dtype)
                geometry[s] = geometry[s] + transition * (trusted - geometry[s])
        # Keep the two objectives separate.  Stage A benefits from information
        # preservation, while a detection fine-tune must be able to anneal both
        # terms instead of silently retaining the old fixed .01/.005 pressure.
        self.embedding_aux_losses = {"reconstruction": auxiliary,
                                     "alignment": alignment_loss}
        self.aux_loss = .01*auxiliary + .005*alignment_loss if self.training else auxiliary.detach()*0
        self.semantic_branch_present = present
        self._semantic_common = aligned_common
        self._semantic_masks = aligned_masks
        self._semantic_flows = flows
        self._evidence_logits = {
            s: list(block.last_evidence_logits)
            for s, block in self.evidence_router.items()
            if hasattr(block, "last_evidence_logits")
        }
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
        if len(self.occlusion_context):
            features[0] = features[0] + self.occlusion_context[0](features[0])
            features[1] = features[1] + self.occlusion_context[1](features[1])
        loc = [x + .15*self.localization_scale[i].tanh()*self.localization[i](geometry[s])
               for i,(s,x) in enumerate(zip(SCALES,features))]
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
    def _objectness_map(targets, size, batch_size, device, dtype):
        """Rasterize complete GT boxes into a spatial object-evidence target."""
        h, w = int(size[0]), int(size[1])
        target = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)
        if not targets["batch_idx"].numel():
            return target
        for bi, box in zip(targets["batch_idx"].long(), targets["bboxes"].float()):
            cx, cy, bw, bh = box
            x0 = max(0, min(w - 1, int(torch.floor((cx - bw / 2) * w).item())))
            y0 = max(0, min(h - 1, int(torch.floor((cy - bh / 2) * h).item())))
            x1 = max(x0 + 1, min(w, int(torch.ceil((cx + bw / 2) * w).item())))
            y1 = max(y0 + 1, min(h, int(torch.ceil((cy + bh / 2) * h).item())))
            target[int(bi), 0, y0:y1, x0:x1] = 1
        return target

    @staticmethod
    def _object_vectors(feature, valid, targets, grid_size=3, min_observed=.5):
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
        return F.normalize(vector, dim=1), weights.mean((1, 2, 3)) >= min_observed

    def semantic_regularization(self, targets, alignment_shift, alignment_supervised,
                                ir_affine_target=None, ir_affine_supervised=None,
                                ir_affine_confidence=None):
        """Return unweighted flow/NCE losses for the current forward pass."""
        zero = next(self.parameters()).new_zeros((), dtype=torch.float32)
        result = {"flow": zero, "nce": zero, "ir_affine": zero, "evidence": zero}
        if not self.training or self._semantic_common is None:
            self.last_semantic_losses = {k: v.detach() for k, v in result.items()}
            return result

        if self.cfg.fusion.flow_supervision_weight > 0:
            terms = []
            shift = alignment_shift.float()
            supervised = alignment_supervised.float()[:, None, None, None]
            identity_weight = float(self.cfg.fusion.flow_identity_weight)
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
                sample_weight = supervised + (1 - supervised) * identity_weight
                mask = self._semantic_masks[s][0].float() * sample_weight
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
                    z, ok = self._object_vectors(
                        self._semantic_common[s][m], self._semantic_masks[s][m], targets,
                        min_observed=.30 if m == 2 else .50)
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

        if self.ir_coarse_aligner is not None and self._ir_affine_prediction is not None:
            if self.cfg.fusion.fusion_strategy == "v511_conditional_ir_v1":
                result["ir_affine"] = self.affine_supervision_loss(
                    ir_affine_target, ir_affine_supervised, ir_affine_confidence)
            else:
                target = (torch.zeros_like(self._ir_affine_prediction) if ir_affine_target is None
                          else ir_affine_target.float())
                supervised = (target.new_zeros(target.shape[0]) if ir_affine_supervised is None
                              else ir_affine_supervised.float().reshape(-1))
                sample_weight = (supervised + (1 - supervised) *
                                 float(self.cfg.fusion.ir_affine_identity_weight))
                error = F.smooth_l1_loss(
                    self._ir_affine_prediction.float(), target, reduction="none", beta=.10).mean(1)
                affine_loss = (error * sample_weight).sum() / sample_weight.sum().clamp_min(1)
                if supervised.any():
                    conf = self._ir_affine_confidence.float().reshape(-1).clamp_min(1e-5)
                    affine_loss = affine_loss + .02 * (
                        -conf.log() * supervised).sum() / supervised.sum().clamp_min(1)
                result["ir_affine"] = affine_loss

        if (self.cfg.fusion.fusion_strategy in ("v47_trusted_evidence_v1",
                                                "v5_ir_quality_evidence",
                                                "v511_conditional_ir_v1") and
                self._evidence_logits):
            evidence_terms = []
            batch_size = int(targets.get("batch_size", 0))
            for s, logits_all in self._evidence_logits.items():
                expected = (1 if self.cfg.fusion.fusion_strategy ==
                            "v511_conditional_ir_v1" else 2)
                if len(logits_all) != expected:
                    continue
                target = self._objectness_map(
                    targets, logits_all[0].shape[-2:], batch_size,
                    logits_all[0].device, logits_all[0].dtype)
                slots = (1,) if expected == 1 else (1, 2)
                for slot, logits in zip(slots, logits_all):
                    valid = self._semantic_masks[s][slot].to(logits.dtype)
                    positive = (target * valid).sum()
                    negative = ((1 - target) * valid).sum()
                    pos_weight = (negative / positive.clamp_min(1)).clamp(1, 20).detach()
                    bce = F.binary_cross_entropy_with_logits(
                        logits.float(), target.float(), reduction="none",
                        pos_weight=pos_weight.float())
                    bce = (bce * valid.float()).sum() / valid.float().sum().clamp_min(1)
                    probability = logits.float().sigmoid() * valid.float()
                    intersection = (probability * target.float()).sum()
                    dice = 1 - (2 * intersection + 1) / (
                        probability.sum() + (target.float() * valid.float()).sum() + 1)
                    evidence_terms.append(bce + .25 * dice)
            result["evidence"] = (torch.stack(evidence_terms).mean()
                                  if evidence_terms else zero)

        self.last_semantic_losses = {k: v.detach() for k, v in result.items()}
        return result
