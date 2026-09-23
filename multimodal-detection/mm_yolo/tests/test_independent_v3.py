import copy
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MM))
from config import default_config
from model import MMYOLO, save_mm_checkpoint, load_mm_checkpoint
from train import (build_optimizer, set_encoder_frozen, apply_bn_policy, make_targets,
                   subset_detection_batch, set_aux_adaptation_mode,
                   set_anchored_joint_mode, set_independent_aux_mode,
                   set_residual_fusion_mode, enable_trainable_defaults,
                   adapt_depth_checkpoint_state, reset_rgb_identity_residuals,
                   reset_incremental_router_additions, reset_v48_additions,
                   fusion_health_measurements)
from data import (MMDataset, AugCfg, collate, scheduled_aug, centered_affine_M,
                  _target_occlusion)
from independent_fusion import (warp, resize_flow, identity_residual_align,
                                LocalCorrespondence, affine_flow,
                                SpatialEvidenceRouter, TrustedEvidenceRouter,
                                EmbeddingComplementPlugin)
from independent_model import depth_reliability_map
from ultralytics.utils.loss import v8DetectionLoss


def config(checkpoint=False):
    c = default_config()
    c.fusion.architecture = "independent_p2_memory_v3"
    c.fusion.bus_dim = 128
    c.encoder.share_tier = "a"
    c.encoder.metric_branch = True
    c.encoder.checkpoint_encoder = checkpoint
    c.depth_resampling = "nearest_valid_v2"
    return c


class IndependentV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(4)
        rgb = torch.rand(2,3,64,96)
        depth = torch.rand(2,4,64,96)
        depth[:,2:] = 1
        return rgb, rgb[:,:1].clone(),depth

    def test_independence_pretraining_optimizer_and_freeze(self):
        m = MMYOLO(config())
        enable_trainable_defaults(m)
        rgb,ir,dep = m.encoder_modules()
        sets = [{p.data_ptr() for p in module.parameters()} for module in (rgb,ir,dep)]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertTrue(torch.equal(rgb[0].conv.weight.sum(1,keepdim=True),ir[0].conv.weight))
        self.assertTrue(torch.equal(rgb[6].state_dict()["cv1.conv.weight"],dep[6].state_dict()["cv1.conv.weight"]))
        opt = build_optimizer(m,.0004,.5)
        ids = {id(p) for g in opt.param_groups if g["role"]=="encoder" for p in g["params"]}
        self.assertTrue(all(id(p) in ids for p in dep.parameters()))
        set_encoder_frozen(m,True)
        apply_bn_policy(m,"adaptive",True)
        self.assertFalse(ir[0].bn.training)
        self.assertFalse(any(p.requires_grad for p in ir.parameters()))
        set_encoder_frozen(m,False)
        self.assertTrue(all(p.requires_grad for p in ir.parameters()))
        self.assertTrue(any(p.requires_grad for p in m.backbone.model[13].parameters()))

    def test_real_loss_gradients_checkpoint_bn_and_p2(self):
        m = MMYOLO(config(checkpoint=True)).train()
        with torch.no_grad():
            for block in m.fusion.values():
                block.residual_scale[1:].fill_(.2)
            m.neck_memory.residual_scale.fill_(.2)
            m.localization_scale.fill_(.2)
        rgb,ir,dep = self.inputs()
        before = int(m.aux_encoders["dep"][0].bn.num_batches_tracked)
        out = m(rgb,ir,dep)
        self.assertEqual([tuple(x.shape[-2:]) for x in out["feats"]],[(16,24),(8,12),(4,6),(2,3)])
        batch = {"boxes":[torch.tensor([[0.,.5,.5,.2,.2]])]*2}
        loss,_ = v8DetectionLoss(m)(out,make_targets(batch,(64,96),torch.device("cpu")))
        (loss.sum()/2+m.aux_loss).backward()
        self.assertEqual(int(m.aux_encoders["dep"][0].bn.num_batches_tracked)-before,1)
        for branch in (m.aux_encoders["ir"],m.aux_encoders["dep"],m.metric_encoder,m.embeddings,m.register_bus,m.p2_neck):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_masking_memory_reset_and_strict_roundtrip(self):
        m = MMYOLO(config()).eval()
        rgb,ir,dep = self.inputs()
        keep = {"rgb":torch.ones(2),"ir":torch.ones(2),"dep":torch.zeros(2)}
        with torch.no_grad():
            a = m(rgb,ir,dep,keep=keep)[0]
            b = m(rgb,ir,dep*5,keep=keep)[0]
            self.assertTrue(torch.equal(a,b))
            c = m(rgb,ir,dep)[0]
            d = m(rgb,ir,dep)[0]
            self.assertTrue(torch.equal(c,d))
        # Training diagnostics are detached/cleared before deepcopy in this test.
        m.aux_loss = torch.tensor(0.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            save_mm_checkpoint(path,m,meta={"modalities":["rgb","ir","dep"],"canvas":[64,96]})
            restored,_ = load_mm_checkpoint(path)
            with torch.no_grad():
                self.assertTrue(torch.equal(c,restored.eval()(rgb,ir,dep)[0]))

    def test_mosaic_depth_units_scene_ids_masks_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("visible","infrared","depth"):
                (root/m).mkdir()
            samples = []
            for i in range(4):
                name = f"{i}.png"
                for m in ("visible","infrared"):
                    cv2.imwrite(str(root/m/name),np.full((48,80,3),90+i,np.uint8))
                cv2.imwrite(str(root/"depth"/name),np.full((24,40),1000*(i+1),np.uint16))
                samples.append({"stem":str(i),"files":{m:name for m in ("visible","infrared","depth")},"boxes":np.array([[0,.5,.5,.25,.25]],np.float32)})
            aug = AugCfg(mosaic_p=1,scale_range=(1,1),translate=0,misalign_px=0,degrade_p=0,
                         rgb_drop_p=0,aux_drop_p=0,depth_resampling="nearest_valid_v2",total_epochs=10,close_aug_frac=.2)
            ds = MMDataset(root,samples,imgsz=(64,96),aug=aug)
            a,b = ds[(0,0,1)],ds[(0,0,1)]
            self.assertTrue(torch.equal(a["rgb"],b["rgb"]))
            self.assertEqual(set(a["quality"]["scene_id"].unique().tolist()),{1.,2.,3.,4.})
            values = a["depth"][1][a["depth"][2]>0].unique().numpy()
            self.assertTrue(all(np.isclose(v,np.array([.05,.1,.15,.2])).any() for v in values))
            batch = collate([a,b])
            self.assertEqual(batch["quality"]["availability"].shape,(2,3,64,96))
            self.assertEqual(batch["quality"]["scene_id"].shape,(2,1,64,96))
            self.assertEqual(scheduled_aug(aug,8).mosaic_p,0.)

    def test_warp_direction_and_flow_scaling(self):
        x = torch.arange(6.).view(1,1,1,6).expand(1,1,4,6)
        flow = torch.zeros(1,2,4,6)
        flow[:,0] = 1
        self.assertTrue(torch.allclose(warp(x,flow)[...,:-1],x[...,1:],atol=1e-6))
        self.assertTrue(torch.allclose(resize_flow(flow,(8,12))[:,0],torch.full((1,8,12),2.)))

    def test_identity_residual_alignment_endpoints(self):
        x = torch.arange(6.).view(1,1,1,6).expand(1,1,4,6)
        flow = torch.zeros(1,2,4,6)
        flow[:,0] = 1
        zero = torch.zeros(1,1,4,6)
        one = torch.ones_like(zero)
        self.assertTrue(torch.equal(identity_residual_align(x,flow,zero),x))
        self.assertTrue(torch.allclose(identity_residual_align(x,flow,one),warp(x,flow)))

    def test_affine_flow_matches_opencv_source_to_destination_geometry(self):
        h, w = 40, 64
        source = np.zeros((h, w), np.float32)
        source[11:20, 17:31] = 1
        cases = ((0., 5., -3., 0.), (4., 0., 0., 0.), (0., 0., 0., .03))
        for angle, dx, dy, ds in cases:
            M = centered_affine_M((h, w), angle, 1 + ds, dx, dy)
            distorted = cv2.warpAffine(source, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            params = torch.tensor([[np.deg2rad(angle), dx / w, dy / h, ds]])
            restored = warp(torch.from_numpy(distorted)[None, None],
                            affine_flow(params, (h, w)))[0, 0].numpy()
            # Rotation/scale interpolation differs slightly between OpenCV and
            # grid_sample.  The interior still has to recover the same object.
            self.assertLess(float(np.mean(np.abs(restored[6:-6, 6:-6] -
                                                 source[6:-6, 6:-6]))), .035)

    def test_spatial_router_is_rgb_identity_then_aux_outputs_receive_gradients(self):
        torch.manual_seed(8)
        router = SpatialEvidenceRouter(32, dim=16, memory_dim=24).train()
        raw = [torch.randn(2, 32, 8, 12) for _ in range(3)]
        common = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        private = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        valid = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        match = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        reliable = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        memory = torch.randn(2, 4, 3, 24)
        a = router(raw, common, private, valid, match, reliable, memory)
        changed = [raw[0], raw[1] * 7, raw[2] - 9]
        b = router(changed, [common[0], common[1] * 5, common[2] - 4],
                   [private[0], private[1] - 6, private[2] * 3],
                   valid, match, reliable, memory)
        self.assertTrue(torch.equal(a, raw[0]))
        self.assertTrue(torch.equal(a, b))
        a.square().mean().backward()
        for output in router.outputs:
            self.assertIsNotNone(output[-1].weight.grad)
            self.assertGreater(float(output[-1].weight.grad.abs().sum()), 0.)

    def test_trusted_router_is_identity_and_uses_target_evidence(self):
        torch.manual_seed(9)
        router = TrustedEvidenceRouter(32, dim=16, memory_dim=24).train()
        raw = [torch.randn(2, 32, 8, 12) for _ in range(3)]
        common = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        private = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        valid = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        match = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        reliable = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        memory = torch.randn(2, 4, 3, 24)
        anchor = torch.randn(2, 32, 8, 12)
        out = router(raw, common, private, valid, match, reliable, memory, anchor=anchor)
        self.assertTrue(torch.equal(out, anchor))
        self.assertTrue(all(torch.allclose(x.sigmoid(), torch.full_like(x, .1), atol=1e-5)
                            for x in router.last_evidence_logits))
        out.square().mean().backward()
        self.assertTrue(all(output[-1].weight.grad is not None and
                            output[-1].weight.grad.abs().sum() > 0
                            for output in router.outputs))

    def test_v48_plugin_is_identity_has_gradients_and_masks_depth(self):
        torch.manual_seed(10)
        plugin = EmbeddingComplementPlugin(32, dim=16, memory_dim=24).train()
        common = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        private = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        valid = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        reliable = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        memory = torch.randn(2, 4, 3, 24)
        anchor = torch.randn(2, 32, 8, 12)
        fused, geometry = plugin(common, private, valid, reliable, memory, anchor)
        self.assertTrue(torch.equal(fused, anchor))
        self.assertEqual(float(geometry.abs().max()), 0.)
        # At exact zero, geometry.square() has a zero derivative.  A detector's
        # regression path supplies a non-zero upstream gradient, modelled here
        # by a linear geometry term.
        (fused.square().mean() + geometry.mean()).backward()
        for output in (plugin.ir_shared, plugin.ir_private, plugin.depth_support):
            self.assertIsNotNone(output[-1].weight.grad)
            self.assertGreater(float(output[-1].weight.grad.abs().sum()), 0.)
        plugin.zero_grad(set_to_none=True)
        valid[2].zero_()
        _, a = plugin(common, private, valid, reliable, memory, anchor)
        common[2].mul_(100)
        private[2].mul_(-100)
        _, b = plugin(common, private, valid, reliable, memory, anchor)
        self.assertTrue(torch.equal(a, b))

    def test_health_logging_accepts_named_scalar_plugin_stats(self):
        class Block:
            last_health = {"ir_route_ratio": torch.tensor(.02)}
            last_stats = {
                "ir_shared_mix": torch.tensor(.6),
                "legacy": (torch.tensor(.7), torch.tensor(.3)),
            }

        values = fusion_health_measurements(Block())
        self.assertEqual(set(values), {
            "ir_route_ratio", "ir_shared_mix", "legacy_match", "legacy_gate"})
        self.assertAlmostEqual(float(values["ir_shared_mix"]), .6, places=5)

    def test_target_occlusion_is_reproducible_and_keeps_labels_external(self):
        rgb = np.full((64, 96, 3), 120, np.uint8)
        ir = np.full_like(rgb, 80)
        boxes = np.array([[2, .5, .5, .5, .5]], np.float32)
        original = boxes.copy()
        import random
        a = _target_occlusion(rgb, ir, boxes, random.Random(17))
        b = _target_occlusion(rgb, ir, boxes, random.Random(17))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        self.assertTrue(np.array_equal(boxes, original))
        self.assertTrue(np.any(a[0] != rgb) ^ np.any(a[1] != ir))

    def test_v44_to_v45_migration_is_explicit_and_strict(self):
        old_cfg = config()
        old = MMYOLO(old_cfg)
        new_cfg = config()
        new_cfg.fusion.fusion_strategy = "evidence_router_v3"
        new_cfg.fusion.ir_coarse_align = True
        new = MMYOLO(new_cfg)
        migrated, changed = adapt_depth_checkpoint_state(old.state_dict(), new)
        self.assertTrue(changed)
        new.load_state_dict(migrated, strict=True)
        broken = dict(migrated)
        broken.pop("backbone.model.0.conv.weight")
        broken, _ = adapt_depth_checkpoint_state(broken, new)
        with self.assertRaises(RuntimeError):
            new.load_state_dict(broken, strict=True)

    def test_v44_incremental_router_preserves_loaded_v44_function(self):
        old_cfg = config()
        old_cfg.fusion.alignment_mode = "identity_residual_v2"
        old_cfg.fusion.depth_reliability = "valid_support_v2"
        old_cfg.fusion.p2_match_refine = True
        old = MMYOLO(old_cfg).train()
        new_cfg = config()
        new_cfg.fusion.fusion_strategy = "v44_incremental_router_v1"
        new_cfg.fusion.alignment_mode = "identity_residual_v2"
        new_cfg.fusion.depth_reliability = "valid_support_v2"
        new_cfg.fusion.p2_match_refine = True
        new = MMYOLO(new_cfg).train()
        state = dict(old.state_dict())
        migrated, changed = adapt_depth_checkpoint_state(state, new)
        self.assertTrue(changed)
        new.load_state_dict(migrated, strict=True)
        reset_incremental_router_additions(new)
        old.infer_canvas = new.infer_canvas = (64, 96)
        rgb, ir, dep = self.inputs()
        with torch.no_grad():
            a = old(rgb, ir, dep)
            b = new(rgb, ir, dep)
        self.assertTrue(torch.equal(a["boxes"], b["boxes"]))
        self.assertTrue(torch.equal(a["scores"], b["scores"]))
        self.assertTrue(torch.equal(new._semantic_flows["p3"][0],
                                    torch.zeros_like(new._semantic_flows["p3"][0])))

    def test_v47_preserves_v44_and_freezes_original_route(self):
        old_cfg = config()
        old_cfg.fusion.alignment_mode = "identity_residual_v2"
        old_cfg.fusion.depth_reliability = "valid_support_v2"
        old_cfg.fusion.p2_match_refine = True
        old = MMYOLO(old_cfg).train()
        new_cfg = copy.deepcopy(old_cfg)
        new_cfg.fusion.fusion_strategy = "v47_trusted_evidence_v1"
        new_cfg.fusion.ir_coarse_align = True
        new = MMYOLO(new_cfg).train()
        migrated, changed = adapt_depth_checkpoint_state(old.state_dict(), new)
        self.assertTrue(changed)
        new.load_state_dict(migrated, strict=True)
        reset_incremental_router_additions(new)
        old.infer_canvas = new.infer_canvas = (64, 96)
        rgb, ir, dep = self.inputs()
        with torch.no_grad():
            a = old(rgb, ir, dep)
            b = new(rgb, ir, dep)
        self.assertTrue(torch.equal(a["boxes"], b["boxes"]))
        self.assertTrue(torch.equal(a["scores"], b["scores"]))
        set_residual_fusion_mode(new, downstream_frozen=False)
        trainable = [name for name, p in new.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("evidence_router.", "ir_coarse_aligner.",
                                             "matchers.p2.0.", "matchers.p3.0."))
                            for name in trainable))

    def test_v48_preserves_v44_then_releases_downstream(self):
        old_cfg = config()
        old_cfg.fusion.alignment_mode = "identity_residual_v2"
        old_cfg.fusion.depth_reliability = "valid_support_v2"
        old_cfg.fusion.p2_match_refine = True
        old = MMYOLO(old_cfg).train()
        new_cfg = copy.deepcopy(old_cfg)
        new_cfg.fusion.fusion_strategy = "v48_embedding_complement_v1"
        new_cfg.fusion.ir_coarse_align = True
        new_cfg.fusion.branch_aux_weights = (0., .04, 0.)
        new = MMYOLO(new_cfg).train()
        migrated, changed = adapt_depth_checkpoint_state(old.state_dict(), new)
        self.assertTrue(changed)
        new.load_state_dict(migrated, strict=True)
        reset_v48_additions(new)
        old.infer_canvas = new.infer_canvas = (64, 96)
        rgb, ir, dep = self.inputs()
        with torch.no_grad():
            a = old(rgb, ir, dep)
            b = new(rgb, ir, dep)
        self.assertTrue(torch.equal(a["boxes"], b["boxes"]))
        self.assertTrue(torch.equal(a["scores"], b["scores"]))
        # A learned, non-zero IR rotation must remain isolated from the
        # protected V4.4 route while the additive plugin is still zero.
        with torch.no_grad():
            new.ir_coarse_aligner.head[-1].bias.copy_(
                torch.tensor((.6, 0., 0., 0., 2.)))
            c = new(rgb, ir, dep)
        self.assertTrue(torch.equal(a["boxes"], c["boxes"]))
        self.assertTrue(torch.equal(a["scores"], c["scores"]))
        set_residual_fusion_mode(new, downstream_frozen=True)
        trainable = [name for name, p in new.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(("evidence_router.", "ir_coarse_aligner."))
                            for name in trainable))
        set_residual_fusion_mode(new, downstream_frozen=False)
        self.assertTrue(any(p.requires_grad for p in new.aux_encoders["ir"].parameters()))
        self.assertTrue(any(p.requires_grad for p in new.backbone.model[13].parameters()))
        self.assertFalse(any(p.requires_grad for p in new.fusion.parameters()))

    def test_v45_full_detection_backward_is_finite_and_aux_sensitive(self):
        c = config(checkpoint=True)
        c.fusion.fusion_strategy = "evidence_router_v3"
        c.fusion.ir_coarse_align = True
        c.fusion.alignment_mode = "identity_residual_v2"
        c.fusion.depth_reliability = "valid_support_v2"
        c.fusion.p2_match_refine = True
        c.fusion.cross_modal_nce_weight = .004
        m = MMYOLO(c).train()
        m.infer_canvas = (64, 96)
        reset_rgb_identity_residuals(m)
        set_residual_fusion_mode(m, downstream_frozen=False)
        rgb, ir, dep = self.inputs()
        with torch.no_grad():
            base = m(rgb, ir, dep)["scores"].clone()
            changed = m(rgb, ir * .1 + .8, dep.roll(5, -1))["scores"]
        self.assertTrue(torch.equal(base, changed))
        out = m(rgb, ir, dep)
        batch = {"boxes":[torch.tensor([[0.,.5,.5,.2,.2]]),
                          torch.tensor([[2.,.4,.4,.3,.2]])]}
        targets = make_targets(batch, (64, 96), torch.device("cpu"))
        det, _ = v8DetectionLoss(m)(out, targets)
        semantic = m.semantic_regularization(
            targets, torch.zeros(2, 2), torch.zeros(2),
            torch.tensor([[.4, -.3, .2, .1], [-.2, .1, -.1, 0.]]),
            torch.ones(2))
        total = det.sum() / 2 + .004 * semantic["nce"] + .08 * semantic["ir_affine"]
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in m.parameters() if p.grad is not None))
        for block in m.evidence_router.values():
            self.assertTrue(all(output[-1].weight.grad is not None and
                                output[-1].weight.grad.abs().sum() > 0
                                for output in block.outputs))
        self.assertGreater(float(m.ir_coarse_aligner.head[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(block[-1].weight.grad is not None and
                            block[-1].weight.grad.abs().sum() > 0
                            for block in m.occlusion_context))

    def test_v42_depth_edges_remain_reliable(self):
        valid = torch.ones(2,1,64,96)
        flat = torch.full_like(valid,.2)
        edged = flat.clone()
        edged[:,:,:,48:] = .8
        metric = valid.clone()
        flat_rel = depth_reliability_map(valid,flat,metric,"valid_support_v2")
        edge_rel = depth_reliability_map(valid,edged,metric,"valid_support_v2")
        # All depth pixels are valid. A large metric edge therefore remains
        # fully reliable instead of falling to the old 0.35 floor.
        self.assertTrue(torch.equal(flat_rel,edge_rel))
        self.assertGreater(float(edge_rel.mean()), .98)

    def test_all_invalid_correspondence_has_finite_zero_gradients(self):
        match = LocalCorrespondence()
        query = torch.randn(2,32,8,12,requires_grad=True)
        key = torch.randn_like(query,requires_grad=True)
        valid = torch.zeros(2,1,8,12)
        flow,conf = match(query,key,torch.ones_like(valid),valid)
        (flow.square().mean()+conf.sum()).backward()
        self.assertEqual(float(conf.sum()),0.)
        self.assertTrue(torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all())

    def test_semantic_heads_flow_and_object_nce_backward(self):
        c = config()
        c.fusion.branch_aux_weight = .15
        c.fusion.flow_supervision_weight = .05
        c.fusion.cross_modal_nce_weight = .03
        c.fusion.p2_match_refine = True
        m = MMYOLO(c).train()
        m.infer_canvas = (64,96)
        rgb,ir,dep = self.inputs()
        out = m(rgb,ir,dep)
        batch = {"boxes":[torch.tensor([[0.,.5,.5,.2,.2]]),
                          torch.tensor([[1.,.4,.4,.3,.2]])]}
        targets = make_targets(batch,(64,96),torch.device("cpu"))
        semantic = m.semantic_regularization(
            targets, torch.tensor([[4.,0.],[0.,-4.]]), torch.ones(2))
        self.assertTrue(torch.isfinite(semantic["flow"]) and torch.isfinite(semantic["nce"]))
        active = m.semantic_branch_present[:,1]
        pred,tgt = subset_detection_batch(m.semantic_branch_prediction("ir"),targets,active)
        branch,_ = v8DetectionLoss(m)(pred,tgt)
        total = out["scores"].sum()*0 + branch.sum()/2 + semantic["flow"] + semantic["nce"]
        total.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in m.semantic_detect.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in m.matchers["p2"].parameters()))

    def test_aux_adaptation_freezes_rgb_detector_but_trains_ir_depth(self):
        c = config()
        c.fusion.branch_aux_weights = (0.,.3,.25)
        c.fusion.alignment_mode = "identity_residual_v2"
        c.fusion.depth_reliability = "valid_support_v2"
        m = MMYOLO(c).train()
        with torch.no_grad():
            for block in m.fusion.values():
                block.residual_scale[1:].fill_(.2)
        set_aux_adaptation_mode(m)
        rgb,ir,dep = self.inputs()
        out = m(rgb,ir,dep)
        out["scores"].mean().backward()
        self.assertFalse(any(p.grad is not None for p in m.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in m.aux_encoders["ir"].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in m.aux_encoders["dep"].parameters()))

    def test_anchored_joint_keeps_rgb_reference_and_uses_role_lrs(self):
        c = config()
        c.fusion.branch_aux_weights = (0.,.03,.025)
        m = MMYOLO(c).train()
        enable_trainable_defaults(m)
        set_anchored_joint_mode(m, detector_frozen=True)
        self.assertFalse(any(p.requires_grad for p in m.backbone.model[:11].parameters()))
        self.assertFalse(any(p.requires_grad for p in m.embeddings["p3"][0].parameters()))
        self.assertTrue(any(p.requires_grad for p in m.embeddings["p3"][1].parameters()))
        self.assertFalse(any(p.requires_grad for p in m.fusion["p3"].query.parameters()))
        self.assertTrue(any(p.requires_grad for p in m.fusion["p3"].gates[1].parameters()))
        self.assertFalse(any(p.requires_grad for p in m.backbone.model[13].parameters()))
        self.assertTrue(any(p.requires_grad for p in m.p2_neck.parameters()))
        self.assertTrue(any(p.requires_grad for p in m.model[-1].cv2[0].parameters()))
        set_anchored_joint_mode(m, detector_frozen=False)
        self.assertTrue(any(p.requires_grad for p in m.backbone.model[13].parameters()))
        self.assertFalse(any(p.requires_grad for p in m.backbone.model[:11].parameters()))

        mults = {"anchor":0., "aux_encoder":.4, "fusion":1.,
                 "p2":.4, "detector":.16, "semantic":.4}
        opt = build_optimizer(m, 2.5e-5, .4, role_mults=mults)
        roles = {g["role"]:g["lr_mult"] for g in opt.param_groups}
        self.assertEqual(roles["anchor"], 0.)
        self.assertEqual(roles["fusion"], 1.)
        self.assertEqual(roles["detector"], .16)

    def test_stage_a_uses_standalone_auxiliary_detectors(self):
        c = config()
        c.fusion.branch_aux_weights = (0., 1., 1.)
        m = MMYOLO(c).train()
        set_independent_aux_mode(m)
        rgb, ir, dep = self.inputs()
        pred, active = m.independent_branch_prediction("ir", ir=ir, depth=dep)
        batch = {"boxes": [torch.tensor([[0., .5, .5, .2, .2]])] * 2}
        targets = make_targets(batch, (64, 96), torch.device("cpu"))
        sub_pred, sub_targets = subset_detection_batch(pred, targets, active)
        loss, _ = v8DetectionLoss(m)(sub_pred, sub_targets)
        loss.sum().backward()
        self.assertFalse(any(p.grad is not None for p in m.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.aux_encoders["ir"].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.independent_aux["ir"].parameters()))
        self.assertFalse(any(p.grad is not None
                             for p in m.independent_aux["dep"].parameters()))

    def test_stage_a_both_branches_receive_gradients_sequentially(self):
        c = config()
        c.fusion.branch_aux_weights = (0., 1., 1.)
        m = MMYOLO(c).train()
        set_independent_aux_mode(m)
        _, ir, dep = self.inputs()
        batch = {"boxes": [torch.tensor([[0., .5, .5, .2, .2]])] * 2}
        targets = make_targets(batch, (64, 96), torch.device("cpu"))
        criterion = v8DetectionLoss(m)
        for name in ("ir", "dep"):
            pred, active = m.independent_branch_prediction(
                name, ir=ir, depth=dep)
            sub_pred, sub_targets = subset_detection_batch(pred, targets, active)
            loss, _ = criterion(sub_pred, sub_targets)
            loss.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.aux_encoders["ir"].parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in m.aux_encoders["dep"].parameters()))

    def test_identity_mode_keeps_registered_ir_on_nominal_grid(self):
        c = config()
        c.fusion.branch_aux_weights = (0., .05, .05)
        c.fusion.alignment_mode = "identity_residual_v2"
        c.fusion.p2_match_refine = True
        m = MMYOLO(c).train()
        rgb, ir, dep = self.inputs()
        m(rgb, ir, dep)
        for scale in ("p2", "p3", "p4", "p5"):
            self.assertEqual(float(m._semantic_flows[scale][0].abs().max()), 0.)

    def test_stage_b_starts_as_exact_rgb_identity(self):
        c = config()
        c.fusion.branch_aux_weights = (0., .05, .035)
        m = MMYOLO(c).eval()
        rgb, ir, dep = self.inputs()
        with torch.no_grad():
            a = m(rgb, ir, dep)[0]
            b = m(rgb, ir.flip(-1), dep.flip(-1))[0]
        self.assertTrue(torch.equal(a, b))
        m.train()
        set_residual_fusion_mode(m, downstream_frozen=True)
        self.assertFalse(any(p.requires_grad for p in m.aux_encoders.parameters()))
        self.assertTrue(all(torch.equal(block.residual_scale.detach(),
                                        torch.zeros_like(block.residual_scale))
                            for block in m.fusion.values()))
        self.assertTrue(all(block.residual_scale.requires_grad
                            for block in m.fusion.values()))
        self.assertFalse(any(p.requires_grad for p in m.backbone.model[11:].parameters()))
        set_residual_fusion_mode(m, downstream_frozen=False)
        self.assertTrue(any(p.requires_grad for p in m.aux_encoders.parameters()))
        self.assertTrue(any(p.requires_grad for p in m.backbone.model[13].parameters()))


if __name__ == "__main__":
    unittest.main()
