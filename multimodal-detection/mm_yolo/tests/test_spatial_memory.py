import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))
from config import default_config, MMConfig
from data import (AugCfg, CoverageRareSampler, MMDataset, collate, scheduled_aug,
                  canvas_boxes_from_norm, canvas_to_orig_norm, letterbox_M)
from memory_fusion import SpatialMemoryFusion, masked_pool
from model import MMYOLO, load_mm_checkpoint, save_mm_checkpoint
from train import apply_bn_policy, ensure_finite_state, reset_fusion_gate_outputs, set_bn_tail_mode, validate_checkpoint
from types import SimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA


def new_config():
    cfg = default_config()
    cfg.fusion.architecture = "spatial_memory_v1"
    cfg.encoder.metric_branch = True
    cfg.depth_resampling = "nearest_valid_v2"
    return cfg


class SpatialMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampler_full_coverage_bounded_and_resume_deterministic(self):
        samples = [{"boxes": [[0 if i < 98 else 11, .5, .5, .1, .1]]} for i in range(100)]
        sampler = CoverageRareSampler(samples, extra_frac=.2)
        sampler.set_epoch(7)
        a = list(sampler)
        self.assertEqual(set(i for i, _, _ in a[:100]), set(range(100)))
        self.assertEqual(len(a), 120)
        self.assertEqual(len(set(a)), 120)
        self.assertLessEqual(max(sum(j == i for j, _, _ in a) for i in range(100)), 4)
        other = CoverageRareSampler(samples, extra_frac=.2)
        other.set_epoch(7)
        self.assertEqual(a, list(other))
        other.set_epoch(8)
        self.assertNotEqual(a, list(other))

    def test_worker_draw_identity_schedule_and_nearest_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mod in ("visible", "infrared", "depth"):
                (root/mod).mkdir()
            rng = np.random.default_rng(0)
            rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
            depth = np.zeros((48, 64), np.uint16)
            depth[:, :20] = 1000
            depth[:, 35:] = 9000
            for mod, arr in (("visible", rgb), ("infrared", rgb), ("depth", depth)):
                cv2.imencode(".png", arr)[1].tofile(str(root/mod/"a.png"))
            sample = {"stem": "a", "files": {m: "a.png" for m in ("visible", "infrared", "depth")},
                      "boxes": np.array([[0, .5, .5, .2, .2]], np.float32)}
            aug = AugCfg(depth_resampling="nearest_valid_v2", total_epochs=10, close_aug_frac=.3,
                         rgb_drop_p=0, aux_drop_p=0, depth_hole_p=0, degrade_p=0)
            ds = MMDataset(root, [sample], imgsz=64, aug=aug, seed=42)
            a, b, again = ds[(0, 0, 2)], ds[(0, 1, 2)], ds[(0, 0, 2)]
            self.assertTrue(torch.equal(a["rgb"], again["rgb"]))
            self.assertFalse(torch.equal(a["rgb"], b["rgb"]))
            self.assertTrue(set(np.round(torch.unique(a["depth"][1]).numpy(), 5)) <= {np.float32(0), np.float32(.05), np.float32(.45)})
            self.assertEqual(float(a["depth"][1][a["depth"][2] == 0].sum()), 0)
            sampler = CoverageRareSampler([sample], extra_frac=0)
            loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=1,
                                persistent_workers=True, collate_fn=collate)
            sampler.set_epoch(2)
            worker_a = next(iter(loader))["rgb"][0]
            self.assertTrue(torch.equal(worker_a, ds[(0, 0, 2)]["rgb"]))
            sampler.set_epoch(9)
            worker_b = next(iter(loader))["rgb"][0]
            self.assertTrue(torch.equal(worker_b, ds[(0, 0, 9)]["rgb"]))
            self.assertFalse(torch.equal(worker_a, worker_b))
            del loader
            tail = scheduled_aug(aug, 9)
            self.assertEqual(tail.scale_range, (.95, 1.05))
            self.assertAlmostEqual(tail.translate, .02)
            self.assertEqual(tail.misalign_px, 0)
            weak = AugCfg(scale_range=(.9, 1.1), translate=.03, total_epochs=20,
                          close_aug_frac=.5)
            weak_tail = scheduled_aug(weak, 19)
            self.assertEqual(weak_tail.scale_range, (.95, 1.05))
            self.assertAlmostEqual(weak_tail.translate, .02)

    def test_masked_pool_and_match_rejection(self):
        x = torch.tensor([[[[10., 999.], [10., 999.]]]])
        valid = torch.tensor([[[[1., 0.], [1., 0.]]]])
        p, f = masked_pool(x, valid, 1)
        self.assertEqual(float(p), 10.)
        self.assertEqual(float(f), .5)
        block = SpatialMemoryFusion(16, dim=8, memory_dim=16)
        q = torch.rand(2, 8, 4, 4)
        out, confidence = block._match(q, q, q, torch.zeros(2, 1, 4, 4), 2)
        self.assertEqual(float(out.abs().sum()), 0.)
        self.assertEqual(float(confidence.sum()), 0.)
        self.assertTrue(torch.isfinite(out).all())

    def test_actual_training_to_submission_coordinate_roundtrip(self):
        boxes = np.array([[3, .45, .55, .10, .15], [8, .6, .4, .08, .09]], np.float32)
        for original in ((1080, 1920), (480, 640)):
            for flip in (False, True):
                canvas = (608, 1088)
                matrix = letterbox_M(*original, canvas, 1.1, .02, -.03, flip)
                transformed = canvas_boxes_from_norm(boxes, matrix, original, canvas)
                self.assertEqual(len(transformed), len(boxes))
                xywh = transformed[:, 1:] * np.array([1088, 608, 1088, 608])
                xyxy = np.concatenate((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), 1)
                recovered = canvas_to_orig_norm(xyxy, matrix, original)
                np.testing.assert_allclose(recovered, boxes[:, 1:], atol=2e-6, rtol=0)

    def test_bounded_context_and_backward(self):
        block = SpatialMemoryFusion(16, dim=8, memory_dim=16, memory_control="bounded_v2")
        evidence = [torch.randn(2, 16, 5, 7, requires_grad=True) for _ in range(3)]
        masks = [torch.ones(2, 1, 5, 7) for _ in range(3)]
        memory = (torch.randn(2, 4, 2, 16)*10000).requires_grad_(True)
        context = block.memory_context(memory)
        self.assertLessEqual(float(context.square().mean(1).sqrt().max()), .250001)
        block(evidence, masks, memory).square().mean().backward()
        self.assertTrue(torch.isfinite(memory.grad).all())
        self.assertGreater(float(memory.grad.abs().sum()), 0)
        for x in evidence:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertLessEqual(float(block.last_health["context_rms"]), .250001)

    def test_bn_rare_tail_does_not_rewrite_stats_but_keeps_gradients(self):
        bn = torch.nn.BatchNorm2d(2).train()
        bn(torch.randn(3, 2, 5, 5))
        mean, var = bn.running_mean.clone(), bn.running_var.clone()
        set_bn_tail_mode(bn, True)
        x = (torch.randn(3, 2, 5, 5)+100).requires_grad_(True)
        bn(x).square().mean().backward()
        self.assertTrue(torch.equal(mean, bn.running_mean))
        self.assertTrue(torch.equal(var, bn.running_var))
        self.assertGreater(float(bn.weight.grad.abs().sum()), 0)
        set_bn_tail_mode(bn, False)
        self.assertEqual(bn.momentum, .03)

    def test_repair_is_explicit_and_only_gate_output_weights_reset(self):
        cfg = new_config()
        old = MMYOLO(cfg)
        ck = {"structure": old.structure_kwargs(), "model_state": old.state_dict(),
              "meta": {"modalities": ["rgb", "ir", "dep"], "canvas": [64,96], "split_digest": "fixed"}}
        cfg2 = MMConfig.from_structure(old.structure_kwargs())
        cfg2.fusion.memory_control = "bounded_v2"
        new = MMYOLO(cfg2)
        with self.assertRaises(ValueError):
            validate_checkpoint(ck, new, ["rgb", "ir", "dep"], [64,96], exact=False)
        args = SimpleNamespace(reset_fusion_gates=True)
        validate_checkpoint(ck, new, ["rgb", "ir", "dep"], [64,96], exact=False, args=args, split_digest="fixed")
        with self.assertRaises(ValueError):
            validate_checkpoint(ck, new, ["rgb", "ir", "dep"], [64,96], exact=False, args=args, split_digest="other")
        new.load_state_dict(old.state_dict(), strict=True)
        reset = set(reset_fusion_gate_outputs(new))
        self.assertEqual(len(reset), 18)
        for key, value in new.state_dict().items():
            if key not in reset:
                self.assertTrue(torch.equal(value, old.state_dict()[key]), key)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"v2.pt"
            save_mm_checkpoint(path, new)
            loaded, _ = load_mm_checkpoint(path)
            self.assertEqual(loaded.cfg.fusion.memory_control, "bounded_v2")
            for key, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[key]))

    def test_model_gradients_memory_reset_rgb_identity_ema_and_checkpoint(self):
        torch.manual_seed(17)
        model = MMYOLO(new_config()).eval()
        self.assertFalse(model.backbone.model[-1].dfl.conv.weight.requires_grad)
        torch.testing.assert_close(model.backbone.model[-1].dfl.conv.weight.flatten(), torch.arange(16).float())
        rgb, ir = torch.rand(2, 3, 64, 96), torch.rand(2, 1, 64, 96)
        depth = torch.rand(2, 4, 64, 96)
        depth[:, 2:] = 1
        with torch.no_grad():
            stock = model.backbone(rgb)[0]
            no_aux = model(rgb)[0]
            self.assertTrue(torch.allclose(stock, no_aux, atol=1e-5, rtol=1e-5))
            first = model(rgb, ir, depth)[0].clone()
            memory = model._last_register_state.clone()
            model(rgb*0, ir, depth)
            again = model(rgb, ir, depth)[0]
            self.assertTrue(torch.equal(first, again))
            self.assertTrue(torch.equal(memory, model._last_register_state))
            self.assertIsNone(model._register_state)
            invalid = depth.clone()
            invalid[:, 2:] = 0
            self.assertTrue(torch.allclose(model(rgb, depth=invalid)[0], no_aux, atol=1e-5, rtol=1e-5))
            ema = ModelEMA(model)
            self.assertTrue(torch.allclose(first, ema.ema(rgb, ir, depth)[0], atol=1e-4, rtol=1e-4))
        model.train()
        apply_bn_policy(model, "adaptive")
        depth.requires_grad_(True)
        outputs = model(rgb, ir, depth)
        sum(v.float().square().mean() for v in (outputs.values() if isinstance(outputs, dict) else outputs)
            if torch.is_tensor(v)).backward()
        for name, module in (("metric", model.metric_encoder), ("fusion", model.fusion),
                             ("memory", model.register_bus), ("neck_memory", model.neck_memory),
                             ("depth", model.dep_stem), ("IR", model.ir_adapter)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, name)
        ensure_finite_state(model)
        self.assertGreater(float(depth.grad[:, 1].abs().sum()), 0, "absolute metric input has no gradient")
        # No doubled state_dict storage (in particular: shared encoder / EMA).
        ptrs = [v.data_ptr() for v in model.state_dict().values()]
        self.assertEqual(len(ptrs), len(set(ptrs)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"new.pt"
            model.eval()
            save_mm_checkpoint(path, model, meta={"modalities": ["rgb", "ir", "dep"], "canvas": [64, 96]})
            loaded, _ = load_mm_checkpoint(path)
            loaded.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(rgb, ir, depth)[0], loaded(rgb, ir, depth)[0]))
            self.assertEqual(loaded.cfg.depth_resampling, "nearest_valid_v2")
        legacy = MMConfig.from_structure({"encoder": {}, "fusion": {}})
        self.assertFalse(legacy.encoder.metric_branch)
        self.assertEqual(legacy.fusion.architecture, "legacy_hook_v1")


if __name__ == "__main__":
    unittest.main()
