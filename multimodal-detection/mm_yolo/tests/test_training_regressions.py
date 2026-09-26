# -*- coding: utf-8 -*-
"""训练语义回归：EFYOLO 环境运行 python -m unittest discover -s code/mm_yolo/tests -p test_training_regressions.py。"""
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

MM = Path(__file__).resolve().parents[1]
for path in (MM, MM.parent / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import FusionCfg  # noqa: E402
from data import collate  # noqa: E402
from fusion import FusionBlock, PersistentRegisterBus  # noqa: E402
from train import (accumulation_loss, build_optimizer, ensure_finite_state, iter_prefetch,
                   set_bn_eval, validate_checkpoint)  # noqa: E402


class TrainingRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bn_default_only_pretrained(self):
        model = nn.Module()
        model.backbone = nn.Sequential(nn.BatchNorm2d(1))
        model.fusion = nn.Sequential(nn.BatchNorm2d(1))
        model.bn_store = nn.Sequential(nn.BatchNorm2d(1))
        self.assertEqual(set_bn_eval(model, include_new=False), 1)
        self.assertTrue(model.fusion[0].training)
        self.assertTrue(model.bn_store[0].training)
        model.train()
        self.assertEqual(set_bn_eval(model, include_new=True), 3)
        self.assertFalse(model.fusion[0].training)

    def test_optimizer_reduces_only_encoder_lr(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                # 前 11 层是编码器；第 11 层代表 Neck/Detect。
                self.backbone.model = nn.Sequential(*[nn.Conv2d(1, 1, 1) for _ in range(12)])
                self.dep_stem = nn.Conv2d(1, 1, 1)
                self.fusion = nn.Conv2d(1, 1, 1)

        model = TinyModel()
        optimizer = build_optimizer(model, lr=1e-3, backbone_mult=0.1)
        group_for = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        encoder = model.backbone.model[0].weight
        task_head = model.backbone.model[11].weight
        depth_stem = model.dep_stem.weight
        fusion = model.fusion.weight
        self.assertEqual(group_for[id(encoder)]["role"], "encoder")
        self.assertAlmostEqual(group_for[id(encoder)]["lr"], 1e-4)
        self.assertEqual(group_for[id(depth_stem)]["role"], "encoder")
        self.assertAlmostEqual(group_for[id(depth_stem)]["lr"], 1e-4)
        self.assertEqual(group_for[id(task_head)]["role"], "task")
        self.assertAlmostEqual(group_for[id(task_head)]["lr"], 1e-3)
        self.assertEqual(group_for[id(fusion)]["role"], "task")
        self.assertAlmostEqual(group_for[id(fusion)]["lr"], 1e-3)

    def test_accumulation_loss_is_invariant_to_physical_batch(self):
        # loss_vec 是 Ultralytics 返回的 batch 总损失。相同逐样本损失在
        # 3x5 与 5x3 累积下都应得到同一 optimizer-step 均值。
        for physical_batch, accum in ((3, 5), (5, 3), (15, 1)):
            contribution = accumulation_loss(
                torch.tensor([2.0, 1.0, 0.5]) * physical_batch,
                physical_batch * accum)
            self.assertAlmostEqual(float(contribution) * accum, 3.5, places=6)

    def test_nonfinite_bn_buffer_is_rejected_before_save(self):
        model = nn.Sequential(nn.BatchNorm2d(2))
        ensure_finite_state(model, "clean")
        model[0].running_var[0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "running_var"):
            ensure_finite_state(model, "dirty")

    def test_prefetch_keeps_every_batch_and_raises_original_error(self):
        for depth in (1, 3):
            self.assertEqual(list(iter_prefetch(range(4), depth=depth)), [0, 1, 2, 3])

        class Broken:
            def __iter__(self):
                yield 0
                raise PermissionError("worker failed")

        it = iter_prefetch(Broken(), depth=1)
        self.assertEqual(next(it), 0)
        with self.assertRaisesRegex(PermissionError, "worker failed"):
            next(it)

    def test_quality_is_per_sample_not_batch_intersection(self):
        def item(q, keep):
            return {"rgb": torch.ones(3, 4, 4), "ir": torch.ones(1, 4, 4),
                    "depth": torch.ones(2, 4, 4), "prior": torch.zeros(4, 2, 2),
                    "quality": q, "boxes": torch.empty(0, 5), "M": torch.eye(2, 3),
                    "orig_hw": torch.tensor([4, 4]), "stem": "s", "keep": keep,
                    "enabled": ["rgb", "ir", "dep"]}
        q = torch.ones(3, 2, 2)
        b = collate([item({"rgb": q, "ir": q, "dep": q}, {"rgb": 1, "ir": 1, "dep": 1}),
                     item({"ir": q}, {"rgb": 0, "ir": 1, "dep": 0})])
        self.assertEqual(set(b["quality"]), {"rgb", "ir", "dep"})
        self.assertEqual(float(b["quality"]["rgb"][0].sum()), 12.0)
        self.assertEqual(float(b["quality"]["rgb"][1].sum()), 0.0)
        self.assertEqual(float(b["quality"]["ir"][1].sum()), 12.0)

    def test_fusion_keep_masks_dropped_aux_and_invalid_depth(self):
        torch.manual_seed(7)
        f = FusionBlock(16, FusionCfg(tier="L2", iters=1)).eval()
        rgb = torch.randn(2, 16, 8, 8)
        ir = torch.randn_like(rgb)
        dep = torch.randn_like(rgb)
        keep = torch.tensor([[1, 0, 1], [0, 1, 0]])
        m = torch.ones(2, 1, 8, 8)
        m[0] = 0
        out = f([rgb, ir, dep], mask=m, keep=keep)
        changed = f([rgb, ir + 100, dep + 100], mask=m, keep=keep)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out[0], changed[0], atol=1e-5))
        self.assertFalse(torch.allclose(out[1], changed[1]))
        with self.assertRaisesRegex(ValueError, "至少"):
            f([rgb, ir, dep], keep=torch.zeros(2, 3))

    def test_register_persists_across_scales_and_keeps_rgb_identity(self):
        bus = PersistentRegisterBus({"p3": 16, "p4": 16, "p5": 32}, n_tokens=4,
                                    dim=16, heads=4, pool=2).eval()
        state = None
        previous = None
        outputs = []
        for scale, channels, hw in (("p3", 16, 8), ("p4", 16, 4), ("p5", 32, 2)):
            feats = [torch.randn(2, channels, hw, hw, requires_grad=True) for _ in range(3)]
            valid = [torch.ones(2, 1, hw, hw, dtype=torch.bool),
                     torch.tensor([1, 0], dtype=torch.bool)[:, None, None, None].expand(-1, 1, hw, hw),
                     torch.tensor([1, 0], dtype=torch.bool)[:, None, None, None].expand(-1, 1, hw, hw)]
            fused = feats[0]
            out, state = bus(scale, feats, valid, fused, state)
            self.assertEqual(state.shape, (2, 4, 16))
            if previous is not None:
                self.assertFalse(torch.allclose(state, previous))
            previous = state.detach().clone()
            self.assertTrue(torch.equal(out[1], fused[1]))  # 纯 RGB 样本严格恒等
            outputs.append(out.mean())
        sum(outputs).backward()
        self.assertIsNotNone(bus.seed.grad)
        self.assertGreater(float(bus.seed.grad.abs().sum()), 0.0)

    def test_register_skips_sample_with_no_token_at_current_scale(self):
        bus = PersistentRegisterBus({"p3": 16, "p4": 16}, n_tokens=4,
                                    dim=16, heads=4, pool=2).eval()
        p3 = [torch.randn(2, 16, 8, 8) for _ in range(3)]
        none = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
        rgb = none.clone()
        rgb[1] = True
        initial = bus.initial_state(2, p3[0]).detach().clone()
        out, state = bus("p3", p3, [rgb, none, none], p3[0], None)
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.equal(state[0], initial[0]))
        self.assertTrue(torch.equal(out[0], p3[0][0]))

        # 到 P4 后该样本的 Depth 接入，register 才开始读取并更新。
        p4 = [torch.randn(2, 16, 4, 4) for _ in range(3)]
        none4 = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        rgb4 = none4.clone(); rgb4[1] = True
        dep4 = none4.clone(); dep4[0] = True
        _, state2 = bus("p4", p4, [rgb4, none4, dep4], p4[0], state)
        self.assertFalse(torch.equal(state2[0], state[0]))

    def test_ema_only_checkpoint_cannot_exact_resume(self):
        class FakeModel:
            def structure_kwargs(self):
                return {"fusion": "L2"}

        ck = {"structure": {"fusion": "L2"}, "model_state": {},
              "meta": {"modalities": ["rgb", "ir", "dep"], "canvas": [544, 960]},
              "train_state": {"optimizer": {}, "ema_state": {}}}
        with self.assertRaisesRegex(ValueError, "raw_model_state"):
            validate_checkpoint(ck, FakeModel(), ("rgb", "ir", "dep"), (544, 960))
        validate_checkpoint(ck, FakeModel(), ("rgb", "ir", "dep"), (544, 960), exact=False)

    def test_checkpoint_weight_path_is_portable(self):
        class FakeModel:
            def structure_kwargs(self):
                return {"weights": "/server/models/yolo11s.pt", "fusion": "L2"}

        ck = {"structure": {"weights": r"C:\models\yolo11s.pt", "fusion": "L2"},
              "model_state": {},
              "meta": {"modalities": ["rgb", "ir", "dep"], "canvas": [544, 960]}}
        validate_checkpoint(ck, FakeModel(), ("rgb", "ir", "dep"), (544, 960), exact=False)


if __name__ == "__main__":
    unittest.main()
