# -*- coding: utf-8 -*-
"""实验开关纯逻辑测试；不会加载数据、权重或启动训练。"""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import cv2

MM = Path(__file__).resolve().parents[1]
for path in (MM, MM.parent / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import MMConfig, default_config  # noqa: E402
from data import MMDataset, _gain_ir, _jitter_rgb  # noqa: E402
from experiments import build_plan  # noqa: E402
from model import MMYOLO, load_mm_checkpoint, resolve_infer_modalities  # noqa: E402
from train import validate_checkpoint  # noqa: E402


class ExperimentControlTests(unittest.TestCase):
    def test_extra_augmentations_preserve_shape_and_ir_order(self):
        ir = np.arange(256, dtype=np.uint8).reshape(16, 16)
        shifted = _gain_ir(ir, random.Random(11))
        self.assertEqual(shifted.shape, ir.shape)
        self.assertTrue(np.all(np.diff(shifted.astype(np.int16).reshape(-1)) >= 0))
        rgb = np.full((16, 16, 3), 128, np.uint8)
        colored = _jitter_rgb(rgb, random.Random(11))
        self.assertEqual(colored.shape, rgb.shape)
        self.assertEqual(colored.dtype, np.uint8)

    def test_depth_only_dataset_uses_rgb_geometry_but_no_rgb_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for m in ("visible", "infrared", "depth"):
                (root / m).mkdir()
            images = (("visible", np.full((64, 128, 3), 100, np.uint8), ".png"),
                      ("infrared", np.full((64, 128), 80, np.uint8), ".png"),
                      ("depth", np.full((64, 128), 1500, np.uint16), ".png"))
            for folder, img, ext in images:
                ok, data = cv2.imencode(ext, img)
                self.assertTrue(ok)
                data.tofile(root / folder / "one.png")
            sample = {"stem": "one", "files": {m: "one.png" for m, _, _ in images},
                      "boxes": np.array([[0, .5, .5, .2, .2]], np.float32)}
            item = MMDataset(root, [sample], imgsz=(64, 128), train=False,
                             enabled=("dep",))[0]
            self.assertEqual(float(item["rgb"].abs().sum()), 0.0)
            self.assertEqual(item["keep"], {"rgb": 0.0, "ir": 0.0, "dep": 1.0})
            self.assertGreater(float(item["depth"][1].mean()), 0.0)

    def test_depth_views_do_not_mutate_input_and_keep_jpg_fallback(self):
        depth = torch.tensor([[[[0.8, 0.4]], [[0.3, 0.0]],
                               [[1.0, 1.0]], [[1.0, 0.0]]]])
        old = depth.clone()
        rel = MMYOLO._depth_view_input(SimpleNamespace(depth_view="relative"), depth)
        self.assertTrue(torch.equal(rel[:, 1], torch.zeros_like(rel[:, 1])))
        metric = MMYOLO._depth_view_input(SimpleNamespace(depth_view="metric_fallback"), depth)
        self.assertTrue(torch.allclose(metric[0, 0, 0], torch.tensor([0.0, 0.4])))
        self.assertTrue(torch.allclose(metric[0, 1, 0], torch.tensor([0.3, 0.0])))
        log_metric = MMYOLO._depth_view_input(
            SimpleNamespace(depth_view="metric_log_fallback"), depth)
        self.assertGreater(float(log_metric[0, 1, 0, 0]), 0.3)
        self.assertAlmostEqual(float(log_metric[0, 0, 0, 1]), 0.4, places=6)
        self.assertTrue(torch.equal(depth, old))

    def test_old_structure_defaults_and_new_structure_roundtrip(self):
        legacy = MMConfig.from_structure({"encoder": {"share_tier": "c"}})
        self.assertEqual(legacy.encoder.depth_input_channels, 2)
        self.assertEqual(legacy.encoder.depth_view, "both")
        self.assertEqual(legacy.encoder.depth_init, "relative")
        cfg = MMConfig()
        cfg.encoder.depth_view = "metric_fallback"
        cfg.encoder.depth_init = "metric_fallback"
        cfg.fusion.depth_scales = (True, True, True)
        restored = MMConfig.from_structure(cfg.structure())
        self.assertEqual(restored.encoder.depth_view, "metric_fallback")
        self.assertEqual(restored.encoder.depth_init, "metric_fallback")

    def test_checkpoint_modalities_cannot_open_untrained_branch(self):
        for trained, expected in ((["rgb"], "rgb"), (["rgb", "ir"], "rgb_ir"),
                                  (["rgb", "dep"], "rgb_dep")):
            model = SimpleNamespace(infer_modalities=trained)
            self.assertEqual(resolve_infer_modalities(model), expected)
            self.assertEqual(resolve_infer_modalities(model, "all"), expected)
        self.assertEqual(resolve_infer_modalities(
            SimpleNamespace(infer_modalities=["rgb", "ir", "dep"]), "rgb_ir"), "rgb_ir")

    def test_plan_is_commands_not_processes_and_keeps_split(self):
        args = SimpleNamespace(stage="modalities", python="python", root="TRAIN",
                               labels="LABELS", weights="PRETRAINED", split_file="SPLIT",
                               out="RUNS", device="cpu", imgsz="608x1088", epochs=80,
                               batch=3, accum=5, workers=0, seed=42, baseline_ckpt="")
        plan = build_plan(args)
        stock = [cmd for title, cmd in plan if title.startswith("校准：")]
        self.assertEqual(len(stock), 1)
        self.assertTrue(stock[0][1].endswith("stock_rgb_baseline.py"))
        train_cmds = [cmd for title, cmd in plan if title.startswith("训练")]
        self.assertEqual(len(train_cmds), 4)
        for cmd in train_cmds:
            self.assertEqual(cmd[cmd.index("--split-file") + 1], "SPLIT")
            self.assertEqual(cmd[cmd.index("--weights") + 1], "PRETRAINED")
            self.assertEqual(cmd[cmd.index("--grad-clip") + 1], "60.0")
            self.assertNotIn("--init-checkpoint", cmd)
            self.assertNotIn("--eval-initial", cmd)
            self.assertEqual(cmd[cmd.index("--val-conf") + 1], "0.01")
        self.assertEqual({cmd[cmd.index("--modalities") + 1] for cmd in train_cmds},
                         {"rgb", "rgb_ir", "rgb_dep", "all"})
        args.run_tag = "optfix_v2"
        args.skip_stock = True
        tagged = build_plan(args)
        self.assertEqual(len(tagged), 8)
        self.assertFalse(any(title.startswith("校准：") for title, _ in tagged))
        self.assertTrue(all("optfix_v2" in " ".join(cmd) for _, cmd in tagged))
        args.run_tag = ""
        args.skip_stock = False
        args.grad_clip = 10.0
        old_recipe = [cmd for title, cmd in build_plan(args) if title.startswith("训练")]
        self.assertTrue(all(cmd[cmd.index("--grad-clip") + 1] == "10.0"
                            for cmd in old_recipe))
        args.grad_clip = 60.0
        args.stage = "standalone"
        alone = [cmd for title, cmd in build_plan(args) if title.startswith("训练")]
        dep = next(cmd for cmd in alone if cmd[cmd.index("--modalities") + 1] == "dep")
        self.assertEqual(dep.count("--depth-scales"), 1)
        self.assertEqual(dep[dep.index("--depth-scales") + 1], "all")
        args.stage = "depth"
        depth = [cmd for title, cmd in build_plan(args) if title.startswith("训练")]
        self.assertEqual(len(depth), 5)
        self.assertTrue(all("--no-prior" in cmd and "--no-quality" in cmd for cmd in depth))
        args.stage = "augment"
        augment = [cmd for title, cmd in build_plan(args) if title.startswith("训练")]
        self.assertTrue(any("--rgb-color-p" in cmd for cmd in augment))
        self.assertTrue(any("--ir-gain-p" in cmd for cmd in augment))

    def test_two_modalities_small_forward_without_training(self):
        weights = MM.parent / "yolo11s.pt"
        if not weights.exists():
            self.skipTest("离线 COCO 权重未安装")
        cfg = default_config()
        cfg.weights = str(weights)
        cfg.encoder.depth_view = "metric_fallback"
        cfg.encoder.depth_init = "metric_fallback"
        model = MMYOLO(cfg).eval()
        rgb = torch.rand(1, 3, 128, 128)
        ir = torch.rand(1, 1, 128, 128)
        dep = torch.rand(1, 4, 128, 128)
        dep[:, 2:] = 1.0
        with torch.no_grad():
            for off, aux_ir, aux_dep in (({"dep"}, ir, None), ({"ir"}, None, dep),
                                         ({"rgb", "ir"}, None, dep),
                                         ({"rgb", "dep"}, ir, None)):
                model.modality_off = off
                main = torch.zeros_like(rgb) if "rgb" in off else rgb
                result = model(main, aux_ir, aux_dep)
                pred = result[0] if isinstance(result, tuple) else result
                self.assertTrue(torch.isfinite(pred).all())
            model.modality_off = {"rgb"}
            keep = {"rgb": torch.zeros(1), "ir": torch.ones(1),
                    "dep": torch.ones(1)}
            a = model(rgb, ir, dep, keep=keep)
            b = model(torch.zeros_like(rgb), ir, dep, keep=keep)
            a = a[0] if isinstance(a, tuple) else a
            b = b[0] if isinstance(b, tuple) else b
            self.assertTrue(torch.equal(a, b), "屏蔽 RGB 后主干不应泄漏原 RGB 信息")

    def test_existing_b2_checkpoint_remains_loadable_but_not_exactly_resumable(self):
        ckpt = MM.parent / "runs" / "b2_depth4_s_safe" / "weights" / "best.pt"
        if not ckpt.exists():
            self.skipTest("本地 B2 权重不存在")
        model, _ck = load_mm_checkpoint(ckpt, device="cpu")
        self.assertEqual(model.depth_view, "both")
        self.assertEqual(model.depth_init, "relative")
        self.assertEqual(resolve_infer_modalities(model), "all")
        # 模型权重仍可评测/迁移；但优化器归组和 loss 标度改变后，旧训练状态
        # 不再冒充“精确续训”。
        old_args = dict(_ck["train_state"]["args"])
        old_args.update(depth_view="both", depth_init="relative",
                        rgb_color_p=0.0, ir_gain_p=0.0)
        with self.assertRaisesRegex(ValueError, "旧训练算法"):
            validate_checkpoint(_ck, model, _ck["meta"]["modalities"],
                                _ck["meta"]["canvas"], exact=True,
                                args=SimpleNamespace(**old_args),
                                split_digest=_ck["meta"]["split_digest"])
        validate_checkpoint(_ck, model, _ck["meta"]["modalities"],
                            _ck["meta"]["canvas"], exact=False)


if __name__ == "__main__":
    unittest.main()
