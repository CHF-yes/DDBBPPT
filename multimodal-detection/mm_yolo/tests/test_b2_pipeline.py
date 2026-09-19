# -*- coding: utf-8 -*-
"""B2 数据表示、增强、均衡采样和 B1→B2 权重迁移回归测试。"""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

MM = Path(__file__).resolve().parents[1]
for path in (MM, MM.parent / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import MMConfig  # noqa: E402
from data import (_apply, _degrade_rgb, _drop_depth_blocks, absolute_metric_depth,
                  balanced_sample_weights, read_depth, target_crop_M)  # noqa: E402
from train import adapt_depth_checkpoint_state  # noqa: E402


class B2PipelineTests(unittest.TestCase):
    def test_depth_metric_detection_and_absolute_channel(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            metric = np.array([[0, 205, 300, 1000, 19999, 20001]], np.uint16)
            ok, enc = cv2.imencode(".png", metric)
            self.assertTrue(ok)
            enc.tofile(td / "深度.png")
            dep, valid, is_metric = read_depth(td / "深度.png", return_metric=True)
            self.assertTrue(is_metric)
            self.assertEqual(valid.tolist(), [[False, False, True, True, True, False]])
            absolute = absolute_metric_depth(dep, valid, is_metric)
            self.assertAlmostEqual(float(absolute[0, 3]), 0.05, places=6)
            self.assertEqual(float(absolute[0, 1]), 0.0)

            pseudo = np.array([[0, 10, 255]], np.uint8)
            ok, enc = cv2.imencode(".jpg", pseudo)
            self.assertTrue(ok)
            enc.tofile(td / "伪深度.jpg")
            dep, valid, is_metric = read_depth(td / "伪深度.jpg", return_metric=True)
            self.assertFalse(is_metric)
            self.assertEqual(float(absolute_metric_depth(dep, valid, is_metric).sum()), 0.0)

    def test_lowlight_really_darkens(self):
        src = np.full((64, 64, 3), 180, np.uint8)
        means = [float(_degrade_rgb(src, random.Random(seed)).mean()) for seed in range(8)]
        self.assertLess(max(means), float(src.mean()) * 0.8)

    def test_target_crop_keeps_selected_target_and_fills_canvas(self):
        boxes = np.array([[0, 0.5, 0.5, 0.10, 0.10]], np.float32)
        canvas = (608, 1088)
        M = target_crop_M(1080, 1920, canvas, boxes, random.Random(3), flip=False)
        centre = _apply(M, np.array([[960.0, 540.0]], np.float32))[0]
        self.assertTrue(0 <= centre[0] < canvas[1] and 0 <= centre[1] < canvas[0])
        self.assertGreater(float(M[0, 0]), canvas[1] / 1920.0)  # 裁剪必然比全图 letterbox 更大

    def test_depth_holes_clear_data_and_valid_together(self):
        dep = np.ones((100, 160), np.float32)
        valid = np.ones_like(dep, bool)
        out, vm = _drop_depth_blocks(dep, valid, random.Random(4))
        removed = valid & ~vm
        self.assertGreater(int(removed.sum()), 0)
        self.assertTrue(np.all(out[removed] == 0))

    def test_balanced_weights_are_bounded(self):
        samples = []
        for i in range(100):
            cls = 0 if i < 90 else (11 if i == 99 else 1)
            samples.append({"boxes": np.array([[cls, .5, .5, .1, .1]], np.float32)})
        weights, counts, factors = balanced_sample_weights(samples, nc=12, max_weight=3)
        self.assertEqual(counts[11], 1)
        self.assertAlmostEqual(float(weights[-1]), 3.0)
        self.assertLessEqual(float(weights.max()), 3.0)
        self.assertGreater(factors[1], 1.0)

    def test_b1_adapter_migrates_to_b2_without_changing_old_semantics(self):
        old = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1)

        class FakeModel:
            @staticmethod
            def state_dict():
                return {"dep_adapter.weight": torch.full((3, 4, 1, 1), 99.0)}

        state, migrated = adapt_depth_checkpoint_state({"dep_adapter.weight": old}, FakeModel())
        got = state["dep_adapter.weight"]
        self.assertTrue(migrated)
        self.assertTrue(torch.equal(got[:, 0], old[:, 0]))
        self.assertTrue(torch.equal(got[:, 2], old[:, 1]))
        self.assertEqual(float(got[:, (1, 3)].abs().sum()), 0.0)

    def test_legacy_structure_rebuilds_two_channel_model_config(self):
        cfg = MMConfig.from_structure({"encoder": {"share_tier": "c"}})
        self.assertEqual(cfg.encoder.depth_input_channels, 2)


if __name__ == "__main__":
    unittest.main()
