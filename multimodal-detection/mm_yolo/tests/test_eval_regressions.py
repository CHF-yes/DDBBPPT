# -*- coding: utf-8 -*-
"""自定义评测完整性回归；不读取正式数据。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

MM = Path(__file__).resolve().parents[1]
for item in (MM, MM.parent / "vendor"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import eval as eval_module  # noqa: E402
import submit as submit_module  # noqa: E402


class _FakeModel(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = prediction
        self.modality_off = set()
        self.nc = 12

    def forward(self, *_args, **_kwargs):
        return self.prediction


class EvalRegressions(unittest.TestCase):
    def test_decode_runs_nms_once_per_image(self):
        pred = torch.zeros(3, 16, 20)
        model = _FakeModel(pred)
        batch = {
            "rgb": torch.zeros(3, 3, 32, 32),
            "ir": torch.zeros(3, 1, 32, 32),
            "depth": torch.zeros(3, 4, 32, 32),
            "quality": {},
            "keep": {},
            "prior": torch.zeros(3, 4, 8, 8),
        }

        def one_result(image, *_args, **_kwargs):
            self.assertEqual(image.shape[0], 1)
            return [torch.tensor([[0., 0., 1., 1., 0.9, 0.]])]

        with patch("eval.non_max_suppression", side_effect=one_result) as mocked:
            decoded = eval_module._decode(model, batch, torch.device("cpu"),
                                          0.001, 0.7, 100, modalities="all")
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(len(decoded), 3)
        self.assertTrue(all(item.shape == (1, 6) for item in decoded))

    def test_submit_enables_explicit_a0_for_v521_checkpoint(self):
        cfg = SimpleNamespace(
            fusion=SimpleNamespace(fusion_strategy="v521_stage_a_v1"),
            encoder=SimpleNamespace(depth_input_channels=4),
            depth_resampling="nearest_valid_v2",
            ir_read_mode="median_channel",
        )
        model = SimpleNamespace(cfg=cfg, eval=lambda: None)
        checkpoint = {"meta": {"ir_a0": {"required": True}}}
        captured = {}

        class StopAfterDatasetConfig(RuntimeError):
            pass

        def capture_dataset(*_args, **kwargs):
            captured["aug"] = kwargs["aug"]
            raise StopAfterDatasetConfig

        argv = [
            "submit.py", "--ckpt", "fake.pt", "--root", "fake-root",
            "--ir-a0-cache", "fake-cache", "--out", "fake-out",
        ]
        with patch.object(sys, "argv", argv), \
             patch.object(submit_module, "load_mm_checkpoint",
                          return_value=(model, checkpoint)), \
             patch.object(submit_module, "resolve_infer_modalities", return_value="all"), \
             patch.object(submit_module, "resolve_infer_canvas", return_value=(32, 32)), \
             patch.object(submit_module, "scan_test",
                          return_value=([{"stem": "sample"}], {"infrared": 0, "depth": 0})), \
             patch.object(submit_module, "MMDataset", side_effect=capture_dataset), \
             self.assertRaises(StopAfterDatasetConfig):
            submit_module.main()

        self.assertTrue(captured["aug"].require_ir_a0)
        self.assertTrue(captured["aug"].v521_explicit)
        self.assertEqual(captured["aug"].ir_a0_cache, "fake-cache")


if __name__ == "__main__":
    unittest.main()
