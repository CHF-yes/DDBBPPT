# -*- coding: utf-8 -*-
"""自定义评测完整性回归；不读取正式数据。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

MM = Path(__file__).resolve().parents[1]
for item in (MM, MM.parent / "vendor"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import eval as eval_module  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
