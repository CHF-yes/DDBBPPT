# -*- coding: utf-8 -*-
"""官方 RGB 校准器的验证完整性；不启动正式训练。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

MM = Path(__file__).resolve().parents[1]
for item in (MM, MM.parent / "vendor"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from stock_rgb_baseline import CompleteDetectionValidator, DetectionValidator  # noqa: E402


class StockRGBRegressions(unittest.TestCase):
    def test_per_image_nms_matches_official_outputs_without_timeouts(self):
        validator = SimpleNamespace(
            args=SimpleNamespace(conf=0.001, iou=0.7, task="detect", single_cls=False,
                                 agnostic_nms=False, max_det=100),
            end2end=False)
        prediction = torch.zeros(3, 16, 20)
        prediction[:, 0, :] = torch.arange(20).float() * 12 + 30
        prediction[:, 1, :] = 40
        prediction[:, 2:4, :] = 8
        prediction[:, 4, :] = 0.9
        expected = DetectionValidator.postprocess(validator, prediction.clone())
        actual = CompleteDetectionValidator.postprocess(validator, prediction.clone())
        for left, right in zip(expected, actual):
            for key in ("bboxes", "conf", "cls", "extra"):
                self.assertTrue(torch.equal(left[key], right[key]))

    def test_every_image_gets_nms_even_if_batched_nms_would_truncate(self):
        validator = SimpleNamespace(
            args=SimpleNamespace(conf=0.001, iou=0.7, task="detect", single_cls=False,
                                 agnostic_nms=False, max_det=100),
            end2end=False)
        pred = torch.zeros(3, 16, 20)

        def timeout_after_first_image(images, *args, **kwargs):
            # 模拟 vendor 提前返回：只填当前批第一张，剩余图片保持空预测。
            return [torch.tensor([[0., 0., 10., 10., 0.9, 0.]])] + [
                torch.empty(0, 6) for _ in range(images.shape[0] - 1)]

        with patch("stock_rgb_baseline.nms.non_max_suppression",
                   side_effect=timeout_after_first_image) as mocked:
            outputs = CompleteDetectionValidator.postprocess(validator, (pred, None))
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(len(out["bboxes"]) == 1 for out in outputs))
        for call in mocked.call_args_list:
            self.assertEqual(call.args[0].shape[0], 1)
            self.assertEqual(call.kwargs["max_det"], 100)
            self.assertTrue(call.kwargs["multi_label"])


if __name__ == "__main__":
    unittest.main()
