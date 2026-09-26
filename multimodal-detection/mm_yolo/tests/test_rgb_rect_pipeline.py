import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import models_config as MC  # noqa: E402
from common import trainer as TR  # noqa: E402


class RGBRectPipelineTests(unittest.TestCase):
    def test_rect_recipe_is_registered_and_has_a_trustworthy_refinement(self):
        cfg = MC.get("rgb_hq_11m_rect")
        self.assertEqual(cfg.hyper.imgsz, 1280)
        self.assertTrue(cfg.hyper.rect_train)
        self.assertEqual(cfg.hyper.aug.mosaic_p, 0.0)
        self.assertGreater(cfg.hyper.localization_epochs, 0)
        self.assertFalse(cfg.hyper.localization_amp)
        self.assertEqual(cfg.hyper.patience, 0)

    def test_trainer_requests_rect_only_when_configured_or_validating(self):
        trainer = object.__new__(TR.HighQualityDetectionTrainer)
        trainer.model = SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0]))
        trainer.args = SimpleNamespace()
        trainer.data = {"names": {0: "person"}, "nc": 1}
        with patch.object(TR, "build_yolo_dataset", return_value="dataset") as build:
            trainer._rect_train = True
            self.assertEqual(trainer.build_dataset("train.txt", "train", 4), "dataset")
            self.assertTrue(build.call_args.kwargs["rect"])
            trainer._rect_train = False
            trainer.build_dataset("train.txt", "train", 4)
            self.assertFalse(build.call_args.kwargs["rect"])
            trainer.build_dataset("val.txt", "val", 4)
            self.assertTrue(build.call_args.kwargs["rect"])

    def test_rect_flag_survives_subprocess_via_environment(self):
        with patch.dict(os.environ, {}, clear=False):
            TR.configure_rgb_trainer(None, rect_train=True)
            self.assertEqual(os.environ["EFYOLO_RECT_TRAIN"], "1")
            TR.configure_rgb_trainer(None, rect_train=False)
            self.assertEqual(os.environ["EFYOLO_RECT_TRAIN"], "0")


if __name__ == "__main__":
    unittest.main()
