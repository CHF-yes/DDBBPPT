import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))

from ir_a0 import (A0_QUALITY_NAMES, affine_model_contract, border_masks,
                   load_sample, quality_maps, save_sample,
                   select_affine_candidate, source_to_sampling)


class IRA0Tests(unittest.TestCase):
    def test_border_connected_dark_region_is_masked_but_dark_object_is_not(self):
        image = np.full((96, 160), 100, np.float32)
        image[:, :12] = 0
        image[35:60, 70:95] = 0
        visible, geometry, invalid = border_masks(image)
        self.assertGreater(float(invalid[:, :8].mean()), .9)
        self.assertLess(float(invalid[40:55, 75:90].mean()), .1)
        self.assertLess(float(geometry[:, :12].mean()), .1)
        self.assertGreater(float(visible[40:55, 75:90].mean()), .9)

    def test_quality_contract_and_cache_roundtrip(self):
        rgb = np.zeros((72, 128, 3), np.uint8)
        rgb[..., 0] = np.arange(128, dtype=np.uint8)[None]
        thermal = np.tile(np.arange(128, dtype=np.float32), (72, 1))
        ir3 = np.repeat(thermal[..., None], 3, 2)
        quality, visible, geometry, meta = quality_maps(rgb, thermal, ir3)
        self.assertEqual(quality.shape[0], len(A0_QUALITY_NAMES))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_sample(root / "samples" / "x.npz", stem="x",
                        params=(1.0, 2.0, -1.0, 1.01), confidence=.8,
                        quality=quality, visible=visible, geometry=geometry,
                        meta=meta, shape=thermal.shape, sequence="s",
                        sequence_prior=(0, 0, 0, 1), sequence_confidence=.7)
            cached = load_sample(root, "x")
            self.assertEqual(tuple(cached["quality_maps"].shape), quality.shape)
            self.assertEqual(int(cached["affine_supervised"]), 1)
            self.assertAlmostEqual(float(cached["quality_maps"][9].mean()), .8, places=2)

    def test_source_to_sampling_restores_rendered_points(self):
        shape = (80, 120)
        params = (2.0, 4.0, -3.0, 1.01)
        source_to_rgb = cv2.getRotationMatrix2D(
            ((shape[1] - 1) / 2, (shape[0] - 1) / 2), params[0], params[3])
        source_to_rgb[:, 2] += (params[1], params[2])
        sampling = source_to_sampling(params, shape)
        points = np.array([[10, 12, 1], [60, 40, 1], [105, 65, 1]], np.float32).T
        destination = source_to_rgb @ points
        restored = sampling @ np.vstack((destination, np.ones(destination.shape[1])))
        self.assertTrue(np.allclose(restored, points[:2], atol=1e-4))

    def test_plain_sample_never_inherits_unrelated_sequence_prior(self):
        coarse = {"params": np.asarray((1., 2., 3., 1.01), np.float32),
                  "confidence": .6}
        refined = {"params": np.asarray((1.1, 2.2, 3.1, 1.01), np.float32),
                   "confidence": .2}
        chosen, source = select_affine_candidate(
            coarse, refined, (-.25, .3, 1.4, 1.01), .8, False)
        self.assertEqual(source, "coarse_individual")
        self.assertTrue(np.allclose(chosen["params"], coarse["params"]))

    def test_real_sequence_can_fall_back_to_robust_prior(self):
        coarse = {"params": np.asarray((1., 2., 3., 1.01), np.float32),
                  "confidence": .6}
        refined = {"params": np.asarray((1.1, 2.2, 3.1, 1.01), np.float32),
                   "confidence": .2}
        prior = np.asarray((-.25, .3, 1.4, 1.01), np.float32)
        chosen, source = select_affine_candidate(coarse, refined, prior, .8, True)
        self.assertEqual(source, "sequence_prior")
        self.assertTrue(np.allclose(chosen["params"], prior))

    def test_affine_contract_matches_canvas_head_limits(self):
        ok, value = affine_model_contract(
            (1.0, 4.0, -3.0, 1.01), (1080, 1920), (736, 1280))
        self.assertTrue(ok)
        self.assertLessEqual(float(np.abs(value).max()), 1.05)
        ok, value = affine_model_contract(
            (1.0, 38.4, 0.0, 1.01), (1080, 1920), (736, 1280))
        self.assertFalse(ok)
        self.assertGreater(float(np.abs(value).max()), 1.05)


if __name__ == "__main__":
    unittest.main()
