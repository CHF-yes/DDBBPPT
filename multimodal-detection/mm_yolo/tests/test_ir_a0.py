import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))

from ir_a0 import (A0_QUALITY_NAMES, SearchCfg, affine_matrix, affine_model_contract,
                   border_masks, compose_source_affine, deghost_for_a0,
                   estimate_affine, load_sample, quality_maps, save_sample,
                   select_affine_candidate, source_to_sampling)


class IRA0Tests(unittest.TestCase):
    def test_rgb_ghost_is_detected_and_reduced_before_border_analysis(self):
        rng = np.random.default_rng(13)
        h, w = 120, 192
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2)
        truth = np.full((h, w), 92, np.float32)
        truth[:, :24] = 4
        truth[:10] = 4
        truth[38:83, 72:132] = 145
        rgbf = rgb.astype(np.float32)
        rgb_high = rgbf - cv2.GaussianBlur(rgbf, (0, 0), 1.5)
        ghost = np.einsum(
            "...c,cd->...d", rgb_high,
            np.asarray([[.18, .03, .02], [.02, .16, .03], [.03, .02, .17]], np.float32))
        ir3 = np.clip(np.repeat(truth[..., None], 3, 2) + ghost, 0, 255)
        observed = np.median(ir3, axis=2).astype(np.float32)
        clean, _, residual_mask, support, meta = deghost_for_a0(
            rgb, observed, ir3)
        fitted = support > .5
        self.assertGreater(meta["ghost_identity_fit"], meta["ghost_control_fit"])
        self.assertGreater(meta["ghost_score"], .05)
        self.assertGreater(meta["ghost_detected_mask_ratio"], 0)
        self.assertLessEqual(meta["ghost_residual_mask_ratio"],
                             meta["ghost_detected_mask_ratio"])
        self.assertEqual(meta["ghost_mask_ratio"],
                         meta["ghost_residual_mask_ratio"])
        self.assertGreaterEqual(float(residual_mask.min()), 0)
        self.assertLess(float(np.abs(clean[fitted] - truth[fitted]).mean()),
                        float(np.abs(observed[fitted] - truth[fitted]).mean()))

    def test_unrelated_rgb_does_not_rewrite_clean_thermal_proxy(self):
        rng = np.random.default_rng(7)
        h, w = 96, 160
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        thermal = np.full((h, w), 80, np.float32)
        thermal[:, :16] = 0
        thermal[25:70, 55:110] = 132
        ir3 = np.repeat(thermal[..., None], 3, 2)
        clean, _, _, _, meta = deghost_for_a0(rgb, thermal, ir3)
        self.assertLess(meta["ghost_score"], .10)
        self.assertLess(float(np.abs(clean - thermal).mean()), .1)

    def test_polygon_exterior_is_masked_but_dark_object_inside_is_not(self):
        image = np.full((120, 192), 4, np.float32)
        polygon = np.asarray([[18, 12], [178, 18], [170, 108], [12, 102]], np.int32)
        cv2.fillConvexPoly(image, polygon, 100)
        image[45:72, 78:108] = 0
        visible, geometry, invalid = border_masks(image)
        self.assertGreater(float(invalid[:6, :6].mean()), .9)
        self.assertLess(float(invalid[50:65, 85:100].mean()), .1)
        self.assertLess(float(geometry[:8, :8].mean()), .1)
        self.assertGreater(float(visible[50:65, 85:100].mean()), .9)

    def test_irregular_shallow_dark_scenery_is_not_a_black_frame(self):
        image = np.full((120, 192), 55, np.float32)
        irregular = np.asarray([
            [10, 24], [62, 10], [88, 31], [128, 12], [181, 28],
            [166, 58], [184, 98], [124, 111], [91, 88], [47, 108],
            [9, 84], [31, 55],
        ], np.int32)
        cv2.fillPoly(image, [irregular], 95)
        image[45:70, 78:111] = 72
        visible, geometry, invalid = border_masks(image)
        self.assertLess(float(invalid.mean()), .01)
        self.assertGreater(float(visible.mean()), .99)
        self.assertGreater(float(geometry.mean()), .99)

    def test_dark_side_bands_are_not_an_enclosing_polygon_frame(self):
        image = np.full((120, 192), 96, np.float32)
        image[:, :22] = 48
        image[:, -25:] = 51
        image[28:91, 72:126] = 128
        visible, geometry, invalid = border_masks(image)
        self.assertLess(float(invalid.mean()), .01)
        self.assertGreater(float(visible.mean()), .99)
        self.assertGreater(float(geometry.mean()), .99)

    def test_wide_rotation_search_exceeds_old_three_degree_limit(self):
        h, w = 160, 240
        gray = np.zeros((h, w), np.uint8)
        cv2.rectangle(gray, (34, 28), (198, 126), 120, 3)
        cv2.line(gray, (45, 115), (184, 42), 230, 4)
        cv2.circle(gray, (150, 92), 22, 180, 3)
        rgb = np.repeat(gray[..., None], 3, 2)
        source_to_rgb = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2),
                                                16.0, 1.0)
        rgb_to_source = cv2.invertAffineTransform(source_to_rgb)
        thermal = cv2.warpAffine(gray, rgb_to_source, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        result = estimate_affine(rgb, thermal, np.ones((h, w), np.float32),
                                 SearchCfg(work_width=w, angle_step=4.0))
        self.assertGreater(abs(float(result["params"][0])), 8.0)
        self.assertLess(abs(float(result["params"][0]) - 16.0), 1.5)

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
            self.assertEqual(tuple(cached["ghost_probability"].shape), quality.shape[1:])
            self.assertEqual(tuple(cached["thermal_confidence_map"].shape), quality.shape[1:])
            self.assertAlmostEqual(float(cached["coarse_candidate_available"]), 1.0)

    def test_rectangle_intersection_border_supports_both_image_sizes(self):
        for h, w in ((120, 192), (240, 320)):
            image = np.full((h, w), 3, np.float32)
            rect = ((w * .51, h * .49), (w * .82, h * .76), 7.0)
            polygon = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillConvexPoly(image, polygon, 104)
            visible, geometry, invalid = border_masks(image)
            self.assertGreater(float(invalid[0, 0]), .9)
            self.assertGreater(float(visible[h // 2, w // 2]), .9)
            self.assertGreater(float(geometry[h // 2, w // 2]), .9)

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

    def test_composed_sampling_matches_base_then_delta_inverse_lookup(self):
        shape = (160, 240)
        base = (11.0, 18.0, -9.0, 1.08)
        delta = (-7.5, -13.0, 15.0, 0.93)
        total_params, total_source = compose_source_affine(base, delta, shape)
        base_sampling = np.vstack((source_to_sampling(base, shape), [0, 0, 1]))
        delta_sampling = np.vstack((source_to_sampling(delta, shape), [0, 0, 1]))
        total_sampling = np.vstack((source_to_sampling(total_params, shape), [0, 0, 1]))
        direct_sampling = np.linalg.inv(
            np.vstack((total_source, [0, 0, 1]))).astype(np.float32)
        self.assertTrue(np.allclose(total_sampling, base_sampling @ delta_sampling,
                                    atol=2e-4))
        self.assertTrue(np.allclose(total_sampling, direct_sampling, atol=2e-4))

    def test_large_rotation_shift_and_scale_are_inside_a0_search(self):
        h, w = 160, 240
        gray = np.zeros((h, w), np.uint8)
        cv2.rectangle(gray, (28, 22), (204, 133), 110, 3)
        cv2.line(gray, (35, 124), (192, 35), 235, 5)
        cv2.circle(gray, (153, 91), 20, 175, 4)
        cv2.rectangle(gray, (69, 49), (101, 78), 205, -1)
        rgb = np.repeat(gray[..., None], 3, 2)
        truth = (22.0, .15 * w, -.12 * h, 1.18)
        source_to_rgb = cv2.getRotationMatrix2D(
            ((w - 1) / 2, (h - 1) / 2), truth[0], truth[3])
        source_to_rgb[:, 2] += truth[1:3]
        thermal = cv2.warpAffine(
            gray, cv2.invertAffineTransform(source_to_rgb), (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        result = estimate_affine(
            rgb, thermal, np.ones((h, w), np.float32),
            SearchCfg(work_width=w, angle_step=5.0))
        found = result["params"]
        self.assertLess(abs(float(found[0]) - truth[0]), 2.0)
        self.assertLess(abs(float(found[1]) - truth[1]), 8.0)
        self.assertLess(abs(float(found[2]) - truth[2]), 8.0)
        self.assertLess(abs(float(found[3]) - truth[3]), .08)

    def test_small_shift_prefers_translation_without_rotation_or_scale(self):
        h, w = 160, 240
        gray = np.zeros((h, w), np.uint8)
        cv2.rectangle(gray, (25, 20), (210, 135), 90, 3)
        cv2.line(gray, (31, 126), (194, 37), 230, 5)
        cv2.circle(gray, (146, 84), 19, 170, 4)
        cv2.rectangle(gray, (70, 48), (103, 76), 205, -1)
        rgb = np.repeat(gray[..., None], 3, 2)
        truth = (0.0, -.05 * w, -.075 * h, 1.0)
        source_to_rgb = affine_matrix(*truth, (h, w))
        thermal = cv2.warpAffine(
            gray, cv2.invertAffineTransform(source_to_rgb), (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        result = estimate_affine(
            rgb, thermal, np.ones((h, w), np.float32), SearchCfg(work_width=w))
        found = result["params"]
        self.assertEqual(result["selected_mode"], "raw_translation")
        self.assertAlmostEqual(float(found[0]), 0.0, places=4)
        self.assertAlmostEqual(float(found[3]), 1.0, places=4)
        self.assertLess(abs(float(found[1]) - truth[1]), 4.0)
        self.assertLess(abs(float(found[2]) - truth[2]), 4.0)

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

        # V5.1.1 keeps a small margin above the observed three-degree search
        # boundary so high-confidence rotations do not saturate the head.
        ok, value = affine_model_contract(
            (3.25, 0.0, 0.0, 1.0), (1080, 1920), (736, 1280),
            angle_limit=4.0)
        self.assertTrue(ok)
        self.assertLessEqual(float(np.abs(value).max()), 1.05)


if __name__ == "__main__":
    unittest.main()
