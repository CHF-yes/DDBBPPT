# -*- coding: utf-8 -*-
"""Regression tests for checkpointed IR channel reduction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_HERE = Path(__file__).resolve().parent
_MM = _HERE.parent
_CODE = _MM.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_MM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import MMConfig  # noqa: E402
from data import read_ir  # noqa: E402


class IRReadModeTests(unittest.TestCase):
    def test_legacy_mode_uses_opencv_first_channel_exactly(self):
        bgr = np.array(
            [[[7, 101, 203], [19, 21, 23]], [[31, 33, 35], [249, 3, 127]]],
            dtype=np.uint8,
        )
        with patch("data.imread_unicode", return_value=bgr):
            got = read_ir("unused.png", mode="legacy_first_channel")
        np.testing.assert_array_equal(got, bgr[:, :, 0])

    def test_median_mode_uses_per_pixel_channel_median(self):
        bgr = np.array(
            [[[7, 101, 203], [19, 21, 23]], [[31, 33, 35], [249, 3, 127]]],
            dtype=np.uint8,
        )
        expected = np.array([[101, 21], [33, 127]], dtype=np.uint8)
        with patch("data.imread_unicode", return_value=bgr):
            got = read_ir("unused.png", mode="median_channel")
        np.testing.assert_array_equal(got, expected)
        self.assertEqual(got.dtype, np.uint8)

    def test_identical_gray_channels_are_bit_exact_in_both_modes(self):
        gray = np.array([[0, 1, 127], [128, 254, 255]], dtype=np.uint8)
        bgr = np.repeat(gray[:, :, None], 3, axis=2)
        with patch("data.imread_unicode", return_value=bgr):
            legacy = read_ir("unused.png", mode="legacy_first_channel")
        with patch("data.imread_unicode", return_value=bgr):
            median = read_ir("unused.png", mode="median_channel")
        np.testing.assert_array_equal(legacy, gray)
        np.testing.assert_array_equal(median, gray)

    def test_uint16_median_keeps_legacy_scale_contract(self):
        bgr = np.array([[[256, 512, 768], [1024, 1280, 1536]]], dtype=np.uint16)
        with patch("data.imread_unicode", return_value=bgr):
            got = read_ir("unused.png", mode="median_channel")
        np.testing.assert_array_equal(got, np.array([[2, 5]], dtype=np.uint8))

    def test_invalid_mode_fails_closed(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch("data.imread_unicode", return_value=image):
            with self.assertRaisesRegex(ValueError, "unknown IR read mode"):
                read_ir("unused.png", mode="not_a_mode")

    def test_structure_roundtrip_and_legacy_default(self):
        cfg = MMConfig(ir_read_mode="median_channel")
        restored = MMConfig.from_structure(cfg.structure())
        self.assertEqual(restored.ir_read_mode, "median_channel")

        legacy = MMConfig.from_structure({"encoder": {}, "fusion": {}})
        self.assertEqual(legacy.ir_read_mode, "legacy_first_channel")


if __name__ == "__main__":
    unittest.main()
