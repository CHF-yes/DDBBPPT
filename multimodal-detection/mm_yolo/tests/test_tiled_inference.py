"""切片坐标、模态同步和全图合并的回归检查。"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data  # noqa: E402
from tiled_inference import (decode_full_and_tiles, merge_detections,
                             tile_to_full_canvas, tile_windows)  # noqa: E402


def test_windows_cover_image_with_overlap():
    windows = tile_windows(80, 120, 0.6, 0.2)
    cover = np.zeros((80, 120), np.uint8)
    for x0, y0, x1, y1 in windows:
        cover[y0:y1, x0:x1] += 1
    assert len(windows) == 4
    assert cover.min() >= 1
    assert cover.max() > 1


def test_crop_keeps_modalities_and_full_image_coordinates(monkeypatch):
    h, w = 96, 128
    yy, xx = np.mgrid[:h, :w]
    rgb = np.stack((xx, yy, xx + yy), axis=2).astype(np.uint8)
    ir = (xx + yy).astype(np.uint8)
    depth = (1000 + xx + yy).astype(np.float32)
    valid = np.ones((h, w), bool)
    monkeypatch.setattr(data, "read_rgb", lambda _: rgb)
    monkeypatch.setattr(data, "read_ir", lambda *args, **kwargs: ir)
    monkeypatch.setattr(data, "read_depth", lambda *args, **kwargs:
                        (depth, valid, True))
    base = {"stem": "one", "files": {"visible": "one.png", "infrared": "one.png",
                                      "depth": "one.png"}, "boxes": None}
    window = (32, 16, 96, 80)
    ds = data.MMDataset(Path("/unused"), [{**base, "crop_xyxy": window}],
                        imgsz=(64, 64), train=False)
    item = ds[0]
    assert item is not None
    assert item["orig_hw"].tolist() == [96.0, 128.0]
    # Crop and canvas have the same size: pixel (0,0) comes from original (32,16).
    assert np.isclose(item["rgb"][0, 0, 0].item() * 255, rgb[16, 32, 0])
    assert np.isclose(item["ir"][0, 0, 0].item() * 255, ir[16, 32])
    assert item["depth"][2].min().item() == 1.0
    assert np.allclose(item["M"].numpy(), [[1, 0, -32], [0, 1, -16]])

    full_M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    detections = np.array([[10, 10, 20, 20, 0.8, 7]], np.float32)
    projected = tile_to_full_canvas(detections, item["M"].numpy(), full_M,
                                    (96, 128), window, (64, 64))
    assert np.allclose(projected[0, :4], [42, 26, 52, 36])


def test_merge_is_class_aware_and_respects_limit():
    full = np.array([[10, 10, 30, 30, .8, 7],
                     [10, 10, 30, 30, .7, 8]], np.float32)
    tile = np.array([[11, 11, 31, 31, .9, 7],
                     [50, 50, 60, 60, .6, 7]], np.float32)
    merged = merge_detections([full, tile], iou=.6, max_det=3)
    assert len(merged) == 3
    assert np.allclose(merged[:, 4], [.9, .7, .6])
    assert merged[:, 5].tolist() == [7, 8, 7]


def test_full_and_tiles_run_through_same_dataset(monkeypatch):
    h, w = 96, 128
    rgb = np.full((h, w, 3), 80, np.uint8)
    ir = np.full((h, w), 120, np.uint8)
    depth = np.full((h, w), 1000, np.float32)
    monkeypatch.setattr(data, "read_rgb", lambda _: rgb)
    monkeypatch.setattr(data, "read_ir", lambda *args, **kwargs: ir)
    monkeypatch.setattr(data, "read_depth", lambda *args, **kwargs:
                        (depth, np.ones((h, w), bool), True))
    sample = {"stem": "one", "files": {"visible": "one.png", "infrared": "one.png",
                                        "depth": "one.png"}, "boxes": None}
    ds = data.MMDataset(Path("/unused"), [sample], imgsz=(64, 64), train=False)
    full_batch = data.collate([ds[0]])
    calls = []

    def fake_decode(model, batch, *args):
        calls.append((batch["rgb"].shape[0], tuple(batch["quality"])))
        return [np.array([[20, 20, 30, 30, .8, 7]], np.float32)
                for _ in range(batch["rgb"].shape[0])]

    monkeypatch.setitem(sys.modules, "eval", types.SimpleNamespace(_decode=fake_decode))
    result = decode_full_and_tiles(None, ds, [sample], full_batch,
                                   [np.zeros((0, 6), np.float32)], "cpu",
                                   .25, .7, 100, "all", None, (64, 64),
                                   tile_batch=2)
    assert calls == [(2, ("rgb", "ir", "dep", "availability", "scene_id")),
                     (2, ("rgb", "ir", "dep", "availability", "scene_id"))]
    assert len(result) == 1
    assert len(result[0]) == 4
    assert np.all(result[0][:, :4] >= 0)


def test_ordinary_training_affine_still_tracks_rotation(monkeypatch):
    h, w = 96, 128
    monkeypatch.setattr(data, "read_rgb", lambda _: np.full((h, w, 3), 80, np.uint8))
    monkeypatch.setattr(data, "read_ir", lambda *args, **kwargs:
                        np.full((h, w), 120, np.uint8))
    monkeypatch.setattr(data, "read_depth", lambda *args, **kwargs:
                        (np.full((h, w), 1000, np.float32), np.ones((h, w), bool), True))
    boxes = np.array([[7, .5, .5, .1, .1]], np.float32)
    sample = {"stem": "one", "files": {"visible": "one.png", "infrared": "one.png",
                                        "depth": "one.png"}, "boxes": boxes}
    aug = data.AugCfg(imgsz=(96, 128), scale_range=(1, 1), translate=0,
                      rotate_deg=10, hflip_p=0, mosaic_p=0)
    item = data.MMDataset(Path("/unused"), [sample], imgsz=(96, 128),
                          train=True, aug=aug, seed=5)[0]
    expected = data.canvas_boxes_from_norm(boxes, item["M"].numpy(),
                                           (h, w), (96, 128),
                                           min_size=aug.box_min_size,
                                           require_center=aug.box_require_center)
    assert np.allclose(item["boxes"].numpy(), expected)
