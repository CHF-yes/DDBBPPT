#!/usr/bin/env python3
"""Render six-column V6.1 A0 audit sheets without training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "mm_yolo"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from mm_yolo.ir_a0 import (SearchCfg, affine_matrix, estimate_affine,
                           quality_maps, read_modalities)


def _paths(root: Path, stem: str):
    return (root / "visible" / f"{stem}.png",
            root / "infrared" / f"{stem}.png")


def _edge(image: np.ndarray):
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    positive = edge[edge > 0]
    scale = float(np.percentile(positive, 97)) if positive.size else 1.0
    return np.clip(edge / max(scale, 1e-6), 0, 1)


def _gray(image: np.ndarray):
    value = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


def _panel(image: np.ndarray, title: str, width: int = 320, height: int = 220):
    canvas = np.zeros((height, width, 3), np.uint8)
    usable_h = height - 30
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), usable_h / max(h, 1))
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    x0 = (width - resized.shape[1]) // 2
    y0 = 30 + (usable_h - resized.shape[0]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    cv2.putText(canvas, title[:52], (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _fill_exterior(clean: np.ndarray, visible: np.ndarray):
    invalid = (visible < .5).astype(np.uint8)
    if invalid.any() and (visible > .5).any():
        return cv2.inpaint(np.clip(clean, 0, 255).astype(np.uint8),
                           invalid * 255, 3, cv2.INPAINT_TELEA).astype(np.float32)
    return clean.astype(np.float32)


def _black_frame_view(clean: np.ndarray, visible: np.ndarray):
    view = _gray(clean).astype(np.float32)
    invalid = visible < .5
    view[invalid] = .30 * view[invalid] + .70 * np.asarray((20, 20, 240), np.float32)
    contours, _ = cv2.findContours((visible > .5).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return np.clip(view, 0, 255).astype(np.uint8)


def _warp(clean: np.ndarray, params):
    h, w = clean.shape
    matrix = affine_matrix(float(params[0]), float(params[1]), float(params[2]),
                           float(params[3]), (h, w))
    return cv2.warpAffine(clean, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _overlay(rgb: np.ndarray, aligned: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # BGR visualization: RGB edge is green, aligned IR edge is red.
    return (np.dstack((np.zeros_like(gray, np.float32), _edge(gray),
                       _edge(aligned))) * 255).astype(np.uint8)


def _old_params(cache: Path | None, stem: str):
    if cache is None:
        return np.asarray((0., 0., 0., 1.), np.float32)
    path = cache / "samples" / f"{stem}.npz"
    if not path.is_file():
        return np.asarray((0., 0., 0., 1.), np.float32)
    with np.load(path, allow_pickle=False) as sample:
        return np.asarray(sample["source_to_rgb_params"], np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-cache")
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--work-width", type=int, default=480)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.out)
    old_cache = Path(args.old_cache) if args.old_cache else None
    output.mkdir(parents=True, exist_ok=False)
    config = SearchCfg(work_width=args.work_width)
    rows, sheets = [], []

    for stem in args.stems:
        rgb_path, ir_path = _paths(root, stem)
        if not rgb_path.is_file() or not ir_path.is_file():
            rows.append({"stem": stem, "missing": True})
            continue

        rgb, thermal, ir3 = read_modalities(rgb_path, ir_path)
        quality, visible, geometry, meta, deghosted = quality_maps(
            rgb, thermal, ir3, return_preprocessed=True)
        geometry_ir = _fill_exterior(deghosted, visible)
        prior = _old_params(old_cache, stem)
        result = estimate_affine(rgb, geometry_ir, geometry, config, prior=prior)
        params = np.asarray(result["params"], np.float32)
        aligned = _warp(geometry_ir, params)
        black_area = int(round(float((visible < .5).sum())))

        panels = [
            _panel(rgb[..., ::-1], "C1 original RGB"),
            _panel(_gray(thermal), "C2 original IR"),
            _panel(_gray(deghosted),
                   f"C3 deghosted IR conf={meta['ghost_score']:.3f}"),
            _panel(_black_frame_view(deghosted, visible),
                   f"C4 black frame area={black_area}px"),
            _panel(_gray(aligned),
                   f"C5 aligned IR {result['selected_mode']} a={params[0]:+.2f}"),
            _panel(_overlay(rgb, aligned),
                   "C6 overlap RGB=green IR=red"),
        ]
        sheet = np.hstack(panels)
        cv2.imwrite(str(output / f"{stem}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        sheets.append(sheet)
        row = {
            "stem": stem,
            "missing": False,
            "params": [float(value) for value in params],
            "selected_mode": result["selected_mode"],
            "confidence": float(result["confidence"]),
            "directional_projection_gain": float(result["directional_projection_gain"]),
            "ghost_confidence": float(meta["ghost_score"]),
            "black_border_area_px": black_area,
            "source_shape": [int(thermal.shape[0]), int(thermal.shape[1])],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if sheets:
        cv2.imwrite(str(output / "overview_12_samples.jpg"), np.vstack(sheets),
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
    (output / "audit.json").write_text(
        json.dumps({"version": "v61-a0-v7-six-columns", "samples": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
