#!/usr/bin/env python3
"""Render a no-training audit of the V6.1 A0 deghost/border ordering."""
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

from mm_yolo.ir_a0 import (SearchCfg, affine_matrix, deghost_for_a0,
                           estimate_affine, quality_maps, read_modalities)


def _paths(root: Path, stem: str):
    return root / "visible" / f"{stem}.png", root / "infrared" / f"{stem}.png"


def _edge(x: np.ndarray):
    gx = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.magnitude(gx, gy)
    scale = float(np.percentile(e[e > 0], 97)) if np.any(e > 0) else 1.0
    return np.clip(e / max(scale, 1e-6), 0, 1)


def _overlay(rgb: np.ndarray, thermal: np.ndarray, params):
    h, w = thermal.shape
    m = affine_matrix(float(params[0]), float(params[1]), float(params[2]),
                      float(params[3]), (h, w))
    aligned = cv2.warpAffine(thermal, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # BGR output: IR edges red, RGB edges green.
    return (np.dstack((np.zeros_like(gray, np.float32), _edge(gray),
                       _edge(aligned))) * 255).astype(np.uint8)


def _aligned(thermal: np.ndarray, params):
    h, w = thermal.shape
    m = affine_matrix(float(params[0]), float(params[1]), float(params[2]),
                      float(params[3]), (h, w))
    return cv2.warpAffine(thermal, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _gray(x: np.ndarray):
    y = np.clip(x, 0, 255).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def _mask(x: np.ndarray, color: int = cv2.COLORMAP_TURBO):
    return cv2.applyColorMap(np.clip(x * 255, 0, 255).astype(np.uint8), color)


def _panel(image: np.ndarray, title: str, width: int = 360):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (width, max(80, round(width * h / w))),
                         interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (resized.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(resized, title[:55], (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (255, 255, 255), 1, cv2.LINE_AA)
    return resized


def _old_row(cache: Path, stem: str):
    path = cache / "samples" / f"{stem}.npz"
    if not path.is_file():
        return np.asarray((0, 0, 0, 1), np.float32), 0.0
    with np.load(path, allow_pickle=False) as z:
        return (np.asarray(z["source_to_rgb_params"], np.float32),
                float(z["affine_confidence"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--old-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stems", nargs="+", required=True)
    p.add_argument("--work-width", type=int, default=480)
    a = p.parse_args()
    root, old_cache, out = Path(a.root), Path(a.old_cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SearchCfg(work_width=a.work_width)
    rows = []
    for stem in a.stems:
        rgb_path, ir_path = _paths(root, stem)
        if not rgb_path.is_file() or not ir_path.is_file():
            rows.append({"stem": stem, "missing": True})
            continue
        rgb, thermal, ir3 = read_modalities(rgb_path, ir_path)
        clean, _, ghost, fit_support, ghost_meta = deghost_for_a0(rgb, thermal, ir3)
        _, visible, geometry, meta, geometry_thermal = quality_maps(
            rgb, thermal, ir3, return_preprocessed=True)
        old_params, old_conf = _old_row(old_cache, stem)
        new = estimate_affine(rgb, geometry_thermal, geometry, cfg,
                              prior=old_params)
        new_params = np.asarray(new["params"], np.float32)
        invalid = 1 - visible
        aligned = _aligned(geometry_thermal, new_params)
        panels = [
            _panel(rgb[..., ::-1], "C1 RGB"),
            _panel(_gray(thermal), "C2 raw IR"),
            _panel(_gray(clean),
                   f"C3 deghosted IR ghost={meta['ghost_score']:.2f} residual={meta.get('ghost_residual_fit', 0):.2f}"),
            _panel(_gray(geometry_thermal),
                   f"C4 black-frame removed IR invalid={invalid.mean():.3f}"),
            _panel(_gray(aligned),
                   f"C5 affine IR {new['selected_mode']} a={new_params[0]:+.2f} s={new_params[3]:.3f}"),
            _panel(_overlay(rgb, geometry_thermal, new_params),
                   "C6 RGB/IR edge overlap"),
        ]
        sheet = np.hstack(panels)
        cv2.imwrite(str(out / f"{stem}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        rows.append({
            "stem": stem, "missing": False,
            "old_params": [float(x) for x in old_params],
            "old_confidence": old_conf,
            "new_params": [float(x) for x in new_params],
            "residual_params": [float(x) for x in new["residual_params"]],
            "search_bounds": new["search_bounds"],
            "new_score": float(new["score"]),
            "new_identity_score": float(new["identity_score"]),
            "new_old_a0_score": float(new["old_a0_score"]),
            "new_baseline_mode": new["baseline_mode"],
            "new_improvement": float(new["improvement"]),
            "new_uniqueness": float(new["uniqueness"]),
            "new_confidence": float(new["confidence"]),
            "selected_mode": new["selected_mode"],
            "mode_diagnostics": new["mode_diagnostics"],
            "common_support": float(new["common_support"]),
            "tile_positive": int(new["tile_positive"]),
            "tile_median_gain": float(new["tile_median_gain"]),
            "tile_worst_gain": float(new["tile_worst_gain"]),
            **{k: float(v) for k, v in ghost_meta.items()},
            "final_black_ratio": float(invalid.mean()),
            "geometry_valid_ratio": float(geometry.mean()),
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (out / "audit.json").write_text(
        json.dumps({"version": "v61-a0-deghost-rotation-v10", "samples": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
