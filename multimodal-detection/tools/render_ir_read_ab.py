#!/usr/bin/env python3
"""Render a local, non-geometric A/B preview for IR channel reduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


def decode(zf: ZipFile, member: str) -> np.ndarray:
    data = np.frombuffer(zf.read(member), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot decode {member}")
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with ZipFile(args.zip) as zf:
        rgb_bgr = decode(zf, f"visible/{args.stem}")
        ir_bgr = decode(zf, f"infrared/{args.stem}")

    if ir_bgr.ndim == 2:
        legacy = ir_bgr
        median = ir_bgr.copy()
        channels = np.repeat(ir_bgr[:, :, None], 3, axis=2)
    else:
        channels = ir_bgr[:, :, :3]
        legacy = channels[:, :, 0]
        median = np.median(channels, axis=2).astype(np.uint8)

    delta = legacy.astype(np.int16) - median.astype(np.int16)
    delta_abs = np.abs(delta).astype(np.uint8)
    amplified = np.clip(delta.astype(np.float32) * 16.0 + 128.0, 0, 255).astype(np.uint8)
    edges_rgb = cv2.Canny(cv2.cvtColor(rgb_bgr[:, :, :3], cv2.COLOR_BGR2GRAY), 80, 160)
    edges_ir = cv2.Canny(median, 80, 160)
    edge_overlay = np.zeros((*median.shape, 3), dtype=np.uint8)
    edge_overlay[:, :, 2] = edges_rgb
    edge_overlay[:, :, 1] = edges_ir

    panel_size = (640, 360)
    panels = [
        label(fit(rgb_bgr, panel_size), "RGB (BGR display)"),
        label(fit(cv2.cvtColor(legacy, cv2.COLOR_GRAY2BGR), panel_size), "IR legacy: channel 0"),
        label(fit(cv2.cvtColor(median, cv2.COLOR_GRAY2BGR), panel_size), "IR median channel"),
        label(fit(cv2.cvtColor(amplified, cv2.COLOR_GRAY2BGR), panel_size),
              "(legacy - median) x16 + 128"),
        label(fit(cv2.cvtColor(delta_abs, cv2.COLOR_GRAY2BGR), panel_size), "absolute difference"),
        label(fit(edge_overlay, panel_size), "edges: RGB red / IR green"),
    ]
    contact = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])

    args.out.mkdir(parents=True, exist_ok=True)
    out_image = args.out / f"{Path(args.stem).stem}_ir_read_ab.jpg"
    ok, encoded = cv2.imencode(".jpg", contact, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"cannot encode {out_image}")
    encoded.tofile(out_image)

    stats = {
        "stem": args.stem,
        "ir_shape": list(ir_bgr.shape),
        "legacy_median_mae_gray": float(delta_abs.mean()),
        "legacy_median_p95_gray": float(np.percentile(delta_abs, 95)),
        "legacy_median_max_gray": int(delta_abs.max()),
        "pixels_changed_fraction": float((delta_abs > 0).mean()),
        "channel_mean_bgr": [float(channels[:, :, i].mean()) for i in range(3)],
        "channel_std_bgr": [float(channels[:, :, i].std()) for i in range(3)],
        "note": "This compares channel reduction only; it does not perform geometric alignment.",
    }
    out_json = args.out / f"{Path(args.stem).stem}_ir_read_ab.json"
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(out_image), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
