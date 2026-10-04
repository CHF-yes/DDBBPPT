"""Render paired RGB/IR/depth validation images for a registration audit.

This only reads source images and a split manifest. It never warps images or
changes annotations, and uses no detector weights.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    low, high = np.percentile(image, [2, 98])
    return np.clip((image - low) * 255.0 / max(high - low, 1.0), 0, 255).astype(np.uint8)


def edge_overlay(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    ref = cv2.Canny(reference, 50, 120)
    alt = cv2.Canny(other, 50, 120)
    out = cv2.cvtColor((reference * 0.3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out[ref > 0] = (0, 220, 0)
    out[alt > 0] = (220, 0, 220)
    out[(ref > 0) & (alt > 0)] = (255, 255, 255)
    return out


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 29), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    ids = list(json.loads(args.split.read_text())["val"])
    random.Random(42).shuffle(ids)
    chosen = ids[:args.count]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    used = []
    for stem in chosen:
        images = []
        for mode in ("visible", "infrared", "depth"):
            filename = next((p for extension in (".jpg", ".png", ".jpeg")
                             if (p := args.root / mode / f"{stem}{extension}").exists()), None)
            if filename is None:
                raise FileNotFoundError(f"{args.root / mode}/{stem}.[jpg|png|jpeg]")
            image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(filename)
            images.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        rgb, ir, depth = images
        rgb_gray, ir_gray, depth_gray = map(gray_u8, images)
        panels = [
            label(rgb, f"{stem} RGB"),
            label(cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR), "IR"),
            label(cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR), "Depth"),
            label(edge_overlay(rgb_gray, ir_gray), "RGB green / IR magenta"),
            label(edge_overlay(rgb_gray, depth_gray), "RGB green / Depth magenta"),
        ]
        rows.append(np.hstack(panels))
        used.append(stem)
    for page in range((len(rows) + 5) // 6):
        sheet = np.vstack(rows[page * 6:(page + 1) * 6])
        cv2.imwrite(str(args.out / f"alignment_contact_{page + 1:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    (args.out / "sample_ids.json").write_text(json.dumps(used, indent=2) + "\n")
    print(json.dumps({"samples": len(used), "pages": (len(rows) + 5) // 6,
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
