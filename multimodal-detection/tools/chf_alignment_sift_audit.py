"""Conservative SIFT/RANSAC diagnostic for paired RGB, IR, and depth images.

Cross-modal SIFT can fail even on correctly aligned pairs. Results are triage
signals only; no warp is applied to training or validation images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def locate(root: Path, mode: str, stem: str) -> Path:
    for suffix in (".jpg", ".png", ".jpeg"):
        path = root / mode / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"{root / mode}/{stem}")


def prepare(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def assess(rgb: np.ndarray, aux: np.ndarray, sift: cv2.SIFT, matcher: cv2.BFMatcher) -> dict:
    kp0, desc0 = sift.detectAndCompute(rgb, None)
    kp1, desc1 = sift.detectAndCompute(aux, None)
    result = {"keypoints_rgb": len(kp0), "keypoints_aux": len(kp1),
              "ratio_matches": 0, "inliers": 0, "status": "insufficient_matches"}
    if desc0 is None or desc1 is None:
        return result
    forward = matcher.knnMatch(desc1, desc0, k=2)
    good = [pair[0] for pair in forward
            if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    result["ratio_matches"] = len(good)
    if len(good) < 12:
        return result
    source = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([kp0[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.99)
    if matrix is None or inlier_mask is None:
        return result
    inliers = inlier_mask[:, 0].astype(bool)
    count = int(inliers.sum())
    result["inliers"] = count
    if count < 12:
        return result
    points = source[inliers, 0]
    span = np.ptp(points, axis=0)
    displacement = target[inliers, 0] - source[inliers, 0]
    transformed = cv2.transform(source[inliers], matrix)[:, 0]
    error = np.linalg.norm(transformed - target[inliers, 0], axis=1)
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    # Repeated textures can make RANSAC collapse to a zero-scale point map.
    # Such a fit has zero reprojection error but carries no alignment evidence.
    if not 0.8 <= scale <= 1.25 or not math.isfinite(angle):
        result.update(status="degenerate_fit", affine_scale=scale,
                      affine_angle_deg=angle)
        return result
    result.update(
        status="spatially_supported" if min(span) >= 90 else "localized_matches",
        median_dx_960=float(np.median(displacement[:, 0])),
        median_dy_540=float(np.median(displacement[:, 1])),
        affine_tx_960=float(matrix[0, 2]),
        affine_ty_540=float(matrix[1, 2]),
        affine_angle_deg=angle,
        affine_scale=scale,
        median_reprojection_px=float(np.median(error)),
        inlier_span_x=float(span[0]),
        inlier_span_y=float(span[1]),
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sample-ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for stem in json.loads(args.sample_ids.read_text()):
        rgb = prepare(locate(args.root, "visible", stem))
        for mode in ("infrared", "depth"):
            aux = prepare(locate(args.root, mode, stem))
            row = {"stem": stem, "mode": mode,
                   **assess(rgb, aux, sift, matcher)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    csv_path = args.out.with_suffix(".csv")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
