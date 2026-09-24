#!/usr/bin/env python3
"""Build the V5.1.1 offline IR A0 cache and visual audit sheets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "mm_yolo"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from mm_yolo.data import build_index, source_group  # noqa: E402
from mm_yolo.ir_a0 import (SearchCfg, estimate_affine, quality_maps, read_modalities,
                           robust_sequence_prior, save_sample, write_manifest)  # noqa: E402


def _paths(root: Path, sample: dict):
    return (root / "visible" / sample["files"]["visible"],
            root / "infrared" / sample["files"]["infrared"])


def _preview(rgb, thermal, visible, geometry, params, title):
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2),
                               float(params[0]), float(params[3]))
    m[:, 2] += (float(params[1]), float(params[2]))
    aligned = cv2.warpAffine(thermal, m, (w, h), flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    def edges(x):
        gx = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        e = cv2.magnitude(gx, gy); return np.clip(e / (np.percentile(e, 97) + 1e-6), 0, 1)
    overlay0 = np.dstack((edges(thermal), edges(gray), np.zeros_like(gray, np.float32)))
    overlay1 = np.dstack((edges(aligned), edges(gray), np.zeros_like(gray, np.float32)))
    panels = [rgb[..., ::-1], cv2.cvtColor(thermal.astype(np.uint8), cv2.COLOR_GRAY2BGR),
              (np.dstack([visible, geometry, np.zeros_like(visible)]) * 255).astype(np.uint8),
              (overlay0 * 255).astype(np.uint8), (overlay1 * 255).astype(np.uint8)]
    panels = [cv2.resize(x, (480, round(480 * h / w)), interpolation=cv2.INTER_AREA) for x in panels]
    sheet = np.hstack(panels)
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(sheet, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--labels", default="",
                   help="optional label directory; omitted for an unlabeled test set")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--work-width", type=int, default=480)
    p.add_argument("--preview-count", type=int, default=80)
    p.add_argument("--min-confidence", type=float, default=.45)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    samples = build_index(root, Path(a.labels) if a.labels else None, limit=a.limit)
    cfg = SearchCfg(work_width=a.work_width, min_confidence=a.min_confidence)
    first, groups, loaded = {}, defaultdict(list), {}
    for i, sample in enumerate(samples):
        stem = sample["stem"]
        rgb, thermal, ir3 = read_modalities(*_paths(root, sample))
        q, visible, geometry, meta = quality_maps(rgb, thermal, ir3)
        estimate = estimate_affine(rgb, thermal, geometry, cfg)
        first[stem] = estimate
        groups[source_group(stem)].append(estimate)
        loaded[stem] = (rgb, thermal, ir3, q, visible, geometry, meta)
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"[A0] coarse {i+1}/{len(samples)}", flush=True)
    priors = {g: robust_sequence_prior(rows) for g, rows in groups.items()}
    rows, previews = [], []
    for i, sample in enumerate(samples):
        stem, group = sample["stem"], source_group(sample["stem"])
        rgb, thermal, ir3, q, visible, geometry, meta = loaded[stem]
        prior, prior_conf = priors[group]
        refined = estimate_affine(rgb, thermal, geometry, cfg, prior=prior)
        # A residual is accepted only if it is at least as trustworthy as the
        # coarse candidate.  Otherwise the robust sequence prior is safer.
        if refined["confidence"] >= max(.35, .8 * first[stem]["confidence"]):
            chosen = refined
            chosen_source = "refined"
        else:
            chosen = {**first[stem], "params": prior,
                      "confidence": min(float(prior_conf), first[stem]["confidence"])}
            chosen_source = "sequence_prior"
        confidence = float(chosen["confidence"] * (.5 + .5 * meta["valid_ratio"]))
        save_sample(out / "samples" / f"{stem}.npz", stem=stem,
                    params=chosen["params"], confidence=confidence,
                    quality=q, visible=visible, geometry=geometry, meta=meta,
                    shape=thermal.shape, sequence=group, sequence_prior=prior,
                    sequence_confidence=prior_conf, min_confidence=a.min_confidence)
        row = {"stem": stem, "sequence": group, "confidence": confidence,
               "supervised": confidence >= a.min_confidence,
               "params": [float(x) for x in chosen["params"]],
               "chosen_source": chosen_source,
               "coarse_score": float(first[stem]["score"]),
               "coarse_identity_score": float(first[stem]["identity_score"]),
               "coarse_improvement": float(first[stem]["improvement"]),
               "coarse_uniqueness": float(first[stem]["uniqueness"]),
               "coarse_phase_response": float(first[stem]["phase_response"]),
               "coarse_confidence": float(first[stem]["confidence"]),
               "refined_score": float(refined["score"]),
               "refined_identity_score": float(refined["identity_score"]),
               "refined_improvement": float(refined["improvement"]),
               "refined_uniqueness": float(refined["uniqueness"]),
               "refined_phase_response": float(refined["phase_response"]),
               "refined_confidence": float(refined["confidence"]),
               **meta}
        rows.append(row)
        if i < a.preview_count or confidence < a.min_confidence:
            previews.append((confidence, stem, _preview(
                rgb, thermal, visible, geometry, chosen["params"],
                f"{stem} conf={confidence:.3f} seq={group} "
                f"a={chosen['params'][0]:+.2f} tx={chosen['params'][1]:+.1f} "
                f"ty={chosen['params'][2]:+.1f} s={chosen['params'][3]:.4f}")))
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"[A0] refine {i+1}/{len(samples)}", flush=True)
    preview_dir = out / "previews"; preview_dir.mkdir(parents=True, exist_ok=True)
    for _, stem, image in sorted(previews, key=lambda x: x[0])[:a.preview_count]:
        cv2.imwrite(str(preview_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    confidences = np.asarray([r["confidence"] for r in rows])
    summary = {
        "version": 1, "n_samples": len(rows), "min_confidence": a.min_confidence,
        "supervised": int(sum(r["supervised"] for r in rows)),
        "confidence_quantiles": {str(q): float(np.quantile(confidences, q))
                                 for q in (0, .1, .25, .5, .75, .9, 1)},
        "angle_quantiles": {str(q): float(np.quantile([r["params"][0] for r in rows], q))
                            for q in (0, .1, .5, .9, 1)},
        "sequence_priors": {g: {"params": [float(x) for x in p], "confidence": float(c)}
                            for g, (p, c) in priors.items()},
        "samples": rows,
    }
    write_manifest(out / "audit_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("n_samples", "supervised",
                                               "confidence_quantiles", "angle_quantiles")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
