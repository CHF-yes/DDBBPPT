#!/usr/bin/env python3
"""Build the V5.1.1 offline IR A0 cache and visual audit sheets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "mm_yolo"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from mm_yolo.data import build_index, source_group  # noqa: E402
from mm_yolo.ir_a0 import (SearchCfg, affine_model_contract, estimate_affine,
                           quality_maps, read_modalities, robust_sequence_prior, save_sample,
                           select_affine_candidate, write_manifest)  # noqa: E402


def _paths(root: Path, sample: dict):
    return (root / "visible" / sample["files"]["visible"],
            root / "infrared" / sample["files"]["infrared"])


def _old_params(cache: Path | None, stem: str):
    if cache is None:
        return None
    path = cache / "samples" / f"{stem}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["source_to_rgb_params"], np.float32)


def _worker_init(opencv_threads: int):
    # The server has a 16-core cgroup quota.  A few processes with a bounded
    # OpenCV thread pool fill that quota more reliably than one Python process
    # iterating the affine candidates while OpenCV repeatedly starts/stops its
    # own workers.
    cv2.setNumThreads(max(1, int(opencv_threads)))


def _coarse_task(payload):
    root, sample, cfg, old_cache = payload
    root = Path(root)
    stem = sample["stem"]
    rgb, thermal, ir3 = read_modalities(*_paths(root, sample))
    _, _, geometry, _, geometry_thermal = quality_maps(
        rgb, thermal, ir3, return_preprocessed=True)
    old = _old_params(Path(old_cache) if old_cache else None, stem)
    return stem, source_group(stem), estimate_affine(
        rgb, geometry_thermal, geometry, cfg, prior=old)


def _refine_task(payload):
    (root, out, sample, cfg, coarse, prior, prior_conf, use_prior,
     min_confidence, contract_canvas, contract_angle,
     contract_shift, contract_scale) = payload
    root, out = Path(root), Path(out)
    stem, group = sample["stem"], source_group(sample["stem"])
    rgb, thermal, ir3 = read_modalities(*_paths(root, sample))
    q, visible, geometry, meta, geometry_thermal = quality_maps(
        rgb, thermal, ir3, return_preprocessed=True)
    refine_origin = prior if use_prior else coarse["params"]
    refined = estimate_affine(
        rgb, geometry_thermal, geometry, cfg, prior=refine_origin)
    chosen, chosen_source = select_affine_candidate(
        coarse, refined, prior, prior_conf, use_prior)
    confidence = float(chosen["confidence"] * (.5 + .5 * meta["valid_ratio"]))
    contract_ok, contract = affine_model_contract(
        chosen["params"], thermal.shape, contract_canvas, contract_angle,
        contract_shift, contract_scale)
    supervised = bool(confidence >= min_confidence and contract_ok)
    save_sample(out / "samples" / f"{stem}.npz", stem=stem,
                params=chosen["params"], confidence=confidence,
                quality=q, visible=visible, geometry=geometry, meta=meta,
                shape=thermal.shape, sequence=group, sequence_prior=prior,
                sequence_confidence=prior_conf, min_confidence=min_confidence,
                affine_supervised=supervised, affine_contract=contract)
    return {
        "stem": stem, "sequence": group, "confidence": confidence,
        "supervised": supervised, "contract_ok": contract_ok,
        "affine_contract": [float(x) for x in contract],
        "params": [float(x) for x in chosen["params"]],
        "chosen_source": chosen_source,
        "coarse_score": float(coarse["score"]),
        "coarse_identity_score": float(coarse["identity_score"]),
        "coarse_improvement": float(coarse["improvement"]),
        "coarse_uniqueness": float(coarse["uniqueness"]),
        "coarse_phase_response": float(coarse["phase_response"]),
        "coarse_confidence": float(coarse["confidence"]),
        "refined_score": float(refined["score"]),
        "refined_identity_score": float(refined["identity_score"]),
        "refined_improvement": float(refined["improvement"]),
        "refined_uniqueness": float(refined["uniqueness"]),
        "refined_phase_response": float(refined["phase_response"]),
        "refined_confidence": float(refined["confidence"]),
        **meta,
    }


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
    p.add_argument("--exclude-stems", default="",
                   help="versioned cross-modal mismatch list")
    p.add_argument("--out", required=True)
    p.add_argument("--old-cache", default="",
                   help="old A0 cache used as a competing baseline, never truth")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--work-width", type=int, default=480)
    p.add_argument("--preview-count", type=int, default=80)
    p.add_argument("--min-confidence", type=float, default=.45)
    p.add_argument("--workers", type=int, default=4,
                   help="sample-level worker processes; 4 is tuned for a 16-core quota")
    p.add_argument("--opencv-threads", type=int, default=4,
                   help="OpenCV threads per worker")
    p.add_argument("--contract-canvas", default="736x1280",
                   help="V5.1.1 training canvas used to validate affine labels")
    p.add_argument("--contract-angle", type=float, default=25.0)
    p.add_argument("--contract-shift", type=float, default=16.0)
    p.add_argument("--contract-scale", type=float, default=.04)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    samples = build_index(
        root, Path(a.labels) if a.labels else None, limit=a.limit,
        exclude_stems=Path(a.exclude_stems) if a.exclude_stems else None)
    cfg = SearchCfg(work_width=a.work_width, min_confidence=a.min_confidence)
    contract_canvas = tuple(int(x) for x in a.contract_canvas.lower().split("x", 1))
    if len(contract_canvas) != 2 or min(contract_canvas) <= 0:
        raise ValueError(f"invalid --contract-canvas: {a.contract_canvas}")
    # Two-pass streaming keeps memory bounded for the full 2,000-image set.
    # Full-resolution RGB/IR arrays and masks are intentionally not retained
    # between the coarse sequence-prior pass and the refinement/write pass.
    first, groups = {}, defaultdict(list)
    with ProcessPoolExecutor(max_workers=max(1, a.workers),
                             initializer=_worker_init,
                             initargs=(a.opencv_threads,)) as pool:
        tasks = ((str(root), sample, cfg, a.old_cache) for sample in samples)
        for i, (stem, group, estimate) in enumerate(
                pool.map(_coarse_task, tasks, chunksize=1)):
            first[stem] = estimate
            groups[group].append(estimate)
            if (i + 1) % 50 == 0 or i + 1 == len(samples):
                print(f"[A0] coarse {i+1}/{len(samples)}", flush=True)
    prior_usable = {g: g != "PLAIN" and len(rows) >= 2 for g, rows in groups.items()}
    priors = {
        g: (robust_sequence_prior(rows) if prior_usable[g]
            else (np.asarray([0., 0., 0., 1.], np.float32), 0.0))
        for g, rows in groups.items()
    }
    rows = []
    sample_by_stem = {sample["stem"]: sample for sample in samples}
    with ProcessPoolExecutor(max_workers=max(1, a.workers),
                             initializer=_worker_init,
                             initargs=(a.opencv_threads,)) as pool:
        tasks = []
        for sample in samples:
            stem, group = sample["stem"], source_group(sample["stem"])
            prior, prior_conf = priors[group]
            tasks.append((str(root), str(out), sample, cfg, first[stem], prior,
                          prior_conf, prior_usable[group], a.min_confidence,
                          contract_canvas, a.contract_angle,
                          a.contract_shift, a.contract_scale))
        for i, row in enumerate(pool.map(_refine_task, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(samples):
                print(f"[A0] refine {i+1}/{len(samples)}", flush=True)
    preview_dir = out / "previews"; preview_dir.mkdir(parents=True, exist_ok=True)
    # Re-read only the lowest-confidence audit samples.  Keeping rendered
    # previews for every weak candidate used several GB without helping the
    # cache itself.
    for row in sorted(rows, key=lambda x: x["confidence"])[:a.preview_count]:
        stem, group = row["stem"], row["sequence"]
        rgb, thermal, ir3 = read_modalities(*_paths(root, sample_by_stem[stem]))
        _, visible, geometry, _, geometry_thermal = quality_maps(
            rgb, thermal, ir3, return_preprocessed=True)
        params = row["params"]
        image = _preview(
            rgb, geometry_thermal, visible, geometry, params,
            f"{stem} conf={row['confidence']:.3f} seq={group} "
            f"a={params[0]:+.2f} tx={params[1]:+.1f} "
            f"ty={params[2]:+.1f} s={params[3]:.4f}")
        cv2.imwrite(str(preview_dir / f"{stem}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    confidences = np.asarray([r["confidence"] for r in rows])
    summary = {
        "version": 6, "n_samples": len(rows), "min_confidence": a.min_confidence,
        "supervised": int(sum(r["supervised"] for r in rows)),
        "model_contract": {
            "canvas": list(contract_canvas), "angle": a.contract_angle,
            "shift": a.contract_shift, "scale": a.contract_scale,
            "rejected": int(sum(not r["contract_ok"] for r in rows)),
            "rejected_above_confidence": int(sum(
                (not r["contract_ok"]) and r["confidence"] >= a.min_confidence
                for r in rows)),
        },
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
