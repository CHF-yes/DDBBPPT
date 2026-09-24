#!/usr/bin/env python3
"""Upgrade a completed V5.1.1 A0 v2 cache to the v3 model contract.

This is intentionally a metadata-only migration: it preserves every quality
map, mask, confidence and affine estimate, then recomputes whether the affine
is representable by the configured train-time alignment head.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mm_yolo.ir_a0 import A0_VERSION, affine_model_contract  # noqa: E402


def _upgrade_one(payload):
    source, target, canvas, angle, shift, scale, min_confidence = payload
    source, target = Path(source), Path(target)
    with np.load(source, allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    version = int(np.asarray(values["version"]).reshape(()))
    if version not in (2, A0_VERSION):
        raise ValueError(f"unsupported A0 cache version {version}: {source}")
    params = np.asarray(values["source_to_rgb_params"], np.float32)
    shape = tuple(int(v) for v in np.asarray(values["orig_hw"]).reshape(-1))
    confidence = float(np.asarray(values["affine_confidence"]).reshape(()))
    contract_ok, contract = affine_model_contract(
        params, shape, canvas, angle, shift, scale)
    supervised = bool(confidence >= min_confidence and contract_ok)
    values["version"] = np.int32(A0_VERSION)
    values["affine_supervised"] = np.uint8(supervised)
    values["affine_contract"] = np.asarray(contract, np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temp, target)
    return source.stem, confidence, supervised, contract_ok, [float(x) for x in contract]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--canvas", default="736x1280")
    parser.add_argument("--angle", type=float, default=4.0)
    parser.add_argument("--shift", type=float, default=16.0)
    parser.add_argument("--scale", type=float, default=.04)
    parser.add_argument("--min-confidence", type=float, default=.45)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    canvas = tuple(int(x) for x in args.canvas.lower().split("x", 1))
    if len(canvas) != 2 or min(canvas) <= 0:
        raise ValueError(f"invalid canvas: {args.canvas}")
    source_files = sorted((source / "samples").glob("*.npz"))
    if not source_files:
        raise FileNotFoundError(f"no A0 samples found under {source / 'samples'}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output is not empty: {out}")
    (out / "samples").mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(path), str(out / "samples" / path.name), canvas, args.angle,
         args.shift, args.scale, args.min_confidence)
        for path in source_files
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, result in enumerate(pool.map(_upgrade_one, tasks, chunksize=4), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"[A0 upgrade] {index}/{len(tasks)}", flush=True)

    old_summary_path = source / "audit_summary.json"
    summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    by_stem = {stem: (confidence, supervised, ok, contract)
               for stem, confidence, supervised, ok, contract in results}
    for row in summary.get("samples", []):
        confidence, supervised, ok, contract = by_stem[row["stem"]]
        row["confidence"] = confidence
        row["supervised"] = supervised
        row["contract_ok"] = ok
        row["affine_contract"] = contract
    summary["version"] = A0_VERSION
    summary["supervised"] = sum(int(item[2]) for item in results)
    summary["model_contract"] = {
        "canvas": list(canvas), "angle": args.angle, "shift": args.shift,
        "scale": args.scale,
        "rejected": sum(int(not item[3]) for item in results),
        "rejected_above_confidence": sum(
            int(item[1] >= args.min_confidence and not item[3]) for item in results),
    }
    (out / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if (source / "previews").is_dir():
        shutil.copytree(source / "previews", out / "previews", dirs_exist_ok=True)
    print(json.dumps({
        "version": A0_VERSION, "samples": len(results),
        "supervised": summary["supervised"],
        "model_contract": summary["model_contract"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
