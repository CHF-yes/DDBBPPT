"""Train one RF-DETR RGB architecture baseline on CHF's fixed COCO split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle", "car",
    "ball", "light", "garbage_can", "uav", "tricycle",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "medium"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    manifest_path = args.dataset / "export_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text())
    if dataset_manifest["classes"] != CLASS_NAMES:
        raise ValueError("COCO class order differs from CHF competition order")
    if (dataset_manifest["splits"]["train"]["images"],
            dataset_manifest["splits"]["valid"]["images"]) != (1600, 400):
        raise ValueError("Expected the fixed 1600/400 split")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a run: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from rfdetr import RFDETRMedium, RFDETRSmall

    torch.set_num_threads(8)
    model_type = {"small": RFDETRSmall, "medium": RFDETRMedium}[args.size]
    record = {
        "model": model_type.__name__, "pretrain_weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "dataset": str(args.dataset), "split_sha256": dataset_manifest["split_sha256"],
        "classes": CLASS_NAMES, "epochs": 24, "resolution": 640, "seed": 42,
        "batch_size": 8, "eval_batch_size": 4, "lr": 0.0001,
        "amp_dtype": "bf16", "skip_best_epochs": 1,
    }
    (args.out / "chf_run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    model = model_type(pretrain_weights=str(args.weights))
    model.train(
        dataset_dir=str(args.dataset), output_dir=str(args.out),
        class_names=CLASS_NAMES, epochs=24, resolution=640,
        batch_size=8, eval_batch_size=4, lr=1e-4,
        amp_dtype="bf16", seed=42, num_workers=8,
        checkpoint_interval=12, skip_best_epochs=1,
        early_stopping=False,
    )


if __name__ == "__main__":
    main()
