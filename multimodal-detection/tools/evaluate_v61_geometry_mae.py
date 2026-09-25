import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "mm_yolo", ROOT / "tools"):
    sys.path.insert(0, str(path))

from data import MMDataset, build_index, collate, load_split
from model import load_mm_checkpoint
from train_v521_geometry_warmup import evaluate, make_aug


def load_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise KeyError(f"model_state missing from {path}")
    return state


def adapter_drift(initial, current):
    total_sq = 0.0
    initial_sq = 0.0
    max_abs = 0.0
    compared = 0
    changed = 0
    largest = []
    for name, initial_value in initial.items():
        if not name.startswith("v52_ir_input.") or name not in current:
            continue
        current_value = current[name]
        if not torch.is_tensor(initial_value) or not torch.is_tensor(current_value):
            continue
        if initial_value.shape != current_value.shape or not initial_value.is_floating_point():
            continue
        delta = current_value.float() - initial_value.float()
        delta_norm = float(delta.norm())
        tensor_max = float(delta.abs().max()) if delta.numel() else 0.0
        total_sq += delta_norm * delta_norm
        initial_norm = float(initial_value.float().norm())
        initial_sq += initial_norm * initial_norm
        max_abs = max(max_abs, tensor_max)
        compared += 1
        if tensor_max > 0:
            changed += 1
        largest.append((delta_norm, name, tensor_max))
    largest.sort(reverse=True)
    return {
        "tensors_compared": compared,
        "tensors_changed": changed,
        "l2_update": total_sq ** 0.5,
        "relative_l2_update": (total_sq / max(initial_sq, 1e-30)) ** 0.5,
        "max_abs_update": max_abs,
        "largest_tensors": [
            {"name": name, "l2_update": norm, "max_abs_update": tensor_max}
            for norm, name, tensor_max in largest[:10]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=/absolute/path/to/checkpoint.pt")
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoints = []
    for item in args.checkpoint:
        label, separator, raw_path = item.partition("=")
        if not separator:
            raise ValueError(f"checkpoint must use LABEL=PATH: {item}")
        checkpoints.append((label, Path(raw_path)))

    cv2.setNumThreads(1)
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    root = Path("/root/autodl-tmp/data/train_extracted")
    index = build_index(
        root,
        Path("/root/autodl-tmp/data/new_labels_2000"),
        exclude_stems=Path("/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt"),
    )
    _, val_rows = load_split(args.cache / "split_s42_v52.json", index)
    synth_ds = MMDataset(
        root, val_rows[:160], imgsz=(736, 1280), train=True,
        aug=make_aug(args.cache, 1.0), enabled=["rgb", "ir"], seed=314,
    )
    identity_ds = MMDataset(
        root, val_rows[:160], imgsz=(736, 1280), train=True,
        aug=make_aug(args.cache, 0.0), enabled=["rgb", "ir"], seed=2718,
    )
    synth_loader = DataLoader(
        synth_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate, pin_memory=True,
    )
    identity_loader = DataLoader(
        identity_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate, pin_memory=True,
    )

    initial_state = load_state(args.initial)
    reports = []
    for label, checkpoint in checkpoints:
        model, _ = load_mm_checkpoint(
            checkpoint, device=device,
            weights="/root/autodl-tmp/weights/yolo11s.pt",
        )
        report = {
            "label": label,
            "checkpoint": str(checkpoint),
            "samples": len(synth_ds),
            "synthetic": evaluate(model, synth_loader, device, synthetic=True),
            "identity": evaluate(model, identity_loader, device, synthetic=False),
            "adapter_drift": adapter_drift(initial_state, load_state(checkpoint)),
        }
        reports.append(report)
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
