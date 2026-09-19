"""CPU-only, fixed-checkpoint fusion diagnostics. Never updates training weights."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))
from data import AugCfg, MMDataset, build_index, load_split, collate
from model import load_mm_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--zero-context", action="store_true", help="diagnostic only; never saves modified weights")
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, ck = load_mm_checkpoint(args.ckpt, device="cpu")
    model.eval()
    cfg = ck["train_state"]["args"]
    index = build_index(Path(cfg["root"]), Path(cfg["labels"]))
    _, val = load_split(Path(cfg["split_file"]), index)
    selected = np.random.default_rng(20260919).choice(len(val), min(args.n, len(val)), replace=False)
    ds = MMDataset(Path(cfg["root"]), [val[i] for i in selected], imgsz=tuple(ck["meta"]["canvas"]),
                   train=False, aug=AugCfg(depth_resampling=model.cfg.depth_resampling))
    stats = defaultdict(list)
    records, handles = [], []

    def record(key, value):
        stats[key].append(float(value))

    for scale, block in model.fusion.items():
        if args.zero_context:
            handles.append(block.memory_query.register_forward_hook(lambda _m, _a, out: torch.zeros_like(out)))
        def context_hook(_m, _a, out, s=scale):
            record(s+".context_rms", out.float().square().mean().sqrt())
        def query_hook(_m, _a, out, s=scale):
            record(s+".local_query_rms", out.float().square().mean().sqrt())
        def fusion_hook(_m, a, out, s=scale):
            ref = a[0][0]
            record(s+".residual_over_rgb_rms", (out-ref).float().norm()/ref.float().norm().clamp_min(1e-8))
        handles.append(block.memory_query.register_forward_hook(context_hook))
        handles.append(block.query.register_forward_hook(query_hook))
        handles.append(block.register_forward_hook(fusion_hook))
        for modality, gate in enumerate(block.gates):
            def gate_hook(_m, a, out, s=scale, m=modality):
                key = f"{s}.gate{m}"
                g = out.float().sigmoid()
                record(key+".mean", g.mean())
                record(key+".above_099", (g > .99).float().mean())
                record(key+".below_001", (g < .01).float().mean())
                record(key+".sigmoid_derivative", (g*(1-g)).mean())
                record(key+".logit_abs", out.float().abs().mean())
            handles.append(gate.register_forward_hook(gate_hook))

    def neck_hook(_m, a, out):
        record("neck.residual_over_input_rms", (out-a[0]).float().norm()/a[0].float().norm().clamp_min(1e-8))
    handles.append(model.neck_memory.register_forward_hook(neck_hook))
    with torch.inference_mode():
        for i in range(len(ds)):
            batch = collate([ds[i]])
            model(batch["rgb"], batch["ir"], batch["depth"], quality=batch["quality"], keep=batch["keep"])
            records.append({"stem": batch["stems"][0], "depth_valid": float(batch["depth"][:, 2].mean()),
                            "fusion": {s: {m: pair.tolist() for m, pair in b.last_stats.items()}
                                       for s, b in model.fusion.items()}})
            print(f"[health] {i+1}/{len(ds)} {batch['stems'][0]}", flush=True)
    for handle in handles:
        handle.remove()
    report = {"checkpoint": str(Path(args.ckpt).resolve()), "epoch": ck["epoch"],
              "n": len(ds), "canvas": ck["meta"]["canvas"], "mode": "EMA eval FP32 CPU, no augment, fixed random validation sample",
              "zero_context": args.zero_context,
              "stats": {k: {"mean": float(np.mean(v)), "min": min(v), "max": max(v)} for k, v in stats.items()},
              "images": records}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "images"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
