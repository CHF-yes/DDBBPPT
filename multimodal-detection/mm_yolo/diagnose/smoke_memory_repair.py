"""Bounded GPU correctness check of v1 checkpoint migration, not an AP experiment."""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import numpy as np

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))
from config import MMConfig
from data import AugCfg, MMDataset, collate, build_index, load_split
from model import MMYOLO
from train import (validate_checkpoint, reset_fusion_gate_outputs, apply_bn_policy,
                   build_optimizer, make_targets, accumulation_loss, ensure_finite_state)
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import ModelEMA
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    source_epoch = ck["epoch"]
    cfg = MMConfig.from_structure(ck["structure"])
    cfg.fusion.memory_control = "bounded_v2"
    model = MMYOLO(cfg)
    meta, old_args = ck["meta"], ck["train_state"]["args"]
    validate_checkpoint(ck, model, meta["modalities"], meta["canvas"], exact=False,
                        args=SimpleNamespace(reset_fusion_gates=True), split_digest=meta["split_digest"])
    model.load_state_dict(ck["model_state"], strict=True)
    reset = reset_fusion_gate_outputs(model)
    del ck
    samples = build_index(Path(old_args["root"]), Path(old_args["labels"]))
    train, _ = load_split(Path(old_args["split_file"]), samples)
    # The lexically first ~100 images are JPG: testing only the first batch
    # would incorrectly demand metric gradients on non-metric input.
    png = [i for i,s in enumerate(train) if Path(s["files"]["depth"]).suffix.lower() == ".png"]
    jpg = [i for i,s in enumerate(train) if Path(s["files"]["depth"]).suffix.lower() == ".jpg"]
    rng = np.random.default_rng(42)
    chosen_png = rng.choice(png, 20, replace=False).tolist()
    chosen_jpg = rng.choice(jpg, 10, replace=False).tolist()
    chosen = [j for i in range(10) for j in (chosen_jpg[i], chosen_png[2*i], chosen_png[2*i+1])]
    ds = MMDataset(Path(old_args["root"]), train, imgsz=tuple(meta["canvas"]), train=True,
                   aug=AugCfg(depth_resampling=cfg.depth_resampling, misalign_px=2,
                              rgb_drop_p=.01, aux_drop_p=.03), seed=42)
    model.cuda().train()
    apply_bn_policy(model, "adaptive_no_tail")
    optimizer = build_optimizer(model, 1e-4, .5, .0001171875)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    criterion = v8DetectionLoss(model)
    ema = ModelEMA(model)
    result = {"source_epoch": source_epoch, "reset_tensors": len(reset), "steps": [], "png":20,"jpg":10}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(5):
            batch = collate([ds[(chosen[3*(step*5+micro)+i], 0, 3)] for i in range(3)])
            with torch.autocast("cuda"):
                pred = model(batch["rgb"].cuda(), batch["ir"].cuda(), batch["depth"].cuda(),
                             quality={k:v.cuda() for k,v in batch["quality"].items()},
                             keep={k:v.cuda() for k,v in batch["keep"].items()})
                losses, _ = criterion(pred, make_targets(batch, meta["canvas"], "cuda"))
                loss = accumulation_loss(losses, 15)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 48.49)
        assert torch.isfinite(norm), norm
        for name, module in (("memory",model.register_bus), ("metric",model.metric_encoder), ("fusion",model.fusion)):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients), name
            assert sum(float(g.abs().sum()) for g in gradients) > 0, name
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= scale_before
        ema.update(model)
        ensure_finite_state(model)
        ensure_finite_state(ema.ema)
        health = {s:{k:float(v) for k,v in block.last_health.items()} for s,block in model.fusion.items()}
        assert all(h["context_rms"] <= .25001 for h in health.values())
        result["steps"].append({"gradient_norm": float(norm), "amp_skip":0, "health":health})
        print(f"[smoke] step {step+1} grad={norm:.3f}", flush=True)
    result["max_allocated_gib"] = torch.cuda.max_memory_allocated()/1024**3
    result["max_reserved_gib"] = torch.cuda.max_memory_reserved()/1024**3
    result["ok"] = True
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
