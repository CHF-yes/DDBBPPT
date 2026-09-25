"""V6.1 T0 audit on one real batch with the protected V4.4 baseline.

All legacy detector/fusion tensors come from the frozen V4.4 Stage-B best
checkpoint.  Only the newly introduced V5.2.1 geometry adapter is overlaid
from the synthetic-only Q warm-up checkpoint.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "mm_yolo"):
    sys.path.insert(0, str(path))

from config import MMConfig
from data import AugCfg, MMDataset, build_index, collate, load_split
from model import MMYOLO, save_mm_checkpoint
from train import (adapt_depth_checkpoint_state, build_optimizer,
                   enable_trainable_defaults, make_targets,
                   set_independent_aux_mode, set_residual_fusion_mode,
                   subset_detection_batch)
from ultralytics.utils.loss import v8DetectionLoss


V44 = Path("/root/autodl-tmp/runs/mm_v44_multimodal_ceiling_rect_s42_b4a4/stage_b/weights/best.pt")
Q_ADAPTER = Path("/root/autodl-tmp/runs/v521_geometry_warmup_ghost_q_20260925_v1/best.pt")
V44_SHA = "0a38f26fcfcf9bc0d909e6aa26f80b6de7d0ae952780d1e2213dbefa625b6ee8"
Q_SHA = "aacd5af4d7d51f5bf7ec6ef3c8c9df46969eacf6849def76c8a1cf66849bf6a7"
SPLIT_SHA = "888ea6b33964aa1562eda30fffc84b00098c87db446c0a352c76be8874ed1824"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_aug(cache):
    return AugCfg(
        imgsz=(736, 1280), scale_range=(1.0, 1.0), translate=0.0,
        hflip_p=0.0, rotate_deg=0.0, mosaic_p=0.0,
        ir_read_mode="median_channel", ir_a0_cache=str(cache),
        require_ir_a0=True, depth_resampling="nearest_valid_v2",
        misalign_px=0.0, rgb_drop_p=0.0, aux_drop_p=0.0,
        degrade_p=0.0, rgb_color_p=0.0, ir_noise_p=0.0,
        ir_gain_p=0.0, depth_hole_p=0.0, target_crop_p=0.0,
        target_occlusion_p=0.0, ir_affine_p=0.0, v521_explicit=True,
        v521_angle_deg=15.0, v521_shift_frac=0.20,
        v521_scale_min=0.80, v521_scale_max=1.25)


def grad_norm(module):
    values = [p.grad.detach().float().square().sum()
              for p in module.parameters() if p.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def update_norm(module, before):
    values = [(p.detach() - old).float().square().sum()
              for p, old in zip(module.parameters(), before)]
    return float(torch.stack(values).sum().sqrt())


def role_multipliers():
    return {
        "anchor": 0.0, "aux_encoder": 0.25, "fusion": 1.0,
        "p2": 1.0, "detector": 1.0, "semantic": 1.0,
        "geometry": 0.1,
    }


def prepare_state():
    assert sha256(V44) == V44_SHA
    assert sha256(Q_ADAPTER) == Q_SHA
    v44 = torch.load(V44, map_location="cpu", weights_only=False)
    q = torch.load(Q_ADAPTER, map_location="cpu", weights_only=False)
    cfg = MMConfig.from_structure(v44["structure"])
    cfg.weights = "/root/autodl-tmp/weights/yolo11s.pt"
    cfg.fusion.fusion_strategy = "v521_stage_a_v1"
    cfg.fusion.branch_aux_weight = 0.0
    cfg.fusion.branch_aux_weights = (1.0, 1.0, 0.6)
    cfg.fusion.quality_channels = 10
    cfg.fusion.ir_coarse_align = False
    cfg.fusion.alignment_mode = "identity_residual_v2"
    cfg.fusion.depth_reliability = "valid_support_v2"
    cfg.fusion.p2_match_refine = True
    cfg.encoder.checkpoint_encoder = True
    cfg.ir_read_mode = "median_channel"

    template = MMYOLO(cfg)
    migrated, migration_used = adapt_depth_checkpoint_state(v44["model_state"], template)
    q_state = q["model_state"]
    adapter_keys = sorted(k for k in migrated if k.startswith("v52_ir_input."))
    if not adapter_keys:
        raise RuntimeError("V6.1 model exposes no v52_ir_input tensors")
    ignored_q_changes = 0
    for key, value in migrated.items():
        if key.startswith("v52_ir_input.") or key not in q_state:
            continue
        if not torch.equal(value, q_state[key]):
            ignored_q_changes += 1
    for key in adapter_keys:
        if key not in q_state or tuple(q_state[key].shape) != tuple(migrated[key].shape):
            raise RuntimeError(f"Q adapter tensor mismatch: {key}")
        migrated[key] = q_state[key].detach().clone()
    template.load_state_dict(migrated, strict=True)
    provenance = {
        "v44_checkpoint": str(V44), "v44_sha256": V44_SHA,
        "q_adapter_checkpoint": str(Q_ADAPTER), "q_adapter_sha256": Q_SHA,
        "adapter_tensor_count": len(adapter_keys),
        "q_non_adapter_changed_tensors_ignored": ignored_q_changes,
        "checkpoint_migration_used": bool(migration_used),
        "legacy_tensors_source": "V4.4 migrated state",
        "new_adapter_source": "Q synthetic-only geometry warm-up",
    }
    return cfg, migrated, template, provenance


def build_model(cfg, state, device):
    model = MMYOLO(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.infer_canvas = (736, 1280)
    model.infer_modalities = ("rgb", "ir", "dep")
    return model


def move_batch(batch, device):
    rgb = batch["rgb"].to(device).requires_grad_(True)
    ir = batch["ir"].to(device)
    depth = batch["depth"].to(device)
    quality = {k: v.to(device) for k, v in batch["quality"].items()}
    keep = {k: v.to(device) for k, v in batch["keep"].items()}
    targets = make_targets(batch, (736, 1280), device)
    return rgb, ir, depth, quality, keep, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    split = args.cache / "split_s42_v52.json"
    assert sha256(split) == SPLIT_SHA

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(61)
    np.random.seed(61)
    device = torch.device("cuda:0")
    cfg, state, template, provenance = prepare_state()

    init_path = args.out / "v61_base_init.pt"
    save_mm_checkpoint(
        init_path, template, epoch=0, best_map=-1.0,
        meta={
            "version": "V6.1", "baseline": "V4.4",
            "modalities": ["rgb", "ir", "dep"],
            "canvas": [736, 1280], "imgsz": [736, 1280],
            "name": "v61_v44_plus_q_adapter_init",
            **provenance,
        })
    del template

    root = Path("/root/autodl-tmp/data/train_extracted")
    index = build_index(
        root, Path("/root/autodl-tmp/data/new_labels_2000"),
        exclude_stems=Path("/root/autodl-tmp/a0_v3d_inputs/pair_reject_stems_v1.txt"))
    train_rows, _ = load_split(split, index)
    dataset = MMDataset(
        root, train_rows[:8], imgsz=(736, 1280), train=True,
        aug=make_aug(args.cache), enabled=["rgb", "ir", "dep"], seed=61)
    dataset.set_epoch(0)
    batch = next(iter(DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate, pin_memory=True)))
    stems = list(batch.get("stems", []))

    model_a = build_model(cfg, state, device).train()
    enable_trainable_defaults(model_a)
    set_independent_aux_mode(model_a, train_v52_ir_input=True)
    optimizer_a = build_optimizer(
        model_a, 1e-4, 0.25, role_mults=role_multipliers())
    rgb, ir, depth, quality, keep, targets = move_batch(batch, device)
    pred, active = model_a.independent_branch_prediction(
        "ir", rgb=rgb, ir=ir, depth=depth, quality=quality, keep=keep)
    pred, active_targets = subset_detection_batch(pred, targets, active)
    loss_vec, _ = v8DetectionLoss(model_a)(pred, active_targets)
    loss_a = loss_vec.sum() / max(1, int(active.sum()))
    before_a = snapshot(model_a.v52_ir_input)
    optimizer_a.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pass
    loss_a.backward()
    grad_a = grad_norm(model_a.v52_ir_input)
    rgb_grad = 0.0 if rgb.grad is None else float(rgb.grad.detach().abs().sum())
    optimizer_a.step()
    update_a = update_norm(model_a.v52_ir_input, before_a)
    if not (grad_a > 0 and update_a > 0 and rgb_grad == 0.0):
        raise RuntimeError("Stage-A real-batch gradient contract failed")
    stage_a = {
        "loss": float(loss_a.detach()), "adapter_grad_norm": grad_a,
        "adapter_update_norm": update_a, "rgb_input_grad_l1": rgb_grad,
        "active_samples": int(active.sum()),
    }
    del model_a, optimizer_a, pred, loss_a
    torch.cuda.empty_cache()

    model_b = build_model(cfg, state, device).train()
    enable_trainable_defaults(model_b)
    set_residual_fusion_mode(
        model_b, downstream_frozen=False, train_v52_ir_input=True)
    optimizer_b = build_optimizer(
        model_b, 1e-4, 0.25, role_mults=role_multipliers())
    geometry_groups = [g for g in optimizer_b.param_groups
                       if g.get("role") == "geometry"]
    if not geometry_groups or any(float(g["lr_mult"]) != 0.1 for g in geometry_groups):
        raise RuntimeError("Stage-B geometry LR group is missing or incorrect")
    rgb, ir, depth, quality, keep, targets = move_batch(batch, device)
    before_b = snapshot(model_b.v52_ir_input)
    optimizer_b.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model_b(rgb, ir, depth, quality=quality, keep=keep)
        loss_vec, _ = v8DetectionLoss(model_b)(prediction, targets)
        loss_b = loss_vec.sum()
    loss_b.backward()
    grad_b = grad_norm(model_b.v52_ir_input)
    optimizer_b.step()
    update_b = update_norm(model_b.v52_ir_input, before_b)
    with torch.no_grad():
        preserve, preserve_active = model_b.independent_branch_prediction(
            "ir", rgb=rgb.detach(), ir=ir, depth=depth,
            quality=quality, keep=keep)
    if not (grad_b > 0 and update_b > 0 and bool(preserve_active.all())):
        raise RuntimeError("Stage-B real-batch gradient or preservation contract failed")
    stage_b = {
        "loss": float(loss_b.detach()), "adapter_grad_norm": grad_b,
        "adapter_update_norm": update_b, "lr_role": "geometry",
        "lr_multiplier": 0.1,
        "preserve_path_has_rgb_ir_quality": preserve is not None,
        "pixel_blend_enabled": bool(model_b.v52_ir_input.last_stats["pixel_blend_enabled"]),
        "accepted_fraction": float(model_b.v52_ir_input.last_stats["accepted_fraction"]),
    }
    if stage_b["pixel_blend_enabled"]:
        raise RuntimeError("V6.1 enabled forbidden pixel blending")

    result = {
        "version": "V6.1", "baseline": "V4.4",
        "passed": True, "sample_stems": stems,
        "split_sha256": SPLIT_SHA,
        "init_checkpoint": str(init_path),
        "init_checkpoint_sha256": sha256(init_path),
        "provenance": provenance,
        "stage_a": stage_a, "stage_b": stage_b,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    (args.out / "real_batch_gradient_audit.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
