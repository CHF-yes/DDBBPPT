"""V6.1 T0 synthetic gradient-chain audit.

This is a mechanism test, not a detection-quality result.  It verifies that the
same explicit A0/residual adapter runs in Stage A and Stage B, that Stage-A RGB
is geometry-only, and that the Stage-B opt-in uses its dedicated low-LR group.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mm_yolo"))

from config import default_config
from model import MMYOLO
from train import (build_optimizer, enable_trainable_defaults, make_targets,
                   set_independent_aux_mode, set_residual_fusion_mode)
from ultralytics.utils.loss import v8DetectionLoss


def config():
    cfg = default_config()
    cfg.fusion.architecture = "independent_p2_memory_v3"
    cfg.fusion.bus_dim = 128
    cfg.encoder.share_tier = "a"
    cfg.encoder.metric_branch = True
    cfg.encoder.checkpoint_encoder = False
    cfg.depth_resampling = "nearest_valid_v2"
    cfg.fusion.fusion_strategy = "v521_stage_a_v1"
    cfg.fusion.branch_aux_weight = 0.0
    cfg.fusion.branch_aux_weights = (1.0, 1.0, 0.6)
    cfg.fusion.alignment_mode = "identity_residual_v2"
    cfg.fusion.depth_reliability = "valid_support_v2"
    cfg.fusion.p2_match_refine = True
    return cfg


def inputs(device):
    torch.manual_seed(61)
    canvas = (64, 96)
    rgb = torch.rand(1, 3, *canvas, device=device, requires_grad=True)
    ir = torch.rand(1, 1, *canvas, device=device)
    depth = torch.rand(1, 4, *canvas, device=device)
    depth[:, 2:] = 1
    q = torch.zeros(1, 10, 16, 24, device=device)
    q[:, 8:10] = 1
    quality = {
        "v52_ir_sampling": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device),
        "availability": torch.ones(1, 3, *canvas, device=device),
        "ir": q,
        "v521_ghost_probability": torch.zeros(1, 1, 16, 24, device=device),
        "v521_ghost_confidence": torch.ones(1, 1, 16, 24, device=device),
        "v521_thermal_confidence": torch.ones(1, 1, 16, 24, device=device),
        "v521_coarse_available": torch.ones(1, 1, 16, 24, device=device),
    }
    keep = {name: torch.ones(1, device=device) for name in ("rgb", "ir", "dep")}
    targets = make_targets(
        {"boxes": [torch.tensor([[0.0, 0.5, 0.5, 0.25, 0.25]], device=device)]},
        canvas, device)
    return canvas, rgb, ir, depth, quality, keep, targets


def grad_norm(module):
    values = [p.grad.detach().float().square().sum()
              for p in module.parameters() if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt())


def parameter_snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def update_norm(module, before):
    values = [(p.detach() - old).float().square().sum()
              for p, old in zip(module.parameters(), before)]
    return float(torch.stack(values).sum().sqrt())


def detection_loss(model, prediction, targets):
    loss, _ = v8DetectionLoss(model)(prediction, targets)
    return loss.sum()


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"version": "V6.1", "baseline": "V4.4", "device": str(device)}

    stage_a = MMYOLO(config()).to(device).train()
    enable_trainable_defaults(stage_a)
    set_independent_aux_mode(stage_a, train_v52_ir_input=True)
    canvas, rgb, ir, depth, quality, keep, targets = inputs(device)
    prediction, active = stage_a.independent_branch_prediction(
        "ir", rgb=rgb, ir=ir, depth=depth, quality=quality, keep=keep)
    assert bool(active.all())
    stage_a_loss = detection_loss(stage_a, prediction, targets)
    stage_a_loss.backward()
    stage_a_grad = grad_norm(stage_a.v52_ir_input)
    rgb_geometry_grad = 0.0 if rgb.grad is None else float(rgb.grad.detach().abs().sum())
    assert stage_a_grad > 0, "Stage-A detection loss did not reach the IR adapter"
    assert rgb_geometry_grad == 0.0, "RGB semantic gradient entered Stage-A IR detection"
    results["stage_a"] = {
        "loss": float(stage_a_loss.detach()),
        "adapter_grad_norm": stage_a_grad,
        "rgb_input_grad_l1": rgb_geometry_grad,
        "adapter_trainable": any(p.requires_grad for p in stage_a.v52_ir_input.parameters()),
    }
    del stage_a, prediction, stage_a_loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage_b_frozen = MMYOLO(config()).to(device).train()
    enable_trainable_defaults(stage_b_frozen)
    set_residual_fusion_mode(
        stage_b_frozen, downstream_frozen=False, train_v52_ir_input=False)
    _, rgb_f, ir_f, depth_f, quality_f, keep_f, _ = inputs(device)
    with torch.no_grad():
        stage_b_frozen(rgb_f, ir_f, depth_f, quality=quality_f, keep=keep_f)
    assert not any(p.requires_grad for p in stage_b_frozen.v52_ir_input.parameters())
    assert stage_b_frozen.v52_ir_input.last_stats
    results["stage_b_frozen"] = {
        "adapter_trainable": False,
        "adapter_forward_executed": True,
    }
    del stage_b_frozen
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage_b = MMYOLO(config()).to(device).train()
    enable_trainable_defaults(stage_b)
    set_residual_fusion_mode(
        stage_b, downstream_frozen=False, train_v52_ir_input=True)
    role_mults = {
        "anchor": 0.0, "aux_encoder": 0.25, "fusion": 1.0,
        "p2": 1.0, "detector": 1.0, "semantic": 1.0, "geometry": 0.1,
    }
    optimizer = build_optimizer(stage_b, 1e-4, 0.25, role_mults=role_mults)
    geometry_groups = [g for g in optimizer.param_groups if g.get("role") == "geometry"]
    geometry_ids = {id(p) for group in geometry_groups for p in group["params"]}
    assert geometry_groups and all(id(p) in geometry_ids for p in stage_b.v52_ir_input.parameters())
    assert all(abs(float(g["lr_mult"]) - 0.1) < 1e-12 for g in geometry_groups)
    before = parameter_snapshot(stage_b.v52_ir_input)
    canvas, rgb, ir, depth, quality, keep, targets = inputs(device)
    prediction = stage_b(rgb, ir, depth, quality=quality, keep=keep)
    stage_b_loss = detection_loss(stage_b, prediction, targets)
    optimizer.zero_grad(set_to_none=True)
    stage_b_loss.backward()
    stage_b_grad = grad_norm(stage_b.v52_ir_input)
    assert stage_b_grad > 0, "Stage-B fused detection loss did not reach the IR adapter"
    optimizer.step()
    stage_b_update = update_norm(stage_b.v52_ir_input, before)
    assert stage_b_update > 0, "Stage-B optimizer did not update the IR adapter"

    preserve, active = stage_b.independent_branch_prediction(
        "ir", rgb=rgb.detach(), ir=ir, depth=depth,
        quality=quality, keep=keep)
    assert bool(active.all()) and preserve is not None
    results["stage_b_trainable"] = {
        "loss": float(stage_b_loss.detach()),
        "adapter_grad_norm": stage_b_grad,
        "adapter_update_norm": stage_b_update,
        "lr_role": "geometry",
        "lr_multiplier": 0.1,
        "preserve_path_has_rgb_ir_quality": True,
        "pixel_blend_enabled": bool(stage_b.v52_ir_input.last_stats["pixel_blend_enabled"]),
    }
    assert not results["stage_b_trainable"]["pixel_blend_enabled"]
    results["passed"] = True
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
