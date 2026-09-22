# -*- coding: utf-8 -*-
"""训练入口：三模态（或 RGB-only 锚点）+ 承重式融合 + 两阶段冻结 + modality dropout。

用法示例
--------
# A0 锚点（RGB-only，与三模态同配置可比）
python code/mm_yolo/train.py --root "<train_extracted>" --labels "<new_labels_2000>" \
    --modalities rgb --imgsz 544x960 --epochs 60 --name a0_rgb --resume

# B1 三模态（L2 + Depth 残差对齐 + 逐模态 BN + 持久 Register）
python code/mm_yolo/train.py --root "<train_extracted>" --labels "<new_labels_2000>" \
    --modalities all --imgsz 544x960 --epochs 100 --name b1_register --register-bus \
    --depth-scales p4p5

审计修复记录（本轮）：
* P0-1 `--modalities rgb` 时数据侧**不生成** IR/Depth 且**关闭** modality dropout
  （旧版 rgb_drop_p=0.25 仍然生效 → 1598 张里 406 张永远黑图）；
* P0-5 训练期验证用**组感知 + 类别均衡**子集（旧版 `va[:limit]` 只有前 N 张，
  完全没有 class 1/7/11），并在日志里打印验证集逐类框数，避免"成绩不可信却不知道"；
* P0-3 验证跑 **EMA 模型**（深拷贝后 hook 已指向自身），best.pt 与日志同源；
* P1 新格式 `--resume` 严格恢复原始模型/optimizer/GradScaler/EMA/RNG；旧 EMA-only
  checkpoint 只可用 `--init-checkpoint` 在新实验中初始化，不能声称精确续训；
* P1 梯度累积**在每轮末尾强制冲刷**（旧版丢弃末组，每轮约 14 张不参与更新）；
* P1 `--freeze-bn` 不再永久冻结：新建的融合 BN / 逐模态 BN 副本默认参与训练
  （`--freeze-new-bn` 可回到旧行为），也可整体关掉 `--no-freeze-bn`；
* P0-7 checkpoint 记录训练模态 + 画布 + 完整训练状态（供 eval/submit/续训使用）。
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import itertools
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultralytics.utils.loss import v8DetectionLoss          # noqa: E402
from ultralytics.utils.torch_utils import ModelEMA          # noqa: E402

from config import default_config                            # noqa: E402
from data import (AugCfg, MMDataset, CoverageRareSampler, scheduled_aug, balanced_sample_weights, build_index, canvas_of,
                  collate, group_split, load_split, parse_imgsz, pick_val_subset,
                  save_split, val_class_stats)   # noqa: E402
from model import MMYOLO, save_mm_checkpoint                 # noqa: E402

BN_TYPES = nn.modules.batchnorm._BatchNorm
TRAINER_RECIPE = "optfix_v2"
MEMORY_RECIPE = "coverage_spatial_memory_v1"


def recipe_for(args):
    if getattr(args, "architecture", "") == "independent_p2_memory_v3":
        if getattr(args, "train_stage", "standard") == "aux_independent":
            return "independent_aux_detectors_v1"
        if getattr(args, "train_stage", "standard") == "residual_fusion":
            return "rgb_identity_residual_fusion_v1"
        if getattr(args, "train_stage", "standard") == "anchored_joint":
            return "independent_p2_anchored_joint_v1"
        if getattr(args, "alignment_mode", "legacy_gate_v1") == "identity_residual_v2":
            return "independent_p2_identity_v2"
        semantic = (getattr(args, "branch_aux_weight", 0) > 0 or
                    bool(getattr(args, "branch_aux_weights", None)) or
                    getattr(args, "flow_supervision_weight", 0) > 0 or
                    getattr(args, "cross_modal_nce_weight", 0) > 0 or
                    getattr(args, "p2_match_refine", False))
        return "independent_p2_semantic_v1" if semantic else "independent_p2_memory_v3"
    if getattr(args, "memory_control", "unbounded_v1") == "bounded_v2":
        return "coverage_spatial_memory_v2"
    return MEMORY_RECIPE if getattr(args, "architecture", "legacy_hook_v1") == "spatial_memory_v1" else TRAINER_RECIPE


def apply_bn_policy(model, policy, frozen=False):
    """Small-batch adaptive stats; affine parameters still follow optimizer groups."""
    if policy not in ("adaptive", "adaptive_no_tail"):
        raise ValueError(policy)
    encoders = model.encoder_modules() if hasattr(model, "encoder_modules") else [model.backbone.model[:11]]
    encoder_bn = {id(m) for enc in encoders for m in enc.modules() if isinstance(m, BN_TYPES)}
    for name, m in model.named_modules():
        if isinstance(m, BN_TYPES):
            m.momentum = .03
            m.train(not (frozen and (id(m) in encoder_bn or name.startswith("dep_stem."))))


def set_bn_tail_mode(model, extra):
    """Extra rare draws still train with batch stats but do not rewrite population stats."""
    for m in model.modules():
        if isinstance(m, BN_TYPES):
            m.momentum = 0.0 if extra else .03


def reset_fusion_gate_outputs(model):
    """Explicit v1->v2 repair only: preserve encoder, values, memory and detection head."""
    if not model.spatial_memory or model.cfg.fusion.memory_control != "bounded_v2":
        raise ValueError("gate repair requires bounded_v2")
    reset = []
    with torch.no_grad():
        for scale, block in model.fusion.items():
            for i, gate in enumerate(block.gates):
                nn.init.normal_(gate[-1].weight, std=.01)
                nn.init.zeros_(gate[-1].bias)
                reset += [f"fusion.{scale}.gates.{i}.2.weight", f"fusion.{scale}.gates.{i}.2.bias"]
    return reset


def reset_rgb_identity_residuals(model):
    """Make the Stage-B hand-off exactly equal to the RGB spatial route."""
    reset = []
    with torch.no_grad():
        for scale, block in model.fusion.items():
            block.residual_scale.zero_()
            reset.append(f"fusion.{scale}.residual_scale")
        model.neck_memory.residual_scale.zero_()
        model.localization_scale.zero_()
        reset += ["neck_memory.residual_scale", "localization_scale"]
    return reset


# ---------------------------------------------------------------- 批次 → 损失输入

def make_targets(batch: dict, canvas, device) -> dict:
    """把 (N,5) 归一化框列表拼成 ultralytics 训练用的 batch 字典。

    v8DetectionLoss 期望：batch_idx / cls / bboxes（**画布像素坐标**，imgsz）。
    框是**画布归一化**的，所以 imgsz 必须传 (H, W) 元组 —— 非正方形画布下
    只传一个标量会让 y 方向的框整体缩放错误。
    """
    bi, cl, bx = [], [], []
    for i, b in enumerate(batch["boxes"]):
        if b is None or len(b) == 0:
            continue
        n = len(b)
        bi.append(torch.full((n,), i, dtype=torch.float32))
        cl.append(b[:, 0])
        bx.append(b[:, 1:5])                                # 保持**归一化 xywh**（损失内部会乘 imgsz）
    dev = device
    if bi:
        batch_idx = torch.cat(bi).to(dev)
        cls = torch.cat(cl).to(dev).float()
        bboxes = torch.cat(bx).to(dev).float()
    else:
        batch_idx = torch.zeros(0, device=dev)
        cls = torch.zeros(0, device=dev)
        bboxes = torch.zeros((0, 4), device=dev)
    return {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes,
            "imgsz": torch.tensor([canvas[0], canvas[1]], device=dev),
            "batch_size": len(batch["boxes"])}


def set_bn_eval(model: nn.Module, include_new: bool = False) -> int:
    """小 batch 下把 BN 切到 eval：用**预训练 running stats**，不用含噪批统计。

    预训练主干/颈部的 BN 是在 COCO 大 batch 上估的；本地 batch 2–8 时批统计噪声极大，
    会把预训练特征带偏。**梯度累积只解决梯度方差，解决不了 BN 批统计**，所以两者要分开处理。

    `include_new=False` 时**跳过本框架新增的 BN**
    （融合块 BN、逐模态 BN 副本）——它们的 running stats 是刚初始化的，
    冻结等于让新模块永远用"均值 0/方差 1"的假统计。旧版把**所有** BN 一律 eval，
    且命令行无法关闭。
    """
    n = 0
    new_prefixes = ("fusion.", "bn_store.", "late_bus.")
    for name, m in model.named_modules():
        if isinstance(m, BN_TYPES):
            if (not include_new) and name.startswith(new_prefixes):
                continue
            m.eval()
            n += 1
    return n


def prevent_sleep(enable: bool = True) -> bool:
    """Windows：由进程自己持有 power request，阻止系统休眠 / Modern Standby。

    实测教训：笔记本会自己进入 Modern Standby（系统事件 506/507 各两次），
    GPU 进程被挂起 → 训练直接死（日志停在 ep11 / ep22，连看门狗任务都被清掉）。
    不依赖电源方案设置（用户可能随时改回省电），这里用 SetThreadExecutionState 最可靠。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | (ES_AWAYMODE_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
        return True
    except Exception:                                            # noqa: BLE001
        return False


# ---------------------------------------------------------------- 冻结/优化器

def set_encoder_frozen(model: MMYOLO, frozen: bool) -> None:
    """两阶段：先冻编码器（主干 0–10 + 辅助 stem + 适配器），只训融合/颈部/头。"""
    targets = model.encoder_modules() if hasattr(model, "encoder_modules") else [model.backbone.model[:11]]
    if hasattr(model, "dep_stem"):
        targets.append(model.dep_stem)
    if hasattr(model, "aux_stages"):
        targets.append(model.aux_stages)
    for mod in targets:
        for p in mod.parameters():
            p.requires_grad_(not frozen)


def enable_trainable_defaults(model: MMYOLO) -> None:
    """Undo checkpoint-side freezes before applying this trainer's stage policy.

    Ultralytics checkpoints may retain ``requires_grad=False`` on pretrained
    neck/head tensors.  The old trainer only toggled encoder layers 0..10, so
    roughly 4.8M neck/detection parameters never adapted to this dataset.  The
    trainer owns freeze policy: start from all trainable, then keep only fixed
    integral DFL projections frozen.
    """
    for p in model.parameters():
        p.requires_grad_(True)
    detectors = [model.model[-1]]
    semantic = getattr(model, "semantic_detect", None)
    if semantic is not None:
        detectors.append(semantic)
    for branch in getattr(model, "independent_aux", {}).values():
        detectors.append(branch.detector)
    for detector in detectors:
        dfl = getattr(detector, "dfl", None)
        if dfl is not None:
            dfl.requires_grad_(False)


def set_aux_adaptation_mode(model: MMYOLO) -> None:
    """Stage A: keep the pretrained RGB detector fixed and adapt auxiliary evidence.

    The frozen detector/neck remains differentiable with respect to its inputs, so
    the main detection loss still teaches IR/Depth features to fit the established
    decision space.  Only auxiliary encoders, their embeddings, residual matching,
    auxiliary fusion projections, and cross-scale memory are trainable.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    trainable = []
    for name in ("aux_encoders", "metric_encoder", "matchers",
                 "register_bus", "neck_memory"):
        module = getattr(model, name, None)
        if module is not None:
            trainable.append(module)
    for module in trainable:
        for p in module.parameters():
            p.requires_grad_(True)

    if hasattr(model, "embeddings"):
        for blocks in model.embeddings.values():
            for block in blocks[1:]:
                for p in block.parameters():
                    p.requires_grad_(True)
    if hasattr(model, "fusion"):
        for block in model.fusion.values():
            # Preserve the RGB anchor/query and the calibrated global gain.  Only
            # IR/Depth content paths learn during auxiliary adaptation.
            for module in list(block.gates[1:]) + list(block.outputs[1:]):
                for p in module.parameters():
                    p.requires_grad_(True)


def set_independent_aux_mode(model: MMYOLO) -> None:
    """Stage A: train IR/Depth as real standalone detectors.

    The RGB encoder, fused route, shared semantic head and memory are completely
    frozen.  Direct GT detection losses update an auxiliary encoder and its own
    full P2--P5 neck/head; the batch loop alternates IR and Depth to fit 24 GB.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (getattr(model, "aux_encoders", None),
                   getattr(model, "metric_encoder", None),
                   getattr(model, "independent_aux", None)):
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(True)
    for branch in getattr(model, "independent_aux", {}).values():
        dfl = getattr(branch.detector, "dfl", None)
        if dfl is not None:
            dfl.requires_grad_(False)


def set_residual_fusion_mode(model: MMYOLO, downstream_frozen: bool) -> None:
    """Stage B: open zero-initialized IR/Depth residuals around a fixed RGB path.

    During the initial fusion-only period even the independently trained
    auxiliary encoders remain fixed.  Afterwards they and the high-resolution
    downstream detector are released at their role-specific low learning rates.
    The RGB encoder/query/embedding and RGB residual switch remain immutable.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def enable(module):
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(True)

    # Learn how much reliable auxiliary evidence to inject.  Slot 0 (RGB) is
    # deliberately excluded, keeping F_out == F_rgb at the hand-off.
    if hasattr(model, "fusion"):
        for block in model.fusion.values():
            for module in list(block.gates[1:]) + list(block.outputs[1:]):
                enable(module)
            block.residual_scale.requires_grad_(True)
    enable(getattr(model, "register_bus", None))
    enable(getattr(model, "neck_memory", None))
    enable(getattr(model, "semantic_adapters", None))
    enable(getattr(model, "semantic_detect", None))

    if not downstream_frozen:
        enable(getattr(model, "aux_encoders", None))
        enable(getattr(model, "metric_encoder", None))
        enable(getattr(model, "matchers", None))
        if hasattr(model, "embeddings"):
            for blocks in model.embeddings.values():
                for block in blocks[1:]:
                    enable(block)
        for name in ("p2_lateral", "p2_neck", "p2_down", "p3_refine", "localization"):
            enable(getattr(model, name, None))
        for name in ("neck_gain", "localization_scale"):
            value = getattr(model, name, None)
            if value is not None:
                value.requires_grad_(True)
        enable(model.backbone.model[11:])

    for detector in (model.model[-1], getattr(model, "semantic_detect", None)):
        dfl = getattr(detector, "dfl", None)
        if dfl is not None:
            dfl.requires_grad_(False)


def set_anchored_joint_mode(model: MMYOLO, detector_frozen: bool) -> None:
    """V4.2c: retain a fixed RGB semantic coordinate system.

    Stage A learned IR/Depth against a frozen RGB detector.  Unfreezing every
    embedding, fusion query and detector tensor at once makes that coordinate
    system move and caused immediate validation collapse.  This policy keeps the
    RGB evidence/query/gain anchor fixed, learns auxiliary evidence plus the new
    P2/localization path, then cautiously releases only the downstream detector.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def enable(module):
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(True)

    for name in ("aux_encoders", "metric_encoder", "matchers",
                 "register_bus", "neck_memory"):
        enable(getattr(model, name, None))

    # Modality 0 is the fixed RGB semantic anchor.  IR/Depth retain enough
    # capacity to learn common/private evidence without rotating the reference.
    if hasattr(model, "embeddings"):
        for blocks in model.embeddings.values():
            for block in blocks[1:]:
                enable(block)
    if hasattr(model, "fusion"):
        for block in model.fusion.values():
            for module in list(block.gates[1:]) + list(block.outputs[1:]):
                enable(module)

    # New high-resolution/localization modules may adapt from the beginning.
    for name in ("p2_lateral", "p2_neck", "p2_down", "p3_refine", "localization"):
        enable(getattr(model, name, None))
    for name in ("neck_gain", "loc_gain"):
        value = getattr(model, name, None)
        if value is not None:
            value.requires_grad_(True)
    detector = model.model[-1]
    for branch_name in ("cv2", "cv3"):
        branches = getattr(detector, branch_name, None)
        if branches is not None and len(branches):
            enable(branches[0])

    enable(getattr(model, "semantic_adapters", None))
    enable(getattr(model, "semantic_detect", None))

    if not detector_frozen:
        enable(model.backbone.model[11:])
    dfl = getattr(detector, "dfl", None)
    if dfl is not None:
        dfl.requires_grad_(False)


def set_frozen_bn_eval(model: nn.Module) -> None:
    """Frozen affine BN parameters must not keep changing population statistics."""
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            local = list(module.parameters(recurse=False))
            if local and not any(p.requires_grad for p in local):
                module.eval()


def build_optimizer(model: MMYOLO, lr: float, backbone_mult: float, wd: float = 5e-4,
                    role_mults: dict | None = None):
    """按真实模块归属分组：仅预训练编码器用小 lr，其余部分用基础 lr。

    ``model.backbone`` 是完整的 Ultralytics DetectionModel，除了编码器还包含 Neck、
    Detect 回归头和刚替换的 12 类分类层。不能用 ``name.startswith("backbone.")``
    判断“预训练主干”：那会把随机初始化的分类层也乘 ``backbone_mult``；RGB-only
    模式下所有实际参与输出的参数都会只拿到日志学习率的十分之一。这里用前 11 层
    编码器的参数 id 做集合判定，Neck/Detect/融合/模态分支均使用基础学习率。

    BN/bias 不施加权重衰减。

    ⚠️ **不要**按 requires_grad 过滤参数：两阶段训练里前 N 轮冻结编码器，
    若建优化器时把它们排除，后面解冻时这些参数根本不在优化器里 → 编码器永远不更新
    （等于全程线性探测）。冻结只靠 requires_grad 控制梯度，参数照进优化器；
    PyTorch 对 grad=None 的参数不会更新，行为正确。

    ⚠️ 参数只从 `model.parameters()` 取（去重）：`stage_shared` 是 backbone 层的别名，
    若再按名字遍历一遍会把同一参数放进两个 group → AdamW 对同一参数更新两次。
    """
    encoder_ids = {id(p) for p in model.backbone.model[:11].parameters()}
    if hasattr(model, "encoder_modules"):
        encoder_ids = {id(p) for module in model.encoder_modules() for p in module.parameters()}
    # Depth stem 是从 COCO RGB stem 深拷贝得到的预训练编码器，不是随机初始化任务层。
    # 把它留在 task 组会让副本以 10× 学习率漂移，而来源相同的 RGB stem 只拿
    # encoder LR。IR/Depth 的 1x1 输入 adapter 是新建层，仍应留在 task 组。
    if hasattr(model, "dep_stem"):
        encoder_ids.update(id(p) for p in model.dep_stem.parameters())
    if getattr(model, "spatial_memory", False) and hasattr(model, "aux_stages"):
        encoder_ids.update(id(p) for p in model.aux_stages.parameters())
    if getattr(model, "spatial_memory", False) and hasattr(model, "bn_store"):
        encoder_ids.update(id(p) for p in model.bn_store.parameters())
    if role_mults is not None:
        role_ids = {name: set() for name in role_mults}

        def add(role, module):
            if module is not None and role in role_ids:
                role_ids[role].update(id(p) for p in module.parameters())

        add("anchor", model.backbone.model[:11])
        add("aux_encoder", getattr(model, "aux_encoders", None))
        add("aux_encoder", getattr(model, "metric_encoder", None))
        add("fusion", getattr(model, "matchers", None))
        add("fusion", getattr(model, "register_bus", None))
        add("fusion", getattr(model, "neck_memory", None))
        if hasattr(model, "embeddings"):
            for blocks in model.embeddings.values():
                add("anchor", blocks[0])
                for block in blocks[1:]:
                    add("fusion", block)
        if hasattr(model, "fusion"):
            for block in model.fusion.values():
                add("anchor", block.query)
                add("anchor", block.context)
                role_ids.get("anchor", set()).update((id(block.identity), id(block.gain)))
                add("anchor", block.gates[0])
                add("anchor", block.outputs[0])
                for module in list(block.gates[1:]) + list(block.outputs[1:]):
                    add("fusion", module)
        for name in ("p2_lateral", "p2_neck", "p2_down", "p3_refine", "localization"):
            add("p2", getattr(model, name, None))
        for name in ("neck_gain", "loc_gain", "localization_scale"):
            value = getattr(model, name, None)
            if value is not None and "p2" in role_ids:
                role_ids["p2"].add(id(value))
        detector = model.model[-1]
        for branch_name in ("cv2", "cv3"):
            branches = getattr(detector, branch_name, None)
            if branches is not None and len(branches):
                add("p2", branches[0])
        add("detector", model.backbone.model[11:])
        add("semantic", getattr(model, "semantic_adapters", None))
        add("semantic", getattr(model, "semantic_detect", None))
        add("semantic", getattr(model, "independent_aux", None))

        # Resolve aliases/overlap by priority: the new P2 head must not inherit
        # the slower pretrained detector rate; the frozen anchor always wins.
        priority = ("anchor", "p2", "semantic", "aux_encoder", "fusion", "detector")
        owner = {}
        for role in reversed(priority):
            for pid in role_ids.get(role, ()):
                owner[pid] = role
        groups = collections.defaultdict(list)
        seen = set()
        for name, p in model.named_parameters():
            if p is None or id(p) in seen:
                continue
            seen.add(id(p))
            role = owner.get(id(p), "fusion")
            no_decay = p.ndim == 1 or name.endswith(".bias")
            groups[(role, no_decay)].append(p)
        params = []
        for (role, no_decay), tensors in groups.items():
            mult = float(role_mults.get(role, 1.0))
            params.append({"params": tensors, "lr": lr * mult, "lr_mult": mult,
                           "role": role, "weight_decay": 0.0 if no_decay else wd})
        return torch.optim.AdamW(params, betas=(0.9, 0.999))

    seen = set()
    groups = {"encoder_decay": [], "encoder_nodecay": [],
              "task_decay": [], "task_nodecay": []}
    for n, p in model.named_parameters():
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        encoder = id(p) in encoder_ids
        nd = p.ndim == 1 or n.endswith(".bias")
        key = ("encoder" if encoder else "task") + ("_nodecay" if nd else "_decay")
        groups[key].append(p)
    params = [
        {"params": groups["encoder_decay"], "lr": lr * backbone_mult,
         "lr_mult": backbone_mult, "role": "encoder", "weight_decay": wd},
        {"params": groups["encoder_nodecay"], "lr": lr * backbone_mult,
         "lr_mult": backbone_mult, "role": "encoder", "weight_decay": 0.0},
        {"params": groups["task_decay"], "lr": lr,
         "lr_mult": 1.0, "role": "task", "weight_decay": wd},
        {"params": groups["task_nodecay"], "lr": lr,
         "lr_mult": 1.0, "role": "task", "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in params if g["params"]], betas=(0.9, 0.999))


def accumulation_loss(loss_vec: torch.Tensor, nominal_samples: int) -> torch.Tensor:
    """把 Ultralytics 返回的 batch 总损失变成有效 batch 的逐样本均值贡献。

    ``v8DetectionLoss`` 已经把三分量乘了当前物理 batch size。每个 micro-batch
    都除以名义有效样本数，累积满后得到整个 optimizer step 的逐样本均值；尾组
    在 step 前再按实际样本数校正。这样 batch/accum 的不同组合不再隐式改变梯度尺度。
    """
    if nominal_samples < 1:
        raise ValueError("nominal_samples 必须为正整数")
    return loss_vec.sum() / float(nominal_samples)


def subset_detection_batch(preds: dict, targets: dict, active: torch.Tensor) -> tuple[dict, dict]:
    """Select observed samples and remap target indices for one auxiliary modality."""
    ids = active.nonzero(as_tuple=True)[0]
    if not ids.numel():
        raise ValueError("subset_detection_batch requires an observed sample")
    mapping = torch.full((active.numel(),), -1, dtype=torch.long, device=active.device)
    mapping[ids] = torch.arange(ids.numel(), device=active.device)
    if targets["batch_idx"].numel():
        target_keep = active.index_select(0, targets["batch_idx"].long())
    else:
        target_keep = active.new_zeros(0)
    sub_targets = {
        "batch_idx": mapping.index_select(0, targets["batch_idx"][target_keep].long()).float(),
        "cls": targets["cls"][target_keep],
        "bboxes": targets["bboxes"][target_keep],
        "imgsz": targets["imgsz"],
        "batch_size": int(ids.numel()),
    }
    sub_preds = {
        "boxes": preds["boxes"].index_select(0, ids),
        "scores": preds["scores"].index_select(0, ids),
        "feats": [x.index_select(0, ids) for x in preds["feats"]],
    }
    return sub_preds, sub_targets


def ensure_finite_state(model: nn.Module, label: str = "model") -> None:
    """拒绝保存被 NaN/Inf 污染的参数或 BN buffer。

    AMP GradScaler 只会跳过非有限梯度，不会回滚 BatchNorm 在 forward 里已更新的
    running_mean/running_var。不在保存边界做检查，看门狗会不断从坏 last.pt 续训。
    """
    bad = []
    for name, value in model.state_dict().items():
        if (torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
                and not torch.isfinite(value).all()):
            bad.append(name)
            if len(bad) >= 12:
                break
    if bad:
        raise FloatingPointError(f"{label} 含非有限状态，拒绝保存：{bad}")


def lr_at(epoch: int, epochs: int, lr0: float, warmup: int = 3, lrf: float = 0.01) -> float:
    if epoch < warmup:
        return lr0 * (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, epochs - warmup)
    return lr0 * (lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def validate_checkpoint(ck: dict, model: MMYOLO, enabled, canvas, exact: bool = True,
                        args=None, split_digest: str = "") -> None:
    """先检查再加载；不能把 EMA-only 权重与原模型的 Adam 状态当成精确续训。"""
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError("checkpoint 缺少 model_state")
    saved_structure = copy.deepcopy(dict(ck.get("structure") or {}))
    current_structure = copy.deepcopy(dict(model.structure_kwargs()))
    # 权重文件路径不是网络结构；Windows→Linux/服务器迁移时允许路径变化。
    saved_structure.pop("weights", None)
    current_structure.pop("weights", None)
    for structure in (saved_structure, current_structure):
        structure.setdefault("depth_resampling", "legacy_bilinear_v1")
        structure.get("encoder", {}).setdefault("metric_branch", False)
        structure.get("encoder", {}).setdefault("checkpoint_encoder", False)
        if isinstance(structure.get("fusion"), dict):
            structure["fusion"].setdefault("spatial_dim", 64)
            structure["fusion"].setdefault("memory_tokens_per_modality", 4)
            structure["fusion"].setdefault("architecture", "legacy_hook_v1")
            structure["fusion"].setdefault("memory_control", "unbounded_v1")
            structure["fusion"].setdefault("branch_aux_weight", 0.0)
            structure["fusion"].setdefault("flow_supervision_weight", 0.0)
            structure["fusion"].setdefault("cross_modal_nce_weight", 0.0)
            structure["fusion"].setdefault("nce_temperature", 0.10)
            structure["fusion"].setdefault("p2_match_refine", False)
    # B1 结构里还没有该字段，其实际语义就是 2ch。先规范化，保证历史
    # last.pt 仍可严格 --resume；B2 则显式记录 4ch。
    if "encoder" in saved_structure:
        saved_structure["encoder"].setdefault("depth_input_channels", 2)
        saved_structure["encoder"].setdefault("depth_view", "both")
        saved_structure["encoder"].setdefault("depth_init", "relative")
    if "encoder" in current_structure:
        current_structure["encoder"].setdefault("depth_input_channels", 2)
        current_structure["encoder"].setdefault("depth_view", "both")
        current_structure["encoder"].setdefault("depth_init", "relative")
    # init-only 允许 B1 2ch Depth → B2 4ch Depth；权重在下方做显式通道迁移。
    if not exact:
        saved_structure.get("encoder", {}).pop("depth_input_channels", None)
        current_structure.get("encoder", {}).pop("depth_input_channels", None)
        if getattr(args, "reset_fusion_gates", False):
            old_f, new_f = saved_structure.get("fusion", {}), current_structure.get("fusion", {})
            if (old_f.get("architecture") != "spatial_memory_v1" or
                    new_f.get("architecture") != "spatial_memory_v1" or
                    old_f.get("memory_control") != "unbounded_v1" or new_f.get("memory_control") != "bounded_v2"):
                raise ValueError("explicit gate repair only allows unbounded_v1 -> bounded_v2")
            if ck.get("meta", {}).get("split_digest") != split_digest:
                raise ValueError("repair must retain the source train/validation split")
            old_f["memory_control"] = "bounded_v2"
    # 损失权重与匹配下限是训练超参，不改变 state_dict 的形状：warm-start
    # （--init-checkpoint）必须允许在保留权重的前提下调整它们。精确 --resume
    # 仍由下方 args 白名单拒绝超参变更。p2_match_refine 会改变匹配器数量，
    # 属于真实结构差异，继续保留在比较里。
    for structure in (saved_structure, current_structure):
        fusion = structure.get("fusion")
        if isinstance(fusion, dict):
            for key in ("branch_aux_weight", "branch_aux_weights", "flow_supervision_weight",
                        "cross_modal_nce_weight", "nce_temperature", "match_floor",
                        "alignment_mode", "depth_reliability", "flow_identity_weight"):
                fusion.pop(key, None)
    if saved_structure != current_structure:
        raise ValueError("checkpoint 结构与当前模型不一致（fusion/share/late-bus 等）")
    meta = ck.get("meta") or {}
    if tuple(meta.get("modalities", ())) != tuple(enabled):
        raise ValueError(f"checkpoint 训练模态 {meta.get('modalities')} 与当前 {enabled} 不一致")
    if exact and tuple(meta.get("canvas", ())) != tuple(canvas):
        raise ValueError(f"checkpoint 画布 {meta.get('canvas')} 与当前 {canvas} 不一致")
    if not exact:
        return
    ts = ck.get("train_state") or {}
    required = ("raw_model_state", "optimizer", "scaler", "ema_state", "ema_updates", "rng")
    missing = [k for k in required if k not in ts]
    if missing:
        raise ValueError(f"旧 checkpoint 无法精确 --resume：缺 {missing}；"
                         "请用新的 --name 加 --init-checkpoint 仅初始化模型")
    if not split_digest or meta.get("split_digest") != split_digest:
        raise ValueError("checkpoint 的训练/验证划分与当前 split.json 不一致，不能精确续训")
    if not all(k in ts["rng"] for k in ("torch", "numpy", "loader")):
        raise ValueError("checkpoint 缺少 RNG 状态，无法精确续训")
    if meta.get("trainer_recipe") != (recipe_for(args) if args is not None else TRAINER_RECIPE):
        raise ValueError("checkpoint 来自旧训练算法（loss/weight-decay/optimizer 分组已变化），"
                         "不能精确 --resume；可用新的 --name + --init-checkpoint 仅初始化")
    if args is not None:
        old = ts.get("args") or {}
        keys = ("modalities", "imgsz", "epochs", "batch", "accum", "lr",
                "backbone_lr_mult", "fusion_lr_mult", "p2_lr_mult", "detector_lr_mult",
                "semantic_lr_mult", "freeze_epochs", "freeze_bn", "freeze_new_bn",
                "fusion_tier", "share_tier", "register_bus", "late_bus", "depth_scales",
                "no_quality", "no_prior", "no_deformable", "no_dropout",
                "rgb_dropout", "aux_dropout", "dropout_start_epoch", "depth_channels",
                "depth_view", "depth_init",
                "misalign_px", "degrade_p", "rgb_color_p", "ir_noise_p", "ir_gain_p", "depth_hole_p",
                "target_crop_p", "rare_sample_max", "seed")
        keys += ("weight_decay", "nominal_batch", "grad_clip")
        keys += ("architecture", "depth_resampling", "metric_branch", "sampler", "rare_extra_frac",
                 "close_aug_frac", "bn_policy", "warmup", "lrf", "calibrate_clip_steps", "memory_control")
        keys += ("scale_min", "scale_max", "translate")
        keys += ("precision", "checkpoint_encoder", "mosaic", "full_data")
        keys += ("branch_aux_weight", "flow_supervision_weight", "cross_modal_nce_weight",
                 "nce_temperature", "p2_match_refine", "match_floor", "branch_aux_weights",
                 "branch_aux_end_weights", "flow_supervision_end_weight",
                 "cross_modal_nce_end_weight", "alignment_mode", "depth_reliability",
                 "flow_identity_weight", "embedding_recon_weight", "embedding_recon_end_weight",
                 "embedding_alignment_weight", "embedding_alignment_end_weight", "train_stage")
        legacy = {"depth_channels": 2, "depth_view": "both", "depth_init": "relative",
                  "misalign_px": 15.0, "degrade_p": 0.3, "rgb_color_p": 0.0,
                  "ir_noise_p": 0.0, "ir_gain_p": 0.0,
                  "depth_hole_p": 0.0, "target_crop_p": 0.0,
                  "scale_min": .75, "scale_max": 1.4, "translate": .1,
                  "rare_sample_max": 1.0, "weight_decay": 5e-4,
                  "nominal_batch": 64, "grad_clip": 10.0}
        legacy.update(architecture="legacy_hook_v1", depth_resampling="legacy_bilinear_v1",
                      metric_branch=False, sampler="legacy", rare_extra_frac=.1,
                      close_aug_frac=0., bn_policy="legacy", warmup=3, lrf=.01, calibrate_clip_steps=0,
                      memory_control="unbounded_v1")
        legacy.update(precision="fp16", checkpoint_encoder=False, mosaic=0., full_data=False)
        legacy.update(branch_aux_weight=0., flow_supervision_weight=0.,
                      cross_modal_nce_weight=0., nce_temperature=.1, p2_match_refine=False)
        legacy.update(match_floor=0., branch_aux_weights=None, branch_aux_end_weights=None,
                      flow_supervision_end_weight=None, cross_modal_nce_end_weight=None,
                      alignment_mode="legacy_gate_v1", depth_reliability="legacy_edge_v1",
                      flow_identity_weight=0., embedding_recon_weight=.01,
                      embedding_recon_end_weight=None, embedding_alignment_weight=.005,
                      embedding_alignment_end_weight=None, fusion_lr_mult=1., p2_lr_mult=1.,
                      detector_lr_mult=1., semantic_lr_mult=1., train_stage="standard")
        changed = [k for k in keys
                   if old.get(k, legacy.get(k)) != getattr(args, k, legacy.get(k))]
        if changed:
            raise ValueError(f"--resume 训练设置发生变化：{changed}；请新开实验")


def adapt_depth_checkpoint_state(state: dict, model: MMYOLO) -> tuple:
    """Explicit init-only migrations; all unrelated differences remain strict."""
    out = dict(state)
    migrated = False
    target_state = model.state_dict()
    # V4.3 adds training-only standalone IR/Depth detectors and exact-zero
    # residual switches.  Old V4.2 checkpoints are the intended initialization
    # source; initialize only these named additions from the freshly constructed
    # COCO-derived modules, never silently accept arbitrary missing tensors.
    for name, target in target_state.items():
        is_new_switch = name.endswith(".residual_scale") or name == "localization_scale"
        if name not in out and (name.startswith("independent_aux.") or is_new_switch):
            out[name] = target.detach().clone()
            migrated = True
    key = "dep_adapter.weight"
    if key not in state:
        return out, migrated
    old, target = state[key], target_state[key]
    if tuple(old.shape) == tuple(target.shape):
        return out, migrated
    if old.ndim != 4 or target.ndim != 4 or old.shape[0] != target.shape[0] or old.shape[2:] != target.shape[2:]:
        raise ValueError(f"{key} 无法迁移：{tuple(old.shape)} -> {tuple(target.shape)}")
    if old.shape[1] == 2 and target.shape[1] == 4:
        w = target.detach().clone().zero_()
        w[:, 0] = old[:, 0]       # relative -> relative
        w[:, 2] = old[:, 1]       # old valid -> new valid
    elif old.shape[1] == 4 and target.shape[1] == 2:
        w = target.detach().clone()
        w[:, 0] = old[:, 0]
        w[:, 1] = old[:, 2]
    else:
        raise ValueError(f"{key} 只支持 2ch↔4ch 迁移：{tuple(old.shape)} -> {tuple(target.shape)}")
    out[key] = w
    return out, True


# ---------------------------------------------------------------- 后台预取

def iter_prefetch(loader, depth: int = 3, stop: Optional["threading.Event"] = None):
    """在后台线程里取数据（含 worker 进程启动），主线程只做 GPU 计算。

    动机（三模态 batch4/544×960 的 I/O/GPU 实测）：
        取数据 314 ms/step，GPU 348 ms/step → 串行 662 ms/step
        完全重叠后 = max(314, 348) ≈ 348 ms/step，**实测加速 1.39×**
    这在 `workers=0` 时是唯一能抢回来的算力（受限环境里 worker 进程起不来），
    在 `workers>0` 时也有收益（掩盖主进程侧 collate / 拷贝）。

    ⚠️ worker 启动异常**发生在 `iter(loader)` 里**，所以迭代器必须在生产线程内部创建，
    这样 `PermissionError` 才能经队列回抛给主线程（train.py 靠它退回单进程）。
    """
    import queue as _queue
    import threading

    q: "_queue.Queue" = _queue.Queue(maxsize=max(1, int(depth)))
    def _produce():
        try:
            for b in loader:
                if stop is not None and stop.is_set():
                    break
                while stop is None or not stop.is_set():
                    try:
                        q.put((b, None), timeout=0.2)
                        break
                    except _queue.Full:
                        continue
        except BaseException as exc:                             # noqa: BLE001
            while stop is None or not stop.is_set():
                try:
                    q.put((None, exc), timeout=0.2)
                    break
                except _queue.Full:
                    continue
        finally:
            while stop is None or not stop.is_set():
                try:
                    q.put((None, None), timeout=0.2)
                    break
                except _queue.Full:
                    continue

    th = threading.Thread(target=_produce, daemon=True, name="mm-prefetch")
    th.start()
    try:
        while True:
            batch, err = q.get()
            if err is not None:
                raise err
            if batch is None:
                break
            yield batch
    finally:
        if stop is not None:
            stop.set()
        while not q.empty():                                     # 排空，避免生产线程卡在 put
            try:
                q.get_nowait()
            except Exception:                                    # noqa: BLE001
                break
        th.join(timeout=5.0)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="含 visible/infrared/depth 的训练目录")
    ap.add_argument("--labels", required=True, help="标签目录（用官方修正版 new_labels_2000）")
    ap.add_argument("--out", default=str(_CODE / "runs"))
    ap.add_argument("--name", default="mm_run")
    ap.add_argument("--weights", default="yolo11s.pt",
                    help="本地 COCO 预训练权重；服务器可传绝对路径（不联网下载）")
    ap.add_argument("--device", default="auto",
                    help="auto/cpu/cuda/cuda:0；服务器多卡时配合 CUDA_VISIBLE_DEVICES")
    ap.add_argument("--modalities", default="all",
                    choices=["all", "rgb", "rgb_ir", "rgb_dep", "ir", "dep"],
                    help="模态对照；ir/dep 是单模态信号实验，不输入 RGB 像素")
    ap.add_argument("--imgsz", default="544x960",
                    help="画布：'960'（正方形）或 'HxW'（如 544x960 匹配 16:9 原图，省约 44%% 算力）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rgb-batch", type=int, default=0, help="仅 RGB 模式的物理 batch 覆盖；累积另设")
    ap.add_argument("--architecture", default="legacy_hook_v1", choices=["legacy_hook_v1", "spatial_memory_v1", "independent_p2_memory_v3"])
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--checkpoint-encoder", action="store_true")
    ap.add_argument("--mosaic", type=float, default=0.)
    ap.add_argument("--full-data", action="store_true", help="final refit; all labeled data, no validation claims")
    ap.add_argument("--memory-control", default="unbounded_v1", choices=["unbounded_v1", "bounded_v2"])
    ap.add_argument("--reset-fusion-gates", action="store_true", help="explicit v1->v2 warm-start, not exact resume")
    ap.add_argument("--depth-resampling", default="legacy_bilinear_v1", choices=["legacy_bilinear_v1", "nearest_valid_v2"])
    ap.add_argument("--metric-branch", action="store_true")
    ap.add_argument("--sampler", default="legacy", choices=["legacy", "coverage"])
    ap.add_argument("--rare-extra-frac", type=float, default=.1)
    ap.add_argument("--close-aug-frac", type=float, default=0.0)
    ap.add_argument("--scale-min", type=float, default=.75,
                    help="同步几何增强的最小缩放；定位精修可设为0.90")
    ap.add_argument("--scale-max", type=float, default=1.40,
                    help="同步几何增强的最大缩放；定位精修可设为1.10")
    ap.add_argument("--translate", type=float, default=.10,
                    help="同步平移上限（相对画布）；定位精修可设为0.03")
    ap.add_argument("--bn-policy", default="legacy", choices=["legacy", "adaptive", "adaptive_no_tail"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lrf", type=float, default=.01)
    ap.add_argument("--calibrate-clip-steps", type=int, default=0,
                    help="warmup 后收集 N 个有限 step，以 1.5×P90 校准，范围 20–200；保存实际阈值")
    ap.add_argument("--val-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积（等效批大小 = batch×accum）")
    ap.add_argument("--freeze-bn", action="store_true", default=True,
                    help="小 batch 时把 BN 切 eval（用预训练 running stats）；大 batch 可关")
    ap.add_argument("--no-freeze-bn", dest="freeze_bn", action="store_false",
                    help="允许所有 BN 训练（batch 足够大时）")
    ap.add_argument("--freeze-new-bn", action="store_true",
                    help="连新建的融合 BN / 逐模态 BN 也冻结（旧行为，不推荐）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--fusion-lr-mult", type=float, default=1.0,
                    help="anchored_joint 中辅助融合/记忆相对基础 lr")
    ap.add_argument("--p2-lr-mult", type=float, default=1.0,
                    help="anchored_joint 中新 P2/定位路径相对基础 lr")
    ap.add_argument("--detector-lr-mult", type=float, default=1.0,
                    help="anchored_joint 中预训练 Neck/Detect 相对基础 lr")
    ap.add_argument("--semantic-lr-mult", type=float, default=1.0,
                    help="anchored_joint 中训练期辅助检测头相对基础 lr")
    ap.add_argument("--weight-decay", type=float, default=5e-4,
                    help="名义 batch 下的 AdamW 衰减；会按有效 batch/nominal-batch 缩放")
    ap.add_argument("--nominal-batch", type=int, default=64,
                    help="weight decay 标定用名义有效 batch（Ultralytics 默认 64）")
    ap.add_argument("--grad-clip", type=float, default=60.0,
                    help="全局梯度范数上限；0=关闭。现有稳定范数约 32–41，60 只拦较大尖峰；"
                         "旧 optfix_v3 权重精确续训须显式设为 10")
    ap.add_argument("--freeze-epochs", type=int, default=5, help="前 N 轮冻结编码器")
    ap.add_argument("--train-stage", default="standard",
                    choices=["standard", "aux_adapt", "anchored_joint",
                             "aux_independent", "residual_fusion"],
                    help="aux_independent 独立预训 IR/Depth 检测器；residual_fusion 从严格 RGB 恒等映射融合")
    ap.add_argument("--fusion-tier", default="L2", choices=["L0", "L1", "L2", "L3"])
    ap.add_argument("--share-tier", default="c", choices=["a", "b", "c"])
    ap.add_argument("--register-bus", dest="register_bus", action="store_true", default=True,
                    help="启用贯穿 P3→P4→P5 的动态 register（默认开）")
    ap.add_argument("--no-register-bus", dest="register_bus", action="store_false",
                    help="关闭 register，做严格消融")
    ap.add_argument("--late-bus", action="store_true",
                    help="额外启用旧的一次性片后 FiLM（仅消融，不建议与 register 同开）")
    ap.add_argument("--depth-scales", default="p4p5", choices=["p4p5", "all", "p3p4", "p4"],
                    help="Depth 进入哪些尺度；有轻微幻影时默认只进 P4/P5")
    ap.add_argument("--depth-channels", type=int, default=4, choices=[2, 4],
                    help="4=B2[相对,绝对,valid,metric]；2=仅用于严格续训旧 B1")
    ap.add_argument("--depth-view", default="both",
                    choices=["both", "relative", "metric_fallback", "metric_log_fallback"],
                    help="Depth 表示：全部 / 相对 / PNG 米制线性或 log + JPG 相对回退")
    ap.add_argument("--depth-init", default="relative",
                    choices=["relative", "balanced", "metric_fallback"],
                    help="Depth 1x1 适配器初始化；balanced 把相对/绝对各初始化为 0.5")
    ap.add_argument("--no-quality", action="store_true", help="关掉质量描述子门控（消融）")
    ap.add_argument("--no-prior", action="store_true", help="关掉深度先验 FiLM（消融）")
    ap.add_argument("--no-deformable", action="store_true", help="关掉可变形残差对齐（消融）")
    ap.add_argument("--no-dropout", action="store_true", help="关掉 modality dropout（消融）")
    ap.add_argument("--rgb-dropout", type=float, default=0.05)
    ap.add_argument("--aux-dropout", type=float, default=0.05)
    ap.add_argument("--dropout-start-epoch", type=int, default=5,
                    help="前 N 轮不做模态 dropout，先学稳定融合")
    ap.add_argument("--misalign-px", type=float, default=5.0,
                    help="Depth 训练时残差错位上限（像素）；幻影轻微时不应过大")
    ap.add_argument("--degrade-p", type=float, default=0.30, help="RGB 真低照退化概率")
    ap.add_argument("--rgb-color-p", type=float, default=0.0, help="RGB 小幅颜色/对比度扰动概率")
    ap.add_argument("--ir-noise-p", type=float, default=0.15, help="IR 轻噪声概率")
    ap.add_argument("--ir-gain-p", type=float, default=0.0, help="IR 单调增益/偏置扰动概率")
    ap.add_argument("--depth-hole-p", type=float, default=0.15, help="Depth 小块失效概率")
    ap.add_argument("--target-crop-p", type=float, default=0.15,
                    help="三模态同步目标感知裁剪概率")
    ap.add_argument("--rare-sample-max", type=float, default=3.0,
                    help="稀有类图像采样权重上限；1=关闭均衡采样")
    ap.add_argument("--prefetch", action="store_true", default=True,
                    help="后台线程预取数据（实测 1.39×；ws=0 时尤其重要）")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个样本（冒烟测试）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=0, help="每 N 轮做一次赛题口径验证（0=不做）")
    ap.add_argument("--eval-initial", action="store_true",
                    help="在第 0 轮评测并保存迁移起点，避免微调退化后丢失初始最佳权重")
    ap.add_argument("--resume", action="store_true",
                    help="从 <out>/<name>/weights/last.pt 断点续训（配合看门狗实现无人值守）")
    ap.add_argument("--init-checkpoint", default="",
                    help="从旧权重仅初始化新实验；不恢复动量/RNG，须使用新的 --name")
    ap.add_argument("--no-resume-split", dest="resume_split", action="store_false", default=True,
                    help="忽略已有 split.json，重新划分")
    ap.add_argument("--split-file", default="",
                    help="实验统一划分：读取指定 split.json，验证完整无重复后复制到本次 run")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="训练期验证用多少张（组感知 + 类别均衡子集；0=全量）")
    ap.add_argument("--val-conf", type=float, default=0.001,
                    help="训练期验证的置信度（0.01 快得多；正式 Test 报数用 0.001）")
    ap.add_argument("--branch-aux-weight", type=float, default=0.0,
                    help="training-only detector shared by RGB/IR/Depth common features")
    ap.add_argument("--flow-supervision-weight", type=float, default=0.0,
                    help="known canvas-shift supervision for Depth correspondence")
    ap.add_argument("--cross-modal-nce-weight", type=float, default=0.0,
                    help="GT-object cross-modal InfoNCE weight")
    ap.add_argument("--nce-temperature", type=float, default=0.10)
    ap.add_argument("--p2-match-refine", action="store_true",
                    help="run an independent local correspondence refinement at P2")
    ap.add_argument("--match-floor", type=float, default=0.0,
                    help="匹配置信度下限（0 = 旧行为）。IR/Depth 的 match 只有约 0.1，"
                         "却被当作共享通路的开关；抬高下限可把相似度与可用性解耦")
    ap.add_argument("--alignment-mode", default="legacy_gate_v1",
                    choices=["legacy_gate_v1", "identity_residual_v2"],
                    help="V4.2 用 match 在原坐标与残差 warp 间插值，而不是抑制整路证据")
    ap.add_argument("--depth-reliability", default="legacy_edge_v1",
                    choices=["legacy_edge_v1", "valid_support_v2"],
                    help="V4.2 仅按无效深度邻域降权，不惩罚真实深度边缘")
    ap.add_argument("--flow-identity-weight", type=float, default=0.0,
                    help="没有人工位移标签时的弱零 flow 先验权重")
    ap.add_argument("--branch-aux-weights", type=float, nargs=3, default=None,
                    metavar=("RGB", "IR", "DEP"),
                    help="按模态指定共享辅助检测支路权重；缺省时三路都用 --branch-aux-weight")
    ap.add_argument("--branch-aux-end-weights", type=float, nargs=3, default=None,
                    metavar=("RGB", "IR", "DEP"),
                    help="末轮辅助分支权重；给出后从起始权重线性退火")
    ap.add_argument("--flow-supervision-end-weight", type=float, default=None)
    ap.add_argument("--cross-modal-nce-end-weight", type=float, default=None)
    ap.add_argument("--embedding-recon-weight", type=float, default=.01,
                    help="common/private 信息重建起始权重；旧固定值为 .01")
    ap.add_argument("--embedding-recon-end-weight", type=float, default=None)
    ap.add_argument("--embedding-alignment-weight", type=float, default=.005,
                    help="common 余弦对齐起始权重；旧固定值为 .005")
    ap.add_argument("--embedding-alignment-end-weight", type=float, default=None)
    args = ap.parse_args()
    if args.branch_aux_weights is not None:
        args.branch_aux_weights = tuple(float(v) for v in args.branch_aux_weights)
    if args.branch_aux_end_weights is not None:
        args.branch_aux_end_weights = tuple(float(v) for v in args.branch_aux_end_weights)
    if args.reset_fusion_gates and (args.resume or not args.init_checkpoint):
        raise ValueError("--reset-fusion-gates requires --init-checkpoint in a new run")
    if args.memory_control != "unbounded_v1" and args.architecture not in ("spatial_memory_v1", "independent_p2_memory_v3"):
        raise ValueError("memory-control applies only to spatial memory")
    if args.bn_policy == "adaptive_no_tail" and args.sampler != "coverage":
        raise ValueError("no-tail BN requires coverage sampler")
    if args.modalities == "rgb" and args.rgb_batch > 0:
        args.batch = args.rgb_batch
    if not (0 <= args.close_aug_frac < 1 and 0 <= args.rare_extra_frac <= 1 and 0 <= args.lrf <= 1):
        raise ValueError("invalid schedule/sampling fractions")
    if not (0 < args.scale_min <= args.scale_max and 0 <= args.translate < 1):
        raise ValueError("invalid geometric augmentation range")
    if args.calibrate_clip_steps < 0 or args.val_batch < 1 or args.warmup < 0:
        raise ValueError("invalid calibration/validation/warmup settings")
    if (min(args.branch_aux_weight, args.flow_supervision_weight, args.cross_modal_nce_weight) < 0
            or args.match_floor < 0
            or (args.branch_aux_weights is not None and min(args.branch_aux_weights) < 0)
            or (args.branch_aux_end_weights is not None and min(args.branch_aux_end_weights) < 0)
            or (args.flow_supervision_end_weight is not None and args.flow_supervision_end_weight < 0)
            or (args.cross_modal_nce_end_weight is not None and args.cross_modal_nce_end_weight < 0)
            or min(args.embedding_recon_weight, args.embedding_alignment_weight) < 0
            or (args.embedding_recon_end_weight is not None and args.embedding_recon_end_weight < 0)
            or (args.embedding_alignment_end_weight is not None and args.embedding_alignment_end_weight < 0)
            or not 0 <= args.flow_identity_weight <= 1
            or args.nce_temperature <= 0):
        raise ValueError("semantic loss weights must be nonnegative and temperature positive")
    if args.calibrate_clip_steps and args.grad_clip <= 0:
        raise ValueError("clip calibration requires a positive initial safety threshold")
    if args.architecture in ("spatial_memory_v1", "independent_p2_memory_v3") and args.sampler != "coverage":
        raise ValueError("new recipe requires --sampler coverage")
    if args.train_stage in ("aux_adapt", "anchored_joint", "aux_independent",
                            "residual_fusion") and args.architecture != "independent_p2_memory_v3":
        raise ValueError("multimodal staged policies 仅支持 independent_p2_memory_v3")
    if args.alignment_mode == "identity_residual_v2" and args.match_floor != 0:
        raise ValueError("identity_residual_v2 不使用 match-floor；请保持 0")
    if args.batch < 1 or args.accum < 1 or args.epochs < 1 or args.save_every < 1:
        raise ValueError("batch/accum/epochs/save-every 必须为正整数")
    if args.nominal_batch < 1 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("nominal-batch 须 >0，weight-decay/grad-clip 须 >=0")
    if min(args.backbone_lr_mult, args.fusion_lr_mult, args.p2_lr_mult,
           args.detector_lr_mult, args.semantic_lr_mult) < 0:
        raise ValueError("所有学习率倍率必须 >=0")
    if not (0.0 <= args.rgb_dropout < 1.0 and 0.0 <= args.aux_dropout < 1.0):
        raise ValueError("dropout 概率必须在 [0,1) 内")
    for name in ("degrade_p", "rgb_color_p", "ir_noise_p", "ir_gain_p",
                 "depth_hole_p", "target_crop_p"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name} 必须在 [0,1] 内")
    if args.misalign_px < 0 or args.rare_sample_max < 1:
        raise ValueError("misalign-px 须 >=0，rare-sample-max 须 >=1")
    if args.late_bus and args.register_bus:
        raise ValueError("持久 register 与旧 --late-bus 不应同时开启；请选择一个做消融")
    if args.modalities == "dep" and args.depth_scales != "all":
        raise ValueError("Depth-only 必须用 --depth-scales all，让 P3 也接收 Depth")
    if args.depth_channels == 2 and (args.depth_view != "both" or args.depth_init != "relative"):
        raise ValueError("Depth 表示实验只支持 4ch；旧 2ch 权重保持原行为")
    if ((args.depth_view == "relative" and args.depth_init != "relative") or
            (args.depth_view in ("metric_fallback", "metric_log_fallback")
             and args.depth_init != "metric_fallback")):
        raise ValueError("depth-view 与 depth-init 不匹配")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # cudnn.benchmark=True 会按"当前时钟下的最快"选卷积算法：实测在省电低时钟（210MHz）时
    # 它选到慢 5 倍的算法（8.71s/iter vs 1.76s/iter）且 workspace 更大（显存 5.4G vs 2.1G）。
    # 画布固定、又要求无人值守稳定 → 一律关掉。
    torch.backends.cudnn.benchmark = False
    device_arg = str(args.device).strip().lower()
    if device_arg == "auto":
        device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_arg == "cuda":
        device_arg = "cuda:0"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求 --device {args.device}，但当前 PyTorch 看不到 CUDA")
    dev = torch.device(device_arg)
    imgsz = parse_imgsz(args.imgsz)
    canvas = canvas_of(imgsz)
    args.canvas = canvas
    enabled = {"rgb": ["rgb"], "ir": ["ir"], "dep": ["dep"],
               "rgb_ir": ["rgb", "ir"], "rgb_dep": ["rgb", "dep"],
               "all": ["rgb", "ir", "dep"]}[args.modalities]
    out_dir = Path(args.out) / args.name
    ck_path = out_dir / "weights" / "last.pt"
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume 与 --init-checkpoint 不能同时使用")
    if args.resume and not ck_path.exists():
        raise FileNotFoundError(f"--resume 指定的 checkpoint 不存在：{ck_path}")
    if ck_path.exists() and not args.resume:
        raise FileExistsError(f"{ck_path} 已存在；请换 --name，或明确使用 --resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()                                    # 不 flush 会卡在缓冲区，看不出进度

    log(f"[train] 设备 {dev} | 配置 {vars(args)}")
    log(f"[train] 防休眠: {prevent_sleep(True)}（SetThreadExecutionState）")

    # ---- 数据 ----
    idx = build_index(Path(args.root), Path(args.labels), limit=args.limit)
    split_path = out_dir / "split.json"
    if args.resume and not split_path.exists():
        raise FileNotFoundError(f"精确续训缺少 split.json：{split_path}")
    if args.full_data:
        if args.val_every or args.eval_initial or args.split_file:
            raise ValueError("full-data refit must disable validation and split-file")
        tr, va = list(idx), []
        if args.resume:
            old_tr, old_va = load_split(split_path,idx)
            if [s["stem"] for s in old_tr] != [s["stem"] for s in tr] or old_va:
                raise ValueError("full-data resume split differs")
        log(f"[train] FINAL REFIT: {len(tr)} labeled images; NO held-out validation")
    elif args.split_file:
        reference = Path(args.split_file)
        if not reference.is_file():
            raise FileNotFoundError(f"--split-file 不存在：{reference}")
        tr, va = load_split(reference, idx)
        all_stems = [s["stem"] for s in tr + va]
        if (len(all_stems) != len(idx) or len(set(all_stems)) != len(idx)
                or set(all_stems) != {s["stem"] for s in idx} or not va):
            raise ValueError("--split-file 必须完整覆盖当前索引且 train/val 互不重复")
        if split_path.exists():
            saved_tr, saved_va = load_split(split_path, idx)
            if ([s["stem"] for s in saved_tr] != [s["stem"] for s in tr] or
                    [s["stem"] for s in saved_va] != [s["stem"] for s in va]):
                raise ValueError("--split-file 与当前 run 的 split.json 不一致，拒绝覆盖已有划分")
        log(f"[train] 固定划分 {reference}：train={len(tr)} val={len(va)}")
    elif args.resume_split and split_path.exists():
        tr, va = load_split(split_path, idx)
        if len(tr) + len(va) == len(idx) and va:
            log(f"[train] 复用已有划分 {split_path.name}：train={len(tr)} val={len(va)}")
        else:
            if args.resume:
                raise ValueError("split.json 与当前索引不匹配，拒绝续训")
            tr, va = group_split(idx, val_ratio=args.val_ratio, seed=args.seed)
    else:
        if args.resume:
            raise ValueError("精确续训不能使用 --no-resume-split")
        tr, va = group_split(idx, val_ratio=args.val_ratio, seed=args.seed)
    split_digest = hashlib.sha256(json.dumps({"train": [s["stem"] for s in tr],
                                               "val": [s["stem"] for s in va]},
                                              ensure_ascii=False).encode("utf-8")).hexdigest()
    cls_va = val_class_stats(va, nc=12)
    if not args.resume:
        save_split(split_path, tr, va, extra={"seed": args.seed, "val_ratio": args.val_ratio,
                                              "val_class_counts": {str(k): v for k, v in cls_va.items()}})
    aug = AugCfg(imgsz=imgsz, scale_range=(args.scale_min, args.scale_max),
                 translate=args.translate, rgb_drop_p=args.rgb_dropout,
                 aux_drop_p=args.aux_dropout, misalign_px=args.misalign_px,
                 degrade_p=args.degrade_p, rgb_color_p=args.rgb_color_p,
                 ir_noise_p=args.ir_noise_p, ir_gain_p=args.ir_gain_p,
                 depth_hole_p=args.depth_hole_p, target_crop_p=args.target_crop_p,
                 legacy_lowlight=args.depth_channels == 2,
                 depth_resampling=args.depth_resampling, total_epochs=args.epochs,
                 close_aug_frac=args.close_aug_frac,
                 mosaic_p=args.mosaic,
                 dropout_start_epoch=max(0, args.dropout_start_epoch))
    if args.modalities == "rgb" or args.no_dropout:
        aug.rgb_drop_p = 0.0
        aug.aux_drop_p = 0.0
    # epoch 通道文件：worker 进程（含 persistent_workers 复用的）靠它拿到"当前第几轮"，
    # 否则 set_epoch 到不了 worker → 随机增强每轮完全相同（P0-1c）。
    ep_file = out_dir / "epoch.txt"
    ep_file.write_text("0", encoding="utf-8")
    ds_tr = MMDataset(Path(args.root), tr, imgsz=imgsz, train=True, aug=aug, seed=args.seed,
                      enabled=enabled, epoch_file=ep_file)
    ds_va = MMDataset(Path(args.root), va, imgsz=imgsz, train=False, aug=aug, seed=args.seed,
                      enabled=enabled, epoch_file=ep_file)
    va_eval = pick_val_subset(va, args.val_limit, seed=args.seed)
    cls_sub = val_class_stats(va_eval, nc=12)
    log(f"[train] 样本 train={len(ds_tr)} val={len(ds_va)}（验证子集 {len(va_eval)}）"
        f" | batch={args.batch}×accum{args.accum} 画布={canvas} 模态={enabled}")
    log(f"[train] 全量 val 逐类框数 {cls_va}")
    log(f"[train] 验证子集逐类框数 {cls_sub}")
    miss_cls = [c for c, v in cls_sub.items() if v == 0]
    if miss_cls and not args.full_data:
        log(f"[train] [!] 验证子集缺少类别 {miss_cls} → 这些类的 AP 记为 nan，"
            f"若要完整 12 类报数请用 --val-limit 0（全量验证）")
    g = torch.Generator()
    g.manual_seed(args.seed)

    sampler = None
    if args.sampler == "coverage":
        sampler = CoverageRareSampler(tr, seed=args.seed, extra_frac=args.rare_extra_frac)
        log(f"[train] 完整覆盖采样：原图 {len(tr)} + 稀有类追加 {sampler.extra}，每图最多追加 {sampler.max_extra} 次")
        log(f"[train] 训练逐类框数 {val_class_stats(tr, nc=12)}")
    elif args.rare_sample_max > 1.0:
        sw, image_counts, class_factors = balanced_sample_weights(
            tr, nc=12, max_weight=args.rare_sample_max)
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                        num_samples=len(sw), replacement=True, generator=g)
        log(f"[train] 稀有类均衡采样 max={args.rare_sample_max:g} "
            f"sample_weight={sw.min():.2f}/{sw.mean():.2f}/{sw.max():.2f}(min/mean/max) "
            f"类图数={image_counts} 类权重="
            f"{ {k: round(v, 2) for k, v in class_factors.items()} }")

    def _make_loader(workers: int) -> DataLoader:
        return DataLoader(ds_tr, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                          num_workers=workers,
                          collate_fn=collate, drop_last=False, generator=g,
                          persistent_workers=workers > 0, pin_memory=dev.type == "cuda")

    dl_tr = _make_loader(args.workers)

    # ---- 模型 ----
    cfg = default_config()
    cfg.weights = args.weights
    cfg.fusion.architecture = args.architecture
    cfg.fusion.memory_control = args.memory_control
    cfg.fusion.branch_aux_weight = float(args.branch_aux_weight)
    cfg.fusion.branch_aux_weights = (tuple(args.branch_aux_weights)
                                     if args.branch_aux_weights else ())
    cfg.fusion.match_floor = float(args.match_floor)
    cfg.fusion.alignment_mode = args.alignment_mode
    cfg.fusion.depth_reliability = args.depth_reliability
    cfg.fusion.flow_identity_weight = float(args.flow_identity_weight)
    # The model only needs to know whether a training-only branch exists.  Use
    # the maximum scheduled weight so an end-heavy schedule cannot omit it.
    flow_end = (args.flow_supervision_weight if args.flow_supervision_end_weight is None
                else args.flow_supervision_end_weight)
    nce_end = (args.cross_modal_nce_weight if args.cross_modal_nce_end_weight is None
               else args.cross_modal_nce_end_weight)
    recon_end = (args.embedding_recon_weight if args.embedding_recon_end_weight is None
                 else args.embedding_recon_end_weight)
    embedding_alignment_end = (
        args.embedding_alignment_weight if args.embedding_alignment_end_weight is None
        else args.embedding_alignment_end_weight)
    cfg.fusion.flow_supervision_weight = float(max(args.flow_supervision_weight, flow_end))
    cfg.fusion.cross_modal_nce_weight = float(max(args.cross_modal_nce_weight, nce_end))
    cfg.fusion.nce_temperature = float(args.nce_temperature)
    cfg.fusion.p2_match_refine = bool(args.p2_match_refine)
    cfg.depth_resampling = args.depth_resampling
    cfg.encoder.metric_branch = args.metric_branch
    cfg.encoder.checkpoint_encoder = args.checkpoint_encoder
    if args.architecture == "independent_p2_memory_v3":
        cfg.fusion.bus_dim = 128
        cfg.fusion.memory_tokens_per_modality = 4
    cfg.imgsz = int(canvas[0])   # 记录用；真实画布由数据侧 (H,W) 决定
    cfg.fusion.tier = args.fusion_tier
    cfg.encoder.share_tier = args.share_tier
    cfg.encoder.depth_input_channels = int(args.depth_channels)
    cfg.encoder.depth_view = args.depth_view
    cfg.encoder.depth_init = args.depth_init
    cfg.fusion.use_bus = bool(args.register_bus and (args.modalities != "rgb" or args.architecture == "spatial_memory_v1"))
    cfg.fusion.late_bus = bool(args.late_bus)
    cfg.fusion.quality_gate = not args.no_quality
    cfg.fusion.prior_film = not args.no_prior
    cfg.fusion.deformable = () if args.no_deformable else ("dep",)
    cfg.fusion.depth_scales = {"p4p5": (False, True, True), "all": (True, True, True),
                               "p3p4": (True, True, False), "p4": (False, True, False)}[args.depth_scales]
    model = MMYOLO(cfg).to(dev)
    enable_trainable_defaults(model)
    model.modality_off = {"rgb", "ir", "dep"} - set(enabled)
    model.infer_modalities = tuple(enabled)
    model.infer_canvas = tuple(canvas)
    pr = model.param_report()
    log(f"[train] parameter_groups={pr}")
    log(f"[train] 参数 总 {pr['total']/1e6:.2f}M（预训练 {pr['pretrained']/1e6:.2f}M + 新增 {pr['new']/1e6:.3f}M）"
        f" | 模态 {args.modalities}")

    crit = v8DetectionLoss(model)
    effective_batch = args.batch * args.accum
    scaled_wd = args.weight_decay * effective_batch / args.nominal_batch
    role_mults = None
    if args.train_stage in ("anchored_joint", "residual_fusion"):
        role_mults = {"anchor": 0.0,
                      "aux_encoder": args.backbone_lr_mult,
                      "fusion": args.fusion_lr_mult,
                      "p2": args.p2_lr_mult,
                      "detector": args.detector_lr_mult,
                      "semantic": args.semantic_lr_mult}
    opt = build_optimizer(model, args.lr, args.backbone_lr_mult, wd=scaled_wd,
                          role_mults=role_mults)
    log(f"[train] 优化器 AdamW：有效 batch={effective_batch}，"
        f"weight_decay={args.weight_decay:g}×{effective_batch}/{args.nominal_batch}"
        f"={scaled_wd:.3g}，grad_clip={args.grad_clip:g}")
    if role_mults is not None:
        group_counts = collections.defaultdict(int)
        for group in opt.param_groups:
            group_counts[group.get("role", "unknown")] += sum(p.numel() for p in group["params"])
        log(f"[train] anchored lr multipliers={role_mults} params="
            f"{ {k: round(v/1e6,3) for k,v in group_counts.items()} }")

    # ---- 断点续训（无人值守必需：半夜崩了不能从头再来）----
    start_ep, best0 = 0, -1.0
    if args.resume or args.init_checkpoint:
        source = ck_path if args.resume else Path(args.init_checkpoint)
        ck = torch.load(str(source), map_location="cpu", weights_only=False)
        validate_checkpoint(ck, model, enabled, canvas, exact=args.resume, args=args,
                            split_digest=split_digest)
        weights = ck["train_state"]["raw_model_state"] if args.resume else ck["model_state"]
        migrated = False
        if not args.resume:
            weights, migrated = adapt_depth_checkpoint_state(weights, model)
        model.load_state_dict(weights, strict=True)
        if args.reset_fusion_gates:
            reset_keys = reset_fusion_gate_outputs(model)
            log(f"[train] v2 迁移：仅重置 {len(reset_keys)} 个门控末层参数张量；其余权重严格迁移")
        if args.train_stage == "residual_fusion":
            reset_keys = reset_rgb_identity_residuals(model)
            log(f"[train] Stage B RGB 恒等起点：重置 {reset_keys}")
        if args.resume:
            start_ep = int(ck["epoch"])
            if start_ep > args.epochs:
                raise ValueError(f"checkpoint epoch={start_ep} 超过请求的 epochs={args.epochs}")
            best0 = float(ck.get("best_map", -1.0))
            log(f"[train] 严格续训：{source} epoch={start_ep} best={best0:.4f}")
        else:
            log(f"[train] 仅初始化模型：{source}；optimizer/EMA/RNG 从新实验开始"
                + (" | 已显式初始化 V4.3 新增训练模块/输入适配" if migrated else ""))

    ema = ModelEMA(model)
    ema.enabled = True
    if args.precision == "bf16" and dev.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested bf16 is unsupported on this GPU")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and dev.type == "cuda" and args.precision == "fp16",
                                 init_scale=1024.0 if model.spatial_memory else 65536.0)
    # 精确续训必须恢复与 optimizer 对应的原始模型权重，不能用 EMA 代替。
    if args.resume:
        ts = ck["train_state"]
        opt.load_state_dict(ts["optimizer"])
        scaler.load_state_dict(ts["scaler"])
        ema.ema.load_state_dict(ts["ema_state"], strict=True)
        ema.updates = int(ts["ema_updates"])
        torch.set_rng_state(ts["rng"]["torch"])
        if dev.type == "cuda" and ts["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(ts["rng"]["cuda"])
        np.random.set_state(ts["rng"]["numpy"])
        g.set_state(ts["rng"]["loader"])
        log(f"[train] optimizer/scaler/EMA/RNG 已严格恢复；ema_updates={ema.updates}")

    meta_base = {"modalities": list(enabled), "canvas": list(canvas), "imgsz": list(canvas),
                 "class_names": list(model.class_names), "nc": int(model.nc),
                 "fusion_tier": args.fusion_tier, "share_tier": args.share_tier,
                 "register_bus": bool(cfg.fusion.use_bus), "late_bus": bool(args.late_bus),
                 "depth_scales": list(cfg.fusion.depth_scales), "name": args.name,
                 "depth_channels": int(args.depth_channels), "recipe": "b2" if args.depth_channels == 4 else "b1",
                 "depth_view": args.depth_view, "depth_init": args.depth_init,
                 "split_digest": split_digest, "trainer_recipe": recipe_for(args),
                 "architecture": args.architecture, "depth_resampling": args.depth_resampling}
    if args.init_checkpoint:
        meta_base["initialization"] = {"checkpoint": str(Path(args.init_checkpoint).resolve()),
                                       "epoch": ck["epoch"], "best_map": ck.get("best_map"),
                                       "reset_fusion_gates": args.reset_fusion_gates}
    elif args.resume and ck.get("meta", {}).get("initialization"):
        meta_base["initialization"] = ck["meta"]["initialization"]
    deployment = _CODE / "v3_deployment.json"
    if args.architecture == "independent_p2_memory_v3" and deployment.is_file():
        source_manifest = json.loads(deployment.read_text(encoding="utf-8"))["files"]
        meta_base["source_manifest_sha256"] = hashlib.sha256(json.dumps(source_manifest,sort_keys=True).encode()).hexdigest()

    clip_state = {"threshold": args.grad_clip, "norms": [], "calibrated": args.calibrate_clip_steps == 0}
    if args.resume:
        clip_state = ck["train_state"].get("clip_state", clip_state)

    def _save(path: Path, ep: int, best_v: float) -> None:
        ensure_finite_state(model, "raw model")
        ensure_finite_state(ema.ema, "EMA model")
        save_mm_checkpoint(path, ema.ema, epoch=ep, best_map=best_v, meta=meta_base,
                           train_state={"optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                                        "raw_model_state": model.state_dict(),
                                        "ema_state": ema.ema.state_dict(),
                                        "ema_updates": ema.updates,
                                        "rng": {"torch": torch.get_rng_state(),
                                                "cuda": (torch.cuda.get_rng_state_all()
                                                         if dev.type == "cuda" else None),
                                                "numpy": np.random.get_state(),
                                                "loader": g.get_state()},
                                        "args": vars(args), "clip_state": clip_state})

    def _evaluate_for_stage(eval_model):
        """Use standalone IR/Depth AP to select Stage A; fused AP elsewhere."""
        from eval import evaluate_model
        if args.train_stage != "aux_independent":
            return evaluate_model(eval_model, Path(args.root), va_eval,
                                  imgsz=imgsz, device=dev, modalities=args.modalities,
                                  conf=args.val_conf, slices=False,
                                  batch_size=args.val_batch)
        branches = {}
        try:
            for branch in ("ir", "dep"):
                eval_model.auxiliary_eval_branch = branch
                branches[branch] = evaluate_model(
                    eval_model, Path(args.root), va_eval, imgsz=imgsz,
                    device=dev, modalities="all", conf=args.val_conf,
                    slices=False, batch_size=args.val_batch)
        finally:
            eval_model.auxiliary_eval_branch = None
        keys = range(int(eval_model.nc))
        combined = {
            "map50_95": sum(v["map50_95"] for v in branches.values()) / 2,
            "map50": sum(v["map50"] for v in branches.values()) / 2,
            "per_class_95": {k: sum(v["per_class_95"][k] for v in branches.values()) / 2
                             for k in keys},
            "per_class_50": {k: sum(v["per_class_50"][k] for v in branches.values()) / 2
                             for k in keys},
            "n_valid_classes": min(v["n_valid_classes"] for v in branches.values()),
            "missing_classes": sorted(set().union(
                *(v["missing_classes"] for v in branches.values()))),
            "n_images": len(va_eval), "canvas": list(canvas),
            "modalities": "independent_ir_depth",
            "selection": "mean_ir_depth_map50_95",
            "branches": branches,
        }
        return combined

    best = best0
    if args.eval_initial and not args.resume:
        log("[train] 开始 ep0 迁移起点验证（未做任何梯度更新）")
        initial = _evaluate_for_stage(ema.ema)
        best = float(initial["map50_95"])
        _save(out_dir / "weights" / "best.pt", 0, best)
        (out_dir / "val_best.json").write_text(json.dumps(
            initial, ensure_ascii=False, indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)),
            encoding="utf-8")
        log(f"[train] ep 0/{args.epochs} initial val mAP50-95={best:.4f} "
            f"mAP50={initial['map50']:.4f}（已保存 best.pt）")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if args.resume and start_ep == args.epochs:
        log(f"[train] checkpoint 已是 ep {start_ep}/{args.epochs}，没有新轮次；保持权重不变")
        return
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        t_ep = time.time()
        frozen = ep < args.freeze_epochs
        if args.train_stage == "aux_independent":
            set_independent_aux_mode(model)
            frozen = True
        elif args.train_stage == "residual_fusion":
            set_residual_fusion_mode(model, downstream_frozen=frozen)
        elif args.train_stage == "aux_adapt":
            set_aux_adaptation_mode(model)
            frozen = True
        elif args.train_stage == "anchored_joint":
            set_anchored_joint_mode(model, detector_frozen=frozen)
        else:
            set_encoder_frozen(model, frozen)
        lr = lr_at(ep, args.epochs, args.lr, warmup=args.warmup, lrf=args.lrf)
        for grp in opt.param_groups:
            grp["lr"] = lr * float(grp.get("lr_mult", 1.0))
        ds_tr.set_epoch(ep)                     # 随机增强每轮不同（可复现）
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(ep)
        model.train()
        if args.bn_policy in ("adaptive", "adaptive_no_tail"):
            apply_bn_policy(model, args.bn_policy, frozen=frozen)
        elif args.freeze_bn:
            nb = set_bn_eval(model, include_new=args.freeze_new_bn)
            if ep == 0:
                log(f"[train] freeze-bn：冻结 {nb} 个 BN；新建 BN "
                    f"{'也冻结' if args.freeze_new_bn else '仍训练'}")
        if args.train_stage in ("aux_adapt", "anchored_joint", "aux_independent",
                                "residual_fusion"):
            set_frozen_bn_eval(model)
        if ep == start_ep:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            log(f"[train] stage={args.train_stage} trainable={trainable/1e6:.3f}M/{total/1e6:.3f}M")

        progress = ep / max(1, args.epochs - 1)
        aux_start = (tuple(args.branch_aux_weights) if args.branch_aux_weights
                     else (float(args.branch_aux_weight),) * 3)
        aux_end = (tuple(args.branch_aux_end_weights) if args.branch_aux_end_weights
                   else aux_start)
        aux_values = tuple(a + (b - a) * progress for a, b in zip(aux_start, aux_end))
        aux_w = dict(zip(("rgb", "ir", "dep"), aux_values))
        flow_weight = (args.flow_supervision_weight +
                       (flow_end - args.flow_supervision_weight) * progress)
        nce_weight = (args.cross_modal_nce_weight +
                      (nce_end - args.cross_modal_nce_weight) * progress)
        recon_weight = (args.embedding_recon_weight +
                        (recon_end - args.embedding_recon_weight) * progress)
        embedding_alignment_weight = (
            args.embedding_alignment_weight +
            (embedding_alignment_end - args.embedding_alignment_weight) * progress)
        aux_w_log = "/".join(f"{v:.3g}" for v in aux_values)
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "n": 0, "micro": 0, "group_samples": 0, "skipped": 0,
               "branch_aux": 0.0, "flow_aux": 0.0, "nce_aux": 0.0,
               "embedding_recon": 0.0, "embedding_alignment": 0.0,
               "grad_steps": 0, "grad_clipped": 0, "grad_norm_sum": 0.0,
               "grad_norm_max": 0.0, "amp_overflow": 0, "norms": [], "stems": set(), "draws": 0}
        health_sum, health_n = {}, 0
        schedule = scheduled_aug(aug, ep)
        log(f"[train] 开始 ep {ep+1}/{args.epochs} samples={len(sampler) if sampler is not None else len(ds_tr)} "
            f"bn={args.bn_policy} clip_limit={clip_state['threshold']:.2f} "
            f"scale={schedule.scale_range} translate={schedule.translate:.3f} crop={schedule.target_crop_p:.3f} mosaic={schedule.mosaic_p:.3f}")
        accum = max(1, args.accum)
        nominal_samples = args.batch * accum

        def _step(actual_samples: int):
            if actual_samples < 1:
                raise RuntimeError("optimizer step 没有有效样本")
            # 每个 micro-batch 都按 nominal_samples 反传；不足一组时恢复成该组
            # 实际样本的均值。也覆盖 collate 过滤坏样本导致的中途小 batch。
            if actual_samples != nominal_samples:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(nominal_samples / actual_samples)
            scaler.unscale_(opt)
            threshold = clip_state["threshold"]
            parameters = [p for p in model.parameters() if p.grad is not None]
            grad_norm_t = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters]))
            grad_norm = float(grad_norm_t.detach())
            if math.isfinite(grad_norm) and threshold > 0 and grad_norm > threshold:
                # Never multiply non-finite gradients by an inf/inf clipping coefficient.
                torch.nn.utils.clip_grad_norm_(parameters, threshold, error_if_nonfinite=True)
            agg["grad_steps"] += 1
            if math.isfinite(grad_norm):
                agg["grad_norm_sum"] += grad_norm
                agg["grad_norm_max"] = max(agg["grad_norm_max"], grad_norm)
                agg["norms"].append(grad_norm)
                if threshold > 0 and grad_norm > threshold:
                    agg["grad_clipped"] += 1
                if not clip_state["calibrated"] and ep >= max(args.warmup, args.freeze_epochs):
                    clip_state["norms"].append(grad_norm)
                    if len(clip_state["norms"]) >= args.calibrate_clip_steps:
                        p90 = float(np.percentile(clip_state["norms"], 90))
                        clip_state["threshold"] = float(np.clip(1.5*p90, 20, 1000 if args.architecture == "independent_p2_memory_v3" else 200))
                        clip_state["calibrated"] = True
                        log(f"[train] 梯度校准 N={len(clip_state['norms'])} P90={p90:.2f} "
                            f"threshold={clip_state['threshold']:.2f}（仅限尖峰保护，不代表最优分数）")
            elif not scaler.is_enabled():
                raise FloatingPointError("non-finite gradient without AMP; refusing optimizer step")
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            overflow = scaler.get_scale() < scale_before
            if overflow:
                agg["amp_overflow"] += 1
            opt.zero_grad(set_to_none=True)
            # GradScaler 检测到 Inf/NaN 时不会执行 optimizer.step；此时 EMA 也不应
            # 假装发生过一次参数更新。首轮自动回退 scale 属正常 AMP 校准，单独记数。
            if not overflow:
                ema.update(model)

        # 后台预取 + worker 起不来时自动退回单进程
        # ⚠️ `iter(loader)` 本身就会创建 worker 进程与队列并抛异常，所以预取线程内部才建迭代器。
        stop_ev = threading.Event()
        prefetch = args.prefetch
        try:
            source = iter_prefetch(dl_tr, depth=3, stop=stop_ev) if prefetch else iter(dl_tr)
            first = next(source)
        except (PermissionError, OSError, RuntimeError) as exc:
            if args.workers <= 0:
                raise
            log(f"[train] [!] DataLoader worker 启动失败（{type(exc).__name__}: {exc}）"
                f"→ 本轮及后续退回 workers=0（单进程 + 后台预取，实测约 1.4× 于纯串行）")
            args.workers = 0
            dl_tr = _make_loader(0)
            stop_ev = threading.Event()
            source = iter_prefetch(dl_tr, depth=3, stop=stop_ev) if prefetch else iter(dl_tr)
            first = next(source)
        for bi, batch in enumerate(itertools.chain((first,), source)):
            if batch is None:
                agg["skipped"] += 1
                continue
            if args.bn_policy == "adaptive_no_tail":
                # Also exclude the one mixed boundary batch (if N % batch != 0).
                set_bn_tail_mode(model, extra=agg["draws"] + len(batch["stems"]) > len(ds_tr))
            agg["stems"].update(batch["stems"])
            agg["draws"] += len(batch["stems"])
            non_blocking = dev.type == "cuda"
            rgb = batch["rgb"].to(dev, non_blocking=non_blocking)
            ir = (None if "ir" not in enabled
                  else batch["ir"].to(dev, non_blocking=non_blocking))
            dep = (None if "dep" not in enabled
                   else batch["depth"].to(dev, non_blocking=non_blocking))
            qual = {k: v.to(dev, non_blocking=non_blocking)
                    for k, v in batch["quality"].items()}
            keep = {k: v.to(dev, non_blocking=non_blocking)
                    for k, v in batch["keep"].items()}
            prior = (None if (args.no_prior or "dep" not in enabled)
                     else batch["prior"].to(dev, non_blocking=non_blocking))
            tgt = make_targets(batch, canvas, dev)
            with torch.autocast("cuda", enabled=bool(args.amp) and dev.type == "cuda", dtype=amp_dtype):
                zero = rgb.new_zeros((), dtype=torch.float32)
                embedding_aux = {"reconstruction": zero, "alignment": zero}
                semantic = {"flow": zero, "nce": zero}
                branch_total, branch_count = zero, 0
                if args.train_stage == "aux_independent":
                    # Alternate two full standalone detectors.  No RGB feature,
                    # fusion tensor or main detector participates in this loss.
                    names = ("ir", "dep")
                    name = names[(ep + bi) % len(names)]
                    branch_aux, active = model.independent_branch_prediction(
                        name, ir=ir, depth=dep, keep=keep)
                    if not active.any():
                        raise RuntimeError(f"Stage A batch has no valid {name} samples")
                    branch_preds, branch_targets = subset_detection_batch(
                        branch_aux, tgt, active)
                    loss_vec, loss_items = crit(branch_preds, branch_targets)
                    branch_total, branch_count = loss_vec.sum(), 1
                    loss = aux_w[name] * accumulation_loss(loss_vec, nominal_samples)
                else:
                    preds = model(rgb, ir, dep, quality=qual, prior=prior, keep=keep)
                    # 本版 ultralytics 的 v8DetectionLoss 返回 (loss*bs 的三分量向量, 分量字典)
                    loss_vec, loss_items = crit(preds, tgt)
                    loss = accumulation_loss(loss_vec, nominal_samples)
                if (args.architecture == "independent_p2_memory_v3" and
                        args.train_stage != "aux_independent"):
                    embedding_aux = model.embedding_aux_losses
                    loss = loss + (recon_weight * embedding_aux["reconstruction"] +
                                   embedding_alignment_weight * embedding_aux["alignment"]) * (
                                       rgb.shape[0] / nominal_samples)
                    semantic = model.semantic_regularization(
                        tgt,
                        batch["alignment_shift"].to(dev, non_blocking=non_blocking),
                        batch["alignment_supervised"].to(dev, non_blocking=non_blocking),
                    )
                    loss = loss + (flow_weight * semantic["flow"] +
                                   nce_weight * semantic["nce"]) * (
                                       rgb.shape[0] / nominal_samples)
                    # 显存修复：每步只跑一个辅助检测支路（原来的三支路同时在图上，
                    # 在 736x1280 / batch=4 时峰值 22.4G 并在第 2 轮 OOM）。按
                    # (epoch+batch) 轮换 rgb->ir->dep，单支路权重不再除以 3，因此
                    # 每步平均辅助梯度量级与原设计一致，只是三条支路轮流受监督。
                    branch_names = ("rgb", "ir", "dep")
                    for off in range(len(branch_names)):
                        name = branch_names[(ep + bi + off) % len(branch_names)]
                        if aux_w[name] <= 0:
                            continue
                        mi = branch_names.index(name)
                        active = model.semantic_branch_present[:, mi]
                        if not active.any():
                            continue
                        branch_aux = model.semantic_branch_prediction(name)
                        if branch_aux is None:
                            continue
                        branch_preds, branch_targets = subset_detection_batch(branch_aux, tgt, active)
                        branch_vec, _ = crit(branch_preds, branch_targets)
                        branch_total = branch_vec.sum()
                        branch_count = 1
                        break
                    if branch_count:
                        loss = loss + aux_w[name] * branch_total / nominal_samples
                    else:
                        branch_total = loss.new_zeros(())
            if not torch.isfinite(loss.detach()):
                stems = batch.get("stems", [])
                log(f"[train][FATAL] ep={ep+1} batch={bi} loss 非有限，"
                    f"样本={stems}；立即停止，不保存污染 checkpoint")
                raise FloatingPointError(f"non-finite loss at ep={ep+1} batch={bi}")
            scaler.scale(loss).backward()
            if (model.spatial_memory and args.memory_control == "bounded_v2" and
                    args.train_stage != "aux_independent"):
                health_n += 1
                for scale, block in model.fusion.items():
                    measurements = dict(block.last_health)
                    for m, pair in block.last_stats.items():
                        measurements[f"{m}_match"] = pair[0]
                        measurements[f"{m}_gate"] = pair[1]
                    for key, value in measurements.items():
                        k = f"{scale}.{key}"
                        health_sum[k] = health_sum.get(k, 0) + value.detach().float()
            # 日志使用 criterion 自身的标度；不要记录为了梯度累积额外缩小后的
            # backward loss，否则只改 accum 也会让曲线失去可比性。
            agg["loss"] += sum(float(v) for v in loss_items.values())
            if args.architecture == "independent_p2_memory_v3":
                agg["branch_aux"] += float(branch_total.detach()) / max(1, rgb.shape[0] * max(1, branch_count))
                agg["flow_aux"] += float(semantic["flow"].detach())
                agg["nce_aux"] += float(semantic["nce"].detach())
                agg["embedding_recon"] += float(embedding_aux["reconstruction"].detach())
                agg["embedding_alignment"] += float(embedding_aux["alignment"].detach())
            for k in ("box_loss", "cls_loss", "dfl_loss"):
                agg[k] = agg.get(k, 0.0) + float(loss_items[k])
            agg["n"] += 1
            agg["micro"] += 1
            agg["group_samples"] += int(rgb.shape[0])
            if agg["micro"] == accum:                           # 累积够了才更新
                _step(agg["group_samples"])
                agg["micro"] = 0
                agg["group_samples"] = 0
            if (bi + 1) % 100 == 0:
                log(f"[train] ep {ep+1} batch {bi+1}/{len(dl_tr)} "
                    f"loss={agg['loss']/max(1,agg['n']):.3f} unique={len(agg['stems'])} "
                    f"amp_skip={agg['amp_overflow']} elapsed={(time.time()-t_ep)/60:.1f}min")
        # ⚠️ 审计修复：每轮末尾**强制冲刷**剩余的累积梯度（旧版不足 accum 的尾部整段丢弃，
        #    实测每轮约 14 张样本（≈1%）从未参与过参数更新）。
        if agg["micro"] > 0:
            _step(agg["group_samples"])
        if args.sampler == "coverage" and (len(agg["stems"]) != len(ds_tr) or agg["draws"] != len(sampler)):
            raise RuntimeError(f"Incomplete epoch coverage: {len(agg['stems'])}/{len(ds_tr)}, draws={agg['draws']}/{len(sampler)}")
        # 验证/保存之前检查 BN buffer：参数有限并不代表 running stats 没被污染。
        ensure_finite_state(model, f"raw model after ep {ep+1}")
        ensure_finite_state(ema.ema, f"EMA model after ep {ep+1}")
        n = max(1, agg["n"])
        mem = ""
        if dev.type == "cuda":
            mem = (f" 显存 {torch.cuda.max_memory_allocated()/1024**3:.2f}G/"
                   f"{torch.cuda.max_memory_reserved()/1024**3:.2f}G")
        finite_steps = agg["grad_steps"] - agg["amp_overflow"]
        msg = (f"[train] ep {ep+1}/{args.epochs} lr={lr:.2e} "
               f"encoder_lr={lr*args.backbone_lr_mult:.2e} frozen={int(frozen)} "
               f"loss={agg['loss']/n:.4f} box={agg.get('box_loss',0)/n:.3f} "
               f"cls={agg.get('cls_loss',0)/n:.3f} dfl={agg.get('dfl_loss',0)/n:.3f}"
               f" grad={agg['grad_norm_sum']/max(1,finite_steps):.2f}/"
               f"{agg['grad_norm_max']:.2f} clip={agg['grad_clipped']}/{finite_steps}"
               f" amp_skip={agg['amp_overflow']}"
               f" coverage={len(agg['stems'])}/{len(ds_tr)} draws={agg['draws']}"
               f" grad_p90={float(np.percentile(agg['norms'],90)) if agg['norms'] else 0:.2f}"
               f" clip_limit={clip_state['threshold']:.2f}"
               f"{mem} 用时 {(time.time()-t_ep)/60:.1f}min/轮 累计 {(time.time()-t0)/60:.1f}min")
        if args.architecture == "independent_p2_memory_v3" and (
                args.branch_aux_weight or args.branch_aux_weights
                or args.branch_aux_end_weights or args.flow_supervision_weight
                or args.cross_modal_nce_weight or args.flow_supervision_end_weight
                or args.cross_modal_nce_end_weight):
            msg += (f" semantic_raw=branch:{agg['branch_aux']/n:.3f} "
                    f"flow:{agg['flow_aux']/n:.4f} nce:{agg['nce_aux']/n:.3f}"
                    f" align:{args.alignment_mode} aux_w:{aux_w_log}"
                    f" flow_w:{flow_weight:.3g} nce_w:{nce_weight:.3g}"
                    f" embed_raw:{agg['embedding_recon']/n:.3f}/{agg['embedding_alignment']/n:.3f}"
                    f" embed_w:{recon_weight:.3g}/{embedding_alignment_weight:.3g}")
        if model.spatial_memory:
            if health_n:
                log(f"[train] fusion epoch-mean={ {k: round(float(v/health_n),4) for k,v in health_sum.items()} }")
            fusion_stats = {s: {m: [round(float(v), 4) for v in pair.cpu()]
                               for m, pair in block.last_stats.items()}
                            for s, block in model.fusion.items()}
            log(f"[train] fusion last-batch [match_conf, injection_gate]={fusion_stats}")
        if args.val_every and ((ep + 1) % args.val_every == 0 or ep == args.epochs - 1):
            try:
                res = _evaluate_for_stage(ema.ema)
                (out_dir / "val_latest.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                    default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
                msg += f" | val mAP50-95={res['map50_95']:.4f} mAP50={res['map50']:.4f}"
                # 选模只在"某类完全缺失"时才有方差风险 → 明确告警，不静默
                nz = [c for c, v in res["per_class_95"].items() if v == v]
                msg += f" (有效类 {len(nz)}/12)"
                if res["map50_95"] > best:
                    best = res["map50_95"]
                    _save(out_dir / "weights" / "best.pt", ep + 1, best)
                    (out_dir / "val_best.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                        default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")
            except Exception as exc:                            # noqa: BLE001
                # 验证出错**不能弄死训练**（曾因 eval 的坐标 bug 让 60 轮跑到 ep10 直接崩）
                import traceback
                msg += f" | val 失败（训练继续）：{type(exc).__name__}: {exc}"
                log("[train][val-error]\n" + traceback.format_exc())
                if model.spatial_memory:
                    _save(out_dir / "weights" / "last.pt", ep + 1, best)
                    raise RuntimeError("新版完整验证失败，已保存安全 last.pt；先修复验证再续训") from exc
            finally:
                # 验证会额外分配一大块显存（实测 reserved 从 2.5G → 4.2G 并一直留着），
                # 无人值守时必须还给缓存，否则后续训练贴着 6G 墙跑，随时 OOM。
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
        log(msg)
        log(f"[train] ep {ep+1} wall_time_with_validation={(time.time()-t_ep)/60:.2f}min")
        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            _save(out_dir / "weights" / "last.pt", ep + 1, best)
    _save(out_dir / "weights" / "last.pt", args.epochs, best)
    log(f"[train] 完成，权重在 {out_dir/'weights'/'last.pt'}"
        + (f" | 最佳 val mAP50-95={best:.4f}" if best >= 0 else ""))


if __name__ == "__main__":
    main()
