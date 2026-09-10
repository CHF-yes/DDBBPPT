# -*- coding: utf-8 -*-
"""
train_loop —— 多模态自定义训练循环（基线模型2 / 实验模型1 共用）。

为什么需要自定义循环
------------------
ultralytics 内建训练器假设"一图固定通道"（3ch RGB 目录）。而：
  - 基线模型2 ：输入是 (B,5,H,W) 融合张量（RGB+IR+D 前期拼接）
  - 实验模型1 ：输入是 (rgb, ir, depth) 三路张量（主流+辅助流）
两者都无法用内建 DataLoader 直接喂。因此本模块提供一个**轻量但完整**的训练循环：

  DataLoader(自定义 collate) → 前向 → ultralytics 损失(复用 v8DetectionLoss/
  DetectionModel.loss) → AMP + EMA + 余弦调度 + 早停 → best/last 权重保存。

本循环不启动训练；由各实例的 main.py train 子命令调用（本次仅接线+静态校验，
实际训练需数据就绪后在服务器执行）。
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import models_config as MC
from common import dataset as DS
from common.multimodal_augment import effective_depth_shift


# ------------------------------------------------------------
# batch / labels 组装
# ------------------------------------------------------------

def make_batch_dict(chw_list: List[np.ndarray], boxes_list: List[Optional[np.ndarray]],
                    stems: List[str], imgsz: Tuple[int, int]) -> Dict:
    """
    把逐样本的 (C,H,W) 张量与归一化框组装成 ultralytics 训练 batch dict。
    boxes 参考系: letterbox 后画布、归一化 [cls,cx,cy,w,h]。
    注意：bboxes **保持归一化 xywh**——vendors v8DetectionLoss 的 Labels.preprocess
    会按 imgsz 缩放后转 xyxy（传像素坐标会越界 → assigner 0 正样本，见实测 bug）。
    返回 dict: cls(N,), bboxes(N,4) 归一化 xywh, batch_idx(N,), img(B,C,H,W),
               ori_shape, imgsz
    """
    device = torch.device("cpu")
    imgs = np.stack(chw_list, axis=0)                     # (B,C,H,W)
    cls_all, box_all, idx_all = [], [], []
    for i, boxes in enumerate(boxes_list):
        if boxes is None or boxes.size == 0:
            continue
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        cls_all.append(b[:, 0].astype(np.float32))
        box_all.append(np.ascontiguousarray(b[:, 1:5]))   # 归一化 xywh（勿转像素！）
        idx_all.append(np.full(len(b), i, dtype=np.int64))
    cls = (np.concatenate(cls_all) if cls_all else np.zeros((0,), np.float32))
    boxes = (np.concatenate(box_all) if box_all else np.zeros((0, 4), np.float32))
    bidx = (np.concatenate(idx_all) if idx_all else np.zeros((0,), np.int64))
    return {
        "cls": torch.from_numpy(cls),
        "bboxes": torch.from_numpy(boxes),
        "batch_idx": torch.from_numpy(bidx),
        "img": torch.from_numpy(np.ascontiguousarray(imgs)).float(),
        "ori_shape": [(imgs.shape[2], imgs.shape[3])] * len(stems),
        "imgsz": int(imgs.shape[2]),
    }


def move_to_device(x, device):
    """递归把 dict/numpy/tensor 迁移到 device（inputs 与 batch 通用）。"""
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    return x


class MultiSampleDataset(Dataset):
    """把样本列表包装为 torch Dataset；__getitem__ 在 DataLoader **worker 进程**中
    执行"读图 + 一致性增强 + 拼装"，实现多进程读图预取。

    支持两种输入模式：
      "5ch"   ：基线2 —— DS.build_consistent_aug_5ch → (5,H,W) 单输入
      "three" ：实验1 —— dataset_adapter.build_model_inputs → (rgb, ir, depth)
    配置全部为可 pickle 的 dataclass/元组，兼容 Linux fork / Windows spawn。
    """

    def __init__(self, samples, mode="5ch", target_size=(1024, 1024),
                 aug=None, depth_shift=(0, 0), align=None, preprocess=None,
                 augment=True, seed=0):
        self.samples = list(samples)
        self.mode = mode
        self.target_size = tuple(target_size)
        self.aug = aug
        self.depth_shift = tuple(depth_shift)
        self.align = align                    # P1-6: 提供时按**原图宽**换算对齐量
        self.preprocess = preprocess
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rng_seed = (self.seed + idx * 7919) % (1 << 31)   # 每样本独立种子（可复现）
        if self.mode == "5ch":
            chw, boxes, stem = DS.build_consistent_aug_5ch(
                s, target_size=self.target_size,
                aug=self.aug if self.augment else None,
                depth_shift=self.depth_shift, align=self.align,
                preprocess=self.preprocess,
                seed=rng_seed)
            return {"img": chw}, boxes, stem
        # "three"：实验1 三路输入（调用方须已注入 实验模型1 目录到 sys.path）
        import dataset_adapter as DA
        rgb, ir, dep, boxes, stem = DA.build_model_inputs(
            s, imgsz=self.target_size,
            aug=self.aug if self.augment else None,
            depth_shift=self.depth_shift, align=self.align,
            preprocess=self.preprocess,
            seed=rng_seed, to_tensor=False,
            mosaic_pool=self.samples if self.augment else None)
        return {"rgb": rgb, "ir": ir, "depth": dep}, boxes, stem


def collate_multimodal(items):
    """[(model_inputs_dict, boxes, stem), ...] → (inputs, batch)。
    batch 与 make_batch_dict 同构（cls/bboxes/batch_idx/ori_shape/imgsz）。"""
    if not items:
        return {}, {}
    first = items[0][0]
    key0 = next(iter(first))
    chw_list = [it[0][key0] for it in items]     # 任一通道张量定 H/W（5ch: img；three: rgb）
    boxes_list = [it[1] for it in items]
    stems = [it[2] for it in items]
    batch = make_batch_dict(chw_list, boxes_list, stems,
                            (chw_list[0].shape[1], chw_list[0].shape[2]))
    inputs = {k: np.stack([it[0][k] for it in items], axis=0) for k in first}
    # 统一转 torch float32：模型前向/设备迁移都按 tensor 处理
    inputs = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in inputs.items()}
    return inputs, batch


# ------------------------------------------------------------
# 训练主循环
# ------------------------------------------------------------

def train_custom(
    model: nn.Module,
    samples_train: list,
    samples_val: list,
    cfg: MC.ModelConfig,
    build_batch: Callable,          # (samples, seed) -> (model_inputs, labels_dict)
    forward_fn: Callable,           # (model, model_inputs) -> preds
    out_dir: Optional[Path] = None,
    verbose: bool = True,
    build_val_batch: Optional[Callable] = None,  # 无增强的 val 构建器；None 时用 build_batch
    conf_thres: float = 0.25,
    iou_nms: float = 0.7,
    use_dataloader: bool = True,     # True: DataLoader 多进程读图(num_workers=h.workers)
    dataset_mode: str = "5ch",       # "5ch"(基线2) | "three"(实验1)
    imgsz_override: Optional[int] = None,   # 快捷覆盖训练分辨率（None=用 cfg.hyper.imgsz）
):
    """
    多模态自定义训练循环（基线2/实验1 共用）。
    验证指标：mAP@50-95（赛题口径，见 common/evaluate.py）；best 按 mAP50-95 最大保存。
    """
    from common import evaluate as EV
    h = cfg.hyper
    isz = int(imgsz_override or h.imgsz)   # 训练/验证分辨率（--imgsz 可覆盖）
    device = torch.device(h.device if h.device != "0" else "cuda:0")
    has_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    device = torch.device("cuda:0" if has_cuda else "cpu")
    model = model.to(device)

    steps_per_epoch = max(1, (len(samples_train) + h.batch - 1) // h.batch)
    opt = _build_optimizer(model, h)
    warmup_steps = max(0, int(round(float(getattr(h, "warmup_epochs", 0.0)) * steps_per_epoch)))
    total_steps = max(1, int(h.epochs) * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=has_cuda and h.amp)
    optimizer_step = 0

    group_desc = ", ".join(
        f"{g.get('group_name', i)}={g['target_lr']:.2e}"
        for i, g in enumerate(opt.param_groups))
    print(f"[{cfg.key}] optimizer={type(opt).__name__} groups[{group_desc}] "
          f"warmup={warmup_steps} steps ({getattr(h, 'warmup_epochs', 0.0):g} epochs) "
          f"lrf={getattr(h, 'lrf', 0.01):g}")

    # EMA（与 ultralytics 一致）
    from ultralytics.utils.torch_utils import ModelEMA  # vendor 版
    ema = ModelEMA(model)

    best_map = float("-inf")     # mAP@50-95（赛题口径）越大越好
    bad_epochs = 0
    out_dir = Path(out_dir) if out_dir else Path.cwd() / "runs" / cfg.key
    wdir = out_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.key}] run epochs={h.epochs} patience={h.patience} imgsz={isz} "
          f"batch={h.batch} workers={h.workers} out={out_dir}")

    for epoch in range(1, h.epochs + 1):
        model.train()
        rng = random.Random(h.seed + epoch)
        order = list(samples_train)
        rng.shuffle(order)
        acc: Dict[str, float] = {}
        n_batches = 0

        # ---- 每轮生效的增强：close_mosaic（末段关闭 mosaic，消除拼接缝的伪几何）----
        # 窗口 = max(close_mosaic_epochs, close_mosaic_frac × epochs)
        # 注意：若训练被 patience 提前截断，窗口可能永远到不了 —— 因此还需
        # 用占比较小的 mosaic_p（见 models_config）+ 足够大的 patience 配合。
        aug_eff = h.aug
        cm = int(getattr(h.aug, "close_mosaic_epochs", 0) or 0)
        cm = max(cm, int(round(float(getattr(h.aug, "close_mosaic_frac", 0.0) or 0.0)
                               * int(h.epochs))))
        if cm > 0 and epoch > int(h.epochs) - cm:
            aug_eff = dataclasses.replace(h.aug, mosaic_p=0.0)
        model._aug_override = aug_eff          # 同步读图路径的 build_batch 会读它

        # ---- 融合课程：前 N 轮整体旁路注入（先让 RGB 通路站稳）----
        if hasattr(model, "set_fusion_enabled"):
            fw = float(getattr(h, "fusion_warmup_epochs", 0.0) or 0.0)
            model.set_fusion_enabled(epoch > fw)

        # 多进程预取 DataLoader（worker 内读图+增强）；workers=0 时退回同步 build_batch
        dl_iter = None
        if use_dataloader and int(getattr(h, "workers", 0) or 0) > 0:
            ds = MultiSampleDataset(
                order, mode=dataset_mode, target_size=(isz, isz),
                aug=aug_eff,
                align=h.align,           # P1-6: 对齐量按原图宽换算（不再是目标 imgsz）
                preprocess=h.preprocess, augment=True, seed=h.seed + epoch)
            dl = DataLoader(ds, batch_size=h.batch, shuffle=False,
                            num_workers=int(h.workers),
                            collate_fn=collate_multimodal,
                            drop_last=False,
                            persistent_workers=int(h.workers) > 0)
            dl_iter = iter(dl)
            print(f"[{cfg.key}] DataLoader 多进程读图: workers={h.workers} "
                  f"mode={dataset_mode}")
        else:
            print(f"[{cfg.key}] 同步单进程读图 (workers={getattr(h, 'workers', 0)})")

        for i in range(0, len(order), h.batch):
            if dl_iter is not None:
                try:
                    inputs, batch = next(dl_iter)
                except StopIteration:
                    break
            else:
                chunk = order[i:i + h.batch]
                inputs, batch = build_batch(chunk, rng)
            inputs = move_to_device(inputs, device)          # rgb/ir/depth 等统一迁 GPU
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}              # cls/bboxes/batch_idx/img
            _set_optimizer_lr(opt, optimizer_step, warmup_steps, total_steps,
                              float(getattr(h, "lrf", 0.01)))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=has_cuda and h.amp):
                preds = forward_fn(model, inputs)
                loss, items = _compute_loss(model, preds, batch)
            # Step3: 每模态辅助头损失（中心分类；λ 线性退火但**保留下限**）
            lam = 0.0
            if int(h.epochs) <= 1:
                aux_factor = 1.0
            else:
                aux_factor = 1.0 - (epoch - 1) / (int(h.epochs) - 1)
            lam = max(float(getattr(h, "aux_lambda_final", 0.0) or 0.0),
                      float(getattr(h, "aux_lambda", 0.1)) * aux_factor)
            if getattr(model, "aux_enabled", False) and model.training and model._aux_logits:
                al = _aux_center_loss(model._aux_logits, batch,
                                      getattr(model, "_aux_keep", None))
                loss = loss + lam * al
                items["aux_loss"] = float(al.detach())
            # Step3b: 稀疏距离头损失（GT = 深度图自身，零额外标注）
            if (getattr(model, "dist_enabled", False) and model.training
                    and getattr(model, "_aux_dist_logits", None) is not None
                    and isinstance(inputs, dict) and "depth" in inputs):
                dl = _aux_distance_loss(model._aux_dist_logits, batch, inputs["depth"],
                                        scale_mm=float(getattr(h.preprocess, "depth_scale_mm", 20000.0)))
                loss = loss + lam * dl
                items["dist_loss"] = float(dl.detach())
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # AMP 溢出时 GradScaler 会跳过 optimizer.step；学习率进度也必须同步跳过。
            if not scaler.is_enabled() or scaler.get_scale() >= scale_before:
                optimizer_step += 1
            for k, v in items.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n_batches += 1

        # ---- 验证：mAP@50-95（赛题口径；用无增强构建器）----
        avg = {k: v / max(n_batches, 1) for k, v in acc.items()}
        map_res = {}
        if samples_val:
            map_res = EV.evaluate_mAP(
                model, samples_val, isz,
                build_val_batch or build_batch, forward_fn,
                nc=cfg.class_num, conf_thres=conf_thres, iou_nms=iou_nms,
                device=str(device))
            cur_map = float(map_res.get("map50_95", float("nan")))
        else:
            cur_map = float("nan")
        if verbose:
            avg_str = " ".join(f"{k} {v:.3f}" for k, v in avg.items())
            map_str = (f"mAP50-95 {cur_map:.3f} mAP50 {map_res.get('map50', float('nan')):.3f}"
                       if map_res else "no val")
            print(f"[{cfg.key}] ep {epoch}/{h.epochs} {avg_str} {map_str} "
                  f"lr[{_format_group_lrs(opt)}]")

        ema.update_attr(model)
        ema.update(model)
        if not samples_val:
            save_ckpt(wdir / "last.pt", ema, epoch, float("nan"), cfg, type(model).__name__)
            continue
        is_best = cur_map > best_map
        if is_best:
            best_map = cur_map
            bad_epochs = 0
            save_ckpt(wdir / "best.pt", ema, epoch, best_map, cfg, type(model).__name__)
        else:
            bad_epochs += 1
        save_ckpt(wdir / "last.pt", ema, epoch, cur_map, cfg, type(model).__name__)

        if h.patience and bad_epochs >= h.patience:
            print(f"[{cfg.key}] early stop @ epoch {epoch} (patience {h.patience})")
            break

    print(f"[{cfg.key}] done. best mAP50-95 {best_map:.3f} -> {wdir / 'best.pt'}")
    return wdir


def save_ckpt(path, ema, epoch, best_map, cfg: MC.ModelConfig, model_type: str) -> None:
    """保存可复现的 checkpoint：state_dict + 训练/模型元数据（不再是裸 state_dict）。"""
    torch.save({
        "model_state": ema.ema.state_dict(),
        "epoch": epoch,
        "best_map": float(best_map),
        "cfg_key": cfg.key,
        "class_num": cfg.class_num,
        "in_channels": cfg.in_channels,
        "model_type": model_type,                 # DetectionModel / Experiment1Model 等
        "class_names": MC.CLASS_NAMES,
    }, path)
    print(f"[train_loop] saved checkpoint (ep {epoch}) -> {path}")


def load_custom_checkpoint(path, model: nn.Module, strict: bool = True) -> dict:
    """
    统一自定义 checkpoint 加载协议（P0-4）——训练保存的 best.pt/last.pt
    不是 ultralytics 原生格式，YOLO() 无法直接 load；必须先构建结构再回填权重。

    支持三种格式：
      1) 本框架自定义格式 {"model_state": ..., ...}（save_ckpt 产出）；
      2) ultralytics 原生训练产物（含 "ema"/"model" 键，如 YOLO.train 的 best.pt）；
      3) 裸 state_dict（旧版兼容）。
    返回 checkpoint dict（含 epoch/best_map/cfg_key 等元数据，供打印）。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"[load_ckpt] 无法识别 checkpoint 格式: {path}")
    state = None
    if "model_state" in ckpt:                     # 本框架自定义格式
        state = ckpt["model_state"]
    elif "ema" in ckpt and hasattr(ckpt["ema"], "state_dict"):
        state = ckpt["ema"].state_dict()          # ultralytics 原生（EMA 版）
    elif "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        state = ckpt["model"].state_dict()        # ultralytics 原生（裸模型）
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt                              # 裸 state_dict
    if state is None:
        raise KeyError(f"[load_ckpt] {path} 中找不到可加载的 state_dict 键")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    meta = {k: ckpt.get(k) for k in ("epoch", "best_map", "cfg_key", "class_num",
                                     "in_channels", "model_type") if k in ckpt}
    print(f"[load_ckpt] {path} 加载完成: {meta}"
          + ("" if strict else f" (strict=False; missing={len(missing)}, "
                               f"unexpected={len(unexpected)})"))
    return ckpt


def _compute_loss(model: nn.Module, preds, batch: Dict):
    """按模型类型取损失：DetectionModel 用自带 .loss()；自定义模型用 v8DetectionLoss。
    返回 (loss, items_dict)；items 值为 float。"""
    # 官方 v8DetectionLoss 需要 model.args 支持属性访问（.box/.cls/.dfl）；
    # YOLO 加载后 args 是 dict，统一转 SimpleNamespace。
    from types import SimpleNamespace
    if isinstance(getattr(model, "args", None), dict):
        _m = dict(model.args)
        for _k, _v in (("box", 7.5), ("cls", 0.5), ("dfl", 1.5)):   # 官方默认损失增益
            _m.setdefault(_k, _v)
        model.args = SimpleNamespace(**_m)
    if hasattr(model, "loss") and not hasattr(model, "criterion_builder"):
        loss, items = model.loss(batch, preds=preds)
        return _scalarize(loss), _to_float_items(items)
    from ultralytics.utils.loss import v8DetectionLoss  # vendor 版
    crit = getattr(model, "_crit_cache", None)
    if crit is None:
        crit = v8DetectionLoss(model)
        model._crit_cache = crit
    loss, items_out = crit(preds, batch)
    return _scalarize(loss), _to_float_items(items_out or crit.loss_names)


def _scalarize(loss):
    """v8DetectionLoss/DetectionModel 返回 (box,cls,dfl) 三分量向量 → 求和为标量（backward 需要）。"""
    import torch
    return loss.sum() if torch.is_tensor(loss) and loss.ndim > 0 else loss


def _aux_center_loss(logits: dict, batch: Dict, keep_masks: Optional[dict] = None):
    """Step3 中心分类辅助损失（CenterNet 风格，P4 stride 网格）：
    GT 框中心映射为 3x3 高斯热图，并用 modified focal loss 处理极端前景稀疏。
    logits 只由 AuxStream→AuxHead 路径产生 → 梯度天然不经过主流 backbone。"""
    device = next(iter(logits.values())).device
    cls = batch["cls"].to(device).long()
    cx, cy = batch["bboxes"][:, 0].to(device), batch["bboxes"][:, 1].to(device)
    bidx = batch["batch_idx"].to(device).long()
    total = next(iter(logits.values())).new_zeros(())
    used_modalities = 0
    for name, raw_logit in logits.items():
        # AMP 下 FP16 的 1-1e-4 会舍入成1，进而 log(1-p)=log(0)=inf；
        # focal loss固定使用FP32，梯度仍会正常回传到原始辅助头。
        logit = raw_logit.float()
        nc_, H4, W4 = logit.shape[1], logit.shape[2], logit.shape[3]
        if keep_masks and name in keep_masks:
            active = keep_masks[name].reshape(logit.shape[0]).to(device=device).bool()
        else:
            active = torch.ones(logit.shape[0], device=device, dtype=torch.bool)
        if not active.any():
            continue
        valid = ((bidx >= 0) & (bidx < logit.shape[0]) &
                 (cls >= 0) & (cls < nc_) & active[bidx.clamp(0, logit.shape[0] - 1)])
        gx = (cx * W4).long().clamp(0, W4 - 1)
        gy = (cy * H4).long().clamp(0, H4 - 1)
        target = torch.zeros_like(logit)
        # 3x3 Gaussian (sigma=1)：中心为1，邻域为软目标；重叠位置取最大值。
        kernel = logit.new_tensor([[0.36787945, 0.60653067, 0.36787945],
                                   [0.60653067, 1.0,        0.60653067],
                                   [0.36787945, 0.60653067, 0.36787945]])
        for bi, ci, yi, xi in zip(bidx[valid].tolist(), cls[valid].tolist(),
                                  gy[valid].tolist(), gx[valid].tolist()):
            y0, y1 = max(0, yi - 1), min(H4, yi + 2)
            x0, x1 = max(0, xi - 1), min(W4, xi + 2)
            ky0, kx0 = y0 - (yi - 1), x0 - (xi - 1)
            patch = kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
            target[bi, ci, y0:y1, x0:x1] = torch.maximum(
                target[bi, ci, y0:y1, x0:x1], patch)

        pred = logit.sigmoid().clamp(1e-4, 1.0 - 1e-4)
        active_map = active.view(-1, 1, 1, 1)
        pos = target.eq(1.0) & active_map
        neg = target.lt(1.0) & active_map
        neg_weights = (1.0 - target).pow(4)
        pos_loss = -(pred.log() * (1.0 - pred).pow(2) * pos).sum()
        neg_loss = -((1.0 - pred).log() * pred.pow(2) * neg_weights * neg).sum()
        num_pos = pos.sum().clamp_min(1).to(logit.dtype)
        total = total + (pos_loss + neg_loss) / num_pos
        used_modalities += 1
    return total / max(used_modalities, 1)


def _aux_distance_loss(logits: torch.Tensor, batch: Dict, depth_input: torch.Tensor,
                       scale_mm: float = 20000.0, win: int = 9, min_valid: float = 0.5):
    """Step3b 稀疏距离回归（深度图自监督，零额外标注）。

    logits      : (B,1,H4,W4) 距离头输出（P4 分辨率，log(米)）
    depth_input : (B,2,H,W)  模型输入的 depth = [归一化距离, 有效掩码]
    GT：GT 框中心处 **win×win 窗口内有效像素的均值距离**（毫米→米→log）。
        为降低噪声：窗口要求有效比例 ≥ min_valid，否则该框不参与
        （否则框中心可能落在背景/深度空洞/mosaic 拼接缝上 → 伪 GT）。
    返回 Huber 损失；梯度只回传距离头所在辅助流。
    """
    device = logits.device
    b, _, h4, w4 = logits.shape
    cls = batch["cls"].to(device)
    cx = batch["bboxes"][:, 0].to(device)
    cy = batch["bboxes"][:, 1].to(device)
    bidx = batch["batch_idx"].to(device).long()
    d_norm = depth_input[:, 0].to(device)              # (B,H,W)
    d_mask = depth_input[:, 1].to(device)              # (B,H,W)
    B, H, W = d_norm.shape

    # 窗口内有效像素的均值距离（向量化，避免逐框循环）
    d_sum = F.avg_pool2d((d_norm * d_mask).unsqueeze(1), win, stride=1,
                         padding=win // 2).squeeze(1)
    m_avg = F.avg_pool2d(d_mask.unsqueeze(1), win, stride=1,
                         padding=win // 2).squeeze(1)
    d_mean = d_sum / m_avg.clamp_min(1e-6)

    valid = (bidx >= 0) & (bidx < B) & (cls >= 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    bi = bidx[valid]
    px = (cx[valid] * W).long().clamp(0, W - 1)
    py = (cy[valid] * H).long().clamp(0, H - 1)
    tgt_m = d_mean[bi, py, px] * (scale_mm / 1000.0)   # 米
    ok = m_avg[bi, py, px] >= float(min_valid)         # 有效比例达标才作为 GT
    if not bool(ok.any()):
        return logits.sum() * 0.0

    gx = (cx[valid] * w4).long().clamp(0, w4 - 1)
    gy = (cy[valid] * h4).long().clamp(0, h4 - 1)
    pred = logits[bi, 0, gy, gx].float()
    target = torch.log(tgt_m.clamp_min(0.05)).float()
    return F.smooth_l1_loss(pred[ok], target[ok], beta=0.2)


def _split_decay_params(named_params):
    """AdamW/SGD 参数拆成 decay 与 no_decay（bias、BN及标量不衰减）。"""
    decay, no_decay = [], []
    for _name, p in named_params:
        if not p.requires_grad:
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return decay, no_decay


def _build_optimizer(model: nn.Module, h):
    """实验1按预训练 backbone / 新模块分组；其余模型保持单一目标学习率。"""
    backbone_mult = float(getattr(h, "backbone_lr_mult", 1.0))
    named = list(model.named_parameters())
    if hasattr(model, "backbone") and backbone_mult != 1.0:
        partitions = [
            ("backbone", [(n, p) for n, p in named if n.startswith("backbone.")], backbone_mult),
            ("new", [(n, p) for n, p in named if not n.startswith("backbone.")], 1.0),
        ]
    else:
        partitions = [("all", named, 1.0)]

    groups = []
    weight_decay = float(getattr(h, "weight_decay", 5e-4))
    for name, params, mult in partitions:
        decay, no_decay = _split_decay_params(params)
        target_lr = float(h.lr0) * mult
        if decay:
            groups.append({"params": decay, "lr": target_lr, "target_lr": target_lr,
                           "weight_decay": weight_decay, "group_name": f"{name}/decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": target_lr, "target_lr": target_lr,
                           "weight_decay": 0.0, "group_name": f"{name}/no_decay"})

    if h.optimizer.lower() in ("sgd", "auto"):
        return torch.optim.SGD(groups, lr=float(h.lr0), momentum=0.937)
    return torch.optim.AdamW(groups, lr=float(h.lr0), betas=(0.9, 0.999))


def _set_optimizer_lr(opt, step: int, warmup_steps: int, total_steps: int, lrf: float):
    """按实际成功的 optimizer 更新次数执行线性 warmup + 非零余弦退火。"""
    if warmup_steps > 0 and step < warmup_steps:
        factor = 0.1 + 0.9 * (step + 1) / warmup_steps
    else:
        remain = max(1, total_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / remain, 0.0), 1.0)
        factor = lrf + 0.5 * (1.0 - lrf) * (1.0 + math.cos(math.pi * progress))
    for group in opt.param_groups:
        group["lr"] = group["target_lr"] * factor


def _format_group_lrs(opt) -> str:
    seen = {}
    for i, group in enumerate(opt.param_groups):
        prefix = str(group.get("group_name", i)).split("/", 1)[0]
        seen[prefix] = group["lr"]
    return ",".join(f"{name}={lr:.2e}" for name, lr in seen.items())


def _to_float_items(items) -> dict:
    """把 v8DetectionLoss 的 items（dict / list / tensor）统一成 {name: float}。"""
    if items is None:
        return {}
    if isinstance(items, dict):
        return {k: float(v) for k, v in items.items()}
    if isinstance(items, (list, tuple)):
        return {f"loss_{i}": float(v) for i, v in enumerate(items)}
    return {"loss": float(items)}
