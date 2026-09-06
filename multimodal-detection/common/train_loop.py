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
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import models_config as MC
from common import dataset as DS


# ------------------------------------------------------------
# batch / labels 组装
# ------------------------------------------------------------

def make_batch_dict(chw_list: List[np.ndarray], boxes_list: List[Optional[np.ndarray]],
                    stems: List[str], imgsz: Tuple[int, int]) -> Dict:
    """
    把逐样本的 (C,H,W) 张量与归一化框组装成 ultralytics 训练 batch dict。
    boxes 参考系: letterbox 后画布、归一化 [cls,cx,cy,w,h]。
    返回 dict: cls(N,), bboxes(N,4) xywh 像素, batch_idx(N,), img(B,C,H,W),
               ori_shape, imgsz
    """
    device = torch.device("cpu")
    imgs = np.stack(chw_list, axis=0)                     # (B,C,H,W)
    cls_all, box_all, idx_all = [], [], []
    for i, boxes in enumerate(boxes_list):
        if boxes is None or boxes.size == 0:
            continue
        H, W = imgs.shape[2], imgs.shape[3]
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        cls_all.append(b[:, 0].astype(np.float32))
        cx, cy, w, h = b[:, 1] * W, b[:, 2] * H, b[:, 3] * W, b[:, 4] * H
        box_all.append(np.stack([cx, cy, w, h], axis=1))
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
                 aug=None, depth_shift=(0, 0), preprocess=None,
                 augment=True, seed=0):
        self.samples = list(samples)
        self.mode = mode
        self.target_size = tuple(target_size)
        self.aug = aug
        self.depth_shift = tuple(depth_shift)
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
                depth_shift=self.depth_shift, preprocess=self.preprocess,
                seed=rng_seed)
            return {"img": chw}, boxes, stem
        # "three"：实验1 三路输入（调用方须已注入 实验模型1 目录到 sys.path）
        import dataset_adapter as DA
        rgb, ir, dep, boxes, stem = DA.build_model_inputs(
            s, imgsz=self.target_size,
            aug=self.aug if self.augment else None,
            depth_shift=self.depth_shift, preprocess=self.preprocess,
            seed=rng_seed, to_tensor=False)
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
):
    """
    多模态自定义训练循环（基线2/实验1 共用）。
    验证指标：mAP@50-95（赛题口径，见 common/evaluate.py）；best 按 mAP50-95 最大保存。
    """
    from common import evaluate as EV
    h = cfg.hyper
    device = torch.device(h.device if h.device != "0" else "cuda:0")
    has_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    device = torch.device("cuda:0" if has_cuda else "cpu")
    model = model.to(device)

    # 优化器 / 调度 / 混合精度
    if h.optimizer.lower() in ("sgd", "auto"):
        opt = torch.optim.SGD(model.parameters(), lr=h.lr0, momentum=0.937,
                              weight_decay=5e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=h.lr0, weight_decay=5e-4)
    steps_per_epoch = max(1, (len(samples_train) + h.batch - 1) // h.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=h.epochs * steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=has_cuda and h.amp)

    # EMA（与 ultralytics 一致）
    from ultralytics.utils.torch_utils import ModelEMA  # vendor 版
    ema = ModelEMA(model)

    best_map = float("-inf")     # mAP@50-95（赛题口径）越大越好
    bad_epochs = 0
    out_dir = Path(out_dir) if out_dir else Path.cwd() / "runs" / cfg.key
    wdir = out_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, h.epochs + 1):
        model.train()
        rng = random.Random(h.seed + epoch)
        order = list(samples_train)
        rng.shuffle(order)
        acc: Dict[str, float] = {}
        n_batches = 0

        # 多进程预取 DataLoader（worker 内读图+增强）；workers=0 时退回同步 build_batch
        dl_iter = None
        if use_dataloader and int(getattr(h, "workers", 0) or 0) > 0:
            ds = MultiSampleDataset(
                order, mode=dataset_mode, target_size=(h.imgsz, h.imgsz),
                aug=h.aug,
                depth_shift=(h.depth_shift_x, h.depth_shift_y),
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
            with torch.cuda.amp.autocast(enabled=has_cuda and h.amp):
                preds = forward_fn(model, inputs)
                loss, items = _compute_loss(model, preds, batch)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in items.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n_batches += 1

        # ---- 验证：mAP@50-95（赛题口径；用无增强构建器）----
        avg = {k: v / max(n_batches, 1) for k, v in acc.items()}
        map_res = {}
        if samples_val:
            map_res = EV.evaluate_mAP(
                model, samples_val, h.imgsz,
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
                  f"lr {sched.get_last_lr()[0]:.2e}")

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


def _to_float_items(items) -> dict:
    """把 v8DetectionLoss 的 items（dict / list / tensor）统一成 {name: float}。"""
    if items is None:
        return {}
    if isinstance(items, dict):
        return {k: float(v) for k, v in items.items()}
    if isinstance(items, (list, tuple)):
        return {f"loss_{i}": float(v) for i, v in enumerate(items)}
    return {"loss": float(items)}
