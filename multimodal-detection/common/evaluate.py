# -*- coding: utf-8 -*-
"""
evaluate —— 多模态自定义模型的 mAP@50-95 评估（与赛题评估口径一致）。

评估逻辑（按赛题细则）：
  1. 预测按置信度降序；对每个 IoU 阈值 t∈{0.50,0.55,...,0.95}、每个类别 c，
     依次把预测框与"尚未匹配"的真实框按 IoU≥t 贪心匹配（TP/FP）；
  2. AP 用 101 点插值（vendor ap_per_class 与 COCO 一致）；
  3. mAP@50-95 = 10 阈值 mAP 平均；mAP@50 = t=0.50 的 mAP。

模型输出适配：vendors 8.4 Detect 的 eval 输出为 (raw, preds)（raw=(B,4+nc,8400)），
经 utils.nms.non_max_suppression 解码成 (B,M,6) xyxy+conf+cls。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import models_config as MC


def _move_to_device(x, dev):
    """递归迁移 dict/tensor（评估用，避免 CPU/GPU 不一致）。"""
    if isinstance(x, dict):
        return {k: _move_to_device(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, dev) for v in x)
    if isinstance(x, torch.Tensor):
        return x.to(dev, non_blocking=True)
    return x


def decode_preds(raw: torch.Tensor, nc: int, conf_thres: float = 0.25,
                 iou_thres: float = 0.7, max_det: int = 300):
    """(B,4+nc,8400) → list[(M,6) xyxy+conf+cls]（vendors NMS，非端到端路径）。

    赛题口径提示：规则按"提交的全部框按置信度排序"计算 AP（每图≤100 框、按置信度截断），
    所以**评测/选优**建议 conf_thres 取很小（如 0.001）+ max_det=100，而非 0.25。
    """
    from ultralytics.utils.nms import non_max_suppression  # vendor 8.4
    if isinstance(raw, (tuple, list)):
        raw = raw[0] if isinstance(raw[0], torch.Tensor) else raw
    return non_max_suppression(raw, conf_thres=conf_thres, iou_thres=iou_thres,
                               max_det=max_det, nc=nc)


def build_tp_matrix(preds: np.ndarray, targets: np.ndarray,
                    iou_thrs=np.arange(0.50, 0.96, 0.05)) -> np.ndarray:
    """
    赛题式逐类贪心匹配：返回 (P, len(iou_thrs)) bool TP 矩阵。
    preds  : (P,6) xyxy+conf+cls（已按 conf 降序）
    targets: (Q,5) xyxy+cls（cls 在第 4 列，与 evaluate_mAP 构造的 g_xyxy 一致）
    """
    from ultralytics.utils.metrics import box_iou
    P = len(preds)
    tp = np.zeros((P, len(iou_thrs)), dtype=bool)
    if P == 0 or len(targets) == 0:
        return tp
    pc, tc = preds[:, 5], targets[:, 4]
    pbox = torch.from_numpy(preds[:, :4]).float()
    tbox = torch.from_numpy(targets[:, :4]).float()
    for ti, t in enumerate(iou_thrs):
        for c in np.unique(pc):
            pi = np.where(pc == c)[0]                     # 已按 conf 降序
            gi = np.where(tc == c)[0]
            if len(gi) == 0:
                continue
            used = set()
            ious = box_iou(pbox[pi], tbox[gi])            # (Pi, Gi)
            for k, i in enumerate(pi):
                if not len(gi):
                    break
                best = int(torch.argmax(ious[k]).item())
                if ious[k, best] >= t:
                    tp[i, ti] = True
                    used.add(best)
                    mask = np.ones(len(gi), dtype=bool)
                    mask[best] = False
                    gi = gi[mask]
                    ious = ious[:, mask]
    return tp


def evaluate_mAP(
    model,
    samples_val: list,
    imgsz: int,
    build_val_batch: Callable,       # (chunk, rng) -> (inputs, batch)；须无增强(flip/hsv/jitter=关)
    forward_fn: Callable,            # (model, inputs) -> raw
    nc: int = MC.CLASS_NUM,
    conf_thres: float = 0.25,
    iou_nms: float = 0.7,
    max_det: int = 300,
    device: Optional[str] = None,
    names: Optional[dict] = None,
) -> Dict[str, float]:
    """
    全量验证集 mAP 评估。返回 dict：
      {"map50_95": float, "map50": float, "precision": float, "recall": float,
       "per_class": {cls_name: ap50_95}}

    conf_thres / max_det：赛题按"全部提交框排序算 AP、每图≤100 框"，因此
    **评测与选优**建议 conf_thres=0.001、max_det=100（默认 0.25/300 仅用于兼容旧结果）。
    """
    import random
    from ultralytics.utils.metrics import DetMetrics  # vendor 版

    names = names or ({i: n for i, n in enumerate(MC.CLASS_NAMES)} if nc == MC.CLASS_NUM
                      else {i: str(i) for i in range(nc)})
    metrics = DetMetrics(names=names)
    model.eval()
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    with torch.no_grad():
        for i in range(0, len(samples_val), 4):          # 小批量评估
            chunk = samples_val[i:i + 4]
            inputs, batch = build_val_batch(chunk, random.Random(0))
            inputs = _move_to_device(inputs, dev)        # 多模态 inputs 统一迁 GPU
            batch = {k: (_move_to_device(v, dev) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            raw = forward_fn(model, inputs)
            dets = decode_preds(raw, nc, conf_thres, iou_nms, max_det=max_det)

            # GT：batch 内 (cls, 归一化 xywh) → xyxy+cls（像素，×imgsz）
            # 注意 batch 已迁到 GPU（可能 cuda）——必须先 detach().cpu() 再 numpy()
            cls = batch["cls"].detach().cpu().numpy().astype(np.float64)
            bidx = batch["batch_idx"].detach().cpu().numpy().astype(np.int64)
            xywh = batch["bboxes"].detach().cpu().numpy().astype(np.float64)
            g_xyxy = np.zeros((len(cls), 5))
            if len(cls):
                cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
                g_xyxy[:, 0] = (cx - w / 2) * imgsz
                g_xyxy[:, 1] = (cy - h / 2) * imgsz
                g_xyxy[:, 2] = (cx + w / 2) * imgsz
                g_xyxy[:, 3] = (cy + h / 2) * imgsz
                g_xyxy[:, 4] = cls

            for si, det in enumerate(dets):              # 逐图像统计
                d = det.cpu().numpy()
                m = (d[:, 4] >= conf_thres)               # 再阈值一次（NMS 内部已滤）
                d = d[m]
                g = g_xyxy[bidx == si]
                tp = build_tp_matrix(d, g)
                metrics.update_stats({
                    "tp": tp,
                    "conf": d[:, 4] if len(d) else np.zeros((0,)),
                    "pred_cls": d[:, 5] if len(d) else np.zeros((0,)),
                    "target_cls": g[:, 4] if len(g) else np.zeros((0,)),
                    "target_img": np.full(len(g), si) if len(g) else np.zeros((0,)),
                    "im_name": f"batch{i}_img{si}",
                })

    if sum(len(t) for t in metrics.stats["target_cls"]) == 0:
        return {"map50_95": float("nan"), "map50": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "per_class": {}}

    metrics.process(save_dir=Path("."), plot=False)
    p, r, map50, map50_95 = metrics.mean_results()
    per_class = {}
    for i, name in names.items():
        if i < len(metrics.maps):
            per_class[name] = float(metrics.maps[i])
    return {"map50_95": float(map50_95), "map50": float(map50),
            "precision": float(p), "recall": float(r), "per_class": per_class}
