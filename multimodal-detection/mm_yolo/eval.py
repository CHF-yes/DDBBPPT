# -*- coding: utf-8 -*-
"""评测：赛题口径 mAP50-95 + 逐类 AP + 条件切片 + **模态屏蔽剖面**。

赛题口径（规则原文）：输出 [class_id, cx, cy, w, h, confidence]，全部框按置信度排序，
每图 ≤100 框、无置信度阈值 → 这里固定 conf=0.001 / iou=0.7 / max_det=100。

关键分析能力：
* **逐类 AP**：12 类长尾（class 11 仅 27 框），必须逐类看；
* **条件切片**：按照度/拥挤度/depth 有效比例分位取最坏切片 —— 决定"是不是靠白天刷分"；
* **模态屏蔽剖面**：同一模型分别屏蔽 IR/depth/RGB 推理 → 给每个模态**定价**
  （零额外训练即可回答"哪个模态在哪种条件下值多少分"）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultralytics.utils.nms import non_max_suppression          # noqa: E402

from config import default_config                               # noqa: E402
from data import (AugCfg, MMDataset, build_index, canvas_of, collate, group_split,
                  load_split, parse_imgsz, pick_val_subset, save_split, val_class_stats)  # noqa: E402
from model import MMYOLO, load_mm_checkpoint, resolve_infer_canvas, resolve_infer_modalities  # noqa: E402

IOU_THRS = np.arange(0.5, 1.0, 0.05)


# ---------------------------------------------------------------- 指标

def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(N,4) b(M,4) xyxy → (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def _ap_101(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO 风格 101 点插值 AP。"""
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, grid, side="left")
    idx = np.clip(idx, 0, mpre.size - 1)
    return float(mpre[idx].mean())


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC 风格（面积法）AP，作为交叉核对。"""
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def compute_map(dets: List[dict], gts: List[dict], nc: int
                ) -> Tuple[float, float, Dict[int, float], Dict[int, float], int, List[int]]:
    """dets/gts: 每图 {'boxes': (n,4) xyxy, 'cls': (n,), 'conf': (n,)}（GT 无 conf）。

    返回 (mAP50-95, mAP50, 逐类 AP50-95, 逐类 AP50, 有效类数, 缺失类列表)。

    性能：先按类把每图的 dets/GT 切好（一次），阈值循环里用 numpy 向量化 IoU +
    `argmax` 贪心匹配（原来是对每个框再套一层 GT 的 Python 双层循环，慢 10 倍）。
    """
    ap95: Dict[int, float] = {}
    ap50: Dict[int, float] = {}
    for c in range(nc):
        per_img = []
        for d, g in zip(dets, gts):
            md = d["cls"] == c
            mg = g["cls"] == c
            conf = d["conf"][md]
            order = np.argsort(-conf) if conf.size else np.array([], int)
            per_img.append((conf[order], d["boxes"][md][order], g["boxes"][mg]))
        npos = sum(len(t[2]) for t in per_img)
        if npos == 0:
            ap95[c] = float("nan")
            ap50[c] = float("nan")
            continue
        aps = []
        for ti, thr in enumerate(IOU_THRS):
            scores, tps, fps = [], [], []
            for conf, db, gb in per_img:
                k = len(conf)
                if k == 0:
                    continue
                ious = _box_iou(db, gb)                       # (k, m)
                used = np.zeros(len(gb), bool)
                for j in range(k):
                    if len(gb) == 0:
                        scores.append(conf[j]); tps.append(0); fps.append(1)
                        continue
                    row = ious[j].copy()
                    row[used] = -1.0
                    gj = int(np.argmax(row))
                    hit = row[gj] >= thr
                    if hit:
                        used[gj] = True
                    scores.append(conf[j]); tps.append(1 if hit else 0); fps.append(0 if hit else 1)
            if not scores:
                aps.append(0.0)
                continue
            s = np.asarray(scores, np.float64)
            o = np.argsort(-s)
            tp = np.cumsum(np.asarray(tps, np.float64)[o])
            fp = np.cumsum(np.asarray(fps, np.float64)[o])
            rec = tp / max(npos, 1)
            prec = tp / np.maximum(tp + fp, 1e-9)
            aps.append(_ap_101(rec, prec))
        valid = [a for a in aps if not np.isnan(a)]
        ap95[c] = float(np.mean(valid)) if valid else float("nan")
        ap50[c] = float(aps[0]) if aps else float("nan")
    vals95 = [v for v in ap95.values() if not np.isnan(v)]
    vals50 = [v for v in ap50.values() if not np.isnan(v)]
    # ⚠️ 审计修复：旧版直接 nanmean 把"验证集里没有这个类"静默排除，
    #    于是"12 类 mAP"实际是"有几类算几类"（前 60 张里 class 1/7/11 一个框都没有）。
    #    现在仍然报 mean-over-present（跨配置可比），但把有效类数与缺失类显式带出来。
    m95 = float(np.mean(vals95)) if vals95 else float("nan")
    m50 = float(np.mean(vals50)) if vals50 else float("nan")
    missing = [int(c) for c, v in ap95.items() if np.isnan(v)]
    return m95, m50, ap95, ap50, len(vals95), missing


# ---------------------------------------------------------------- 推理

def _decode(model: MMYOLO, batch: dict, device, conf: float, iou: float, max_det: int,
            modalities: str = "all", off: Optional[Sequence[str]] = None,
            imgsz: int = 960) -> List[np.ndarray]:
    """返回每图 (n,6) [x1,y1,x2,y2,conf,cls]（画布坐标）。

    `modalities="rgb"` 时 IR/Depth 分支**完全不前向**（也不吃它们的内存/算力），
    与"用 RGB-only 权重"等价；`off` 用于诊断性屏蔽某一路。
    """
    prev = set(model.modality_off)
    model.modality_off = set(prev) | set(off or [])
    rgb = batch["rgb"].to(device)
    ir = batch["ir"].to(device) if modalities in ("all", "rgb_ir", "ir") else None
    dep = batch["depth"].to(device) if modalities in ("all", "rgb_dep", "dep") else None
    qual = {k: v.to(device) for k, v in batch["quality"].items()}
    prior = batch["prior"].to(device) if dep is not None else None
    model.eval()
    try:
        with torch.no_grad():
            out = model(rgb, ir, dep, quality=qual, prior=prior,
                        keep={k: v.to(device) for k, v in batch["keep"].items()})
    finally:
        model.modality_off = prev
    pred = _extract_pred(out)
    # vendor NMS 使用整个 batch 共用的时间预算，超时会 ``break``，使后续图片
    # 悄悄保留为空预测。逐图调用保证每张图都被处理；预测和阈值完全不变。
    dets = []
    for image_pred in pred.split(1, dim=0):
        one = non_max_suppression(image_pred, conf, iou, max_det=max_det, nc=model.nc)
        if len(one) != 1:
            raise RuntimeError(f"逐图 NMS 应返回 1 项，实际 {len(one)}")
        dets.append(one[0])
    return [d.cpu().numpy() if d is not None and len(d) else np.zeros((0, 6), np.float32)
            for d in dets]


def _extract_pred(out):
    """取出 NMS 可用的预测张量 (B, 4+nc, A)。

    本版 `Detect.forward` 在 eval 下返回 `(y, preds)`：
      * `y`     = `_inference()` 解码后的张量（xywh + 类别分数）→ **NMS 要的就是它**；
      * `preds` = {'boxes': DFL 距离(4*reg_max), 'scores': 类别 logits, 'feats'} → **不是** NMS 输入。
    踩坑记录：早期版本误把 boxes+scores 拼起来喂 NMS（64+12=76 通道），
    结果类别索引越界（12/13/44）—— 是 DFL 距离被当成了类别分数。
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if torch.is_tensor(item) and item.ndim == 3 and item.shape[1] <= 4 + 200:
                return item
        for item in out:
            if isinstance(item, dict) and "boxes" in item and "scores" in item:
                raise RuntimeError("只拿到原始 DFL 输出（未解码）→ 无法直接送 NMS")
    if isinstance(out, dict):
        if "boxes" in out and "scores" in out:
            raise RuntimeError("只拿到原始 DFL 输出（未解码）→ 无法直接送 NMS")
        for v in out.values():
            if torch.is_tensor(v):
                return v
    raise RuntimeError(f"无法从输出中提取预测张量: {type(out)}")


def evaluate_model(model: MMYOLO, root: Path, samples: List[dict], imgsz=960,
                   device="cuda:0", conf: float = 0.001, iou: float = 0.7, max_det: int = 100,
                   modalities: str = "all", batch_size: int = 4, workers: int = 4,
                   profile: bool = False, slices: bool = True,
                   ablate_absolute: bool = False, by_depth_format: bool = False) -> dict:
    """在给定样本上按赛题口径评测；可选模态屏蔽剖面与条件切片。

    ⚠️ 坐标必须走**同一套 letterbox 映射**：GT 取 dataset 内部已变换到画布的框
    （`item["boxes"] × (Wc,Hc)`），预测本来就在画布上 —— 两者同空间才能算 IoU。
    早期版本直接拿原图归一化标签 × imgsz（忽略 letterbox 的黑边与缩放），
    在 16:9 数据上 y 方向整体错位 → mAP 无意义。

    `imgsz` 支持 int（正方形）或 (H, W)（如 (544,960)）—— 非正方形画布是本项目的默认配置。
    """
    ds = MMDataset(Path(root), samples, imgsz=imgsz, train=False,
                   aug=AugCfg(imgsz=imgsz,
                              depth_resampling=getattr(getattr(model, "cfg", None), "depth_resampling", "legacy_bilinear_v1"),
                              legacy_lowlight=int(model.cfg.encoder.depth_input_channels) == 2),
                   enabled={"rgb": ("rgb",), "ir": ("ir",), "dep": ("dep",),
                            "rgb_ir": ("rgb", "ir"),
                            "rgb_dep": ("rgb", "dep"),
                            "all": ("rgb", "ir", "dep")}[modalities])
    Hc, Wc = ds.canvas                                     # 画布是 (H, W)，别再当标量用
    nc = model.nc
    gts: List[dict] = []

    def _gt_canvas(item) -> dict:
        """dataset 里的框是**画布归一化**，按 (Wc,Hc) 还原成画布像素。"""
        bb = item["boxes"].numpy()
        if not len(bb):
            return {"boxes": np.zeros((0, 4), np.float32), "cls": np.zeros(0, int)}
        b = bb[:, 1:5].copy()
        b[:, [0, 2]] *= Wc
        b[:, [1, 3]] *= Hc
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        return {"boxes": xyxy.astype(np.float32), "cls": bb[:, 0].astype(int)}

    dets_runs: Dict[str, List[dict]] = {}
    modes = {"all": []}
    if profile:
        modes = {"all": []}
        if modalities in ("all", "rgb_ir"):
            modes["no_ir"] = ["ir"]
        if modalities in ("all", "rgb_dep"):
            modes["no_dep"] = ["dep"]
        if modalities in ("all", "rgb_ir", "rgb_dep"):
            modes["no_rgb"] = ["rgb"]
    if ablate_absolute:
        if int(model.cfg.encoder.depth_input_channels) != 4 or modalities not in ("all", "rgb_dep", "dep"):
            raise ValueError("绝对距离通道消融只支持启用 Depth 的 4ch checkpoint")
        modes["no_absolute"] = []
    sl = {"lum": [], "crowd": [], "dvalid": []}

    def _run(off, tag):
        out = []
        gts.clear()
        for i in range(0, len(samples), batch_size):
            chunk = [ds[j] for j in range(i, min(i + batch_size, len(samples)))]
            if any(item is None for item in chunk):
                raise RuntimeError("验证集读图失败；拒绝跳图，否则分组/总体 AP 无法一一对应")
            batch = collate(chunk)
            if batch is None:
                continue
            if tag == "no_absolute":
                batch["depth"] = batch["depth"].clone()
                batch["depth"][:, 1] = 0.0
            dec = _decode(model, batch, device, conf, iou, max_det, modalities, off, imgsz)
            for d, item in zip(dec, chunk):
                out.append({"boxes": d[:, :4], "conf": d[:, 4], "cls": d[:, 5].astype(int)})
                gts.append(_gt_canvas(item))
            if tag == "all" and slices:
                rgb = batch["rgb"]
                sl["lum"] += rgb.mean(dim=(1, 2, 3)).tolist()
                sl["crowd"] += [float(len(b["boxes"])) for b in chunk]
                valid_channel = 2 if int(model.cfg.encoder.depth_input_channels) == 4 else 1
                sl["dvalid"] += batch["depth"][:, valid_channel].mean(dim=(1, 2)).tolist()
        return out, list(gts)

    res: Dict[str, object] = {}
    gts_all: List[dict] = []
    for tag, off in modes.items():
        d, g = _run(off, tag)
        dets_runs[tag] = d
        gts_all = g
        m95, m50, per95, per50, n_valid, miss = compute_map(d, g, nc)
        res[tag] = {"map50_95": m95, "map50": m50, "per_class_95": per95, "per_class_50": per50,
                    "n_valid_classes": n_valid, "missing_classes": miss}

    out = {"map50_95": res["all"]["map50_95"], "map50": res["all"]["map50"],
           "per_class_95": res["all"]["per_class_95"], "per_class_50": res["all"]["per_class_50"],
           "n_valid_classes": res["all"]["n_valid_classes"],
           "missing_classes": res["all"]["missing_classes"],
           "n_images": len(samples), "canvas": list(ds.canvas),
           "modalities": modalities}
    if profile or ablate_absolute:
        out["profile"] = {k: {"map50_95": v["map50_95"], "map50": v["map50"]}
                          for k, v in res.items()}
        base = res["all"]["map50_95"]
        out["profile_drop"] = {k: base - v["map50_95"] for k, v in res.items() if k != "all"}
    if by_depth_format:
        out["depth_format"] = {}
        for ext in (".png", ".jpg"):
            indices = [i for i, s in enumerate(samples)
                       if Path(s["files"].get("depth", "")).suffix.lower() == ext]
            out["depth_format"][ext] = {"n_images": len(indices)}
            if indices:
                for tag, dets in dets_runs.items():
                    d = [dets[i] for i in indices]
                    g = [gts_all[i] for i in indices]
                    m95, m50, _pc95, _pc50, n_valid, _miss = compute_map(d, g, nc)
                    out["depth_format"][ext][tag] = {"map50_95": m95, "map50": m50,
                                                     "n_valid_classes": n_valid}
    if slices and sl["lum"]:
        out["slices"] = _slice_report(dets_runs["all"], gts_all, nc, sl)
    return out


def _slice_report(dets: List[dict], gts: List[dict], nc: int, sl: Dict[str, List[float]]) -> dict:
    """按照度/拥挤度/depth 有效比例分位切片，报告最坏 20% 与最好 20%。"""
    out = {}
    for key, name in (("lum", "照度"), ("crowd", "拥挤度"), ("dvalid", "depth有效")):
        v = np.array(sl[key], np.float32)
        if v.size < 20:
            continue
        lo, hi = np.quantile(v, 0.2), np.quantile(v, 0.8)
        idx_lo = np.where(v <= lo)[0]
        idx_hi = np.where(v >= hi)[0]
        for tag, idx in ((f"{name}最低20%", idx_lo), (f"{name}最高20%", idx_hi)):
            d = [dets[i] for i in idx]
            g = [gts[i] for i in idx]
            m95, m50, _, _, _, _ = compute_map(d, g, nc)
            out[tag] = {"n": int(len(idx)), "map50_95": m95, "map50": m50}
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-weights", default="",
                    help="服务器上本地预训练权重路径；用于重建 checkpoint 结构")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--imgsz", default="", help="空=用 checkpoint 记录的训练画布；支持 'HxW'")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张（0=全量 val）")
    ap.add_argument("--split", default="", help="指定 split.json（默认找 <ckpt>/../../<name>/split.json）")
    ap.add_argument("--profile", action="store_true", help="模态屏蔽剖面")
    ap.add_argument("--ablate-absolute", action="store_true",
                    help="另外评测一个置零绝对深度通道的状态（仅诊断，不能代替重新训练）")
    ap.add_argument("--by-depth-format", action="store_true",
                    help="按 Depth 文件 PNG/JPG 分别报分；小子集仅供方向判断")
    ap.add_argument("--no-slices", action="store_true", help="不算条件切片")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--modalities", default="",
                    choices=["", "all", "rgb", "rgb_ir", "rgb_dep", "ir", "dep"],
                    help="空=按 checkpoint 记录的模态（RGB-only 权重不会被误开三模态）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device_arg = ("cuda:0" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else ("cuda:0" if args.device == "cuda" else args.device)
    dev = torch.device(device_arg)
    overrides = {"weights": args.base_weights} if args.base_weights else {}
    model, ck = load_mm_checkpoint(args.ckpt, device=dev, **overrides)
    model.eval()
    modalities = resolve_infer_modalities(model, args.modalities or None)
    imgsz = parse_imgsz(resolve_infer_canvas(model, parse_imgsz(args.imgsz) if args.imgsz
                                             else None))
    if not args.imgsz:
        print(f"[eval] 使用 checkpoint 记录的训练画布 {canvas_of(imgsz)}")
    idx = build_index(Path(args.root), Path(args.labels) if args.labels else None, limit=0)
    # 验证集必须与训练时**完全一致**（否则成绩不可比）：优先用训练产物里的 split.json
    split_path = Path(args.split) if args.split else None
    if split_path is None:
        cand = Path(args.ckpt).resolve().parent.parent / "split.json"
        split_path = cand if cand.exists() else None
    if split_path is not None and split_path.exists():
        tr, va = load_split(split_path, idx)
        print(f"[eval] 复用划分 {split_path}：val={len(va)}")
    else:
        tr, va = group_split(idx, val_ratio=0.2)
        print(f"[eval] [!] 未找到 split.json，按默认 0.2 重新划分：val={len(va)}"
              f"（与训练时的验证集可能不同，成绩不可直接比较）")
    if args.limit:
        va = pick_val_subset(va, args.limit, seed=42)
        print(f"[eval] 只评测前 {len(va)} 张")
    cls_cnt = val_class_stats(va, nc=model.nc)
    print(f"[eval] 评测集逐类框数 {cls_cnt}")
    res = evaluate_model(model, Path(args.root), va, imgsz=imgsz, device=dev,
                         modalities=modalities, profile=args.profile,
                         slices=not args.no_slices, conf=args.conf, batch_size=args.batch,
                         ablate_absolute=args.ablate_absolute,
                         by_depth_format=args.by_depth_format)
    res["ckpt"] = str(args.ckpt)
    res["epoch"] = int(ck.get("epoch", 0))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
