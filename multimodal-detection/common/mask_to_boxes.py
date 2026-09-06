# -*- coding: utf-8 -*-
"""
mask_to_boxes —— 把二值前景掩码 GT 转成 YOLO 检测框标签 txt。

用途：VDT-2048 的 GT 是二值前景掩码（无类别）。为把它用于**检测框架验证**，
本工具将每个连通域转为一个框，输出赛题/YOLO 格式 txt：
    每行: <cls> <cx> <cy> <w> <h>   （归一化 0~1；无类别信息 → 默认单类 cls=0）

处理流程：读掩码 → 二值化(>127) → 形态学闭运算(合并碎片) → 连通域 →
过滤小面积噪声 → bbox 归一化 → 写同名 txt。

用法：
    python common/mask_to_boxes.py --gt-dir "VDT/Train/GT" --out-dir "VDT/Train/labels"
    python common/mask_to_boxes.py --gt-dir "VDT/Test/GT"  --out-dir "VDT/Test/labels"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np


def read_mask(path: Path) -> np.ndarray:
    """读取掩码（兼容 2D/3D、中文路径），返回二值 uint8 (H,W) {0,1}。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return (img > 127).astype(np.uint8)


def mask_to_boxes(mask: np.ndarray, min_area: int = 20, close_k: int = 5,
                  cls: int = 0) -> np.ndarray:
    """
    掩码 → (N,5) float32 [cls, cx, cy, w, h]（全部以图像宽高归一化，YOLO 格式）。
    - close_k: 形态学闭运算核（合并被遮挡分裂的前景碎片；0=跳过）
    - min_area: 低于该面积的连通域视为噪声丢弃
    """
    H, W = mask.shape[:2]
    m = mask.copy()
    if close_k and close_k > 1:
        k = np.ones((close_k, close_k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, num):                       # 0 是背景
        x, y, w, h, area = stats[i]
        if area < max(min_area, 1):
            continue
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        boxes.append([float(cls), cx, cy, w / W, h / H])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 5) if boxes else np.zeros((0, 5), np.float32)


def convert_dir(gt_dir: Path, out_dir: Path, min_area: int = 20,
                close_k: int = 5, cls: int = 0) -> Tuple[int, int]:
    """转换一个 GT 目录 → 输出同名 txt；返回 (转换文件数, 总目标数)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files, n_targets = 0, 0
    for p in sorted(gt_dir.glob("*")):
        if p.suffix.lower() != ".png":
            continue
        mask = read_mask(p)
        boxes = mask_to_boxes(mask, min_area=min_area, close_k=close_k, cls=cls)
        lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
        (out_dir / (p.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_files += 1
        n_targets += len(boxes)
    return n_files, n_targets


def main() -> None:
    ap = argparse.ArgumentParser(description="二值前景掩码 GT → YOLO bbox txt")
    ap.add_argument("--gt-dir", required=True, help="GT 掩码目录（如 VDT/Train/GT）")
    ap.add_argument("--out-dir", required=True, help="输出 txt 目录（如 VDT/Train/labels）")
    ap.add_argument("--min-area", type=int, default=20, help="面积阈值（px），低于视为噪声")
    ap.add_argument("--close-k", type=int, default=5, help="闭运算核（默认 5，0=不合并）")
    ap.add_argument("--cls", type=int, default=0, help="类别 id（无类别掩码 → 默认 0）")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir).expanduser().resolve()
    if not gt_dir.is_dir():
        raise SystemExit(f"[mask_to_boxes] GT 目录不存在: {gt_dir}")
    nf, nt = convert_dir(gt_dir, Path(args.out_dir).expanduser().resolve(),
                         args.min_area, args.close_k, args.cls)
    print(f"[mask_to_boxes] done: {nf} 个掩码 → {nt} 个目标框 -> {args.out_dir}")


if __name__ == "__main__":
    main()
