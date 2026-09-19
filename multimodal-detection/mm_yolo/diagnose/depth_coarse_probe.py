# -*- coding: utf-8 -*-
"""depth↔RGB 粗尺度对齐探针：把边缘图下采样到 1/4、1/8、1/16 后再做堆叠相关。

动机：FHD 原生分辨率上 depth 边缘与 RGB 边缘的相关只有 ~0.01–0.04（近乎噪声）。
若 depth 只是"低频几何线索"，那么在粗尺度上应当能看到清晰的相关峰。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent.parent
for _p in (str(_CODE / "mm_yolo"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from io_utils import imread_unicode                        # noqa: E402
from align import _edge_mag, _fill_holes_depth             # noqa: E402


def main(root: Path, n: int = 80, r: int = 20) -> None:
    vis = sorted((root / "visible").glob("*.png"))
    picks = []
    for p in vis:
        img = imread_unicode(p)
        if img is not None and img.shape[0] >= 1000:
            picks.append(p)
        if len(picks) >= n:
            break
    print(f"[coarse] 样本 {len(picks)} 张，搜索 ±{r} 像素（各自尺度）", flush=True)

    for down in (4, 8, 16, 32):
        acc = None
        zero = None
        for p in picks:
            rgb = imread_unicode(p)
            dep = imread_unicode(root / "depth" / (p.stem + p.suffix))
            if rgb is None or dep is None:
                continue
            gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_BGR2GRAY)
            d = dep.astype(np.float32)
            e_rgb = _edge_mag(gray)
            e_dep = _edge_mag(_fill_holes_depth(d, d > 0))
            H, W = e_rgb.shape
            h, w = max(24, H // down), max(24, W // down)
            a = cv2.resize(e_rgb, (w, h), interpolation=cv2.INTER_AREA)
            b = cv2.resize(e_dep, (w, h), interpolation=cv2.INTER_AREA)
            a = a / (a.max() + 1e-6)
            b = b / (b.max() + 1e-6)
            m = max(12, int(0.6 * min(h, w)))
            cy, cx = h // 2, w // 2
            tmpl = a[cy - m // 2: cy + m // 2, cx - m // 2: cx + m // 2]
            pad = cv2.copyMakeBorder(b, r, r, r, r, cv2.BORDER_CONSTANT, value=0.0)
            if pad.shape[0] < m or pad.shape[1] < m:
                continue
            s = cv2.matchTemplate(pad, tmpl, cv2.TM_CCOEFF_NORMED)
            acc = s if acc is None else acc + s
            zero = (cx + r - m // 2, cy + r - m // 2)
        if acc is None:
            print(f"  1/{down}: 无有效样本")
            continue
        acc /= len(picks)
        _mn, mx, _ml, ml = cv2.minMaxLoc(acc)
        zx, zy = zero
        near = acc[max(0, zy - 3):zy + 4, max(0, zx - 3):zx + 4].max()
        prof = " ".join(f"{acc[ml[1], j]:+.3f}" for j in
                        range(max(0, ml[0] - 4), min(acc.shape[1], ml[0] + 5)))
        print(f"  1/{down:<3} 面尺寸 {acc.shape} | 堆叠峰 ({ml[0]-zx:+d},{ml[1]-zy:+d}) 值={mx:.4f} | "
              f"零偏移±3 峰值={near:.4f} | 峰剖面 {prof}", flush=True)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"C:\Users\35482\Desktop\人工智能精英\2\初赛数据集-面向城市场景的多模态目标检测\train_extracted")
    main(root, int(sys.argv[2]) if len(sys.argv) > 2 else 80)
