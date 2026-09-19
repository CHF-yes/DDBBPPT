# -*- coding: utf-8 -*-
"""对齐探针：判断 (a) 是否存在**系统性**模态偏移，(b) 逐图估计为何失效。

做法：
  1. 取 N 张样本，算边缘图（RGB 灰度 / IR / depth 去洞后）
  2. 每张算 NCC 相关面（matchTemplate），先看逐图峰值与置信度
  3. 把 N 张相关面**堆叠平均**（stacked correlation）——若存在固定偏移，
     单张的噪声会被平均掉，堆叠面会出现明显单峰
  4. 顺带检查 IR 三通道是否同源（伪彩色 / 灰度复制）、depth 表示的影响

用法：python code/mm_yolo/diagnose/align_probe.py --root <提取后的可见光父目录> --n 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent.parent
for _p in (str(_CODE / "mm_yolo"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from io_utils import imread_unicode                       # noqa: E402
from align import _edge_mag, _fill_holes_depth            # noqa: E402


def _surface(a_edge: np.ndarray, b_edge: np.ndarray, r: int, tmpl_frac: float = 0.5
             ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """b（被搜索）在 a（模板来源）上的 NCC 面；返回 (面, 模板中心在面坐标里的位置)。"""
    m = max(16, int(tmpl_frac * min(a_edge.shape)))
    cy, cx = a_edge.shape[0] // 2, a_edge.shape[1] // 2
    tmpl = a_edge[cy - m // 2: cy + m // 2, cx - m // 2: cx + m // 2]
    pad = cv2.copyMakeBorder(b_edge, r, r, r, r, cv2.BORDER_CONSTANT, value=0.0)
    surf = cv2.matchTemplate(pad, tmpl, cv2.TM_CCOEFF_NORMED)
    return surf, (cx + r - m // 2, cy + r - m // 2)      # 零偏移时峰应在该处


def probe(root: Path, n: int, work_w: int, max_shift: int, lab_dir: Optional[Path]) -> None:
    vis = sorted((root / "visible").glob("*"))
    fhd = [p for p in vis if imread_unicode(p) is not None and imread_unicode(p).shape[0] >= 1000][:n]
    print(f"[probe] 使用 {len(fhd)} 张 FHD 样本（工作宽度 {work_w}）", flush=True)

    acc: Dict[str, Optional[np.ndarray]] = {"dep": None, "ir": None, "ir_gray": None}
    peaks: Dict[str, List[Tuple[float, float, float]]] = {"dep": [], "ir": [], "ir_gray": []}
    ir_ch_stats = []
    zero_pos = None

    for i, p in enumerate(fhd):
        rgb = imread_unicode(p)
        ir = imread_unicode(root / "infrared" / (p.stem + p.suffix))
        dep = imread_unicode(root / "depth" / (p.stem + p.suffix))
        if rgb is None or ir is None or dep is None:
            continue
        s = work_w / rgb.shape[1]
        sz = (work_w, max(8, int(round(rgb.shape[0] * s))))
        rgb_s = cv2.resize(rgb, sz, interpolation=cv2.INTER_AREA)
        ir_s = cv2.resize(ir, sz, interpolation=cv2.INTER_AREA)
        dep_s = cv2.resize(dep, sz, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(rgb_s[:, :, :3], cv2.COLOR_BGR2GRAY)
        ir_g = ir_s[:, :, 0] if ir_s.ndim == 3 else ir_s
        dep_v = (dep_s[:, :, 0] if dep_s.ndim == 3 else dep_s).astype(np.float32)
        valid = dep_v > 0
        e_rgb = _edge_mag(gray)
        e_ir = _edge_mag(ir_g)
        e_dep = _edge_mag(_fill_holes_depth(dep_v, valid))
        if ir_s.ndim == 3:
            d01 = float(np.abs(ir_s[:, :, 0].astype(np.int16) - ir_s[:, :, 1]).mean())
            d12 = float(np.abs(ir_s[:, :, 1].astype(np.int16) - ir_s[:, :, 2]).mean())
            ir_ch_stats.append((d01, d12))

        for tag, e in (("dep", e_dep), ("ir", e_ir)):
            surf, zpos = _surface(e_rgb, e, max_shift)
            zero_pos = zpos
            acc[tag] = surf if acc[tag] is None else acc[tag] + surf
            # 逐图峰值（相对零偏移位置）
            mn, mx, _ml, ml = cv2.minMaxLoc(surf)
            conf = float(mx - np.median(surf))
            peaks[tag].append((ml[0] - zpos[0], ml[1] - zpos[1], conf))
        if i % 50 == 0:
            print(f"[probe] {i}/{len(fhd)}", flush=True)

    print(f"\n[probe] IR 通道差（|c0-c1|, |c1-c2| 均值）: "
          f"{np.mean([a for a, _ in ir_ch_stats]):.2f} / {np.mean([b for _, b in ir_ch_stats]):.2f} "
          f"→ {'灰度复制' if np.mean([a for a, _ in ir_ch_stats]) < 1 else '三通道不同（伪彩色或双光融合图）'}")

    for tag in ("dep", "ir"):
        arr = np.asarray(peaks[tag], np.float32)
        if arr.size == 0:
            continue
        print(f"\n=== {tag} 逐图估计 ===")
        print(f"  dx: p5={np.percentile(arr[:,0],5):.1f} p50={np.percentile(arr[:,0],50):.1f} "
              f"p95={np.percentile(arr[:,0],95):.1f} | dy: p5={np.percentile(arr[:,1],5):.1f} "
              f"p50={np.percentile(arr[:,1],50):.1f} p95={np.percentile(arr[:,1],95):.1f}")
        print(f"  峰值−中位(conf): p50={np.percentile(arr[:,2],50):.3f} p95={np.percentile(arr[:,2],95):.3f}")
        # 堆叠面
        surf = acc[tag] / len(fhd)
        mn, mx, _ml, ml = cv2.minMaxLoc(surf)
        zx, zy = zero_pos
        # 在零偏移附近 ±5 找局部峰，与全局峰对比
        win = surf[max(0, zy - 5):zy + 6, max(0, zx - 5):zx + 6]
        print(f"  堆叠面：全局峰 ({ml[0]-zx:+d}, {ml[1]-zy:+d}) 值={mx:.4f}；"
              f"零偏移附近峰值={win.max():.4f}")
        # 峰形（沿 dx 剖面）
        row = surf[ml[1], :]
        print(f"  堆叠面 dx 剖面（峰附近 ±6）: "
              + " ".join(f"{row[j]:+.3f}" for j in range(max(0, ml[0]-3), min(row.size, ml[0]+4))))
        np.save(_CODE / "runs" / "d0_audit" / f"stack_{tag}.npy", surf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--work-w", type=int, default=480)
    ap.add_argument("--max-shift", type=int, default=40)
    ap.add_argument("--labels", default="")
    args = ap.parse_args()
    (_CODE / "runs" / "d0_audit").mkdir(parents=True, exist_ok=True)
    probe(Path(args.root), args.n, args.work_w, args.max_shift,
          Path(args.labels) if args.labels else None)


if __name__ == "__main__":
    main()
