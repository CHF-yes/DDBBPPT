# -*- coding: utf-8 -*-
"""
实验模型1 —— Depth↔RGB 对齐**定量标定**工具。

为什么需要
----------
`models_config.AlignConfig` 的 shift_x/shift_y 目前是一个手填常数（-22px@1920）。
但"官方数据有偏移/本地数据没有偏移"从未被定量验证过；IR-RGB 的强度相关又极弱
（实测 0.02~0.20），不能作为依据。

本工具用**边缘域**（跨模态更稳）逐样本估计最优平移：
  1. RGB  → 灰度 → Sobel 幅值 → 高斯模糊
  2. Depth→ 1/d（近处响应大）→ Sobel 幅值 → 高斯模糊
  3. 在 ±range 像素内扫描 (sx, sy)，最大化重叠区归一化互相关
  4. 汇总中位数/IQR，并给出"归一化到 ref_size(1920) 基准"的建议值

输出语义：**把 depth 平移 (sx, sy) 后与 RGB 最一致** —— 也就是要写进
`AlignConfig.shift_x/shift_y` 的值（`align_depth` 的语义正是如此）。

用法：
  python 实验模型1/check_align.py --root "<数据根>" [--imgsz 480] [--range 16]
  数据根可以是 VDT (\Train 或 \Test)、也可以是官方样例根（含 visible/infrared/depth/labels）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_CODE_ROOT / "vendor"), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

import models_config as MC                                  # noqa: E402
from common import scan_data as SD                          # noqa: E402
from common import dataset as DS                            # noqa: E402


def _edge_map(gray: np.ndarray, blur: int = 9) -> np.ndarray:
    """Sobel 幅值 + 高斯模糊（跨模态相关更稳）。"""
    import cv2
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    ex = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    ey = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    e = np.sqrt(ex * ex + ey * ey)
    return cv2.GaussianBlur(e, (blur, blur), 0)


def _corr_at(a: np.ndarray, b: np.ndarray, sx: int, sy: int) -> float:
    """b 平移 (sx,sy) 后与 a 在重叠区的归一化互相关。"""
    H, W = a.shape
    x0, x1 = max(0, sx), min(W, W + sx)
    y0, y1 = max(0, sy), min(H, H + sy)
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return -1.0
    aa = a[y0:y1, x0:x1].ravel()
    bb = b[y0 - sy:y1 - sy, x0 - sx:x1 - sx].ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    d = aa.std() * bb.std()
    return float((aa * bb).mean() / d) if d > 1e-8 else -1.0


def estimate(sample, work_w: int = 480, rng: int = 16):
    """粗到精两段搜索，返回 (最优 sx, sy, 峰相关, 0 位移相关, 原始宽, 缩放比)。"""
    import cv2
    rgb = cv2.cvtColor(DS.read_rgb_bgr(sample.img["rgb"]), cv2.COLOR_RGB2RGB
                       if False else cv2.COLOR_BGR2RGB)
    dep = DS.read_depth_mm(sample.img["depth"]).astype(np.float32)
    if rgb.shape[:2] != dep.shape[:2]:
        raise ValueError(f"RGB{rgb.shape[:2]} 与 Depth{dep.shape[:2]} 尺寸不同（跳过）")
    W0 = rgb.shape[1]
    sc = work_w / W0
    rgb_s = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    dep_s = cv2.resize(dep, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)
    valid = dep_s > 1
    inv = np.zeros_like(dep_s)
    inv[valid] = 1.0 / dep_s[valid]
    if valid.any():                                  # 归一化到 [0,1]
        lo, hi = inv[valid].min(), inv[valid].max()
        inv[valid] = (inv[valid] - lo) / max(1e-6, hi - lo)
    e_rgb = _edge_map(gray)
    e_dep = _edge_map(inv)

    def scan(cx, cy, half, step):
        best = (-2.0, cx, cy)
        for sy in range(cy - half, cy + half + 1, step):
            for sx in range(cx - half, cx + half + 1, step):
                c = _corr_at(e_rgb, e_dep, sx, sy)
                if c > best[0]:
                    best = (c, sx, sy)
        return best

    coarse = scan(0, 0, rng, max(4, rng // 8))       # 粗扫（步长 ~1/8 半径）
    fine = scan(coarse[1], coarse[2], max(2, rng // 8), 1)   # 精扫（±1/8 半径，步长 1）
    return fine[1], fine[2], fine[0], _corr_at(e_rgb, e_dep, 0, 0), W0, sc


def main():
    ap = argparse.ArgumentParser(prog="实验模型1/check_align")
    ap.add_argument("--root", required=True, help="数据根（VDT Train/Test 或官方样例根）")
    ap.add_argument("--limit", type=int, default=60, help="最多统计多少个样本")
    ap.add_argument("--work-w", type=int, default=480, help="估计时的工作宽度")
    ap.add_argument("--range", type=int, default=16, help="搜索半径（工作尺度像素）")
    ap.add_argument("--min-peak", type=float, default=0.15,
                    help="只统计峰值相关 ≥ 该值的样本（低置信样本是离群点）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    scanned = SD.scan_samples_auto(root)
    samples = scanned.get("train") or scanned.get("val") or next(iter(scanned.values()))
    samples = samples[: args.limit]
    print(f"[align] {root.name}: {len(samples)} 组样本 | 工作宽 {args.work_w} | 搜索 ±{args.range}px")

    rows = []
    for s in samples:
        try:
            sx, sy, peak, c0, W0, sc = estimate(s, args.work_w, args.range)
        except Exception as e:                          # 坏样本跳过
            print(f"  {s.stem}: 跳过（{e}）")
            continue
        # 换算回**原图**像素，并归一化到 ref_size(1920) 基准
        k = 1920.0 / (args.work_w / sc)
        rows.append((s.stem, sx, sy, peak, c0, W0, sx * 1920.0 / W0, sy * 1920.0 / W0))
        print(f"  {s.stem:<22} 原图{W0:>5}px  最优 ({sx:+.0f},{sy:+.0f})@{args.work_w}  "
              f"峰相关 {peak:+.3f}  0位移相关 {c0:+.3f}  → 等价原图 ({sx * W0 / args.work_w:+.1f},"
              f"{sy * W0 / args.work_w:+.1f})px  ref1920 基准 ({sx * 1920.0 / W0:+.1f},"
              f"{sy * 1920.0 / W0:+.1f})px")

    if not rows:
        print("[align] 无有效样本"); return
    all_rows = rows
    rows = [r for r in all_rows if r[3] >= args.min_peak]
    print(f"\n（峰值相关 ≥ {args.min_peak} 的样本 {len(rows)}/{len(all_rows)} 个参与汇总）")
    if not rows:
        rows = all_rows
    sx1920 = np.array([r[6] for r in rows]); sy1920 = np.array([r[7] for r in rows])
    peaks = np.array([r[3] for r in rows]); c0s = np.array([r[4] for r in rows])
    print("===== 汇总（换算到 ref_size=1920 基准，可直接填 AlignConfig）=====")
    print(f"  shift_x: 中位 {np.median(sx1920):+.1f}px  IQR [{np.percentile(sx1920,25):+.1f}, "
          f"{np.percentile(sx1920,75):+.1f}]  范围 [{sx1920.min():+.1f}, {sx1920.max():+.1f}]")
    print(f"  shift_y: 中位 {np.median(sy1920):+.1f}px  IQR [{np.percentile(sy1920,25):+.1f}, "
          f"{np.percentile(sy1920,75):+.1f}]  范围 [{sy1920.min():+.1f}, {sy1920.max():+.1f}]")
    print(f"  峰值相关 中位 {np.median(peaks):+.3f}（0 位移中位 {np.median(c0s):+.3f}）"
          f" → 对齐带来的相关提升 中位 {np.median(peaks - c0s):+.3f}")
    near0 = float(np.mean((np.abs(sx1920) < 2.0) & (np.abs(sy1920) < 2.0)))
    print(f"  最优位移落在 ±2px(1920 基准) 内的样本比例: {near0*100:.0f}%")
    print(f"\n  当前 config: shift_x={MC.EXPERIMENT1.hyper.align.shift_x} "
          f"shift_y={MC.EXPERIMENT1.hyper.align.shift_y} "
          f"ref_size={MC.EXPERIMENT1.hyper.align.ref_size} "
          f"mode={MC.EXPERIMENT1.hyper.align.mode}")
    print("ALIGN_CHECK_DONE")


if __name__ == "__main__":
    main()
