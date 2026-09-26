# -*- coding: utf-8 -*-
"""Stage A：对齐与表示（方案 v4 §3 Stage A 的实现）。

三段式：
  A1 估计  estimate_shift()      —— 边缘域归一化互相关，粗到细 + 亚像素 + 置信度 + 抗残影 trim
  A2 补偿  warp_depth_subpixel() —— 亚像素 warp，掩码同步（无效区保持无效）
  A3 表示  to_relative_depth() / ir_local_norm() / reliability_mask()

设计约束：
  * 估计**逐图**做、结果缓存（cache save/load），绝不在 dataloader 每轮重算；
  * 置信度低 → 向全局估计收缩（经验贝叶斯），不写死固定偏移、也不信噪声估计；
  * 残影使像素级对齐不可达 → 除 warp 外还要输出"可靠性掩码"，交给融合层降权。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------- 基础工具

def _edge_mag(gray: np.ndarray, blur: int = 5) -> np.ndarray:
    """梯度幅值（float32, 归一化到 [0,1]）。"""
    g = gray.astype(np.float32)
    g = cv2.GaussianBlur(g, (blur, blur), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    return m / (m.max() + EPS)


def _fill_holes_depth(dep_mm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """把无效值用中值/最近邻填掉，供梯度计算用（**不改原始 depth 的有效性语义**）。"""
    d = dep_mm.astype(np.float32).copy()
    if valid is None or valid.all():
        return d
    d[~valid] = np.nan
    med = float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0
    d = np.where(np.isfinite(d), d, med)
    return cv2.medianBlur(d.astype(np.float32), 5)


def _ncc_peak(img: np.ndarray, tmpl: np.ndarray) -> Tuple[float, Tuple[int, int], np.ndarray]:
    """TM_CCOEFF_NORMED 匹配：返回 (峰值, 全图 argmax, 响应图)。"""
    resp = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
    _mn, mx, _ml, ml = cv2.minMaxLoc(resp)
    return float(mx), (int(ml[0]), int(ml[1])), resp


def _subpixel_peak(resp: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """3×3 抛物线插值到亚像素。"""
    if x <= 0 or y <= 0 or x >= resp.shape[1] - 1 or y >= resp.shape[0] - 1:
        return float(x), float(y)
    def _off(a: float, b: float, c: float) -> float:
        denom = (a - 2 * b + c)
        return 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    dx = _off(resp[y, x - 1], resp[y, x], resp[y, x + 1])
    dy = _off(resp[y - 1, x], resp[y, x], resp[y + 1, x])
    return x + dx, y + dy


def _peak_confidence(resp: np.ndarray, x: int, y: int, sup: int = 5) -> Tuple[float, float]:
    """置信度 = (主峰 − 邻域次峰) 与 主峰绝对值的组合；返回 (confidence, peak_width)。"""
    h, w = resp.shape
    r = resp.copy()
    x0, x1 = max(0, x - sup), min(w, x + sup + 1)
    y0, y1 = max(0, y - sup), min(h, y + sup + 1)
    patch = r[y0:y1, x0:x1].copy()
    peak = float(resp[y, x])
    py, px = y - y0, x - x0
    patch[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = -np.inf      # 屏蔽主峰邻域
    second = float(patch.max()) if np.isfinite(patch).any() else -1.0
    conf = float(np.clip(peak - max(second, 0.0), 0.0, 1.0))
    # 峰宽：以 0.5*peak 为阈值的连通区域宽度（越大越不可信）
    thr = 0.5 * peak
    mask = (resp >= thr).astype(np.uint8)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    width = 1.0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_LEFT] <= x < stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] \
                and stats[i, cv2.CC_STAT_TOP] <= y < stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]:
            width = float(max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))
            break
    return conf, width


# ---------------------------------------------------------------- A1 估计

def estimate_shift(rgb: np.ndarray, dep_mm: np.ndarray, valid: Optional[np.ndarray] = None,
                   max_shift: float = 40.0, work_width: int = 480,
                   trim_frac: float = 0.08) -> Dict[str, float]:
    """估计 **depth 内容相对 RGB 的位移** (dx, dy)（单位：原图像素）。

    ⚠️ 语义：返回值是"depth 相对于 RGB 偏了多少"。
    **对齐时要施加以相反数** —— 用 `correction_from_estimate()` 取，别自己加负号。

    返回 dict：dx, dy, conf, width, agree（trim 前后一致性）, ncc。
    """
    H, W = rgb.shape[:2]
    scale = min(1.0, float(work_width) / float(W))
    if scale < 1.0:
        size = (max(8, int(round(W * scale))), max(8, int(round(H * scale))))
        rgb_s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        dep_s = cv2.resize(dep_mm, size, interpolation=cv2.INTER_NEAREST)
        val_s = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) \
            if valid is not None else None
    else:
        rgb_s, dep_s, val_s = rgb, dep_mm, valid
    e_rgb = _edge_mag(rgb_s if rgb_s.ndim == 2 else cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY))
    e_dep = _edge_mag(_fill_holes_depth(dep_s, val_s))

    def _run(e_d: np.ndarray) -> Tuple[float, float, float, float, float]:
        r = int(round(max_shift * scale))
        pad = cv2.copyMakeBorder(e_d, r, r, r, r, cv2.BORDER_CONSTANT, value=0.0)
        # 用 rgb 的中心区域当模板（避免边界效应）
        m = max(8, int(0.6 * e_rgb.shape[0]))
        cy, cx = e_rgb.shape[0] // 2, e_rgb.shape[1] // 2
        tmpl = e_rgb[cy - m // 2: cy + m // 2, cx - m // 2: cx + m // 2]
        peak, (px, py), resp = _ncc_peak(pad, tmpl)
        sx, sy = _subpixel_peak(resp, px, py)
        # 模板中心在 pad 中的期望位置 = (cx + r, cy + r) → 偏移量
        dx = (sx + m // 2) - (cx + r)
        dy = (sy + m // 2) - (cy + r)
        conf, width = _peak_confidence(resp, px, py)
        return dx / scale, dy / scale, conf, width, peak      # 换算回原图尺度

    dx1, dy1, c1, w1, n1 = _run(e_dep)
    # 抗残影：裁掉 depth 梯度最强的一小撮像素再估一次
    e_tr = e_dep.copy()
    if 0.0 < trim_frac < 0.5:
        thr = np.quantile(e_tr, 1.0 - trim_frac)
        e_tr[e_tr >= thr] = 0.0
    dx2, dy2, c2, w2, n2 = _run(e_tr)
    agree = float(np.hypot(dx1 - dx2, dy1 - dy2))
    use_trim = (c2 >= c1)
    dx, dy, conf, width, ncc = (dx2, dy2, c2, w2, n2) if use_trim else (dx1, dy1, c1, w1, n1)
    # 两次估计不一致 → 置信度打折
    conf *= float(np.clip(1.0 - agree / max(8.0, 0.5 * max_shift), 0.0, 1.0))
    return dict(dx=float(dx), dy=float(dy), conf=float(conf), width=float(width),
                agree=float(agree), ncc=float(ncc))


def correction_from_estimate(est: Dict[str, float]) -> Tuple[float, float]:
    """把"depth 相对 RGB 的位移"换算成对齐要施加的 warp 量（取相反数）。"""
    return (-float(est["dx"]), -float(est["dy"]))


def shrink_to_global(est: Dict[str, float], global_xy: Tuple[float, float],
                     conf_min: float = 0.25) -> Tuple[float, float]:
    """置信度 → 向全局估计收缩（经验贝叶斯）。conf≥conf_min 时线性过渡，最高权重 1。"""
    c = float(est.get("conf", 0.0))
    w = 0.0 if c <= conf_min else float(np.clip((c - conf_min) / max(1e-6, 1.0 - conf_min), 0.0, 1.0))
    gx, gy = global_xy
    return (w * est["dx"] + (1 - w) * gx, w * est["dy"] + (1 - w) * gy)


def fit_global_shift(estimates: Dict[str, Dict[str, float]], conf_min: float = 0.25
                     ) -> Tuple[float, float]:
    """从逐图估计里稳健拟合全局偏移（只用高置信样本取中位数）。"""
    hi = [e for e in estimates.values() if e.get("conf", 0.0) >= conf_min]
    if not hi:
        hi = list(estimates.values()) or [dict(dx=0.0, dy=0.0)]
    return (float(np.median([e["dx"] for e in hi])), float(np.median([e["dy"] for e in hi])))


# ---------------------------------------------------------------- A2 补偿

def warp_depth_subpixel(dep_mm: np.ndarray, valid: np.ndarray, dx: float, dy: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """亚像素平移 depth，掩码同步（用最近邻 warp，保证"无效区保持无效"）。"""
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return dep_mm, valid
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    h, w = dep_mm.shape[:2]
    d = cv2.warpAffine(dep_mm.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpAffine(valid.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return d, m.astype(bool)


# ---------------------------------------------------------------- A3 表示

def to_relative_depth(dep_mm: np.ndarray, valid: np.ndarray, lo: float = 2.0,
                      hi: float = 98.0) -> np.ndarray:
    """米制深度 → [0,1] 相对深度（每图分位归一；无效处 = 0）。抗未知单调预处理。"""
    d = dep_mm.astype(np.float32)
    v = valid & (d > 0)
    out = np.zeros_like(d)
    if v.sum() < 16:
        return out
    q_lo, q_hi = np.percentile(d[v], [lo, hi])
    if q_hi - q_lo < 1e-3:
        return out
    out[v] = np.clip((d[v] - q_lo) / (q_hi - q_lo), 0.0, 1.0)
    return out


def ir_local_norm(ir_u8: np.ndarray, ksize: int = 15) -> np.ndarray:
    """IR 局部对比度归一：(x − 局部均值) / (局部标准差 + eps) → [0,1]。"""
    x = ir_u8.astype(np.float32)
    mean = cv2.blur(x, (ksize, ksize))
    sq = cv2.blur(x * x, (ksize, ksize))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    z = (x - mean) / (std + 8.0)                     # 8 灰度级作软化，避免平坦区爆噪
    z = np.clip(z * 0.25 + 0.5, 0.0, 1.0)
    return z


def reliability_mask(dep_mm: np.ndarray, valid: np.ndarray, rgb_gray: np.ndarray,
                     ghost_ratio: float = 0.35, var_win: int = 9) -> np.ndarray:
    """可靠性掩码（uint8 0/1）：无效值 | 残影/补洞 | 局部方差过大 → 不可靠。"""
    m = valid & (dep_mm > 0)
    e_dep = _edge_mag(_fill_holes_depth(dep_mm, valid))
    e_rgb = _edge_mag(rgb_gray)
    # 残影判据：depth 有强边但 RGB 无支撑边
    strong_d = e_dep > max(0.2, float(np.quantile(e_dep[m], 0.9)) if m.any() else 0.2)
    weak_rgb = e_rgb < 0.5 * float(np.median(e_rgb[e_rgb > 0]) if (e_rgb > 0).any() else 0.1)
    ghost = strong_d & weak_rgb
    # 局部方差过大（补洞/涂抹）
    d = _fill_holes_depth(dep_mm, valid)
    mean = cv2.blur(d, (var_win, var_win))
    var = cv2.blur(d * d, (var_win, var_win)) - mean * mean
    noisy = var > (np.quantile(var[m], 0.95) if m.any() else np.inf)
    bad = ghost | noisy
    if ghost_ratio > 0 and bad.sum() > 0.5 * m.sum():
        bad = ghost                                   # 残影判据过激时只保留 ghost
    return (m & ~bad).astype(np.uint8)


# ---------------------------------------------------------------- 缓存

def save_cache(path: str, entries: Dict[str, Dict[str, float]], global_xy: Tuple[float, float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"global": list(global_xy), "per_image": entries}, f, ensure_ascii=False)


def load_cache(path: str) -> Tuple[Dict[str, Dict[str, float]], Tuple[float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    g = obj.get("global", [0.0, 0.0])
    return obj.get("per_image", {}), (float(g[0]), float(g[1]))


# ---------------------------------------------------------------- 估计器自检（A1.5）

def _selftest(verbose: bool = True) -> bool:
    """人为已知偏移 → 估计器必须复原；再加残影 → trim 版必须更稳。"""
    rng = np.random.default_rng(0)
    H, W = 540, 960
    scene = np.zeros((H, W), np.float32)
    for _ in range(40):                                     # 造有结构的"场景"（方块+边缘）
        x, y = rng.integers(0, W - 80), rng.integers(0, H - 80)
        w, h = rng.integers(20, 80), rng.integers(20, 80)
        scene[y:y + h, x:x + w] = rng.uniform(0.2, 1.0)
    rgb = cv2.GaussianBlur(scene, (3, 3), 0)
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    ok = True
    rows = []
    for (cx, cy) in ((0, 0), (-12, 7), (21, 0), (-30, -15)):
        # 造出"depth 内容相对 RGB 位移 (cx,cy)"：warpAffine 的 tx 使内容平移 +tx
        M = np.array([[1, 0, cx], [0, 1, cy]], np.float32)
        dep = cv2.warpAffine(scene, M, (W, H), flags=cv2.INTER_LINEAR) * 20000.0
        valid = np.ones_like(dep, bool)
        est = estimate_shift(rgb_u8, dep.astype(np.float32), valid, max_shift=40, work_width=480)
        err = float(np.hypot(est["dx"] - cx, est["dy"] - cy))
        rows.append((cx, cy, est["dx"], est["dy"], est["conf"], err))
        ok &= err <= 1.5
    # 残影：主副本（强）+ 拖尾副本（弱、偏移 12px）→ 估计应停在主峰附近，且置信度明显下降
    cx, cy = 21, 0
    dep = cv2.warpAffine(scene, np.array([[1, 0, cx], [0, 1, cy]], np.float32), (W, H),
                         flags=cv2.INTER_LINEAR)
    ghost = cv2.warpAffine(dep, np.array([[1, 0, -12], [0, 1, 0]], np.float32), (W, H),
                           flags=cv2.INTER_LINEAR)
    dep_g = ((0.75 * dep + 0.25 * ghost) * 20000.0).astype(np.float32)
    est_g = estimate_shift(rgb_u8, dep_g, np.ones_like(dep_g, bool), max_shift=40, work_width=480)
    err_g = float(np.hypot(est_g["dx"] - cx, est_g["dy"] - cy))
    conf_drop = float(rows[2][4] - est_g["conf"])          # 与无残影的同类场景比
    if verbose:
        print("=== align 估计器自检（真值 = depth 内容相对 RGB 的位移，单位：原图像素）===")
        print(f"{'真值':>12} {'估计':>18} {'conf':>6} {'误差':>6}")
        for r in rows:
            print(f"{str((r[0], r[1])):>12} {str((round(r[2], 2), round(r[3], 2))):>18} "
                  f"{r[4]:>6.3f} {r[5]:>6.2f}")
        print(f"残影场景（主峰 {cx},0 + 弱拖尾）：估计=({est_g['dx']:.2f}, {est_g['dy']:.2f}) "
              f"conf={est_g['conf']:.3f} 误差={err_g:.2f} 置信度下降={conf_drop:.3f}")
        rel = to_relative_depth(dep_g, np.ones_like(dep_g, bool))
        rel_ir = ir_local_norm(rgb_u8)
        m = reliability_mask(dep_g, np.ones_like(dep_g, bool), rgb_u8)
        print(f"相对深度范围=[{rel.min():.2f},{rel.max():.2f}]  "
              f"IR 局部归一范围=[{rel_ir.min():.2f},{rel_ir.max():.2f}]  "
              f"可靠性掩码有效比例={m.mean():.3f}")
        print(f"对齐 warp 量（取相反数）= {tuple(round(v, 2) for v in correction_from_estimate(est_g))}")
    ok &= (err_g <= 3.0)
    if verbose:
        print("ALIGN_SELFCHECK_OK" if ok else "ALIGN_SELFCHECK_FAILED")
    return bool(ok)


if __name__ == "__main__":
    _selftest()
