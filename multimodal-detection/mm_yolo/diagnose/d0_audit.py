# -*- coding: utf-8 -*-
"""D0 数据审计（方案 v4 §7）——数据到手当天跑一次，产出决定后续所有配方的报告。

审计内容：
  1. 规模与一致性：三模态配对、每模态分辨率是否一致（错位风险的第一道门）
  2. 模态格式与语义：depth 的 dtype/值域/无效比例（决定"相对深度 vs 米制"）
  3. 条件轴：照度（RGB 低分位亮度）、清晰度、拥挤度、目标尺度 → 分位切片定义
  4. 对齐：逐图估计 + 全局稳健拟合 + 置信度分布 + 三对组合（depth 到底对齐谁）
  5. 残影指标：depth 强边而 RGB 无支撑边的比例
  6. 类别与尺度：长尾、每类尺度、每图框数（>100 框比例 → 是否触发框预算问题）
  7. 结论：直接给出应写入 config 的具体值 + 最坏切片阈值

用法：
  python code/mm_yolo/diagnose/d0_audit.py --root "<数据根>" --split train --workers 8 \
      --out code/runs/d0_audit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent.parent
for _p in (str(_CODE / "mm_yolo"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from align import estimate_shift, _edge_mag, _fill_holes_depth          # noqa: E402
from io_utils import imread_unicode                                     # noqa: E402

MODS = ("visible", "infrared", "depth")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------- 单样本

def _read_any(path: Path):
    """读图：必须走 imread_unicode —— 赛题数据根目录含中文，cv2.imread 会全部失败。"""
    return imread_unicode(path)


def audit_one(args) -> Optional[Dict]:
    stem, paths, lab_path, work_w = args
    out: Dict = {"stem": stem}
    try:
        rgb = _read_any(Path(paths["visible"]))
        ir = _read_any(Path(paths["infrared"]))
        dep = _read_any(Path(paths["depth"]))
    except Exception as exc:                                  # noqa: BLE001
        return {"stem": stem, "error": str(exc)}
    if rgb is None or ir is None or dep is None:
        return {"stem": stem, "error": "读取失败"}

    out["shape_visible"] = list(rgb.shape)
    out["shape_infrared"] = list(ir.shape)
    out["shape_depth"] = list(dep.shape)
    out["depth_dtype"] = str(dep.dtype)
    out["shapes_equal"] = bool(rgb.shape[:2] == ir.shape[:2] == dep.shape[:2])
    # 模态存储形式（本数据集实测混了两种来源：640×360 的 jpg 组 vs 1920×1080 的 png 组）
    out["depth_channels"] = int(dep.shape[2]) if dep.ndim == 3 else 1
    out["depth_metric"] = bool(dep.dtype == np.uint16 and int(dep.max()) > 1000)
    if ir.ndim == 3:
        ch_eq = bool(np.array_equal(ir[:, :, 0], ir[:, :, 1]) and np.array_equal(ir[:, :, 1], ir[:, :, 2]))
    else:
        ch_eq = True
    out["ir_is_gray"] = ch_eq
    out["ext_visible"] = Path(paths["visible"]).suffix.lower()
    out["ext_depth"] = Path(paths["depth"]).suffix.lower()

    H, W = rgb.shape[:2]
    s = float(work_w) / float(W)
    if s < 1.0:
        size = (max(8, int(round(W * s))), max(8, int(round(H * s))))
        rgb_s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        ir_s = cv2.resize(ir, size, interpolation=cv2.INTER_AREA)
        dep_s = cv2.resize(dep, size, interpolation=cv2.INTER_NEAREST)
    else:
        rgb_s, ir_s, dep_s = rgb, ir, dep

    bgr = rgb_s[:, :, :3] if rgb_s.ndim == 3 else cv2.cvtColor(rgb_s, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ir_g = ir_s[:, :, 0] if ir_s.ndim == 3 else ir_s
    if dep_s.ndim == 3:
        dep_v = dep_s[:, :, 0]
    else:
        dep_v = dep_s

    # ---- 条件轴 ----
    out["lum_mean"] = float(gray.mean())
    out["lum_p5"] = float(np.percentile(gray, 5))
    out["lum_p50"] = float(np.percentile(gray, 50))
    out["ir_mean"] = float(ir_g.mean())
    out["ir_std"] = float(ir_g.std())
    out["sharp_rgb"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # ---- depth 语义 ----
    valid = dep_v > 0
    out["depth_valid_ratio"] = float(valid.mean())
    if valid.any():
        vv = dep_v[valid].astype(np.float32)
        out["depth_p5"] = float(np.percentile(vv, 5))
        out["depth_p50"] = float(np.percentile(vv, 50))
        out["depth_p95"] = float(np.percentile(vv, 95))
        out["depth_max"] = float(vv.max())
    else:
        out.update(depth_p5=0.0, depth_p50=0.0, depth_p95=0.0, depth_max=0.0)

    # ---- 残影 / 模态关系 ----
    e_rgb = _edge_mag(gray)
    e_ir = _edge_mag(ir_g)
    e_dep = _edge_mag(_fill_holes_depth(dep_v.astype(np.float32), valid))
    d_strong = e_dep > max(0.2, float(np.quantile(e_dep[valid], 0.9)) if valid.any() else 0.2)
    r_weak = e_rgb < 0.5 * float(np.median(e_rgb[e_rgb > 0]) if (e_rgb > 0).any() else 0.1)
    out["ghost_ratio"] = float((d_strong & r_weak).mean())
    # 边缘相关性（对齐前的粗指标）
    def _corr(a, b):
        a = a.ravel(); b = b.ravel()
        a = a - a.mean(); b = b - b.mean()
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / d) if d > 1e-9 else 0.0
    out["edge_corr_rgb_ir"] = _corr(e_rgb, e_ir)
    out["edge_corr_rgb_dep"] = _corr(e_rgb, e_dep)
    out["edge_corr_ir_dep"] = _corr(e_ir, e_dep)

    # ---- 对齐（depth→RGB；必要时也测 IR→RGB 作对照）----
    for tag, src, val in (("dep", dep_v.astype(np.float32), valid),
                          ("ir", ir_g.astype(np.float32), None)):
        try:
            est = estimate_shift(gray, src, val, max_shift=60.0, work_width=work_w)
            for k in ("dx", "dy", "conf"):
                out[f"align_{tag}_{k}"] = float(est[k])
        except Exception as exc:                              # noqa: BLE001
            out[f"align_{tag}_err"] = str(exc)

    # ---- 标签 ----
    if lab_path and Path(lab_path).exists():
        rows = [l.split() for l in Path(lab_path).read_text().strip().splitlines() if l.strip()]
        out["n_box"] = len(rows)
        out["classes"] = sorted({int(r[0]) for r in rows})
        if rows:
            wh = np.array([[float(r[3]), float(r[4])] for r in rows], np.float32)
            out["area_med"] = float(np.median(wh[:, 0] * wh[:, 1]))
            out["area_p10"] = float(np.percentile(wh[:, 0] * wh[:, 1], 10))
    else:
        out["n_box"] = None
    return out


# ---------------------------------------------------------------- 主流程

def collect(root: Path, split: str, lab_dir: Optional[Path]) -> List[tuple]:
    jobs = []
    names = None
    for m in MODS:
        d = root / m
        if not d.is_dir():
            raise SystemExit(f"缺少模态目录: {d}")
        files = {p.stem: p for p in d.iterdir() if p.suffix.lower() in EXTS}
        names = set(files) if names is None else (names & set(files))
        jobs.append(files)
    stems = sorted(names or [])
    labs = {}
    if lab_dir and lab_dir.is_dir():
        labs = {p.stem: p for p in lab_dir.iterdir() if p.suffix == ".txt"}
    return [(s, {m: str(jobs[i][s]) for i, m in enumerate(MODS)}, str(labs.get(s, "")), 480)
            for s in stems]


def _q(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], np.float64)
    if a.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean()),
                                                               "min": float(a.min()),
                                                               "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="含 visible/infrared/depth 的目录")
    ap.add_argument("--labels", default="", help="标签目录（缺省 = root/labels）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(_CODE / "runs" / "d0_audit"))
    args = ap.parse_args()

    root = Path(args.root)
    lab = Path(args.labels) if args.labels else (root / "labels")
    jobs = collect(root, args.split, lab)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[D0] 样本 {len(jobs)} 个（三模态配对后）| 标签目录 {lab} | workers={args.workers}", flush=True)

    rows: List[Dict] = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(audit_one, jobs, chunksize=8), 1):
            if r:
                rows.append(r)
            if i % 200 == 0:
                print(f"[D0] 已处理 {i}/{len(jobs)}", flush=True)
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    print(f"[D0] 成功 {len(ok)}，失败 {len(err)}", flush=True)

    rep: Dict = {"n_total": len(rows), "n_ok": len(ok), "n_error": len(err),
                 "errors": err[:20]}

    # ---- 一致性 ----
    rep["shapes_equal_ratio"] = float(np.mean([r["shapes_equal"] for r in ok]))
    rep["shape_visible"] = sorted({tuple(r["shape_visible"]) for r in ok})[:6]
    rep["shape_depth"] = sorted({tuple(r["shape_depth"]) for r in ok})[:6]
    rep["depth_dtype"] = sorted({r["depth_dtype"] for r in ok})
    rep["depth_channels"] = {int(k): int(v) for k, v in
                             zip(*np.unique([r["depth_channels"] for r in ok], return_counts=True))}
    rep["depth_metric_ratio"] = float(np.mean([r["depth_metric"] for r in ok]))
    rep["ir_is_gray_ratio"] = float(np.mean([r["ir_is_gray"] for r in ok]))
    _ext = [f'{r["ext_visible"]}|{r["ext_depth"]}' for r in ok]
    rep["ext_pairs"] = {}
    for e in set(_ext):
        rep["ext_pairs"][e] = _ext.count(e)

    # ---- 模态语义 ----
    rep["depth_valid_ratio"] = _q([r["depth_valid_ratio"] for r in ok])
    rep["depth_p50"] = _q([r["depth_p50"] for r in ok])
    rep["depth_max"] = _q([r["depth_max"] for r in ok])
    rep["ghost_ratio"] = _q([r["ghost_ratio"] for r in ok])
    rep["edge_corr"] = {k: _q([r[k] for r in ok]) for k in
                        ("edge_corr_rgb_ir", "edge_corr_rgb_dep", "edge_corr_ir_dep")}

    # ---- 条件轴 ----
    rep["lum_p50"] = _q([r["lum_p50"] for r in ok])
    rep["lum_mean"] = _q([r["lum_mean"] for r in ok])
    rep["sharp_rgb"] = _q([r["sharp_rgb"] for r in ok])
    rep["n_box"] = _q([r["n_box"] for r in ok if r.get("n_box") is not None])
    rep["over_100_box"] = int(sum(1 for r in ok if (r.get("n_box") or 0) > 100))

    # ---- 对齐 ----
    for tag in ("dep", "ir"):
        dx = [r.get(f"align_{tag}_dx") for r in ok]
        dy = [r.get(f"align_{tag}_dy") for r in ok]
        cf = [r.get(f"align_{tag}_conf") for r in ok]
        hi = [i for i, c in enumerate(cf) if c is not None and c >= 0.25]
        rep[f"align_{tag}"] = {
            "dx": _q(dx), "dy": _q(dy), "conf": _q(cf),
            "n_conf_ge_0.25": len(hi),
            "global_dx_median_hi": float(np.median([dx[i] for i in hi])) if hi else None,
            "global_dy_median_hi": float(np.median([dy[i] for i in hi])) if hi else None,
            "global_dx_median_all": float(np.median([d for d in dx if d is not None])) if dx else None,
            "global_dy_median_all": float(np.median([d for d in dy if d is not None])) if dy else None,
        }

    # ---- 类别 ----
    cls_cnt: Dict[int, int] = {}
    for r in ok:
        for c in (r.get("classes") or []):
            cls_cnt[c] = cls_cnt.get(c, 0) + 1
    rep["class_image_count"] = dict(sorted(cls_cnt.items()))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"audit_{args.split}.json").write_text(
        json.dumps({"summary": rep, "per_image": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[D0] 明细已写入 {out_dir / f'audit_{args.split}.json'}", flush=True)

    # ---- 控制台摘要 ----
    print("\n=== D0 摘要 ===")
    print(f"配对成功 {len(ok)}/{len(rows)}；三模态分辨率一致比例 {rep['shapes_equal_ratio']:.3f}")
    print(f"visible 形状 {rep['shape_visible']}；depth 形状 {rep['shape_depth']}；depth dtype {rep['depth_dtype']}")
    print(f"depth 有效比例 {rep['depth_valid_ratio']}")
    print(f"depth 中位值 {rep['depth_p50']}；最大值 {rep['depth_max']}")
    print(f"残影比例 {rep['ghost_ratio']}")
    print(f"照度（RGB p50）{rep['lum_p50']}")
    print(f"清晰度（Laplacian var）{rep['sharp_rgb']}")
    print(f"每图框数 {rep['n_box']}；>100 框的图 {rep['over_100_box']} 张")
    for tag in ("dep", "ir"):
        a = rep[f"align_{tag}"]
        print(f"对齐 {tag}: 高置信样本 {a['n_conf_ge_0.25']}；全局 dx={a['global_dx_median_hi']} "
              f"dy={a['global_dy_median_hi']}；逐图 dx {a['dx']}")
    print(f"含该类的图片数 {rep['class_image_count']}")


if __name__ == "__main__":
    main()
