# -*- coding: utf-8 -*-
"""
vis_boxes —— 数据标注框可视化（检查标签/对齐/数据质量）。

对每个样本把赛题 txt 标签的 bbox 画到三种模态上（RGB 彩色 / IR 灰度伪彩 / Depth 伪彩），
水平拼接保存为一张图，便于目检：
  * 框是否贴合目标（标签质量）
  * 三模态框位是否一致（对齐情况）
  * 类别颜色：每类固定颜色（12 类，按 default_table 映射）

用法：
    python common/vis_boxes.py --root <数据根> --stems 00000008,000016 --out vis_out
    python common/vis_boxes.py --root <数据根> --all --max-n 10 --out vis_out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np

import models_config as MC

# 12 类固定配色（BGR），与 CLASS_NAMES 顺序一致
_CLASS_COLORS = [
    (255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0),
    (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255),
    (255, 0, 255), (255, 0, 128),
]
_SCALE = 1.0  # 输出缩放（大图可 0.5 减小体积）


def _read(path: Path, kind: str) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(path)
    if img.ndim == 3 and kind == "rgb":
        return img
    if img.ndim == 3:
        img = img[..., 0]
    return img


def _depth_pseudo(depth: np.ndarray) -> np.ndarray:
    """16bit 毫米 → 伪彩 BGR（无效区黑）。"""
    d = depth.astype(np.float32)
    m = float(d[d > 1].mean()) if bool((d > 1).any()) else 1.0
    d[d <= 1] = np.nan
    d = np.nan_to_num(d, nan=0.0) / max(float(np.nanmax(d)), 1.0)
    vis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth <= 1] = 0
    return vis


def _draw_boxes(img: np.ndarray, labels_path: Path, cls_names) -> np.ndarray:
    """按赛题 txt 画框（BGR 原地）。"""
    if not labels_path.exists():
        return img
    H, W = img.shape[:2]
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = line.split()
            if len(v) < 5:
                continue
            cls = int(float(v[0]))
            cx, cy, w, h = (float(x) for x in v[1:5])
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            name = cls_names[cls] if cls < len(cls_names) else str(cls)
            cv2.putText(img, name, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def visualize_sample(sample, out_dir: Path, cls_names=None, save=True):
    """一个样本 → 三模态画框拼接图。返回拼接图 (np.ndarray)。"""
    cls_names = cls_names or MC.CLASS_NAMES
    rgb = _read(sample.img["rgb"], "rgb")
    ir = _read(sample.img["ir"], "ir")
    dep = _read(sample.img["depth"], "depth")

    def norm_size(a: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        if a.shape[:2] != (H, W):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
        return a

    ir_bgr = cv2.cvtColor(norm_size(ir), cv2.COLOR_GRAY2BGR)
    dep_bgr = _depth_pseudo(norm_size(dep))
    rgb_d = _draw_boxes(rgb.copy(), sample.label, cls_names)
    ir_d = _draw_boxes(ir_bgr, sample.label, cls_names)
    dep_d = _draw_boxes(dep_bgr, sample.label, cls_names)

    canvas = np.concatenate([rgb_d, ir_d, dep_d], axis=1)   # 3 倍宽
    if _SCALE != 1.0:
        canvas = cv2.resize(canvas, None, fx=_SCALE, fy=_SCALE)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", canvas)
        if ok:
            buf.tofile(str(out_dir / f"{sample.stem}_vis_ir_depth.jpg"))
    return canvas


def pair_by_dirs(root: Path, use_limits: bool = True) -> list:
    """
    目录约定配对（文件名无模态关键字的布局，如 visible/infrared/depth 或 V/T/D 平铺）：
      按目录名找到各模态目录 → 同名 stem 配对 → 构造 Sample。
    支持 root 下平铺目录，或 root/<split>/<mod> 两层结构。
    """
    from common.scan_data import Sample
    mod_aliases = {
        "rgb": ("visible", "rgb", "color", "V"),
        "ir": ("infrared", "ir", "thermal", "T"),
        "depth": ("depth", "d", "D"),
    }
    label_aliases = ("labels", "label", "GT", "gt")

    def find_mod_dir(base: Path, aliases) -> Optional[Path]:
        for a in aliases:
            p = base / a
            if p.is_dir():
                return p
        return None

    # 1) 探测布局：平铺（root/visible ...）或两层（root/<split>/visible ...）
    candidates = [root]
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if find_mod_dir(sub, mod_aliases["rgb"]) is not None:
            candidates.append(sub)
    out, seen = [], set()
    for base in candidates:
        rgb_dir = find_mod_dir(base, mod_aliases["rgb"])
        ir_dir = find_mod_dir(base, mod_aliases["ir"])
        dep_dir = find_mod_dir(base, mod_aliases["depth"])
        lab_dir = find_mod_dir(base, label_aliases)
        if not (rgb_dir and ir_dir and dep_dir):
            continue
        stems = sorted({p.stem for p in rgb_dir.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")})
        for stem in stems:
            if stem in seen:
                continue
            seen.add(stem)

            def find_in(d: Path):
                for ext in (".png", ".jpg", ".jpeg"):
                    p = d / (stem + ext)
                    if p.exists():
                        return p
                return None

            lab = None
            if lab_dir is not None:
                for ext in (".txt",):
                    p = lab_dir / (stem + ext)
                    if p.exists():
                        lab = p
                        break
            if not use_limits:
                pass
            out.append(Sample(stem=stem, img={
                "rgb": find_in(rgb_dir), "ir": find_in(ir_dir), "depth": find_in(dep_dir),
            }, label=lab))
    return out


def main() -> None:
    from common import scan_data as SD
    ap = argparse.ArgumentParser(description="三模态标注框可视化")
    ap.add_argument("--root", type=str, default=str(MC.DATA_ROOT))
    ap.add_argument("--stems", type=str, default="", help="逗号分隔样本名（留空则 --all）")
    ap.add_argument("--all", action="store_true", help="处理全部采样样本")
    ap.add_argument("--max-n", type=int, default=10, help="--all 时最多处理数量")
    ap.add_argument("--out", type=str, default="vis_out")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"[vis_boxes] 数据根不存在: {root}")
    samples = pair_by_dirs(root)

    wanted: List = []
    if args.stems:
        wanted = [s for s in samples if s.stem in args.stems.split(",")]
    elif args.all:
        wanted = samples[:args.max_n]
    if not wanted:
        raise SystemExit("[vis_boxes] 未匹配到样本（用 --stems 指定名字或 --all）")

    out_dir = Path(args.out).expanduser().resolve()
    n = 0
    for s in wanted:
        if s.img.get("rgb") and s.img.get("ir") and s.img.get("depth"):
            visualize_sample(s, out_dir)
            n += 1
            print(f"[vis_boxes] {s.stem} -> {out_dir / (s.stem + '_vis_ir_depth.jpg')}")
    print(f"[vis_boxes] done: {n} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
