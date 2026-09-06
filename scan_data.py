# -*- coding: utf-8 -*-
"""
scan_data —— RGBTD 三模态数据布局自动探测 + 生成 ultralytics data.yaml

适配本项目的「目录分离」布局（与赛题样例一致）：
    <root>/
    ├── visible/      可见光 RGB 图
    ├── infrared/     红外图（单通道灰度堆叠 3 份）
    ├── depth/        深度图（16bit 单通道毫米）
    └── labels/       YOLO 标签 txt（class_id cx cy w h 归一化）

三种模态文件名一一对应（同一 base name），加载器靠目录名字符串替换配对。
同时兼容「train/val/test 分组」布局：
    <root>/train/visible/ ...   <root>/val/visible/ ...

用法：
    python scan_data.py --root 数据根目录 --out data.yaml
    python scan_data.py --root 数据根目录 --split-dirs train,val --out data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

# 赛题官方 12 类（id 0-11，顺序严格对应标签 txt 数值）
CLASS_NAMES: List[str] = [
    "person",       # 0
    "boat",         # 1
    "animal",       # 2
    "seat",         # 3
    "sign",         # 4  路牌/标语/标志
    "bicycle",      # 5  双轮车：自行车/双轮电动车
    "car",          # 6  四轮汽车
    "ball",         # 7
    "light",        # 8  路灯/室内照明灯
    "garbage_can",  # 9
    "uav",          # 10 无人机
    "tricycle",     # 11 三轮车
]

# 模态目录别名（探测时按此顺序匹配，优先级高在前）
MODALITY_DIR_ALIASES: Dict[str, List[str]] = {
    "visible": ["visible", "rgb", "color", "vis", "vis_img"],
    "infrared": ["infrared", "ir", "thermal", "tir", "ir_img"],
    "depth": ["depth", "d", "disp", "depth_img"],
}

LABEL_DIR_ALIASES: List[str] = ["labels", "label", "annotations", "anns"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Sample:
    """一组配对完成的空间对齐三模态样本。"""

    def __init__(self, stem: str, img: Dict[str, Path], label: Optional[Path] = None):
        self.stem = stem
        self.img = img          # {"visible": Path, "infrared": Path, "depth": Path}
        self.label = label      # 标签 txt Path（可为 None）


# ------------------------------------------------------------
# 目录树打印
# ------------------------------------------------------------

def print_tree(root: Path, max_depth: int = 3, prefix: str = "") -> None:
    if not root.exists():
        print(f"[!] 路径不存在: {root}")
        return
    if prefix == "":
        print(str(root))
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return
    for i, child in enumerate(entries):
        last = i == len(entries) - 1
        branch = "└── " if last else "├── "
        size = f"  ({child.stat().st_size} B)" if child.is_file() else ""
        print(f"{prefix}{branch}{child.name}{size}")
        if child.is_dir() and max_depth > 1:
            print_tree(child, max_depth - 1, prefix + ("    " if last else "│   "))


# ------------------------------------------------------------
# 模态目录探测
# ------------------------------------------------------------

def _match_dir(base: Path, aliases: List[str]) -> Optional[Path]:
    """在 base 下按别名找目录（大小写不敏感）。"""
    if not base.is_dir():
        return None
    lowered = {p.name.lower(): p for p in base.iterdir() if p.is_dir()}
    for a in aliases:
        if a.lower() in lowered:
            return lowered[a.lower()]
    return None


def find_modality_dirs(root: Path) -> Dict[str, Path]:
    """在 root 下探测 visible/infrared/depth 三个模态目录。"""
    found = {}
    for mod, aliases in MODALITY_DIR_ALIASES.items():
        p = _match_dir(root, aliases)
        if p:
            found[mod] = p
    return found


def find_label_dir(root: Path) -> Optional[Path]:
    return _match_dir(root, LABEL_DIR_ALIASES)


# ------------------------------------------------------------
# 样本配对
# ------------------------------------------------------------

def _scan_one_split(base: Path) -> List[Sample]:
    """在一个 split 目录内按 base name 配对三模态 + 标签。"""
    mod_dirs = find_modality_dirs(base)
    if "visible" not in mod_dirs:
        return []
    label_dir = find_label_dir(base)

    vis_dir = mod_dirs.get("visible")
    ir_dir = mod_dirs.get("infrared")
    dep_dir = mod_dirs.get("depth")

    def _imgs(d: Optional[Path]) -> Dict[str, Path]:
        if not d or not d.is_dir():
            return {}
        return {p.stem: p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS}

    vis = _imgs(vis_dir)
    ir = _imgs(ir_dir)
    dep = _imgs(dep_dir)
    labels = {}
    if label_dir and label_dir.is_dir():
        labels = {p.stem: p for p in label_dir.iterdir()
                  if p.is_file() and p.suffix.lower() == ".txt"}

    # 以 visible 为准（有可见光才算有效样本），标签可选
    samples = []
    for stem in sorted(vis):
        samples.append(Sample(
            stem=stem,
            img={
                "visible": vis[stem],
                "infrared": ir.get(stem),
                "depth": dep.get(stem),
            },
            label=labels.get(stem),
        ))
    return samples


def scan_samples(root: Path, split_subdirs: Optional[List[str]] = None) -> Dict[str, List[Sample]]:
    """扫描数据根，返回 {split_name: [Sample, ...]}。

    - 若根下直接有 visible/infrared/depth，则当作一个"扁平"分组（split 名为 root）
    - 若指定 --split-dirs train,val，则扫描 <root>/train、<root>/val
    - 未指定且探测到 train/val 等子目录含模态目录时自动识别
    """
    root = Path(root)
    if split_subdirs is None:
        # 自动：根下直接有 visible 目录 → 扁平；否则识别含 visible 的子目录
        direct = find_modality_dirs(root)
        if "visible" in direct:
            return {"root": _scan_one_split(root)}
        split_subdirs = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and "visible" in find_modality_dirs(p)
        )
        if not split_subdirs:
            return {}

    out = {}
    for sd in split_subdirs:
        base = root if sd == "root" else root / sd
        out[sd] = _scan_one_split(base)
        print(f"[scan] split '{sd}': {len(out[sd])} 组样本")
    return out


# ------------------------------------------------------------
# 生成 data.yaml
# ------------------------------------------------------------

def _split_dir_of(samples: List[Sample], root: Path) -> str:
    """取某 split 的 visible 目录路径（ultralytics data.yaml 的 train/val 指向该目录）。"""
    for s in samples:
        v = s.img.get("visible")
        if v:
            return str(v.parent.resolve())
    return ""


def build_data_yaml(root: Path, scan: Dict[str, List[Sample]], out_path: Path,
                    train: str = "train", val: str = "val") -> Path:
    """根据扫描结果生成 data.yaml。"""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))

    # 分组名兼容：扁平(root)时 train/val 都指向同一目录，并在注释提醒切分
    train_dir = _split_dir_of(scan.get(train, scan.get("root", [])), root)
    val_dir = _split_dir_of(scan.get(val, scan.get("root", [])), root)

    lines = [
        "# auto-generated by scan_data.py",
        f"train: {train_dir}",
        f"val: {val_dir}",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        names,
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[scan] data.yaml 写入 -> {out_path.resolve()}")
    if train_dir == val_dir and train_dir:
        print("[scan] 提示: train 与 val 指向同一目录。若数据未分组，请用 split_data.py 先做 train/val 切分。")
    return out_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RGBTD 三模态数据布局探测 + 生成 data.yaml")
    ap.add_argument("--root", type=str, required=True, help="数据根目录")
    ap.add_argument("--split-dirs", type=str, default="",
                    help="逗号分隔的分组目录名，如 train,val（默认自动探测）")
    ap.add_argument("--out", type=str, default=None, help="输出 data.yaml 路径")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[!] 数据根不存在: {root}")
        return

    print("== 目录树 ==")
    print_tree(root, max_depth=2)

    split_dirs = [s.strip() for s in args.split_dirs.split(",") if s.strip()] or None
    scan = scan_samples(root, split_dirs)

    if not scan:
        print("[!] 未扫描到任何样本。请确认根下存在 visible/（及 infrared/depth/labels）目录。")
        return

    for name, samples in scan.items():
        miss = sum(1 for s in samples if s.img.get("infrared") is None or s.img.get("depth") is None)
        lab = sum(1 for s in samples if s.label is not None)
        print(f"[scan] {name}: {len(samples)} 组, 缺红外/深度 {miss} 组, 有标签 {lab} 组")

    if args.out:
        build_data_yaml(root, scan, Path(args.out))


if __name__ == "__main__":
    main()
