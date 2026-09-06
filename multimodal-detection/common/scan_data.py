# -*- coding: utf-8 -*-
"""
scan_data —— 数据布局自动探测 + 生成 ultralytics data.yaml。

背景：赛题数据布局未知（train 目录结构、是否 train/val 分离、文件命名规则未定），
本模块提供一个"探针"：
  1) 遍历数据根目录并打印目录树（先看清结构）；
  2) 按多套常见命名规则自动把某样本的 RGB/IR/Depth 三模态 + 标签 txt 配对；
  3) 生成 ultralytics 训练所需的 data.yaml。

用法（拿到正式数据后，把 DATA_ROOT 换成实际路径）：
    python -m common.scan_data --root 你的数据根目录 --out data.yaml
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import models_config as MC


# ------------------------------------------------------------
# 命名规则表格：文件名含的"形态关键字"优先匹配，其次位置关键字
# ------------------------------------------------------------
MODALITY_HINTS: Dict[str, List[str]] = {
    "rgb": ["rgb", "visible", "color", "visible_image"],
    "ir": ["ir", "infrared", "thermal", "tir"],
    "depth": ["depth", "disp", "d_"],
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
LABEL_EXTS = {".txt"}


def _pick_stem(root: Path) -> dict:
    """对某个候选文件，判断它属于哪个模态、以及相对 stem。返回 None 若不对。"""
    return root


@dataclass
class Sample:
    """一组配对完成的空间对齐三模态样本。"""
    stem: str
    img: Dict[str, Path]        # {"rgb": Path, "ir": Path, "depth": Path}
    label: Optional[Path] = None


# ------------------------------------------------------------
# 递归目录树打印（便于人工核对布局）
# ------------------------------------------------------------

def print_dataset_tree(root: Path, max_depth: int = 3, prefix: str = "") -> None:
    """打印目录树到 max_depth 层。"""
    if not root.exists():
        print(f"[!] 路径不存在: {root}")
        return
    if prefix == "":
        print(f"{root}")
    try:
        entries = sorted([p for p in root.iterdir()], key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return
    for i, child in enumerate(entries):
        last = i == len(entries) - 1
        branch = "└── " if last else "├── "
        print(f"{prefix}{branch}{child.name}{'' if child.is_dir() else '  (' + str(child.stat().st_size) + ' B)'}")
        if child.is_dir() and max_depth > 1:
            print_dataset_tree(child, max_depth - 1, prefix + ("    " if last else "│   "))


# ------------------------------------------------------------
# 模态识别 + 样本配对
# ------------------------------------------------------------

def _classify_image(path: Path) -> Optional[str]:
    """按文件名是否含模态关键字推断它是 rgb/ir/depth 之一；否则 None。"""
    name = path.stem.lower()
    if path.suffix.lower() not in IMAGE_EXTS:
        return None
    # IR/DEDPTH 关键字优先，避免 depth 子串撞词
    order = ["depth", "ir", "rgb"]
    for mod in order:
        for hint in MODALITY_HINTS[mod]:
            if hint in name:
                return mod
    return None


def _strip_modality_tag(stem: str) -> str:
    """把 stem 中的模态标记去掉，得到"基名"用于配对（把不同模态文件对齐到同一 stem）。"""
    lower = stem.lower()
    for hints in MODALITY_HINTS.values():
        for hint in hints:
            lower = lower.replace(hint, "")
    # 清理分隔符与多下划线
    for sep in ("__", "--"):
        while sep in lower:
            lower = lower.replace(sep, "_")
    lower = lower.strip("_- .")
    return lower or stem


def scan_samples(data_root: Path, split_subdirs: Optional[list] = None) -> Dict[str, List[Sample]]:
    """
    扫描并按 `split_subdirs` 聚合样本。

    data_root 下假设形如:
        <root>/train/xxx_rgb.png, xxx_ir.png, xxx_depth.png, xxx.txt
        <root>/val/...
    若 split_subdirs=None 则默认探测 root 的直接子目录(如 train/val)；
    并将返回 {split_name: [Sample, ...]}。

    Sample.img 可能缺某一模态(如 IR 缺失)，返回后由 dataset 层按 cfg 取舍。
    """
    if split_subdirs is None:
        # 自动推断：root 的直接子目录且看起来是 split（含图片集）
        split_subdirs = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and any(c.suffix.lower() in IMAGE_EXTS for c in p.iterdir())
        )
        if not split_subdirs:
            # 退化为整个 root 当作一个分组（此时不入 train/val 拆分）
            text = "root"
            # 让 scan_all 函数处理"扁平"案例：把 root 本身当作一组合并收集
            split_subdirs = [text]

    out: Dict[str, List[Sample]] = {}
    for sd in split_subdirs:
        if sd == "root":
            base = data_root
        else:
            base = data_root / sd
        samples = _scan_one_split(base)
        out[sd] = samples
        print(f"[scan] split '{sd}': {len(samples)} 组样本")
    return out


def _scan_one_split(base: Path) -> List[Sample]:
    """在一个 split 目录内按 stem 配对三模态 + 标签。"""
    # 3) 收集所有图片文件 → 分类
    img_by_mod: Dict[str, Dict[str, Path]] = {  # base_stem -> {mod: path}
        "rgb": {}, "ir": {}, "depth": {}
    }
    label_map: Dict[str, Path] = {}

    if not base.exists():
        return []

    # 遍历 base 及一层子目录(很多 layout 是 images/ 标 签同级 或 labels/ 子目录)
    all_imgs: List[Path] = []
    all_txts: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS:
                all_imgs.append(p)
            elif p.suffix.lower() in LABEL_EXTS:
                all_txts.append(p)

    for imgp in all_imgs:
        mod = _classify_image(imgp)
        if mod is None:
            continue
        base_stem = _strip_modality_tag(imgp.stem)
        img_by_mod[mod][base_stem] = imgp

    # 标签 txt：文件名即基名（赛题 txt 与图同名，无模态后缀）
    for txtp in all_txts:
        label_map[txtp.stem] = txtp

    # 4) 以"有 rgb 且有 label 的 stem"为准（若全缺则退回任意）
    stems = set(img_by_mod["rgb"])
    if not stems:
        stems = set(label_map.keys())
    stems = stems & set(label_map.keys())  # 有标签才算有效样本
    samples = []
    for s in sorted(stems):
        sample = Sample(
            stem=s,
            img={
                mod: img_by_mod[mod].get(s)
                for mod in ("rgb", "ir", "depth")
            },
            label=label_map.get(s),
        )
        samples.append(sample)
    return samples


# ------------------------------------------------------------
# 生成 ultralytics data.yaml
# ------------------------------------------------------------

def build_data_yaml(data_root: Path, split_info: Dict[str, List[Sample]],
                    out_path: Path,
                    train: str = "train", val: str = "val") -> Path:
    """
    根据扫描结果生成 data.yaml。
    说明：ultralytics 训练需要 train/val 两个分组的图像目录；若原数据没有 val，
    需自行先划分（例如 8:1 拆出 val）。本函数只写 data.yaml，不代为划分——不足时由 README 指导。
    """
    # ultralytics 惯例：data.yaml 的路径是相对它自身所在目录或绝对。
    root_abs = data_root.resolve()
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(MC.CLASS_NAMES))

    # 取各 split 的图像目录（找出同一组内图片所在目录）
    def samples_dir(split_samples: List[Sample]) -> str:
        if not split_samples:
            return ""
        # 取第一个样本 rgb 图的父目录作为该 split 的图目录
        rgb = next((s.img["rgb"] for s in split_samples if s.img.get("rgb")), None)
        return str(rgb.parent.resolve()) if rgb else ""

    # 允许配置里显式给 train/val 子目录名
    lines = [
        f"# auto-generated by common.scan_data (from {MC.get_keys()[0]})",
        f"train: {samples_dir(split_info.get(train, []))}",
        f"val: {samples_dir(split_info.get(val, []))}",
        "",
        f"nc: {MC.CLASS_NUM}",
        "names:",
        names,
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[scan] data.yaml 写入 -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描多模态检测数据布局并(可选)生成 data.yaml")
    ap.add_argument("--root", type=str, default=str(MC.DATA_ROOT), help="数据根目录")
    ap.add_argument("--split-dirs", type=str, default="", help="逗号分隔的分组目录名，如 train,val")
    ap.add_argument("--out", type=str, default=None,
                    help="可选：输出 data.yaml 路径")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[!] 数据根不存在: {root}")
        return

    print("== 目录树 ==")
    print_dataset_tree(root, max_depth=2)

    split_dirs = [s.strip() for s in args.split_dirs.split(",") if s.strip()] or None
    res = scan_samples(root, split_dirs or None)

    if args.out:
        # 若无 val 分组，打印提醒
        has_val = "val" in res or "validation" in res or any(len(v) for v in res.values())
        if not has_val:
            print("\n[注意] 扫描不到 val/validation 分组。ultralytics 训练需要 val 子集，"
                  "请先做 train/val 划分，或提供 --split-dirs train,val。")
        build_data_yaml(root, res, Path(args.out))


if __name__ == "__main__":
    main()
