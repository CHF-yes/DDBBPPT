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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 直接 python common/scan_data.py 运行时也可找到 code 根（-m 方式同样兼容）
_CODE_ROOT = str(Path(__file__).resolve().parent.parent)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import models_config as MC


# ------------------------------------------------------------
# 命名规则表格：文件名含的"形态关键字"优先匹配，其次位置关键字
# ------------------------------------------------------------
MODALITY_HINTS: Dict[str, List[str]] = {
    "rgb": ["rgb", "visible", "color", "visible_image"],
    "ir": ["ir", "infrared", "thermal", "tir"],
    "depth": ["depth", "disp", "d_"],
}
# 按「目录名」配对（文件名无模态后缀时用，如 VDT-2048 的 V/T/D 布局）
MODALITY_DIR_TABLE: Dict[str, List[str]] = {
    "rgb": ["v", "vis", "visible", "rgb"],
    "ir": ["t", "ir", "thermal", "infrared", "tir"],
    "depth": ["d", "depth"],
}
# 标签目录候选名（目录布局模式下）
LABEL_DIR_NAMES = ("labels_multi", "labels", "label", "gt")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
LABEL_EXTS = {".txt"}


def classify_by_dir(dir_name: str) -> Optional[str]:
    """按目录名判定模态（精确匹配，返回 rgb/ir/depth 之一）。"""
    n = dir_name.lower()
    for mod, names in MODALITY_DIR_TABLE.items():
        if n in names:
            return mod
    return None


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


def scan_samples(data_root: Path, split_subdirs: Optional[list] = None,
                 layout: bool = False) -> Dict[str, List[Sample]]:
    """
    扫描并按 `split_subdirs` 聚合样本。

    data_root 下假设形如:
        <root>/train/xxx_rgb.png, xxx_ir.png, xxx_depth.png, xxx.txt
        <root>/val/...
    若 split_subdirs=None 则默认探测 root 的直接子目录(如 train/val)；
    并将返回 {split_name: [Sample, ...]}。

    layout=True（目录布局模式，如 VDT-2048 的 V/T/D/labels_multi）：
      把 data_root 本身当作"一组"，按目录名 V/T/D→rgb/ir/depth、labels_*→标签 配对，
      文件名无需携带模态后缀。
    """
    if layout:
        return {data_root.name: _scan_one_split_layout(data_root)}

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


def _looks_like_layout_root(root: Path) -> bool:
    """判断 root 是否为目录布局根（如 VDT 的 V/T/D/labels_multi、visible/infrared/depth/labels）。"""
    names = {p.name.lower() for p in root.iterdir() if p.is_dir()}
    mod_names = set(MODALITY_DIR_TABLE["rgb"] + MODALITY_DIR_TABLE["ir"]
                    + MODALITY_DIR_TABLE["depth"])
    lab_names = {n.lower() for n in LABEL_DIR_NAMES}
    has_mod_dirs = bool(names & mod_names)
    has_label_dirs = bool(names & lab_names)
    return has_mod_dirs and has_label_dirs


def scan_samples_auto(data_root: Path) -> Dict[str, List[Sample]]:
    """
    P0-5: 主训练入口用的自动探测：
      先按默认 split 子目录扫描；若 0 样本且 root 是 V/T/D/labels_multi 型
      目录布局，自动降级为 layout=True 重扫（无需用户手动传 layout）。
    """
    res = scan_samples(data_root, layout=False)
    total = sum(len(v) for v in res.values())
    if total == 0 and data_root.is_dir() and _looks_like_layout_root(data_root):
        print("[scan] 默认 split 扫描 0 样本；检测到 V/T/D/labels_multi 目录布局，"
              "自动切换 layout 模式重扫 ...")
        res = scan_samples(data_root, layout=True)
    return res


def _scan_one_split_layout(base: Path) -> List[Sample]:
    """目录布局模式：按子目录名配对三模态 + 标签（如 VDT 的 V/T/D/labels_multi）。"""
    img_by_mod: Dict[str, Dict[str, Path]] = {"rgb": {}, "ir": {}, "depth": {}}
    label_map: Dict[str, Path] = {}
    if not base.exists():
        return []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        mod = classify_by_dir(sub.name)
        if mod is not None:
            for p in sub.glob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    img_by_mod[mod][p.stem] = p
        elif sub.name.lower() in LABEL_DIR_NAMES:
            for p in sub.glob("*.txt"):
                label_map[p.stem] = p
    stems = set(img_by_mod["rgb"]) if img_by_mod["rgb"] else set(label_map)
    stems &= set(label_map)
    return [Sample(stem=s, img={m: img_by_mod[m].get(s) for m in ("rgb", "ir", "depth")},
                   label=label_map.get(s)) for s in sorted(stems)]


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
                    train: str = "train", val: str = "val",
                    names: Optional[List[str]] = None) -> Path:
    """
    根据扫描结果生成 data.yaml。
    说明：ultralytics 训练需要 train/val 两个分组的图像目录；若原数据没有 val，
    需自行先划分（例如 8:1 拆出 val）。本函数只写 data.yaml，不代为划分——不足时由 README 指导。
    names: 类别名列表（每行一个 / 直接传 list）；None 时用赛题 12 类。
    """
    # ultralytics 惯例：data.yaml 的路径是相对它自身所在目录或绝对。
    root_abs = data_root.resolve()
    names = list(names) if names else []
    names = names or list(MC.CLASS_NAMES)
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))

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
        f"nc: {len(names)}",
        "names:",
        names_block,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[scan] data.yaml 写入 -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描多模态检测数据布局并(可选)生成 data.yaml")
    ap.add_argument("--root", type=str, default=str(MC.DATA_ROOT), help="数据根目录")
    ap.add_argument("--split-dirs", type=str, default="", help="逗号分隔的分组目录名，如 train,val")
    ap.add_argument("--layout", action="store_true",
                    help="目录布局模式：按目录名配对（如 VDT 的 V/T/D/labels_multi）")
    ap.add_argument("--names", type=str, default=None,
                    help="类别名文件（每行一个类别名）；默认赛题 12 类")
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
    res = scan_samples(root, split_dirs or None, layout=args.layout)

    names_list = None
    if args.names:
        names_list = [l.strip() for l in
                      Path(args.names).read_text(encoding="utf-8").splitlines()
                      if l.strip() and not l.startswith("#")]
        print(f"[scan] 使用类别名文件: {args.names} ({len(names_list)} 类)")

    if args.out:
        # P0-5: has_val 只应反映是否存在 val/validation 分组数据
        # （旧逻辑 "any split 有数据" 会把只有 train 的情形误判为有 val）
        has_val = bool(res.get("val") or res.get("validation"))
        if not has_val:
            print("\n[注意] 扫描不到 val/validation 分组。ultralytics 训练需要 val 子集，"
                  "请先做 train/val 划分，或提供 --split-dirs train,val。")
        build_data_yaml(root, res, Path(args.out), names=names_list)


if __name__ == "__main__":
    main()
