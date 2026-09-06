# -*- coding: utf-8 -*-
"""
split_data —— RGBTD 三模态数据 train/val/test 划分（类别近似分层，可复现）

赛题 2000 组数据默认不分 train/val。为保证：
  1) 训练可复现（随机种子可设置）；
  2) 每个类别在 train/val/test 中的占比尽量一致（分层抽样），
    避免"某类全部落进 test"导致验证失真；
本脚本把 scan_data 扫描到的样本按比例拆成 train/val/test，并输出
ultralytics 可用的图片清单 train.txt/val.txt/test.txt + 完整 data.yaml。

只写清单、绝不移动/复制原文件，原始数据目录保持只读。

用法：
    python scan_data.py --root 数据根 --out data.yaml            # 先看布局
    python split_data.py --root 数据根 --ratios 0.8,0.1,0.1 --seed 42 --out splits
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scan_data import CLASS_NAMES, Sample, scan_samples

DEFAULT_SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")
DEFAULT_RATIOS: Tuple[float, ...] = (0.8, 0.1, 0.1)
DEFAULT_SEED: int = 42


# ------------------------------------------------------------
# 标签读取
# ------------------------------------------------------------

def read_label_cls(label_path: Path) -> np.ndarray:
    """读取 YOLO 标签 txt 的所有类别 id（去重、升序）。无标签/空则空。"""
    if not label_path or not Path(label_path).exists():
        return np.array([], dtype=np.int64)
    ids = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                ids.append(int(float(parts[0])))
    return np.unique(np.asarray(ids, dtype=np.int64)) if ids else np.array([], dtype=np.int64)


def sample_class_ids(sample: Sample) -> np.ndarray:
    return read_label_cls(sample.label)


def class_distribution(samples: Sequence[Sample]) -> Dict[int, int]:
    dist: Dict[int, int] = {}
    for s in samples:
        for c in sample_class_ids(s).tolist():
            dist[int(c)] = dist.get(int(c), 0) + 1
    return dist


# ------------------------------------------------------------
# 核心划分（类别近似分层的"负载均衡"实现）
# ------------------------------------------------------------

def _largest_remainder(amount: int, ratios: Sequence[float]) -> List[int]:
    total = float(sum(ratios))
    exact = [amount * r / total for r in ratios]
    floor = [int(v) for v in exact]
    remainder = amount - sum(floor)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - floor[i], reverse=True)
    for i in range(remainder):
        floor[order[i % len(order)]] += 1
    return floor


def split_samples(
    samples: Sequence[Sample],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
    split_order: Sequence[str] = DEFAULT_SPLIT_ORDER,
) -> Dict[str, List[Sample]]:
    """按比例划分样本。seed 可复现；stratify=True 时按类别近似分层。"""
    samples = list(samples)
    n = len(samples)
    if n == 0:
        return {name: [] for name in split_order}

    quotas = _largest_remainder(n, ratios)
    buckets: Dict[str, List[Sample]] = {name: [] for name in split_order}

    if not stratify:
        rng = random.Random(seed)
        order = list(samples)
        rng.shuffle(order)
        ptr = 0
        for name, q in zip(split_order, quotas):
            buckets[name] = order[ptr:ptr + q]
            ptr += q
        return buckets

    rng = random.Random(seed)
    sample_ids_cache: Dict[int, Tuple[int, ...]] = {
        id(s): tuple(sorted(int(c) for c in sample_class_ids(s).tolist()))
        for s in samples}
    freq: Dict[int, int] = {}
    class_of: Dict[int, List[Sample]] = {}
    for s in samples:
        for c in sample_ids_cache[id(s)]:
            freq[c] = freq.get(c, 0) + 1
            class_of.setdefault(c, []).append(s)

    quotas_map = dict(zip(split_order, quotas))
    size = {name: 0 for name in split_order}
    placed: set = set()

    def open_buckets() -> List[str]:
        return [nm for nm in split_order if size[nm] < quotas_map[nm]]

    def pick_bucket() -> str:
        cand = open_buckets()
        if not cand:
            raise RuntimeError("无可用桶——容量已耗尽（不该发生）")
        return min(cand, key=lambda nm: (size[nm] / quotas_map[nm], rng.random()))

    # 稀有类（freq 小）优先落桶
    rarest_keys = sorted(
        {c for ids in sample_ids_cache.values() for c in ids},
        key=lambda c: (freq[c], c))
    for c in rarest_keys:
        members = [s for s in class_of[c] if id(s) not in placed]
        rng.shuffle(members)
        for s in members:
            dest = pick_bucket()
            buckets.setdefault(dest, []).append(s)
            size[dest] += 1
            placed.add(id(s))

    # 无类别标签或遗漏的样本：按负载补进最空的桶
    rest = [s for s in samples if id(s) not in placed]
    rng.shuffle(rest)
    for s in rest:
        dest = min(open_buckets(), key=lambda nm: (size[nm] / quotas_map[nm], rng.random()))
        buckets.setdefault(dest, []).append(s)
        size[dest] += 1
        placed.add(id(s))

    assert sum(len(v) for v in buckets.values()) == len(samples)
    return {nm: buckets.get(nm, []) for nm in split_order}


def report(splits: Dict[str, List[Sample]]) -> str:
    lines = []
    total = sum(len(v) for v in splits.values())
    for name in DEFAULT_SPLIT_ORDER:
        if name not in splits:
            continue
        sub = splits[name]
        dist = class_distribution(sub)
        dist_str = " ".join(f"{c}:{dist.get(c, 0)}" for c in sorted(dist))
        lines.append(f"[{name}] 样本数={len(sub)} ({len(sub) / max(total, 1):.1%})  类别分布: {dist_str}")
    return "\n".join(lines)


# ------------------------------------------------------------
# 落盘：图片清单 + data.yaml
# ------------------------------------------------------------

def _sample_image_path(s: Sample) -> Optional[Path]:
    """取样本主图(visible)路径用于清单；缺则回退红外/深度。"""
    for mod in ("visible", "infrared", "depth"):
        p = s.img.get(mod)
        if p is not None:
            return Path(p)
    return None


def write_split_txts(splits: Dict[str, List[Sample]], out_dir: Path) -> Dict[str, Path]:
    """为每个 split 写一行一个可见光图绝对路径的 txt。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name in DEFAULT_SPLIT_ORDER:
        if name not in splits:
            continue
        lines = []
        for s in splits[name]:
            img = _sample_image_path(s)
            if img is not None:
                lines.append(str(img.resolve()))
        p = out_dir / f"{name}.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[name] = p
        print(f"[split_data] {name}.txt 写入 {len(lines)} 条 -> {p.resolve()}")
    return written


def write_split_data_yaml(splits: Dict[str, List[Sample]], out_dir: Path,
                          data_yaml: Optional[Path] = None) -> Path:
    """基于划分清单生成完整 data.yaml（train/val/test 指向同名 txt）。"""
    txts = write_split_txts(splits, out_dir)
    yaml_path = Path(data_yaml) if data_yaml else out_dir / "data.yaml"

    def line_for(split: str) -> str:
        p = txts.get(split)
        return str(p.resolve()) if p else ""

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    parts = [
        "# auto-generated by split_data.py",
        f"train: {line_for('train')}",
        f"val: {line_for('val')}",
        f"test: {line_for('test')}",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        names,
    ]
    yaml_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[split_data] data.yaml 写入 -> {yaml_path.resolve()}")
    return yaml_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _parse_ratios(text: str) -> Tuple[float, ...]:
    vals = [float(x) for x in text.split(",") if x.strip()]
    if len(vals) != len(DEFAULT_SPLIT_ORDER):
        raise SystemExit(
            f"--ratios 需给 {len(DEFAULT_SPLIT_ORDER)} 个值(train,val,test)，"
            f"例如 0.8,0.1,0.1；收到: {text!r}")
    if sum(vals) <= 0:
        raise SystemExit("--ratios 之和必须 > 0")
    return tuple(vals)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="split_data",
        description="train/val/test 划分（可设种子，按类别近似分层）并生成 ultralytics 清单/yaml")
    ap.add_argument("--root", type=str, required=True, help="数据根目录")
    ap.add_argument("--ratios", type=str, default=",".join(map(str, DEFAULT_RATIOS)),
                    help="train,val,test 比例")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    ap.add_argument("--no-stratify", action="store_true", help="关闭按类别分层，纯随机切分")
    ap.add_argument("--out", type=str, default=None, help="清单/yaml 输出目录（默认 <数据根>/splits）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[!] 数据根不存在: {root}")
        return

    ratios = _parse_ratios(args.ratios)
    scan = scan_samples(root, split_subdirs=None)

    all_samples: List[Sample] = []
    for split_name, sub in scan.items():
        all_samples.extend(sub)
    if not all_samples:
        print("[!] 未扫描到任何样本。先运行 scan_data.py 确认布局可被识别。")
        return

    print(f"== 输入: 共 {len(all_samples)} 组样本, seed={args.seed}, "
          f"ratios(train,val,test)={ratios}, stratify={not args.no_stratify} ==")
    buckets = split_samples(all_samples, ratios=ratios, seed=args.seed,
                            stratify=not args.no_stratify)
    print(report(buckets))

    out_dir = Path(args.out) if args.out else (root / "splits")
    write_split_data_yaml(buckets, out_dir)


if __name__ == "__main__":
    main()
