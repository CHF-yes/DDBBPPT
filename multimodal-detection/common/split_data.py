# -*- coding: utf-8 -*-
"""
split_data —— 训练/验证/测试 划分结构（train/val/test 三分）。

背景
----
赛题 2000 组数据未划分。为保证：
  1) 训练可复现（随机种子可设置）；
  2) 每个类别在 train/val/test 中的占比尽量一致（分层抽样），
    避免"某类全部落进 test"导致验证失真或测试分异常；
本模块提供：

  split_samples()            : 纯函数，把一个 split 的样本列表按比例拆成
                               {"train": [...], "val": [...], "test": [...]}
  write_split_txts()         : 把划分结果写为 ultralytics 可用的图片清单
                               train.txt / val.txt / test.txt（每行一个图片绝对路径）
  write_split_data_yaml()    : 直接生成一份完整 data.yaml（train/val/test 指向上述 txt）
  main() / CLI               : python -m common.split_data --root ... --seed 42 ...

说明
----
- 分层实现为 iterative stratification 的近似版（多标签可用，无需额外依赖）；
  样本含多个类别时按"稀有类别优先分配"逐个落桶，尽量保持各类别三份占比一致。
- 种子：默认 42，可用 --seed 覆盖；同一数据 + 同一种子 → 同一划分（确定性）。
- 只写清单、绝不移动/复制原文件，原始数据目录保持只读。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import models_config as MC
from .scan_data import Sample, scan_samples

# 划分组顺序与默认比例
DEFAULT_SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")
DEFAULT_RATIOS: Tuple[float, ...] = (0.8, 0.1, 0.1)   # train / val / test
DEFAULT_SEED: int = 42


# ------------------------------------------------------------
# 类别分布统计
# ------------------------------------------------------------

def sample_class_ids(sample: Sample) -> np.ndarray:
    """读取一个样本标签 txt 的所有类别 id（去重、升序）。无标签则空。"""
    if not sample.label or not Path(sample.label).exists():
        return np.array([], dtype=np.int64)
    from .dataset import read_label_txt
    lab = read_label_txt(sample.label)          # (N,5) cls 在第 0 列
    if lab.size == 0:
        return np.array([], dtype=np.int64)
    return np.unique(lab[:, 0].astype(np.int64))


def class_distribution(samples: Sequence[Sample]) -> Dict[int, int]:
    """统计一批样本中每个类别的样本出现次数（多标签样本会重复计入）。"""
    dist: Dict[int, int] = {}
    for s in samples:
        for c in sample_class_ids(s).tolist():
            dist[int(c)] = dist.get(int(c), 0) + 1
    return dist


# ------------------------------------------------------------
# 核心划分
# ------------------------------------------------------------

def _largest_remainder(amount: int, ratios: Sequence[float]) -> List[int]:
    """把 amount 按 ratios 拆分出整数配额（最大余数法，保证总和=amount）。"""
    total = float(sum(ratios))
    exact = [amount * r / total for r in ratios]
    floor = [int(v) for v in exact]
    remainder = amount - sum(floor)
    # 按小数部分从大到小补足
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
    """
    把样本列表按比例划分为 {split_order[0]: [...], ...}。

    - seed: 随机种子（可复现；seed=None 时每次不同）
    - stratify=True: 按标签类别近似分层，尽量让每类在三份中的占比一致
    - 无标签或 stratify=False 时退化为纯随机洗牌后切分

    返回 dict 保持 split_order 顺序。
    """
    samples = list(samples)
    n = len(samples)
    if n == 0:
        return {name: [] for name in split_order}

    quotas = _largest_remainder(n, ratios)          # 各桶容量
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

    # ---- 类别近似分层的"负载均衡"实现（保证精确容量 + 可复现）----
    #
    # 要点：
    #   1) 读出每个样本的类别集合并缓存（只读一次标签文件）；
    #   2) 各样本按其"含概的最稀有类别"作为分层主键，稀有类样本先落桶；
    #   3) 每个带类别样本落到"当前负载 < 目标占比"最小的桶（负载=size/quota），
    #      让每个类大致按比例铺开，同时绝不超各桶容量 → 总量精确；
    #   4) 无类别标签的样本最后按负载补进最空的桶。
    #   5) 全程用 seed 控制的 rng 决定同类内顺序与并列时的随机打破，保证同种子=同结果。
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
    size = {name: 0 for name in split_order}          # 每桶已装样本数
    placed: set[int] = set()

    def open_buckets() -> List[str]:
        return [nm for nm in split_order if size[nm] < quotas_map[nm]]

    def pick_bucket(classes: Tuple[int, ...], restrict_to: Optional[List[str]]) -> str:
        """在可装桶中选负载最低；并列(容量充足都空)时用 rng 随机挑一个可装桶。"""
        cand = restrict_to if restrict_to is not None else open_buckets()
        if not cand:
            raise RuntimeError("无可用桶——容量已耗尽（不该发生）")
        # 只为当前样本的类别覆盖作依据：取该样本类别里能改善最大的桶太复杂，
        # 简化成"负载率最低 + rng 打破"，已能获得类别近似的均衡铺开。
        best = min(cand, key=lambda nm: (size[nm] / quotas_map[nm], rng.random()))
        return best

    # 稀有类（freq 小）优先落桶，保证冷门类不被后续"挤到角"
    rarest_keys = sorted(
        {c for ids in sample_ids_cache.values() for c in ids},
        key=lambda c: (freq[c], c))
    for c in rarest_keys:
        members = [s for s in class_of[c] if id(s) not in placed]
        rng.shuffle(members)
        for s in members:
            dest = pick_bucket(sample_ids_cache[id(s)], None)
            buckets.setdefault(dest, []).append(s)
            size[dest] += 1
            placed.add(id(s))

    # 无类别标签或已漏的样本：按负载补进最空的桶
    rest = [s for s in samples if id(s) not in placed]
    rng.shuffle(rest)
    for s in rest:
        dest = min(open_buckets(), key=lambda nm: (size[nm] / quotas_map[nm], rng.random()))
        buckets.setdefault(dest, []).append(s)
        size[dest] += 1
        placed.add(id(s))

    # 兜底断言：总量必须精确
    assert sum(len(v) for v in buckets.values()) == len(samples)
    assert all(size[nm] == len(buckets.get(nm, [])) for nm in split_order)
    return {nm: buckets.get(nm, []) for nm in split_order}


def report(splits: Dict[str, List[Sample]]) -> str:
    """返回划分统计报告：各份样本数与类别分布（便于核对分层是否均衡）。"""
    lines = []
    total = sum(len(v) for v in splits.values())
    for name in DEFAULT_SPLIT_ORDER:
        if name not in splits:
            continue
        sub = splits[name]
        dist = class_distribution(sub)
        dist_str = " ".join(f"{c}:{dist.get(c,0)}" for c in sorted(dist))
        lines.append(f"[{name}] 样本数={len(sub)} ({len(sub)/max(total,1):.1%})  类别分布: {dist_str}")
    return "\n".join(lines)


# ------------------------------------------------------------
# 落盘：ultralytics 图片清单 + data.yaml
# ------------------------------------------------------------

def _sample_image_path(s: Sample) -> Optional[Path]:
    """取样本的"主图"路径用于清单：优先 rgb，缺则 ir/depth。"""
    for mod in ("rgb", "ir", "depth"):
        p = s.img.get(mod)
        if p is not None:
            return Path(p)
    return None


def write_split_txts(splits: Dict[str, List[Sample]], out_dir: Path) -> Dict[str, Path]:
    """
    为每个 split 写一行一个图片绝对路径的 txt（ultralytics 支持 txt 清单）。
    返回 {split: txt路径}。
    """
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
        print(f"[split_data] {name}.txt 写入 {len(lines)} 条 -> {p}")
    return written


def write_split_data_yaml(
    splits: Dict[str, List[Sample]],
    out_dir: Path,
    data_yaml: Optional[Path] = None,
) -> Path:
    """基于划分清单生成一份完整 data.yaml（train/val/test 指向同名 txt）。"""
    txts = write_split_txts(splits, out_dir)
    yaml_path = Path(data_yaml) if data_yaml else out_dir / "data.yaml"

    def line_for(split: str) -> str:
        p = txts.get(split)
        return str(p.resolve()) if p else ""       # 空则不给该字段

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(MC.CLASS_NAMES))
    parts = [
        "# auto-generated by common.split_data",
        f"train: {line_for('train')}",
        f"val: {line_for('val')}",
        f"test: {line_for('test')}",
        "",
        f"nc: {MC.CLASS_NUM}",
        "names:",
        names,
    ]
    yaml_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[split_data] data.yaml 写入 -> {yaml_path}")
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
    ap.add_argument("--root", type=str, default=str(MC.DATA_ROOT), help="数据根目录")
    ap.add_argument("--split-dirs", type=str, default="train",
                    help="对根目录下哪些分组做划分，逗号分隔（默认 train）")
    ap.add_argument("--ratios", type=str, default=",".join(map(str, DEFAULT_RATIOS)),
                    help="train,val,test 比例")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"随机种子（默认 {DEFAULT_SEED}；同一数据+同一种子=同一划分）")
    ap.add_argument("--no-stratify", action="store_true", help="关闭按类别分层，纯随机切分")
    ap.add_argument("--out", type=str, default=None,
                    help="清单/yaml 输出目录（默认 <数据根>/splits）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[!] 数据根不存在: {root}")
        return

    ratios = _parse_ratios(args.ratios)
    splits_in = [s.strip() for s in args.split_dirs.split(",") if s.strip()]
    scanned = scan_samples(root, splits_in or None)

    all_samples: List[Sample] = []
    for split_name, sub in scanned.items():
        all_samples.extend(sub)
    if not all_samples:
        print("[!] 未扫描到任何样本。先运行 common.scan_data 确认布局可被识别。")
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
