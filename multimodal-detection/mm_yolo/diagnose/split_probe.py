# -*- coding: utf-8 -*-
"""近重复/来源分组检测：为"按组划分验证集"提供依据（避免同源帧泄漏进验证集）。

做法：对每张 visible 图算 32×32 灰度缩略图 + dHash（64bit），
按汉明距离 ≤ 阈值的并查集聚类，输出簇数量/大小分布 + 组划分建议。

用法：python code/mm_yolo/diagnose/split_probe.py --root <train_extracted> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent.parent
for _p in (str(_CODE / "mm_yolo"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from io_utils import imread_unicode                        # noqa: E402


def dhash(gray32: np.ndarray) -> int:
    d = gray32[:, 1:] > gray32[:, :-1]
    v = 0
    for b in d.flatten():
        v = (v << 1) | int(b)
    return v


def prefix_group(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) == 1:
        return "PLAIN"
    if parts[0] in ("hehe", "shuming"):
        return parts[0]
    return parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.root)
    files = sorted((root / "visible").glob("*"))
    stems, hashes, thumbs = [], [], []
    for p in files:
        img = imread_unicode(p)
        if img is None:
            continue
        g = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        g32 = cv2.resize(g, (33, 32), interpolation=cv2.INTER_AREA)
        stems.append(p.stem)
        hashes.append(dhash(g32))
        thumbs.append(cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel())
    n = len(stems)
    print(f"[split] 读取 {n} 张")

    # 额外：16×16 缩略图的余弦相似（对亮度/对比度不敏感的粗判）
    T = np.stack(thumbs)
    T = T - T.mean(1, keepdims=True)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)
    Sim = T @ T.T
    H = np.array([h & ((1 << 64) - 1) for h in hashes], dtype=np.uint64)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 有效近似：先按前缀分组强制成簇，再用哈希/相似度合并跨组重复
    Hb = np.array([[(int(H[i]) ^ int(H[j])).bit_count() for j in range(n)] for i in range(n)],
                  dtype=np.uint8)
    near = (Hb <= args.hamming) | (Sim > 0.995)
    pref = [prefix_group(s) for s in stems]
    same_named = np.array([[pref[i] == pref[j] and pref[i] != "PLAIN" for j in range(n)]
                           for i in range(n)])
    adj = near | same_named
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(stems[i])
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"[split] 簇数 {len(clusters)}；最大簇 {sizes[:10]}；单例簇 {sum(1 for s in sizes if s == 1)} 个")
    big = [v for v in clusters.values() if len(v) >= 2]
    print(f"[split] ≥2 张的簇 {len(big)} 个，覆盖 {sum(len(v) for v in big)} 张（{sum(len(v) for v in big)/n:.1%}）")
    for v in sorted(big, key=len, reverse=True)[:5]:
        print(f"   例({len(v)}): {v[:6]}")
    # 按前缀看规模
    pref = Counter(prefix_group(s) for s in stems)
    print(f"[split] 前缀分组: {pref.most_common(12)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "n": n,
            "clusters": {str(k): v for k, v in clusters.items()},
            "sizes": sizes,
            "prefix_counts": dict(pref),
        }, ensure_ascii=False), encoding="utf-8")
        print(f"[split] 已写出 {out}")


if __name__ == "__main__":
    main()
