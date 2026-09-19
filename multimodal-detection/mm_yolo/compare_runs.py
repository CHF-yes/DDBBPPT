# -*- coding: utf-8 -*-
"""A0 vs B0 同轮次对比（不依赖 matplotlib，直接出 Markdown/CSV）。

训练日志里每次验证都会写：
    [train] ep 5/60 lr=... loss=... | val mAP50-95=0.1234 mAP50=0.2345 (有效类 11/12)

本脚本把两份日志按 epoch 对齐，输出：
  * 控制台表格（同轮次并排 + 差值 Δ）；
  * `compare.md`（技术报告可直接用）+ `compare.csv`（画曲线用）。

用法：
  python code/mm_yolo/compare_runs.py --a code/runs/a0_rgb --b code/runs/b0_mm \
      --out code/runs/compare_a0_b0
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

LINE = re.compile(
    r"ep (\d+)/(\d+).*?loss=([\d.]+).*?"
    r"(?:val mAP50-95=([\d.]+) mAP50=([\d.]+)\s*\(有效类 (\d+)/12\))?")
VAL = re.compile(r"ep (\d+)/\d+.*?val mAP50-95=([\d.]+) mAP50=([\d.]+)\s*\(有效类 (\d+)/12\)")


def parse_log(path: Path) -> dict:
    """返回 {epoch: {'loss':float,'map50_95':float,'map50':float,'valid_classes':int}}。"""
    out: dict = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.search(raw)
        if not m:
            continue
        ep = int(m.group(1))
        rec = out.setdefault(ep, {})
        rec["loss"] = float(m.group(3))
        v = VAL.search(raw)
        if v:
            rec["map50_95"] = float(v.group(2))
            rec["map50"] = float(v.group(3))
            rec["valid_classes"] = int(v.group(4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A0（RGB-only）run 目录")
    ap.add_argument("--b", required=True, help="B0（三模态）run 目录")
    ap.add_argument("--la", default="A0(RGB)")
    ap.add_argument("--lb", default="B0(三模态)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    a, b = parse_log(Path(args.a) / "train.log"), parse_log(Path(args.b) / "train.log")
    if not a or not b:
        print(f"[compare] 日志缺失或为空：{args.a}={'有' if a else '无'}，"
              f"{args.b}={'有' if b else '无'}（训练还没跑到验证轮？）")
    eps = sorted(set(a) | set(b))
    rows = []
    for ep in eps:
        ra, rb = a.get(ep, {}), b.get(ep, {})
        va, vb = ra.get("map50_95"), rb.get("map50_95")
        rows.append({
            "epoch": ep,
            "a_loss": ra.get("loss"), "b_loss": rb.get("loss"),
            "a_map50_95": va, "b_map50_95": vb,
            "delta": (None if (va is None or vb is None) else round(vb - va, 4)),
            "a_map50": ra.get("map50"), "b_map50": rb.get("map50"),
            "a_valid": ra.get("valid_classes"), "b_valid": rb.get("valid_classes"),
        })

    def fmt(x, n=4):
        return "-" if x is None else f"{x:.{n}f}"

    print(f"| ep | {args.la} loss | {args.lb} loss | {args.la} mAP50-95 | "
          f"{args.lb} mAP50-95 | Δ(B-A) | {args.la} 有效类 | {args.lb} 有效类 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['epoch']} | {fmt(r['a_loss'],3)} | {fmt(r['b_loss'],3)} | "
              f"{fmt(r['a_map50_95'])} | {fmt(r['b_map50_95'])} | {fmt(r['delta'])} | "
              f"{r['a_valid'] or '-'} | {r['b_valid'] or '-'} |")

    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    if deltas:
        best_a = max((r["a_map50_95"] for r in rows if r["a_map50_95"]), default=None)
        best_b = max((r["b_map50_95"] for r in rows if r["b_map50_95"]), default=None)
        print(f"\n共同验证轮次 {len(deltas)} 个：Δ 平均 {sum(deltas)/len(deltas):+.4f}、"
              f"最好 {max(deltas):+.4f}、最差 {min(deltas):+.4f}")
        print(f"峰值：{args.la} {fmt(best_a)} | {args.lb} {fmt(best_b)} | "
              f"Δ峰值 {fmt(None if (best_a is None or best_b is None) else best_b-best_a)}")
        print("[!] 单 run 噪声约 +/-0.005；差距 >0.01 才值得下结论（2 种子更稳）。")
    else:
        print("\n还没有共同轮次的两份验证结果（两份日志需跑到同一 val-every 轮次）。")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["epoch"])
            w.writeheader()
            w.writerows(rows)
        md = [f"# {args.la} vs {args.lb}（同轮次）", ""]
        md.append("| ep | A loss | B loss | A mAP50-95 | B mAP50-95 | Δ(B-A) | A 有效类 | B 有效类 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['epoch']} | {fmt(r['a_loss'],3)} | {fmt(r['b_loss'],3)} | "
                      f"{fmt(r['a_map50_95'])} | {fmt(r['b_map50_95'])} | {fmt(r['delta'])} | "
                      f"{r['a_valid'] or '-'} | {r['b_valid'] or '-'} |")
        if deltas:
            md += ["", f"共同轮次 {len(deltas)}：Δ 平均 {sum(deltas)/len(deltas):+.4f}，"
                       f"峰值 A={fmt(best_a)} / B={fmt(best_b)}。",
                   "", "> 单 run 噪声约 ±0.005；>0.01 才值得下结论。"]
        (out / "compare.md").write_text("\n".join(md), encoding="utf-8")
        print(f"[compare] 已写出 {out/'compare.csv'} 与 {out/'compare.md'}")


if __name__ == "__main__":
    main()
