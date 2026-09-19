# -*- coding: utf-8 -*-
"""训练结束后的完整分析：A0 vs B0 + 模态定价 + 逐类 AP + 条件切片 -> 一份 Markdown 报告。

分析设计（口径统一，全部走 `eval.py` 的赛题口径 conf=0.001/iou=0.7/max_det=100）：

1. **A0 与 B0 都在同一个 400 张全量 val 上评测**（同一个 split.json）——训练期日志里的
   120 张子集只用于看趋势，不作为结论；
2. **A0 按 RGB-only 评测**（checkpoint 自描述模态，`eval.py` 自动解析）；
3. **B0 按三模态评测 + 模态屏蔽剖面**（all / no_ir / no_dep / no_rgb）→ 给每个模态定价；
4. **逐类 AP 并排**（12 类长尾，class 11 全局仅 27 框）；
5. **条件切片**（照度/拥挤度/depth 有效比例 的最坏 20%）；
6. 两份 `train.log` 的**同轮次曲线**（辅助证据，注明是 120 张子集）。

用法：
  python code/mm_yolo/analyze_final.py --root <train_extracted> --labels <lab> \
      --a0 code/runs/a0_rgb --b0 code/runs/b0_mm --out code/runs/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import build_index, group_split, load_split, val_class_stats          # noqa: E402
from eval import evaluate_model                                                # noqa: E402
from model import load_mm_checkpoint, resolve_infer_canvas, resolve_infer_modalities  # noqa: E402
from compare_runs import parse_log                                             # noqa: E402

CLASS_NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
               "light", "garbage_can", "uav", "tricycle"]


def fmt(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, float) and (np.isnan(x)):
        return "nan"
    return f"{x:.{n}f}"


def pick_ckpt(run_dir: Path, which: str) -> Path:
    p = run_dir / "weights" / f"{which}.pt"
    if not p.exists():
        raise SystemExit(f"缺少权重 {p}")
    return p


def eval_ckpt(ckpt: Path, root: Path, samples, dev, profile: bool, tag: str) -> dict:
    t0 = time.time()
    model, ck = load_mm_checkpoint(ckpt, device=dev)
    model.eval()
    mod = resolve_infer_modalities(model, None)
    imgsz = resolve_infer_canvas(model, None)
    res = evaluate_model(model, root, samples, imgsz=imgsz, device=dev, modalities=mod,
                         profile=profile, slices=True, conf=0.001, batch_size=4)
    res["_tag"] = tag
    res["_ckpt"] = str(ckpt)
    res["_epoch"] = int(ck.get("epoch", 0))
    res["_modalities"] = mod
    res["_canvas"] = list(imgsz) if isinstance(imgsz, (tuple, list)) else [imgsz, imgsz]
    res["_seconds"] = round(time.time() - t0, 1)
    print(f"[analysis] {tag}: mAP50-95={fmt(res['map50_95'])} mAP50={fmt(res['map50'])} "
          f"有效类={res['n_valid_classes']}/12 模态={mod} 用时={res['_seconds']}s", flush=True)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--a0", default="code/runs/a0_rgb")
    ap.add_argument("--b0", default="code/runs/b0_mm")
    ap.add_argument("--out", default="code/runs/analysis")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张（0=全量 val）")
    ap.add_argument("--skip-last", action="store_true", help="跳过 last.pt（省时间）")
    args = ap.parse_args()

    root = Path(args.root)
    a0d, b0d = Path(args.a0), Path(args.b0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx = build_index(root, Path(args.labels))
    split = a0d / "split.json"
    if split.exists():
        _, va = load_split(split, idx)
        print(f"[analysis] 复用 {split} 的 val={len(va)}")
    else:
        _, va = group_split(idx, 0.2, seed=42)
        print(f"[analysis] 未找到 split.json，按 0.2 重新划分 val={len(va)}")
    if args.limit:
        va = va[: args.limit]
    cls_cnt = val_class_stats(va, 12)
    print(f"[analysis] 评测集逐类框数 {cls_cnt}")

    results: dict = {"val_class_counts": {str(k): v for k, v in cls_cnt.items()},
                     "n_val": len(va), "runs": {}}

    # ---- A0：RGB-only 锚点（best；可选 last）----
    jobs = [("a0_best", a0d, "best", False)]
    if not args.skip_last:
        jobs.append(("a0_last", a0d, "last", False))
    # ---- B0：三模态（best 带剖面，last 不带）----
    jobs.append(("b0_best", b0d, "best", True))
    if not args.skip_last:
        jobs.append(("b0_last", b0d, "last", False))

    for tag, d, which, prof in jobs:
        ck = d / "weights" / f"{which}.pt"
        if not ck.exists():
            print(f"[analysis] 跳过 {tag}：{ck} 不存在")
            continue
        try:
            results["runs"][tag] = eval_ckpt(ck, root, va, dev, prof, tag)
        except Exception as exc:                                       # noqa: BLE001
            import traceback
            print(f"[analysis] {tag} 评测失败：{type(exc).__name__}: {exc}")
            traceback.print_exc()
            results["runs"][tag] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- 训练日志曲线（120 张子集，辅助证据）----
    results["curve_a0"] = parse_log(a0d / "train.log")
    results["curve_b0"] = parse_log(b0d / "train.log")

    (out / "analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=1,
                                                  default=str), encoding="utf-8")

    # ---------------------------------------------------------------- Markdown
    md: list = ["# A0(RGB) vs B0(三模态) 分析报告", ""]
    md.append(f"- 评测集：**{len(va)} 张**（同一个 `split.json` 的 val；逐类框数 {cls_cnt}）")
    md.append("- 赛题口径：`conf=0.001 / iou=0.7 / max_det=100`，mAP50-95 为 101 点插值")
    md.append("- ⚠️ 单 run 噪声约 ±0.005；差距 >0.01 才值得下结论")
    md.append("")

    md.append("## 1. 主表（全量 val）")
    md.append("")
    md.append("| 权重 | 模态 | epoch | mAP50-95 | mAP50 | 有效类 |")
    md.append("|---|---|---|---|---|---|")
    for tag, r in results["runs"].items():
        if "error" in r:
            md.append(f"| {tag} | - | - | 失败 | {r['error']} | - |")
            continue
        md.append(f"| {tag} | {r['_modalities']} | {r['_epoch']} | {fmt(r['map50_95'])} | "
                  f"{fmt(r['map50'])} | {r['n_valid_classes']}/12 |")
    md.append("")

    b0 = results["runs"].get("b0_best", {})
    a0 = results["runs"].get("a0_best", {})
    if "map50_95" in a0 and "map50_95" in b0:
        d = b0["map50_95"] - a0["map50_95"]
        md.append(f"**B0 - A0 = {d:+.4f} mAP50-95**（{fmt(a0['map50_95'])} → "
                  f"{fmt(b0['map50_95'])}）")
        md.append("")

    # 模态定价
    if "profile" in b0:
        md.append("## 2. 模态定价（B0 推理期屏蔽，全量 val）")
        md.append("")
        md.append("| 配置 | mAP50-95 | mAP50 | 相对 all 掉点 |")
        md.append("|---|---|---|---|")
        base = b0["profile"]["all"]["map50_95"]
        names = {"all": "三模态全开", "no_ir": "屏蔽 IR", "no_dep": "屏蔽 depth",
                 "no_rgb": "屏蔽 RGB（只剩 IR+depth）"}
        for k, v in b0["profile"].items():
            md.append(f"| {names.get(k, k)} | {fmt(v['map50_95'])} | {fmt(v['map50'])} | "
                      f"{base - v['map50_95']:+.4f} |")
        md.append("")
        md.append("> 掉点越大 = 该模态越不可替代；掉点≈0 = 该模态当前被浪费。")
        md.append("")

    # 逐类 AP
    if "per_class_95" in a0 and "per_class_95" in b0:
        md.append("## 3. 逐类 AP50-95（全量 val）")
        md.append("")
        md.append("| cls | 名称 | A0(RGB) | B0(三模态) | Δ(B-A) | val 框数 |")
        md.append("|---|---|---|---|---|---|")
        for c in range(12):
            ca = a0["per_class_95"].get(str(c), a0["per_class_95"].get(c))
            cb = b0["per_class_95"].get(str(c), b0["per_class_95"].get(c))
            ca = float(ca) if ca is not None else float("nan")
            cb = float(cb) if cb is not None else float("nan")
            dd = "-" if (np.isnan(ca) or np.isnan(cb)) else f"{cb - ca:+.4f}"
            md.append(f"| {c} | {CLASS_NAMES[c]} | {fmt(ca)} | {fmt(cb)} | {dd} | "
                      f"{cls_cnt.get(c, cls_cnt.get(str(c), 0))} |")
        md.append("")

    # 条件切片
    if "slices" in b0:
        md.append("## 4. 条件切片（B0，全量 val 的最坏/最好 20%）")
        md.append("")
        md.append("| 切片 | 张数 | mAP50-95 | mAP50 |")
        md.append("|---|---|---|---|")
        for k, v in b0["slices"].items():
            md.append(f"| {k} | {v['n']} | {fmt(v['map50_95'])} | {fmt(v['map50'])} |")
        md.append("")
        md.append("> 关注「照度最低 20%」——它能回答\"是不是靠白天刷分\"。")
        md.append("")

    # 曲线（parse_log 的键是 int；json 里会变成 str，两种都兼容）
    ca, cb = results["curve_a0"], results["curve_b0"]

    def _at(curve, ep):
        r = curve.get(ep)
        if r is None:
            r = curve.get(str(ep))
        return r or {}
    common = sorted({int(k) for k in ca} & {int(k) for k in cb})
    md.append("## 5. 同轮次曲线（训练期 120 张子集，辅助证据）")
    md.append("")
    md.append("| ep | A0 mAP50-95 | B0 mAP50-95 | Δ(B-A) | A0 loss | B0 loss |")
    md.append("|---|---|---|---|---|---|")
    for ep in common:
        ra, rb = _at(ca, ep), _at(cb, ep)
        va_, vb_ = ra.get("map50_95"), rb.get("map50_95")
        dd = "-" if (va_ is None or vb_ is None) else f"{vb_ - va_:+.4f}"
        md.append(f"| {ep} | {fmt(va_)} | {fmt(vb_)} | {dd} | {fmt(ra.get('loss'), 3)} | "
                  f"{fmt(rb.get('loss'), 3)} |")
    if not common:
        md.append("| - | （两份日志还没有共同验证轮次） | | | | |")
    md.append("")

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[analysis] 报告已写出 {out/'report.md'}（原始数据 {out/'analysis.json'}）")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
