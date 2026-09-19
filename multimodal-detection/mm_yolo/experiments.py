# -*- coding: utf-8 -*-
"""三模态实验命令清单。只打印命令，绝不启动训练或评测。

运行示例（PowerShell/Linux 均可）：
python code/mm_yolo/experiments.py --stage all --root TRAIN_ROOT \
    --labels LABEL_ROOT --weights code/yolo11s.pt \
    --split-file code/runs/b2_depth4_s_safe/split.json \
    --baseline-ckpt code/runs/b2_depth4_s_safe/weights/best.pt

逐组执行打印的命令。所有训练组从相同 COCO 权重独立开始，不能从 B1/B2
checkpoint 初始化，否则模态之间的比较会被迁移历史混淆。
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _run_name(args, base: str) -> str:
    """为新配方建立独立目录，空 tag 保持历史命名兼容。"""
    tag = str(getattr(args, "run_tag", "") or "").strip()
    return f"{base}_{tag}_s{args.seed}" if tag else f"{base}_s{args.seed}"


def _format(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def _train(args, name: str, extra: list[str]) -> list[str]:
    cmd = [args.python, str(HERE / "train.py"),
            "--root", args.root, "--labels", args.labels,
            "--weights", args.weights, "--split-file", args.split_file,
            "--out", args.out, "--name", name,
            "--device", args.device, "--imgsz", args.imgsz,
            "--epochs", str(args.epochs), "--batch", str(args.batch),
            "--accum", str(args.accum), "--workers", str(args.workers),
            "--lr", "0.0001", "--backbone-lr-mult", "0.1",
            "--grad-clip", str(getattr(args, "grad_clip", 60.0)),
            "--freeze-epochs", "3", "--fusion-tier", "L2", "--share-tier", "c",
            "--depth-scales", "p4p5", "--depth-channels", "4",
            "--rgb-dropout", "0.02", "--aux-dropout", "0.02",
            "--dropout-start-epoch", "10", "--misalign-px", "5",
            "--degrade-p", "0.15", "--ir-noise-p", "0.1",
            "--depth-hole-p", "0.1", "--target-crop-p", "0.08",
            "--rare-sample-max", "1.5", "--seed", str(args.seed),
            # 训练期只用于一致选模：0.01 显著减少早期海量低置信候选；正式评测
            # 仍在 _eval 中使用 0.001。修复逐图 NMS 后不再用会极慢的 ep0 全量评测。
            "--val-every", "5", "--val-limit", "0", "--val-conf", "0.01"]
    i = 0
    while i < len(extra):
        flag = extra[i]
        has_value = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        if flag in cmd:
            pos = cmd.index(flag)
            del cmd[pos:pos + (2 if has_value else 1)]
        cmd += extra[i:i + (2 if has_value else 1)]
        i += 2 if has_value else 1
    return cmd


def _eval(args, ckpt: str, name: str, profile: bool = False,
          ablate_absolute: bool = False) -> list[str]:
    cmd = [args.python, str(HERE / "eval.py"),
           "--ckpt", ckpt, "--root", args.root, "--labels", args.labels,
           "--split", args.split_file, "--device", args.device,
           "--conf", "0.001", "--batch", "1", "--by-depth-format",
           "--out", str(Path(args.out) / name / "formal_eval.json")]
    if not profile:
        cmd.append("--no-slices")
    if profile:
        cmd.append("--profile")
    if ablate_absolute:
        cmd.append("--ablate-absolute")
    return cmd


def _stock_rgb(args) -> list[str]:
    """官方 Ultralytics 基线；先校准单模态训练器，再解释融合增益。"""
    return [args.python, str(HERE / "stock_rgb_baseline.py"),
            "--root", args.root, "--labels", args.labels,
            "--weights", args.weights, "--split-file", args.split_file,
            "--out", args.out, "--name", _run_name(args, "stock_rgb_yolo11s"),
            "--device", args.device, "--epochs", str(max(100, args.epochs)),
            "--batch", "8", "--imgsz", "640", "--workers", str(args.workers),
            "--seed", str(args.seed)]


def build_plan(args) -> list[tuple[str, list[str]]]:
    plan: list[tuple[str, list[str]]] = []
    if args.stage in ("all", "diagnose"):
        if not args.baseline_ckpt:
            raise ValueError("diagnose 阶段须指定 --baseline-ckpt")
        plan.append(("诊断：现有 B2 模态/绝对深度消融（分组评测）",
                     _eval(args, args.baseline_ckpt, "diagnose_b2",
                           profile=True, ablate_absolute=True)))
    if args.stage in ("all", "modalities"):
        if not bool(getattr(args, "skip_stock", False)):
            plan.append(("校准：官方 Ultralytics YOLO11s RGB 基线", _stock_rgb(args)))
        for mode in ("rgb", "rgb_ir", "rgb_dep", "all"):
            name = _run_name(args, f"exp_mod_{mode}")
            plan.append((f"训练：{mode}（同一起点）", _train(args, name, ["--modalities", mode])))
            plan.append((f"正式评测：{mode}",
                         _eval(args, str(Path(args.out) / name / "weights" / "best.pt"), name)))
    if args.stage in ("all", "standalone"):
        for mode in ("ir", "dep"):
            name = _run_name(args, f"exp_signal_{mode}")
            extra = ["--modalities", mode]
            if mode == "dep":
                extra += ["--depth-scales", "all"]
            plan.append((f"训练：仅 {mode} 像素信号（RGB 全零）", _train(args, name, extra)))
            plan.append((f"正式评测：仅 {mode}",
                         _eval(args, str(Path(args.out) / name / "weights" / "best.pt"), name)))
    if args.stage in ("all", "depth"):
        variants = (
            ("both_relative", "both", "relative"),
            ("relative_only", "relative", "relative"),
            ("metric_fallback", "metric_fallback", "metric_fallback"),
            ("metric_log_fallback", "metric_log_fallback", "metric_fallback"),
            ("both_balanced", "both", "balanced"),
        )
        for label, view, init in variants:
            name = _run_name(args, f"exp_depth_{label}")
            # 隔离直接 Depth 通道：关闭以相对深度计算的先验和质量图。
            extra = ["--modalities", "all", "--depth-view", view,
                     "--depth-init", init, "--no-prior", "--no-quality"]
            plan.append((f"训练：Depth {label}（无相对深度先验旁路）",
                         _train(args, name, extra)))
            plan.append((f"正式评测：Depth {label}",
                         _eval(args, str(Path(args.out) / name / "weights" / "best.pt"), name)))
    if args.stage in ("all", "augment"):
        # 单变量对照；基线是 modalities 阶段的 all。Mosaic/copy-paste 未接线，不伪称已测试。
        variants = (("crop20", ["--target-crop-p", "0.20"]),
                    ("rgbdegrade30", ["--degrade-p", "0.30"]),
                    ("rgbcolor15", ["--rgb-color-p", "0.15"]),
                    ("irgain15", ["--ir-gain-p", "0.15"]),
                    ("depthholes20", ["--depth-hole-p", "0.20"]),
                    ("no_jitter", ["--misalign-px", "0"]))
        for label, changes in variants:
            name = _run_name(args, f"exp_aug_{label}")
            plan.append((f"训练：增强 {label}（只改一个变量）",
                         _train(args, name, ["--modalities", "all", *changes])))
            plan.append((f"正式评测：增强 {label}",
                         _eval(args, str(Path(args.out) / name / "weights" / "best.pt"), name)))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("all", "diagnose", "modalities", "standalone",
                                        "depth", "augment"),
                    default="all")
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", required=True, help="同一份本地 COCO 权重；不能填 B1/B2 checkpoint")
    ap.add_argument("--split-file", required=True, help="固定的 train/val 划分")
    ap.add_argument("--baseline-ckpt", default="", help="仅诊断阶段使用的现有 B2 best.pt")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=str(HERE.parent / "runs"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", default="608x1088")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-clip", type=float, default=60.0,
                    help="新配方梯度裁剪阈值；旧 optfix_v3 精确续训须设为 10")
    ap.add_argument("--run-tag", default="", help="追加到 run 名称，隔离不同训练配方")
    ap.add_argument("--skip-stock", action="store_true",
                    help="复用已完成的官方 RGB 基线，只运行自定义四组训练/评测")
    args = ap.parse_args()
    if min(args.epochs, args.batch, args.accum) < 1 or args.workers < 0:
        ap.error("epochs/batch/accum 须为正数，workers 须为非负数")
    if args.grad_clip < 0:
        ap.error("grad-clip 须 >=0")
    for title, cmd in build_plan(args):
        print(f"\n# {title}\n{_format(cmd)}")
    print("\n# 以上仅打印命令；脚本没有执行任何训练或评测。")


if __name__ == "__main__":
    main()
