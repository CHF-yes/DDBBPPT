"""One portable training recipe, not the suspended ablation queue.

Windows: EFYOLO python -u code/mm_yolo/run_spatial_memory.py
Linux:   python -u code/mm_yolo/run_spatial_memory.py --root /data/train --labels /data/labels
Resume: repeat the SAME command with --resume. Paths may change across hosts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
WORK = CODE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(WORK / "初赛数据集-面向城市场景的多模态目标检测" / "train_extracted"))
    ap.add_argument("--labels", default=str(WORK / "初赛数据集-面向城市场景的多模态目标检测" / "训练集" / "new_labels_2000"))
    ap.add_argument("--split-file", default=str(CODE / "runs/b2_depth4_s_safe/split.json"))
    ap.add_argument("--weights", default=str(CODE / "yolo11s.pt"))
    ap.add_argument("--out", default=str(CODE / "runs"))
    ap.add_argument("--name", default="spatial_memory_s_v1_s42")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", default="608x1088")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--repair-v2", action="store_true", help="bounded memory + gate warm-start; new run")
    ap.add_argument("--localization-finetune", action="store_true",
                    help="bounded-v2 best-weight fine-tune with weak geometry; does not reset fusion gates")
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--wait-pid", type=int, default=0,
                    help="wait for an existing trainer/launcher PID before starting this run")
    ap.add_argument("--smoke", action="store_true", help="32 samples / 2 epochs, separate correctness run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Child runs from CODE; resolve caller-relative paths before changing cwd.
    for field in ("root", "labels", "split_file", "weights", "out", "init_checkpoint"):
        if getattr(args, field):
            setattr(args, field, str(Path(getattr(args, field)).expanduser().resolve()))
    if args.repair_v2 and args.localization_finetune:
        raise ValueError("choose repair-v2 or localization-finetune, not both")
    v2_mode = args.repair_v2 or args.localization_finetune
    if args.localization_finetune and args.name == "spatial_memory_s_v1_s42":
        args.name = "spatial_memory_s_v2_locft_s42"
    elif args.repair_v2 and args.name == "spatial_memory_s_v1_s42":
        args.name = "spatial_memory_s_v2_repair_s42"
    if args.localization_finetune and args.epochs == 100:
        args.epochs = 20
    elif args.repair_v2 and args.epochs == 100:
        args.epochs = 80
    if args.init_checkpoint and not v2_mode:
        raise ValueError("checkpoint initialization requires --repair-v2 or --localization-finetune")
    if v2_mode and not args.resume and not args.init_checkpoint:
        raise ValueError("v2 warm-start/fine-tune requires an explicit --init-checkpoint")
    if args.resume and args.init_checkpoint:
        raise ValueError("resume does not reinitialize weights")
    if args.smoke and args.name == "spatial_memory_s_v1_s42":
        args.name = "_smoke_spatial_memory_v1"
    command = [sys.executable, "-u", str(HERE/"train.py"),
        "--root", args.root, "--labels", args.labels, "--out", args.out, "--name", args.name,
        "--weights", args.weights, "--modalities", "all", "--imgsz", args.imgsz,
        "--epochs", str(2 if args.smoke else args.epochs), "--batch", str(args.batch),
        "--accum", str(args.accum), "--workers", str(args.workers),
        "--architecture", "spatial_memory_v1", "--sampler", "coverage", "--rare-extra-frac", "0.10",
        "--depth-resampling", "nearest_valid_v2", "--metric-branch", "--depth-channels", "4",
        "--register-bus", "--depth-scales", "p4p5", "--share-tier", "c",
        "--no-prior", "--no-deformable",
        "--bn-policy", "adaptive", "--lr", "0.0003", "--backbone-lr-mult", "0.5",
        "--freeze-epochs", "1", "--warmup", str(0 if args.smoke else 5), "--lrf", "0.02",
        "--grad-clip", "60", "--calibrate-clip-steps", str(1 if args.smoke else 128),
        "--scale-min", "0.75", "--scale-max", "1.40", "--translate", "0.10",
        "--close-aug-frac", "0.20", "--target-crop-p", "0.15", "--misalign-px", "2",
        "--degrade-p", "0.12", "--rgb-color-p", "0.3", "--ir-noise-p", "0.08", "--ir-gain-p", "0.2",
        "--depth-hole-p", "0.08", "--rgb-dropout", "0.01", "--aux-dropout", "0.03",
        "--dropout-start-epoch", "5", "--seed", "42", "--val-every", "1", "--val-limit", "0",
        "--val-batch", "2", "--val-conf", "0.001", "--save-every", "1"]
    command += ["--limit", "32"] if args.smoke else ["--split-file", args.split_file]
    if v2_mode:
        replacements = {"--lr": "0.0001", "--freeze-epochs": "0", "--warmup": "0" if args.smoke else "3",
                        "--grad-clip": "48.49", "--bn-policy": "adaptive_no_tail"}
        for flag, value in replacements.items():
            command[command.index(flag)+1] = value
        command += ["--memory-control", "bounded_v2"]
        if args.repair_v2 and not args.resume:
            command += ["--init-checkpoint", args.init_checkpoint, "--reset-fusion-gates"]
    if args.localization_finetune:
        replacements = {"--lr": "0.00002", "--backbone-lr-mult": "0.5",
                        "--warmup": "1", "--lrf": "0.10", "--grad-clip": "26.35",
                        "--calibrate-clip-steps": "0", "--close-aug-frac": "0.50",
                        "--scale-min": "0.90", "--scale-max": "1.10", "--translate": "0.03",
                        "--target-crop-p": "0", "--misalign-px": "0", "--degrade-p": "0.03",
                        "--rgb-color-p": "0.10", "--ir-noise-p": "0.02", "--ir-gain-p": "0.05",
                        "--depth-hole-p": "0", "--rgb-dropout": "0", "--aux-dropout": "0"}
        for flag, value in replacements.items():
            command[command.index(flag)+1] = value
        if not args.resume:
            command += ["--init-checkpoint", args.init_checkpoint, "--eval-initial"]
    if args.resume:
        command += ["--resume"]
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS="4")
    run = Path(args.out) / args.name
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "weights/last.pt"
    if ckpt.exists() and not args.resume:
        raise FileExistsError(f"Existing checkpoint: {ckpt}; use --resume or a new --name")
    (run / "launch.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    status_file = run / "launcher_status.json"
    if args.wait_pid:
        import psutil
        status = {"state": "waiting", "pid": os.getpid(), "name": args.name,
                  "wait_for_pid": args.wait_pid}
        status_file.write_text(json.dumps(status), encoding="utf-8")
        while psutil.pid_exists(args.wait_pid):
            time.sleep(5)
    status = {"state": "running", "pid": os.getpid(), "name": args.name}
    status_file.write_text(json.dumps(status), encoding="utf-8")
    with (run / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        result = subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
    status.update(state="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
    status_file.write_text(json.dumps(status), encoding="utf-8")
    print(json.dumps(status), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
