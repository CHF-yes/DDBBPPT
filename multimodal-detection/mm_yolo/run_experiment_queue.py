# -*- coding: utf-8 -*-
"""顺序执行某一实验阶段；失败即停止，已有 last.pt 走严格续训。

与 experiments.py 的只打印入口分开。此脚本会正式启动训练，必须显式传
--stage modalities/standalone/depth/augment，不存在 --stage all，避免误跑全部实验。
队列每组完成后使用 conf=.001、batch=1 全量验证，再进入下一组。
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiments import build_plan


HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("modalities", "standalone", "depth", "augment"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", default=str(HERE.parent / "runs"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", default="608x1088")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--accum", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-clip", type=float, default=60.0,
                    help="新配方默认 60；续训旧 optfix_v3 队列须显式指定 10")
    ap.add_argument("--run-tag", default="", help="追加到 run/队列名称，防止覆盖旧配方")
    ap.add_argument("--skip-stock", action="store_true",
                    help="复用既有官方 RGB 基线，只跑四组自定义模态对照")
    ap.add_argument("--start-index", type=int, default=1,
                    help="从队列第 N 项续跑；之前各项必须已有 completed 记录及产物")
    args = ap.parse_args()
    args.baseline_ckpt = ""
    if min(args.epochs, args.batch, args.accum) < 1 or args.workers < 0:
        ap.error("epochs/batch/accum 须为正数，workers 须为非负数")
    if args.grad_clip < 0:
        ap.error("grad-clip 须 >=0")
    if args.run_tag and re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag) is None:
        ap.error("--run-tag 只允许字母、数字、下划线和连字符")
    for field in ("root", "labels", "weights", "split_file", "python"):
        value = Path(getattr(args, field)).resolve()
        if not value.exists():
            ap.error(f"{field} 路径不存在：{value}")
        setattr(args, field, str(value))
    if Path(args.weights).name.lower() in ("best.pt", "last.pt"):
        ap.error("模态因果对照必须从同一公开预训练权重开始，不能把已有实验 best/last 当 --weights")
    args.out = str(Path(args.out).resolve())
    plan = build_plan(args)
    if not 1 <= args.start_index <= len(plan):
        ap.error(f"--start-index 必须在 1..{len(plan)} 之间")
    tag_part = f"_{args.run_tag}" if args.run_tag else ""
    qdir = Path(args.out) / f"experiment_queue_{args.stage}{tag_part}_s{args.seed}"
    qdir.mkdir(parents=True, exist_ok=True)
    log_path = qdir / "queue.log"
    if args.start_index > 1:
        if not log_path.is_file():
            ap.error(f"续跑缺少队列历史：{log_path}")
        completed = set()
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("state") == "completed" and record.get("exit_code") == 0:
                completed.add((record.get("index"), record.get("title")))
        for index, (title, command) in enumerate(plan[:args.start_index - 1], start=1):
            if (index, title) not in completed:
                ap.error(f"第 {index} 项未完成，不能跳过：{title}")
            is_train = Path(command[1]).name in ("train.py", "stock_rgb_baseline.py")
            if is_train:
                name = command[command.index("--name") + 1]
                required = [Path(args.out) / name / "weights" / item
                            for item in ("last.pt", "best.pt")]
            else:
                required = [Path(command[command.index("--out") + 1])]
            if any(not path.is_file() for path in required):
                ap.error(f"第 {index} 项完成产物缺失，不能跳过：{required}")
    lock_path = qdir / "active.lock"
    try:
        handle = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        ap.error(f"队列锁已存在：{lock_path}；先确认没有另一队列在运行，勿并发覆盖实验")
    with os.fdopen(handle, "w", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    atexit.register(lambda: lock_path.unlink(missing_ok=True))
    status_path = qdir / "status.json"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    def report(index: int, title: str, state: str, exit_code=None):
        record = {"stage": args.stage, "index": index, "total": len(plan),
                  "title": title, "state": state, "exit_code": exit_code,
                  "time": datetime.now().isoformat(timespec="seconds"),
                  "pid": os.getpid()}
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(status_path)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[queue] {index}/{len(plan)} {state}: {title}" +
              (f" code={exit_code}" if exit_code is not None else ""), flush=True)

    for i, (title, original) in enumerate(plan, start=1):
        if i < args.start_index:
            continue
        command = list(original)
        script_name = Path(command[1]).name
        is_train = script_name in ("train.py", "stock_rgb_baseline.py")
        if is_train:
            name = command[command.index("--name") + 1]
            run_dir = Path(args.out) / name
            last = run_dir / "weights" / "last.pt"
            if last.exists():
                command.append("--resume")
            run_dir.mkdir(parents=True, exist_ok=True)
            console = run_dir / "console.log"
        else:
            ckpt = Path(command[command.index("--ckpt") + 1])
            if not ckpt.is_file():
                report(i, title, "missing_checkpoint", 2)
                return 2
            run_dir = ckpt.parent.parent
            console = run_dir / "eval_console.log"
        report(i, title, "running")
        started = time.monotonic()
        with console.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[queue] {datetime.now().isoformat()} START {title}\n")
            stream.flush()
            result = subprocess.run(command, cwd=str(HERE.parent.parent), env=env,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
            stream.write(f"[queue] END code={result.returncode} elapsed={time.monotonic()-started:.1f}s\n")
        if result.returncode != 0:
            report(i, title, "failed", result.returncode)
            return result.returncode
        if is_train:
            train_log = run_dir / "train.log"
            if not train_log.is_file() or any(
                    marker in train_log.read_text(encoding="utf-8", errors="replace")
                    for marker in ("[val-error]", "[FATAL]")):
                report(i, title, "training_log_error", 3)
                return 3
        report(i, title, "completed", 0)
    report(len(plan), "阶段全部完成", "complete", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
