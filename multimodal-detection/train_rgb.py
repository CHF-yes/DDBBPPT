# -*- coding: utf-8 -*-
"""统一 RGB 高质量训练入口。

这个入口只负责 RGB 单模态，并有意绕开多模态自定义训练循环。检测 loss、AdamW、
有效 batch/weight-decay 缩放、AMP、EMA 和 checkpoint 恢复均沿用钉死版本的
Ultralytics；本文件只增加可审计的数据视图、稀有类追加采样、逐图 NMS、防硬裁剪和
末段定位精修。

示例（Windows / EFYOLO）::

    python train_rgb.py --data-root <train_extracted> --labels <new_labels_2000> \
        --split-file configs/split_s42.json --weights yolo11m.pt

Linux 服务器使用相同参数，只需覆盖 --device/--batch/--workers，不含 Windows 写死路径。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
for entry in (str(VENDOR), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

import models_config as MC  # noqa: E402
from common import trainer as TR  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DATA_VIEW_VERSION = 1


def _utf8_console() -> None:
    """Keep redirected logs UTF-8 and avoid Windows console encoding crashes."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _keep_awake(enabled: bool) -> None:
    if os.name != "nt":
        return
    continuous, system_required, display_required = 0x80000000, 0x00000001, 0x00000002
    flags = continuous | system_required | (display_required if enabled else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _add_file_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = str(path.resolve())
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return
    handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)


def _resolve_rgb_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = (root / "visible", root / "train_extracted" / "visible")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"找不到 RGB 目录。--data-root 应指向包含 visible/ 的训练数据目录；收到：{root}")


def _read_classes(label: Path) -> tuple[int, ...]:
    classes: list[int] = []
    for line_no, raw in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"标签列数不是5：{label}:{line_no} -> {raw!r}")
        values = [float(x) for x in fields]
        cls = int(values[0])
        if values[0] != cls or not 0 <= cls < MC.CLASS_NUM:
            raise ValueError(f"类别越界：{label}:{line_no} -> {values[0]}")
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"标签含 NaN/Inf：{label}:{line_no}")
        x, y, w, h = values[1:]
        # 与钉死的 Ultralytics 校验口径一致，容忍标注浮点取整造成的 1% 边界外溢。
        if not (-0.01 <= x <= 1.01 and -0.01 <= y <= 1.01
                and 0 < w <= 1.01 and 0 < h <= 1.01):
            raise ValueError(f"非法归一化框：{label}:{line_no} -> {values[1:]}")
        classes.append(cls)
    return tuple(classes)


def _safe_link(src: Path, dst: Path, stage: Path) -> None:
    """Create a generated dataset view; only replaces files inside the exact stage directory."""
    src, stage = src.resolve(), stage.resolve()
    # resolve() 会跟随已有符号链接跳到源数据目录；这里只规范化父目录，保留目标链接本身。
    dst = dst.parent.resolve() / dst.name
    if stage not in dst.parents:
        raise RuntimeError(f"拒绝写到数据视图之外：{dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        if dst.is_dir():
            raise RuntimeError(f"生成目标意外是目录：{dst}")
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def _repeat_stems(stems: list[str], classes: dict[str, tuple[int, ...]],
                  target_images: int, max_repeat: int) -> tuple[list[str], dict[int, int]]:
    """Keep every unique image once, then append deterministic rare-class repeats."""
    image_freq = Counter(c for stem in stems for c in set(classes[stem]))
    repeat_for_class = {
        c: min(max_repeat, max(1, math.ceil(target_images / count)))
        for c, count in image_freq.items()
    } if target_images > 0 and max_repeat > 1 else {}
    expanded = list(stems)
    for stem in stems:
        factor = max((repeat_for_class.get(c, 1) for c in set(classes[stem])), default=1)
        expanded.extend([stem] * (factor - 1))
    return expanded, dict(sorted(image_freq.items()))


def _write_list(path: Path, stems: Iterable[str], linked_images: dict[str, Path]) -> None:
    lines = [linked_images[stem].resolve().as_posix() for stem in stems]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_yaml(path: Path, train_list: str, val_list: str) -> None:
    path.write_text(yaml.safe_dump({
        "path": path.parent.resolve().as_posix(),
        "train": train_list,
        "val": val_list,
        "names": {i: name for i, name in enumerate(MC.CLASS_NAMES)},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _prepare_smoke_yaml(stage: Path, train_n: int = 64, val_n: int = 2) -> Path:
    """Create a tiny real-data view for forward/backward/EMA/validation smoke tests."""
    def unique_head(path: Path, amount: int) -> list[str]:
        seen, result = set(), []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and line not in seen:
                seen.add(line)
                result.append(line)
            if len(result) >= amount:
                break
        return result

    train = unique_head(stage / "train.txt", train_n)
    val = unique_head(stage / "val.txt", val_n)
    if len(train) < train_n or len(val) < val_n:
        raise RuntimeError("真实数据不足，无法建立 smoke 清单")
    (stage / "smoke_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (stage / "smoke_val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    path = stage / "dataset_smoke.yaml"
    _write_yaml(path, "smoke_train.txt", "smoke_val.txt")
    return path


def prepare_dataset(data_root: Path, labels_dir: Path, split_file: Path, stage: Path,
                    target_images: int, max_repeat: int) -> dict:
    visible = _resolve_rgb_dir(data_root)
    labels_dir = labels_dir.expanduser().resolve()
    split_file = split_file.expanduser().resolve()
    if not labels_dir.is_dir() or not split_file.is_file():
        raise FileNotFoundError(f"标签目录或 split 不存在：{labels_dir} | {split_file}")

    split = json.loads(split_file.read_text(encoding="utf-8"))
    train, val = list(split.get("train", [])), list(split.get("val", []))
    if not train or not val or len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("split 的 train/val 必须非空且各自无重复")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f"split 泄漏：train/val 重叠 {overlap[:8]}")

    images = {p.stem: p for p in visible.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in labels_dir.glob("*.txt")}
    all_stems = sorted(set(train) | set(val))
    missing_images = [s for s in all_stems if s not in images]
    missing_labels = [s for s in all_stems if s not in labels]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"固定划分缺文件：images={missing_images[:8]} labels={missing_labels[:8]}")
    extra_images = sorted(set(images) - set(all_stems))
    if extra_images:
        raise ValueError(
            f"split 未覆盖 visible 中的 {len(extra_images)} 张图，前几项：{extra_images[:8]}")

    classes = {stem: _read_classes(labels[stem]) for stem in all_stems}
    linked_images: dict[str, Path] = {}
    for stem in all_stems:
        image_dst = stage / "images" / "all" / images[stem].name
        label_dst = stage / "labels" / "all" / f"{stem}.txt"
        _safe_link(images[stem], image_dst, stage)
        _safe_link(labels[stem], label_dst, stage)
        linked_images[stem] = image_dst

    train_expanded, train_image_freq = _repeat_stems(
        train, classes, target_images=target_images, max_repeat=max_repeat)
    full_expanded, full_image_freq = _repeat_stems(
        all_stems, classes, target_images=target_images, max_repeat=max_repeat)
    _write_list(stage / "train.txt", train_expanded, linked_images)
    _write_list(stage / "val.txt", val, linked_images)
    _write_list(stage / "all_train.txt", full_expanded, linked_images)
    _write_yaml(stage / "dataset_dev.yaml", "train.txt", "val.txt")
    _write_yaml(stage / "dataset_full.yaml", "all_train.txt", "val.txt")

    box_counts = Counter(c for stem in all_stems for c in classes[stem])
    report = {
        "version": DATA_VIEW_VERSION,
        "visible": str(visible),
        "labels": str(labels_dir),
        "split": str(split_file),
        "unique_train": len(train),
        "unique_val": len(val),
        "unique_all": len(all_stems),
        "train_entries_after_rare_append": len(train_expanded),
        "full_entries_after_rare_append": len(full_expanded),
        "rare_target_images": target_images,
        "rare_max_repeat": max_repeat,
        "train_class_image_counts": {str(k): v for k, v in train_image_freq.items()},
        "all_class_image_counts": {str(k): v for k, v in full_image_freq.items()},
        "all_class_box_counts": {str(k): v for k, v in sorted(box_counts.items())},
    }
    _json_dump(stage / "dataset_report.json", report)
    return report


def _metrics_payload(metrics, checkpoint: Path, split_file: Path, imgsz: int) -> dict:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "split": str(split_file.resolve()),
        "imgsz": imgsz,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class_95": {str(i): float(v) for i, v in enumerate(metrics.box.maps)},
        "names": {str(k): str(v) for k, v in metrics.names.items()},
    }


def _common_overrides(cfg: MC.ModelConfig, args, data_yaml: Path, project: Path,
                      name: str) -> dict:
    kw = TR.build_train_kwargs(cfg)
    kw.update({
        "data": str(data_yaml.resolve()),
        "project": str(project.resolve()),
        "name": name,
        "exist_ok": True,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "pretrained": True,
        "max_det": 100,
        "conf": 0.001,
        "iou": 0.7,
        "plots": False,
        "verbose": True,
        "save_period": args.save_period,
        "freeze": None,
    })
    # 极小 smoke 数据不足以达到正式 nbs=64 的累积步数；仅 smoke 改成每批更新一次，
    # 以实际覆盖 optimizer/scaler/EMA。正式训练仍严格使用注册表 nbs。
    if args.smoke:
        kw["nbs"] = args.batch
        kw["warmup_epochs"] = 0.0
    return kw


def _run_dev(cfg: MC.ModelConfig, args, run_root: Path, stage: Path, trainer_type,
             validator_type, state: dict) -> Path:
    phase_dir = run_root / "phase1_dev"
    last, best = phase_dir / "weights" / "last.pt", phase_dir / "weights" / "best.pt"
    if args.resume and last.is_file() and not state.get("phase1_complete"):
        LOGGER.info(f"Phase 1 断点续训（恢复 optimizer/scaler/EMA）：{last}")
        YOLO(str(last)).train(resume=True, trainer=trainer_type)
    elif not state.get("phase1_complete"):
        kw = _common_overrides(cfg, args, stage / args.dev_yaml, run_root, "phase1_dev")
        LOGGER.info(f"Phase 1 开始：固定 train/val，epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
        YOLO(str(args.weights)).train(trainer=trainer_type, **kw)
    if not best.is_file():
        raise FileNotFoundError(f"Phase 1 完成但不存在 best.pt：{best}")

    if not state.get("phase1_complete"):
        metrics = YOLO(str(best)).val(
            validator=validator_type,
            data=str((stage / args.dev_yaml).resolve()), split="val",
            imgsz=args.imgsz, batch=max(1, args.batch), workers=args.workers,
            device=args.device, conf=0.001, iou=0.7, max_det=100,
            plots=False, verbose=True)
        report = _metrics_payload(metrics, best, args.split_file, args.imgsz)
        _json_dump(phase_dir / "formal_eval.json", report)
        state.update({"phase1_complete": True, "phase1_best": str(best.resolve()),
                      "phase1_metrics": report})
        _json_dump(run_root / "state.json", state)
    return best


def _run_full(cfg: MC.ModelConfig, args, run_root: Path, stage: Path, trainer_type,
              source: Path, state: dict) -> Path:
    phase_dir = run_root / "phase2_full"
    last = phase_dir / "weights" / "last.pt"
    if args.resume and last.is_file() and not state.get("phase2_complete"):
        LOGGER.info(f"Phase 2 断点续训（恢复 optimizer/scaler/EMA）：{last}")
        YOLO(str(last)).train(resume=True, trainer=trainer_type)
    elif not state.get("phase2_complete"):
        h = cfg.hyper
        kw = _common_overrides(cfg, args, stage / "dataset_full.yaml", run_root, "phase2_full")
        kw.update({
            "epochs": args.full_finetune_epochs,
            "optimizer": "AdamW",
            "lr0": args.full_finetune_lr,
            "lrf": 0.20,
            "warmup_epochs": 1.0,
            "patience": 0,
            "val": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 0,
            "scale": 0.12,
            "translate": 0.03,
            "hsv_h": min(h.aug.hsv_h, 0.01),
            "hsv_s": min(h.aug.hsv_s, 0.25),
            "hsv_v": min(h.aug.hsv_v, 0.20),
        })
        LOGGER.info(
            f"Phase 2 开始：从 Phase 1 best 新建 AdamW/EMA，用全量数据低 LR 精修 "
            f"{args.full_finetune_epochs} 轮；此阶段验证集已进入训练，指标不作泛化成绩。")
        YOLO(str(source)).train(trainer=trainer_type, **kw)
    if not last.is_file():
        raise FileNotFoundError(f"Phase 2 完成但不存在 last.pt：{last}")
    if not state.get("phase2_complete"):
        state.update({"phase2_complete": True, "final_checkpoint": str(last.resolve())})
        _json_dump(run_root / "state.json", state)
    return last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="rgb_hq_11m", choices=MC.get_keys())
    parser.add_argument("--data-root", type=Path, required=True,
                        help="包含 visible/ 的训练数据目录；也可传其上一级目录")
    parser.add_argument("--labels", type=Path, required=True, help="修正版 YOLO 标签目录")
    parser.add_argument("--split-file", type=Path, required=True, help="固定 train/val split.json")
    parser.add_argument("--weights", type=Path, default=None, help="COCO 预训练权重；默认读统一注册表")
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--name", default="rgb_hq_11m_s42")
    parser.add_argument("--device", default=None, help="本机常用 0；服务器可用 0 或 0,1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="<=0 表示关闭；默认使用注册表（正式配置为关闭但记录范数）")
    parser.add_argument("--rare-target-images", type=int, default=None)
    parser.add_argument("--rare-max-repeat", type=int, default=None)
    parser.add_argument("--full-finetune-epochs", type=int, default=None)
    parser.add_argument("--full-finetune-lr", type=float, default=None)
    parser.add_argument("--no-full-finetune", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="只检查数据并生成清单，不加载模型、不训练")
    parser.add_argument("--smoke", action="store_true",
                        help="用少量真实图跑1轮前后向，不启动正式训练")
    parser.add_argument("--smoke-images", type=int, default=64,
                        help="smoke 使用的训练图数（默认64，验证固定2张）")
    return parser.parse_args()


def main() -> None:
    _utf8_console()
    args = parse_args()
    cfg = MC.get(args.config)
    if cfg.modality is not MC.Modality.RGB_ONLY:
        raise SystemExit("train_rgb.py 只接受 RGB_ONLY 注册项")
    h = cfg.hyper
    args.device = args.device if args.device is not None else h.device
    args.epochs = args.epochs if args.epochs is not None else h.epochs
    args.imgsz = args.imgsz if args.imgsz is not None else h.imgsz
    args.batch = args.batch if args.batch is not None else h.batch
    args.workers = args.workers if args.workers is not None else h.workers
    args.seed = args.seed if args.seed is not None else h.seed
    args.rare_target_images = (args.rare_target_images if args.rare_target_images is not None
                               else h.rare_target_images)
    args.rare_max_repeat = (args.rare_max_repeat if args.rare_max_repeat is not None
                            else h.rare_max_repeat)
    args.full_finetune_epochs = (args.full_finetune_epochs
                                 if args.full_finetune_epochs is not None
                                 else h.full_finetune_epochs)
    args.full_finetune_lr = (args.full_finetune_lr if args.full_finetune_lr is not None
                             else h.full_finetune_lr)
    clip = h.grad_clip_norm if args.grad_clip is None else args.grad_clip
    clip = None if clip is None or clip <= 0 else float(clip)
    if args.batch < 1 or args.epochs < 1 or args.imgsz < 320:
        raise SystemExit("batch/epochs 必须为正，imgsz 必须 >= 320")
    args.dev_yaml = "dataset_dev.yaml"
    if args.smoke:
        args.epochs = 1
        args.workers = 0
        args.no_full_finetune = True
        args.dev_yaml = "dataset_smoke.yaml"

    run_root = (args.out / args.name).expanduser().resolve()
    stage = run_root / "dataset"
    state_path = run_root / "state.json"
    if not args.resume and any((run_root / p / "weights" / "last.pt").exists()
                               for p in ("phase1_dev", "phase2_full")):
        raise FileExistsError(f"运行目录已有 checkpoint；请换 --name 或显式 --resume：{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    _add_file_log(run_root / "train.log")

    report = prepare_dataset(
        args.data_root, args.labels, args.split_file, stage,
        target_images=max(0, args.rare_target_images),
        max_repeat=max(1, args.rare_max_repeat))
    if args.smoke:
        _prepare_smoke_yaml(stage, train_n=max(args.batch, args.smoke_images))
    LOGGER.info(
        f"数据检查通过：unique train/val={report['unique_train']}/{report['unique_val']}，"
        f"追加后每轮 train entries={report['train_entries_after_rare_append']}")

    launch = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config": args.config,
        "command": sys.argv,
        "resolved": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gradient_clip_norm": clip,
        "ultralytics_source": str(Path(sys.modules["ultralytics"].__file__).resolve()),
        "dataset_report": report,
    }
    _json_dump(run_root / "launch.json", launch)
    LOGGER.info(TR.train_config_prints(cfg))
    if args.prepare_only:
        LOGGER.info("--prepare-only 完成：未加载模型、未开始训练。")
        return

    args.weights = (args.weights.expanduser().resolve() if args.weights is not None
                    else MC.resolve_pretrained_weights(h.pretrained_weights))
    if not args.weights.is_file():
        raise FileNotFoundError(f"预训练权重不存在：{args.weights}")
    TR.configure_rgb_trainer(clip)
    validator_type, trainer_type = TR.get_rgb_trainer_types()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "phase1_complete": False, "phase2_complete": False}

    _keep_awake(True)
    try:
        best = _run_dev(cfg, args, run_root, stage, trainer_type, validator_type, state)
        if args.no_full_finetune or args.full_finetune_epochs <= 0:
            final = best
            state.update({"final_checkpoint": str(final.resolve()), "phase2_skipped": True})
            _json_dump(state_path, state)
        else:
            final = _run_full(cfg, args, run_root, stage, trainer_type, best, state)
        _json_dump(run_root / "final_model.json", {
            "checkpoint": str(final.resolve()),
            "single_model": True,
            "single_forward": True,
            "nms": "one standard NMS per image",
            "tta": False,
            "ensemble": False,
        })
        LOGGER.info(f"训练流程完成，最终单模型权重：{final}")
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    main()
