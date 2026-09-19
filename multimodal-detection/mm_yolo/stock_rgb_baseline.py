# -*- coding: utf-8 -*-
"""用未经多模态封装的官方 Ultralytics YOLO11s 校准 RGB 单模态上限。

该入口只使用官方训练集的 visible 图像和标签，并严格复用多模态实验的固定
train/val stem 划分。它会在运行目录旁建立硬链接（不复制图像内容）；硬链接不可用时
依次回退到符号链接和复制。训练完成后用 best.pt、conf=.001、max_det=100 再做一次
全量验证，将结果写入 formal_eval.json。
"""
from __future__ import annotations

import argparse
from copy import copy
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for item in (str(CODE), str(CODE / "vendor"), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402
from ultralytics.utils import nms  # noqa: E402

from config import COMPETITION_CLASS_NAMES  # noqa: E402
from train import prevent_sleep  # noqa: E402


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class CompleteDetectionValidator(DetectionValidator):
    """仅防止批量 NMS 超时后跳过剩余图片；不改模型、阈值或匹配算法。"""

    def postprocess(self, preds):
        prediction = preds[0] if isinstance(preds, (tuple, list)) else preds
        outputs = []
        for image in prediction.split(1, dim=0):
            # vendor NMS 在处理完当前图片之后才检查时间并 break。
            # 每次只喂一张图，即使某张超时也不会漏掉同批后面的图片。
            outputs.extend(nms.non_max_suppression(
                image, self.args.conf, self.args.iou,
                nc=0 if self.args.task == "detect" else self.nc,
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
                end2end=self.end2end,
                rotated=self.args.task == "obb",
            ))
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5],
                 "extra": x[:, 6:]} for x in outputs]


class CompleteDetectionTrainer(DetectionTrainer):
    """保留官方训练流程，只替换不完整的批量 NMS 验证路径。"""

    def get_validator(self):
        return CompleteDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks)


def _link(src: Path, dst: Path) -> None:
    """建立零拷贝数据视图；已有正确目标保持不动。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
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


def prepare_dataset(root: Path, labels: Path, split_file: Path, stage: Path) -> Path:
    visible = root / "visible"
    if not visible.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"RGB/标签目录不存在：{visible} | {labels}")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    train, val = list(split.get("train", [])), list(split.get("val", []))
    if not train or not val or set(train) & set(val):
        raise ValueError("split.json 必须含非空且互斥的 train/val")

    images = {p.stem: p for p in visible.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    label_files = {p.stem: p for p in labels.glob("*.txt")}
    requested = train + val
    missing_images = [s for s in requested if s not in images]
    missing_labels = [s for s in requested if s not in label_files]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"固定划分找不到文件：images={missing_images[:8]} labels={missing_labels[:8]}")

    image_dir, label_dir = stage / "images" / "all", stage / "labels" / "all"
    lines: dict[str, list[str]] = {"train": [], "val": []}
    for part, stems in (("train", train), ("val", val)):
        for stem in stems:
            image_dst = image_dir / images[stem].name
            label_dst = label_dir / f"{stem}.txt"
            _link(images[stem], image_dst)
            _link(label_files[stem], label_dst)
            lines[part].append(image_dst.resolve().as_posix())
        (stage / f"{part}.txt").write_text("\n".join(lines[part]) + "\n", encoding="utf-8")

    data_yaml = stage / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": stage.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "names": {i: name for i, name in enumerate(COMPETITION_CLASS_NAMES)},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", default=str(CODE / "runs"))
    parser.add_argument("--name", default="stock_rgb_yolo11s_s42")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(f"[stock-rgb] 防休眠: {prevent_sleep(True)} | 逐图 NMS 防漏算已启用", flush=True)

    project = Path(args.out).resolve()
    run_dir = project / args.name
    stage = project / f"_{args.name}_dataset"
    data_yaml = prepare_dataset(Path(args.root).resolve(), Path(args.labels).resolve(),
                                Path(args.split_file).resolve(), stage)
    last = run_dir / "weights" / "last.pt"
    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume 但不存在：{last}")
        print(f"[stock-rgb] 官方断点续训（恢复 optimizer/scaler/EMA） {last}", flush=True)
        YOLO(str(last)).train(resume=True, trainer=CompleteDetectionTrainer)
    else:
        print("[stock-rgb] 官方 YOLO11s RGB 校准："
              f"train=1600 val=400 imgsz={args.imgsz} batch={args.batch}", flush=True)
        YOLO(str(Path(args.weights).resolve())).train(
            trainer=CompleteDetectionTrainer,
            data=str(data_yaml), project=str(project), name=args.name, exist_ok=True,
            epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, workers=args.workers,
            device=args.device, seed=args.seed, deterministic=True, optimizer="auto",
            pretrained=True, amp=True, patience=30, max_det=100, conf=0.001,
            iou=0.7, plots=False, verbose=True)

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"训练完成但不存在 best.pt：{best}")
    metrics = YOLO(str(best)).val(
        validator=CompleteDetectionValidator,
        data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, conf=0.001, iou=0.7,
        max_det=100, plots=False, verbose=True)
    report = {
        "model": "official_ultralytics_yolo11s_rgb",
        "checkpoint": str(best),
        "split": str(Path(args.split_file).resolve()),
        "train_images": 1600,
        "val_images": 400,
        "imgsz": args.imgsz,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class_95": {str(i): float(v) for i, v in enumerate(metrics.box.maps)},
        "names": {str(k): str(v) for k, v in metrics.names.items()},
    }
    (run_dir / "formal_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "train.log").write_text(
        f"[stock-rgb] 完成 mAP50-95={report['map50_95']:.6f} "
        f"mAP50={report['map50']:.6f} best={best}\n", encoding="utf-8")
    print((run_dir / "train.log").read_text(encoding="utf-8"), end="", flush=True)


if __name__ == "__main__":
    main()
