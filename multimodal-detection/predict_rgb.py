# -*- coding: utf-8 -*-
"""Generate and strictly validate a competition RGB submission.

One checkpoint, one forward pass per image and one standard NMS are used. The
archive contains exactly one root-level TXT per visible test image, including
empty files when no detection survives NMS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for entry in (str(HERE / "vendor"), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils import nms, ops  # noqa: E402

import models_config as MC  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _parse_imgsz(value: str) -> int | list[int]:
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        size = int(text)
        return size
    height, width = (int(item) for item in text.split("x", 1))
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("imgsz 必须为正整数或 HxW")
    return [height, width]


def _images(source: Path) -> list[Path]:
    files = sorted(p for p in source.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"测试图像目录为空: {source}")
    stems = [p.stem for p in files]
    duplicates = sorted(k for k, v in Counter(stems).items() if v > 1)
    if duplicates:
        raise ValueError(f"不同扩展名存在同名测试图，无法生成唯一TXT: {duplicates[:10]}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_prediction(path: Path, xywhn: torch.Tensor, conf: torch.Tensor,
                      cls: torch.Tensor, nc: int) -> Counter:
    rows: list[tuple[int, float, float, float, float, float]] = []
    if len(conf):
        xywhn = xywhn.detach().float().cpu()
        conf = conf.detach().float().cpu()
        cls = cls.detach().long().cpu()
        order = conf.argsort(descending=True)[:100]
        for index in order.tolist():
            class_id = int(cls[index])
            if not 0 <= class_id < nc:
                raise ValueError(f"模型输出非法类别 {class_id}: {path.name}")
            x, y, w, h = (float(v) for v in xywhn[index].clamp_(0.0, 1.0))
            score = float(conf[index])
            values = (x, y, w, h, score)
            if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
                raise ValueError(f"模型输出非法数值 {values}: {path.name}")
            if w <= 0.0 or h <= 0.0:
                continue
            rows.append((class_id, x, y, w, h, score))
    text = "".join(
        f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f} {score:.8f}\n"
        for class_id, x, y, w, h, score in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    return Counter(row[0] for row in rows)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取测试图: {path}")
    return image


def _device(value: str) -> torch.device:
    text = str(value).strip().lower()
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    return torch.device(text)


def validate_submission(images: list[Path], labels: Path, nc: int) -> dict:
    expected = {p.stem for p in images}
    files = list(labels.glob("*.txt"))
    actual = {p.stem for p in files}
    if actual != expected:
        raise ValueError(
            f"TXT集合不匹配: missing={sorted(expected-actual)[:10]}, "
            f"extra={sorted(actual-expected)[:10]}"
        )
    detections = 0
    empty = 0
    classes: Counter = Counter()
    for path in files:
        previous = float("inf")
        count = 0
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            parts = raw.split()
            if len(parts) != 6:
                raise ValueError(f"{path.name}:{line_number} 需要6列，实际{len(parts)}列")
            class_value = float(parts[0])
            class_id = int(class_value)
            values = [float(v) for v in parts[1:]]
            if class_value != class_id or not 0 <= class_id < nc:
                raise ValueError(f"{path.name}:{line_number} 非法类别 {parts[0]}")
            if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
                raise ValueError(f"{path.name}:{line_number} 坐标或置信度越界")
            if values[2] <= 0.0 or values[3] <= 0.0:
                raise ValueError(f"{path.name}:{line_number} 框宽高必须为正")
            if values[4] > previous + 1e-12:
                raise ValueError(f"{path.name}:{line_number} 未按置信度降序")
            previous = values[4]
            classes[class_id] += 1
            count += 1
        if count > 100:
            raise ValueError(f"{path.name} 有{count}个框，超过100")
        detections += count
        empty += count == 0
    return {
        "images": len(images),
        "txt_files": len(files),
        "detections": detections,
        "empty_images": empty,
        "class_detections": {str(k): classes[k] for k in range(nc)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="测试集 visible 目录")
    parser.add_argument("--out", type=Path, required=True,
                        help="新建的TXT输出目录；为防混入旧文件，必须不存在或为空")
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--imgsz", default="544x960",
                        help="推理画布，默认544x960以匹配全体16:9测试图和正式验证")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    args.weights = args.weights.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.zip_path = args.zip_path.expanduser().resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not args.source.is_dir():
        raise NotADirectoryError(args.source)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝混入旧结果: {args.out}")
    if args.zip_path.exists():
        raise FileExistsError(f"压缩包已存在，拒绝覆盖: {args.zip_path}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)

    images = _images(args.source)
    imgsz = _parse_imgsz(args.imgsz)
    shape = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
    device = _device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA推理，但torch.cuda不可用")
    model = YOLO(str(args.weights)).model.to(device).eval()
    use_half = bool(args.half and device.type == "cuda")
    model.half() if use_half else model.float()
    letterbox = LetterBox(new_shape=shape, auto=False, scale_fill=False,
                          scaleup=True, stride=32)
    seen: set[str] = set()
    for start in range(0, len(images), args.batch):
        paths = images[start:start + args.batch]
        originals = [_read_image(path) for path in paths]
        resized = [letterbox(image=image) for image in originals]
        batch = np.stack([image[..., ::-1].transpose(2, 0, 1) for image in resized])
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        tensor = tensor.half() if use_half else tensor.float()
        tensor /= 255.0
        # 官方NMS会原位转换xywh→xyxy；使用no_grad而非inference_mode，避免
        # inference tensor离开上下文后禁止原位更新。
        with torch.no_grad():
            predictions = model(tensor)[0]
        outputs = []
        # 单张调用官方NMS，规避批级time_limit提前break而静默漏掉后续图片。
        for prediction in predictions.split(1, dim=0):
            outputs.extend(nms.non_max_suppression(
                prediction, args.conf, args.iou, nc=MC.CLASS_NUM,
                multi_label=True, agnostic=False, max_det=100,
            ))
        if len(outputs) != len(paths):
            raise RuntimeError(f"NMS返回{len(outputs)}张，当前批应为{len(paths)}张")
        for path, original, detections in zip(paths, originals, outputs):
            stem = path.stem
            if stem in seen:
                raise RuntimeError(f"模型重复返回测试图: {stem}")
            seen.add(stem)
            if len(detections):
                boxes = ops.scale_boxes(shape, detections[:, :4].clone(),
                                        original.shape[:2])
                xywhn = ops.xyxy2xywh(boxes)
                xywhn[:, (0, 2)] /= original.shape[1]
                xywhn[:, (1, 3)] /= original.shape[0]
                conf, cls = detections[:, 4], detections[:, 5]
            else:
                xywhn = detections.new_zeros((0, 4))
                conf = detections.new_zeros((0,))
                cls = detections.new_zeros((0,))
            _write_prediction(args.out / f"{stem}.txt", xywhn, conf, cls, MC.CLASS_NUM)
        print(f"processed {min(start + len(paths), len(images))}/{len(images)}", flush=True)
    missing_results = {p.stem for p in images} - seen
    if missing_results:
        raise RuntimeError(f"模型未返回{len(missing_results)}张图: {sorted(missing_results)[:10]}")

    report = validate_submission(images, args.out, MC.CLASS_NUM)
    with zipfile.ZipFile(args.zip_path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(args.out.glob("*.txt")):
            archive.write(path, arcname=path.name)
    report.update({
        "weights": str(args.weights),
        "weights_sha256": _sha256(args.weights),
        "source": str(args.source),
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": 100,
        "single_model": True,
        "single_forward": True,
        "tta": False,
        "ensemble": False,
        "zip": str(args.zip_path),
        "zip_sha256": _sha256(args.zip_path),
        "zip_bytes": args.zip_path.stat().st_size,
    })
    manifest = args.zip_path.with_suffix(args.zip_path.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
