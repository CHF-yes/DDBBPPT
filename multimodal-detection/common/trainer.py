# -*- coding: utf-8 -*-
"""正式 RGB 训练所需的 Ultralytics 参数、验证器和梯度审计。"""

from __future__ import annotations

import math
import os
from copy import copy
from pathlib import Path
from typing import Optional

import models_config as MC

try:  # 保持纯配置工具在未装训练依赖时仍可 import；真正训练时会在下方给出明确错误。
    import torch
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import LOGGER, nms
    from ultralytics.utils.torch_utils import unwrap_model
except ImportError:  # pragma: no cover - 仅用于轻量配置环境
    torch = None
    DetectionTrainer = object
    DetectionValidator = object
    LOGGER = None
    nms = None

# code 根目录（models_config.py 所在目录）
CODE_ROOT: Path = Path(MC.__file__).resolve().parent


def build_train_kwargs(cfg: MC.ModelConfig) -> dict:
    """把 cfg.hyper 转成 ultralytics YOLO.train(**kw) 的参数 dict。"""
    h = cfg.hyper
    return dict(
        data=str(MC.DATA_ROOT),       # 占位；训练前用它复盖为实际 data.yaml 绝对路径
        epochs=h.epochs,
        imgsz=h.imgsz,
        batch=h.batch,
        device=h.device,
        workers=h.workers,
        optimizer=h.optimizer,
        lr0=h.lr0,
        lrf=h.lrf,
        momentum=h.momentum,
        weight_decay=h.weight_decay,
        warmup_epochs=h.warmup_epochs,
        nbs=h.nbs,
        cos_lr=h.cos_lr,
        amp=h.amp,
        seed=h.seed,
        deterministic=h.deterministic,
        patience=h.patience,
        cache=h.cache,
        multi_scale=h.multi_scale,
        box=h.box,
        cls=h.cls,
        dfl=h.dfl,
        hsv_h=h.aug.hsv_h if h.aug.hsv_rgb else 0.0,
        hsv_s=h.aug.hsv_s if h.aug.hsv_rgb else 0.0,
        hsv_v=h.aug.hsv_v if h.aug.hsv_rgb else 0.0,
        degrees=h.aug.degrees,
        translate=h.aug.translate,
        scale=h.aug.scale,
        shear=h.aug.shear,
        perspective=h.aug.perspective,
        flipud=h.aug.vflip_p,
        fliplr=h.aug.flip_p,
        mosaic=h.aug.mosaic_p,
        mixup=h.aug.mixup_p,
        cutmix=h.aug.cutmix_p,
        copy_paste=h.aug.copy_paste_p,
        close_mosaic=max(
            int(h.aug.close_mosaic_epochs),
            int(round(h.epochs * h.aug.close_mosaic_frac)),
        ),
        project=str(CODE_ROOT / "runs" / cfg.key),
        name="train",                 # 输出 runs/<key>/train
        exist_ok=True,
    )


def train_config_prints(cfg: MC.ModelConfig) -> str:
    """打印将被用于训练的配置清单（便于技术报告/复现）。"""
    h = cfg.hyper
    lines = [
        f"== train config [{cfg.key}] {cfg.name} ==",
        f"  in_channels : {cfg.in_channels}",
        f"  modality    : {cfg.modality.value}",
        f"  fusion      : {cfg.fusion.value}",
        f"  imgsz/batch : {h.imgsz} / {h.batch}",
        f"  optimizer   : {h.optimizer}  lr0={h.lr0}  lrf={h.lrf}  amp={h.amp}",
        f"  grad clip   : {h.grad_clip_norm if h.grad_clip_norm else 'disabled (monitor only)'}",
        f"  epochs      : {h.epochs}  patience={h.patience}",
        f"  device      : {h.device}",
        f"  class_num   : {cfg.class_num}",
    ]
    return "\n".join(lines)


def configure_rgb_trainer(grad_clip_norm: Optional[float], localization_scale: float = 0.12,
                          localization_translate: float = 0.03,
                          rect_train: bool = False) -> None:
    """Configure custom RGB trainer in an environment-safe way (also survives Ultralytics DDP spawn)."""
    value = 0.0 if grad_clip_norm is None else float(grad_clip_norm)
    os.environ["EFYOLO_GRAD_CLIP_NORM"] = str(value)
    os.environ["EFYOLO_LOCALIZATION_SCALE"] = str(float(localization_scale))
    os.environ["EFYOLO_LOCALIZATION_TRANSLATE"] = str(float(localization_translate))
    os.environ["EFYOLO_RECT_TRAIN"] = "1" if rect_train else "0"


def _report_gradient_health(trainer) -> None:
    from ultralytics.utils import LOGGER

    count = int(getattr(trainer, "_grad_count", 0))
    if not count:
        return
    finite_count = count - int(trainer._grad_nonfinite)
    mean = float(trainer._grad_sum / finite_count) if finite_count else float("nan")
    maximum = float(trainer._grad_max)
    clipped = int(trainer._grad_clipped)
    limit = trainer._grad_clip_norm
    LOGGER.info(
        f"gradient health: mean={mean:.3f} max={maximum:.3f} "
        f"nonfinite={trainer._grad_nonfinite}/{count} "
        f"amp_scale={trainer.scaler.get_scale():g} "
        f"clip={'off' if limit is None else f'{limit:g}'} triggered={clipped}/{count}"
    )
    trainer._grad_count = 0
    trainer._grad_sum = 0.0
    trainer._grad_max = 0.0
    trainer._grad_clipped = 0
    trainer._grad_nonfinite = 0


class CompleteDetectionValidator(DetectionValidator):
    """Run standard NMS once per image so a batch time limit cannot drop later images."""

    def postprocess(self, preds):
        prediction = preds[0] if isinstance(preds, (tuple, list)) else preds
        outputs = []
        for image in prediction.split(1, dim=0):
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


class HighQualityDetectionTrainer(DetectionTrainer):
    """Official detector trainer with audited NMS, gradient policy and late localization phase."""

    def __init__(self, *args, **kwargs):
        # Set before BaseTrainer builds datasets; the environment also survives
        # Ultralytics' optional DDP subprocess launch.
        self._rect_train = os.environ.get("EFYOLO_RECT_TRAIN", "0") == "1"
        super().__init__(*args, **kwargs)
        raw = float(os.environ.get("EFYOLO_GRAD_CLIP_NORM", "0"))
        self._grad_clip_norm = raw if raw > 0 and math.isfinite(raw) else None
        self._grad_count = 0
        self._grad_sum = 0.0
        self._grad_max = 0.0
        self._grad_clipped = 0
        self._grad_nonfinite = 0
        LOGGER.info(
            f"[rgb-hq] audited trainer active; gradient clip="
            f"{'off (monitor only)' if self._grad_clip_norm is None else self._grad_clip_norm}; "
            f"rect_train={self._rect_train}"
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Use true rectangular batches for the all-16:9 RGB data when requested.

        Ultralytics' detection trainer hard-codes rectangular batches to validation.
        Here all images have exactly the same aspect ratio, so training remains
        shuffleable while an ``imgsz=1280`` request produces a 736x1280 batch
        instead of a 1280x1280 canvas with 44% padding.
        """
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        rect = mode == "val" or (mode == "train" and self._rect_train)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=rect, stride=stride)

    def get_validator(self):
        return CompleteDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks)

    def optimizer_step(self):
        """Keep AMP/AdamW/EMA official semantics; make clipping explicit and measurable."""
        self.scaler.unscale_(self.optimizer)
        gradients = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
        if gradients:
            # 只读计算，绝不能用 max_norm=inf：若某步溢出，inf/inf 会把梯度污染为 NaN。
            norms = torch.stack([torch.linalg.vector_norm(g.float(), 2) for g in gradients])
            norm = torch.linalg.vector_norm(norms, 2)
            value = float(norm.detach().cpu())
        else:
            value = 0.0
        self._grad_count += 1
        if math.isfinite(value):
            self._grad_sum += value
            self._grad_max = max(self._grad_max, value)
            if self._grad_clip_norm is not None and value > self._grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip_norm)
                self._grad_clipped += 1
        else:
            # GradScaler.step() 会检测到非有限梯度、跳过该步并自动下调 scale。
            self._grad_nonfinite += 1
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def save_metrics(self, metrics):
        """Persist the per-epoch gradient audit next to official metrics."""
        if not self._grad_count:
            LOGGER.warning("[rgb-hq] epoch ended without an optimizer step; gradient audit is empty")
        _report_gradient_health(self)
        return super().save_metrics(metrics)

    def _close_dataloader_mosaic(self):
        """At the official close-mosaic boundary, also soften geometry for box refinement."""
        loc_scale = float(os.environ.get("EFYOLO_LOCALIZATION_SCALE", "0.12"))
        loc_translate = float(os.environ.get("EFYOLO_LOCALIZATION_TRANSLATE", "0.03"))
        self.args.scale = min(float(self.args.scale), loc_scale)
        self.args.translate = min(float(self.args.translate), loc_translate)
        self.args.hsv_s = min(float(self.args.hsv_s), 0.25)
        self.args.hsv_v = min(float(self.args.hsv_v), 0.20)
        LOGGER.info(
            f"Localization phase: mosaic/mixup off, scale={self.args.scale:g}, "
            f"translate={self.args.translate:g}, hsv_s={self.args.hsv_s:g}, "
            f"hsv_v={self.args.hsv_v:g}"
        )
        super()._close_dataloader_mosaic()


def get_rgb_trainer_types():
    if torch is None or nms is None:
        raise RuntimeError("训练依赖不可用：请在装有 torch 与 ultralytics 的 EFYOLO 环境运行。")
    return CompleteDetectionValidator, HighQualityDetectionTrainer


__all__ = [
    "build_train_kwargs", "train_config_prints", "configure_rgb_trainer",
    "get_rgb_trainer_types",
]
