# -*- coding: utf-8 -*-
"""正式 RGB 训练入口的最小配置层。

多模态模型拥有独立且自描述的 ``mm_yolo.config``，不再在这里保留已经停用的
早期融合基线和实验队列配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


CLASS_NAMES: List[str] = [
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage_can", "uav", "tricycle",
]
CLASS_NUM = len(CLASS_NAMES)
DATA_ROOT = Path(os.environ.get("MULTIMODAL_DATA_ROOT", r"D:\datasets\multimodal_det"))
_CODE_ROOT = Path(__file__).resolve().parent


def resolve_pretrained_weights(name: str | Path) -> Path:
    """解析本地权重；缺失时明确报错，避免静默联网下载。"""
    path = Path(str(name)).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(f"预训练权重不存在: {path}")
    for candidate in (_CODE_ROOT / path, Path.cwd() / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"找不到预训练权重 {name!r}；请放到 {_CODE_ROOT} 或传入绝对路径。"
    )


class Modality(Enum):
    RGB_ONLY = "rgb_only"


class FusionScheme(Enum):
    NONE = "none"


@dataclass
class AugmentParams:
    flip_p: float = 0.5
    vflip_p: float = 0.0
    scale: float = 0.45
    translate: float = 0.10
    mosaic_p: float = 0.80
    close_mosaic_epochs: int = 25
    close_mosaic_frac: float = 0.0
    degrees: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
    mixup_p: float = 0.05
    cutmix_p: float = 0.0
    copy_paste_p: float = 0.0
    hsv_rgb: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.60
    hsv_v: float = 0.35


@dataclass
class HyperParams:
    pretrained_weights: str = "yolo11m.pt"
    epochs: int = 200
    imgsz: int = 960
    batch: int = 4
    workers: int = 4
    device: str = "0"
    optimizer: str = "AdamW"
    lr0: float = 8e-4
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 5.0
    nbs: int = 64
    cos_lr: bool = True
    amp: bool = True
    seed: int = 42
    deterministic: bool = True
    patience: int = 50
    cache: object = False
    multi_scale: float = 0.0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    grad_clip_norm: Optional[float] = None
    rare_target_images: int = 80
    rare_max_repeat: int = 6
    full_finetune_epochs: int = 18
    full_finetune_lr: float = 1.5e-4
    aug: AugmentParams = field(default_factory=AugmentParams)


@dataclass
class ModelConfig:
    key: str
    name: str
    description: str
    in_channels: int = 3
    modality: Modality = Modality.RGB_ONLY
    fusion: FusionScheme = FusionScheme.NONE
    enabled: bool = True
    class_num: int = CLASS_NUM
    hyper: HyperParams = field(default_factory=HyperParams)
    notes: str = ""
    data_prep: str = ""


RGB_HQ_11M = ModelConfig(
    key="rgb_hq_11m",
    name="RGB高质量模型",
    description="YOLO11m COCO预训练的正式RGB单模态模型",
    notes="单模型、单权重、单次前向和标准逐图NMS；不使用TTA/WBF/投票。",
    data_prep="固定group-aware 1600/400划分；训练结束后全量2000图低学习率精修。",
)

_MODELS = [RGB_HQ_11M]
MODELS: Dict[str, ModelConfig] = {model.key: model for model in _MODELS}


def get_keys() -> List[str]:
    return list(MODELS)


def get(key: str) -> ModelConfig:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(f"未注册模型 {key!r}；可用模型: {get_keys()}") from exc


def summarize() -> str:
    return "\n".join(f"{model.key}: {model.description}" for model in _MODELS)


if __name__ == "__main__":
    print(summarize())
