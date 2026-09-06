# -*- coding: utf-8 -*-
"""
model_builder —— 基线模型1 = 3 通道原版 YOLO（仅可见光 RGB）的构建。

不改网络结构、不改首层通道，仅仅是"载入 COCO 预训练权重 + 把类头做成赛题 12 类"。
作为单模态对照基线，用于衡量多模态融合的增益。

本文件与 基线模型2/model_builder.py 保持**对称**，便于两人/两种结构消融对照。
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                      # noqa: E402
from common import model_utils as MU            # noqa: E402


def build_baseline1(weights: str = None, class_num: int = MC.CLASS_NUM):
    """载入预训练权重（3ch 单模态即可），只修类头到赛题 12 类。"""
    cfg = MC.BASELINE1_3CH
    w = weights or cfg.hyper.pretrained_weights
    model = MU.build_base_model(w)
    # 3 通道无需重建首层，仅提示类头设定
    MU.ensure_detect_classes(model, class_num)
    print(f"[baseline1] 已构建 {cfg.name}  in_channels={cfg.in_channels} 类={cfg.class_num}")
    return model


if __name__ == "__main__":
    build_baseline1()
    print("基线模型1 = 原版 3 通道 YOLO。载入权重见 main.py / README。")
