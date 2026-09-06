# -*- coding: utf-8 -*-
"""
model_builder —— 基线模型2 = YOLO 三模态「5 通道前期融合」骨架。

改造要点（最小侵入）：
  - 载入 COCO 预训练 YOLO11（如 yolo11s.pt）
  - 首层卷积 3 → 5 通道（RGB 前 3 + IR/Depth 后 2），权重继承自预训练 RGB
  - 检测头由 COCO 80 类 → 赛题 12 类（经 data nc 或训练参数设定，见 ensure_detect_classes）
运行方式：
    python main.py train|predict   （main.py 会先注入 code 根 sys.path）
或作为子模块被 main.py import:
    # 需要其所在目录在 sys.path
    import model_builder
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- 路径引导：让本文件可独立运行/被 import 时都能 use common & config ----
_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                       # noqa: E402
from common import model_utils as MU              # noqa: E402


def build_baseline2(weights: str = None, class_num: int = MC.CLASS_NUM):
    """
    构建 5 通道前期融合模型。
    weights: 预训练权重路径，默认取 cfg.hyper.pretrained_weights。
    """
    cfg = MC.BASELINE2_5CH
    w = weights or cfg.hyper.pretrained_weights
    model = MU.build_base_model(w)              # ultralytics YOLO 对象
    # 首层 3→5
    MU.rebuild_first_conv(model, cfg.in_channels, strategy="mean_rgb")
    MU.ensure_detect_classes(model, class_num)
    print(f"[baseline2] 已构建 {cfg.name}  首层输入={cfg.in_channels} 类={cfg.class_num}")
    return model


if __name__ == "__main__":
    model = build_baseline2()
    print("构建成功示例 —— 权重默认取 cfg.hyper 的 yolo11s.pt（可在 code 根放置本地权重）。")
    print("请在 EFYOLO conda 环境执行；进一步内容见 main.py / README。")
