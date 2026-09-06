# -*- coding: utf-8 -*-
"""
trainer —— 封装训练入口（统一读取 models_config 超参）。

分两种情形：
  * 3 通道（基线模型1，单模态 RGB）：可直接喂给 ultralytics 内建训练器，
    使用 common.scan_data.build_data_yaml() 生成的 ultralytics data.yaml。
  * 5 通道（基线模型2/实验模型1，三模态前期融合）：ultralytics 内建 DataLoader
    只支持"单图固定通道"，无法直接返回融合张量；因此正确定位是：
        - 先在本框架构造 5 通道模型（见基线模型2/model_builder.py 改首层）
        - 再以自定义 Dataset(common.dataset.MultimodalDetectionDataset) 喂训练。
      本模块提供对应 hook，具体在主程序里接线（见各实例 README）。

建议执行环境：EFYOLO conda 环境（已装 ultralytics + CUDA）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import models_config as MC

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
        amp=h.amp,
        seed=h.seed,
        patience=h.patience,
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
        f"  optimizer   : {h.optimizer}  lr0={h.lr0}  amp={h.amp}",
        f"  epochs      : {h.epochs}  patience={h.patience}",
        f"  device      : {h.device}",
        f"  class_num   : {cfg.class_num}",
    ]
    return "\n".join(lines)


__all__ = ["build_train_kwargs", "train_config_prints"]
