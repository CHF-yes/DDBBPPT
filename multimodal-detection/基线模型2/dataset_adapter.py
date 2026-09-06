# -*- coding: utf-8 -*-
"""
dataset_adapter —— 基线模型2 的数据适配（针对根目录中的三模态 + 标签格式）。

职责：把赛题"一个样本 = 空间对齐 RGB + IR + Depth + 同名标签 txt"组织成
5 通道可训练/可推理的形态。核心复用 common.dataset 的三模态读取与通道拼装，
这里给出与 根目录结构对接的入口。

「适配根目录格式」:
  - 扫描根目录找到样本集（用 common.scan_data）
  - 用 common.dataset.build_input_channels(+cfg.in_channels=5) 拼接 5 通道
  - 标签为同名 .txt(赛题格式 cls cx cy w h)

典型流程见 main.py 与 README；也可单独运行本文件打印一个样本通道概览。
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for p in (str(_DIR.parent), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                              # noqa: E402
import models_config as MC                       # noqa: E402
from common import scan_data as SD               # noqa: E402
from common import dataset as DS                 # noqa: E402


def build_5ch_from_sample_paths(paths: dict, imgsz=(1024, 1024),
                                depth_shift=None) -> np.ndarray:
    """
    入口：给样本三模态路径(如 {"rgb":..., "ir":..., "depth":...}) + 目标尺寸，
    返回 (5,H,W) float32 归一化张量（RGB·IR·Depth）。见 common.dataset.build_input_channels。
    depth_shift: 固定平移对齐(dx,dy)；None 时自动读 models_config 的 depth_shift_x/y。
    """
    if depth_shift is None:
        h = MC.BASELINE2_5CH.hyper
        depth_shift = (h.depth_shift_x, h.depth_shift_y)
    return DS.build_input_channels(paths, 5, target_size=imgsz,
                                   depth_shift=depth_shift)


def preview_sample(data_root: Path, stem: str) -> None:
    """扫描 data_root, 打印关于某一 stem 样本的可视信息(shape/含哪几模态)。"""
    res = SD.scan_samples(data_root)
    for split, samples in res.items():
        for s in samples:
            if s.stem == stem:
                chw = DS.build_input_channels(s.img, MC.BASELINE2_5CH.in_channels)
                info = {m: (p.name if p else None) for m, p in s.img.items()}
                print(f"[split={split}] {s.stem} -> 存在: { {k: v for k,v in info.items()} }")
                print(f"    合成 shape={chw.shape} dtype={chw.dtype} label={s.label.name if s.label else None}")
                return
    print(f"未找到 stem={stem}")


if __name__ == "__main__":
    # 简易自检：若 DATA_ROOT 存在则预览第一个样本
    root = Path(MC.DATA_ROOT)
    if not root.exists():
        print(f"数据根未就绪：{root}。请写入正确 DATA_ROOT（环境变量 {MC._DATA_ROOT_ENV}）")
    else:
        preview_sample(root, "0000001")
