# -*- coding: utf-8 -*-
"""
dataset_adapter —— 基线模型1（3 通道/仅 RGB）的数据适配。

基线1 只吃可见光，因此数据适配最简单：
  - 扫描根目录得到样本，过滤出 RGB 图；
  - 需要的是"一个只含 RGB 的图片目录 + 同名标签 txt"，正好是
    ultralytics 原版训练的输入形态（把 data.yaml 的 train/val 指向 RGB 图目录即可）。

本模块复现 基线模型2 里"适配根目录格式"的同款入口，
但因只用 RGB，不涉及多通道拼装；泛化成 common 的 scan 直接可用。
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for p in (str(_DIR.parent), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                      # noqa: E402
from common import scan_data as SD              # noqa: E402
from common import dataset as DS                # noqa: E402


def rgb_images_suggestion(data_root: Path) -> Path:
    """
    返回一个"应作为 RGB 图目录"的参考路径：
    由于 ultralytics 希望 train/val 各自是一个包含图片 + 同名 label，
    官方 layout 可将 RGB 图就放 split 目录（或 labels/ 目录承载 txt）。
    具体由 scan_data 输出目录结构与 data.yaml 决定。
    返回 data_root 本身（示意），真正使用前请以 scan 结果为准。
    """
    print("[提示] 真正训练时：把 data.yaml 的 train/val 指向'含 RGB 图与同名 txt'的目录。")
    return data_root


def preview_one(data_root: Path, stem: str) -> None:
    res = SD.scan_samples(data_root)
    for split, samples in res.items():
        for s in samples:
            if s.stem == stem and s.img.get("rgb"):
                print(f"[split={split}] {s.stem} rgb={s.img['rgb'].name} "
                      f"label={s.label.name if s.label else None}")
                return
    print(f"未找到 stem={stem}")


if __name__ == "__main__":
    root = Path(MC.DATA_ROOT)
    if not root.exists():
        print(f"数据根未就绪：{root}。请配置 DATA_ROOT。")
    else:
        preview_one(root, "0000001")
