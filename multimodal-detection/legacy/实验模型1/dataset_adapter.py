# -*- coding: utf-8 -*-
"""
实验模型1 数据适配 —— 三模态**独立输入**（区别于基线模型2 的 5 通道拼接）。

模型 forward 需要 (rgb(B,3,H,W), ir(B,1,H,W), depth(B,1,H,W)) 三路输入，
本模块复用 common 的一致性增强（flip/letterbox 三图同步、仅 RGB HSV、Depth 最近邻）
与 depth 固定对齐/随机抖动，产出与模型接口对齐的三个张量。

标签：始终绑定 RGB 坐标系（bbox 同步变换；depth 对齐/抖动不改变标签）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for p in (str(_DIR.parent), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import random                                   # noqa: E402
import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
import models_config as MC                      # noqa: E402
from common import dataset as DS                # noqa: E402
from common import multimodal_augment as MA     # noqa: E402


def _read_source(sample, align):
    """读一个样本的原始三模态 + 归一化框（**增强前**、原图坐标系）。

    ⚠️ 三模态**必须同一分辨率**：mosaic/letterbox 用同一组仿射矩阵处理 RGB/IR/Depth，
       若某个模态分辨率不同（例如曾被优化成"RGB/IR 半分辨率解码、Depth 全分辨率"），
       同一仿射会把 Depth 放错尺度 → **三模态像素错位**（实测框内深度中位数偏差 70%）。
       因此这里一律全分辨率解码（实测半分辨率只快 ~5%，完全不值得）。
    返回 (rgb(BGR→RGB,H,W,3), ir(H,W), dep_mm(H,W), boxes(N,5)|None)。"""
    import cv2
    h = MC.EXPERIMENT1.hyper
    rgb = DS.read_rgb_bgr(sample.img["rgb"])                  # (H,W,3) BGR
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)                # P1-7: 统一 RGB 序
    ir = DS.read_ir_gray(sample.img["ir"])                    # (H,W) uint8
    dep_mm = DS.read_depth_mm(sample.img["depth"])            # (H,W) uint16 mm
    # 对齐量按 **depth 自身的原图宽** 换算（= 原图宽度；不能用降采样后的宽度）
    shift = MA.effective_depth_shift(align if align is not None else h.align,
                                     dep_mm.shape[1])         # P1-6
    dep_mm = MA.align_depth(dep_mm, *shift)                   # 固定平移对齐到 RGB 坐标系
    assert rgb.shape[:2] == ir.shape[:2] == dep_mm.shape[:2], \
        f"三模态分辨率不一致: rgb{rgb.shape[:2]} ir{ir.shape[:2]} depth{dep_mm.shape[:2]}（{sample.stem}）"
    boxes = None
    if sample.label is not None and Path(sample.label).exists():
        lab = DS.read_label_txt(sample.label)
        boxes = lab if lab.size else None
    return rgb, ir, dep_mm, boxes


def build_model_inputs(sample, imgsz=(1024, 1024), aug=None,
                       depth_shift=None, align=None, preprocess=None, seed=None,
                       to_tensor: bool = True, mosaic_pool=None):
    """
    读一个样本的三模态，按统一 aug 配置做一致性增强 + depth 固定对齐，
    返回 (rgb, ir, depth, boxes_out, stem)：
      rgb   : (3,H,W) RGB 序 或 tensor(B=1,3,H,W)
      ir    : (1,H,W)
      depth : (2,H,W)  = [归一化距离, 有效掩码]（Step1）
      boxes_out : (N,5) 归一化框(letterbox 画布) 或 None
    depth_shift / align：对齐平移；**align（AlignConfig）优先**——提供时按原图宽
    等比换算（P1-6: -22px@1920 是原图坐标系实测值），depth_shift 仅作显式覆盖。
    aug=None 时仅同步 letterbox（验证/推理路径）；preprocess=None 时读总配置。
    mosaic_pool：样本列表；当 aug.mosaic_p 命中时从池中随机取 3 个其它样本做 4 图拼接
    （三模态共用同一拼接布局，深度最近邻 + 0 填充）。
    """
    if sample.img.get("ir") is None or sample.img.get("depth") is None:
        raise ValueError(f"实验模型1 需要 ir/depth；样本 {sample.stem} 缺失")
    h = MC.EXPERIMENT1.hyper
    pp = preprocess if preprocess is not None else h.preprocess
    rng = random.Random(seed) if seed is not None else random

    rgb, ir, dep_mm, boxes = _read_source(sample, align)

    # 4 图拼接所需的额外 3 个源（仅训练、且 aug.mosaic_p>0 时）
    extras = None
    if (aug is not None and getattr(aug, "mosaic_p", 0.0) > 0
            and mosaic_pool and len(mosaic_pool) > 4):
        pool = [s for s in mosaic_pool if s.stem != sample.stem]
        if len(pool) >= 3:
            # 辅助源与主样本同样全分辨率读取（必须三模态同分辨率，见 _read_source 注释）
            extras = [_read_source(o, align) for o in rng.sample(pool, 3)]

    rgb_o, ir_o, dep_o, boxes_o = MA.consistent_augment_full(
        rgb, ir, dep_mm, boxes,
        new_size=imgsz, aug=aug, seed=seed, extras=extras, rng=rng)

    # 值域按 PreprocessParams 模式处理（赛题数据未归一化；默认 div255 / mm_unit）
    rgb_t = DS.norm_array(rgb_o, pp.rgb_mode).transpose(2, 0, 1)      # (3,H,W)
    ir_t = DS.norm_array(ir_o, pp.ir_mode)[None, ...]
    dep_sc = DS.depth_mm_to_scaled(dep_o, pp.depth_mode,
                                   pp.depth_scale_mm, pp.depth_invalid_zero)
    dep_mask = (dep_o > 1).astype(np.float32)                          # 有效掩码(增强后同步)
    dep_t = np.stack([dep_sc, dep_mask], axis=0)                       # (2,H,W)
    if to_tensor:
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_t)).float()
        ir_t = torch.from_numpy(np.ascontiguousarray(ir_t)).float()
        dep_t = torch.from_numpy(np.ascontiguousarray(dep_t)).float()
    return rgb_t, ir_t, dep_t, boxes_o, sample.stem
