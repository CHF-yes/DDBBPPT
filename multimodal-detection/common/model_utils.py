# -*- coding: utf-8 -*-
"""
model_utils —— 跨版本的 YOLO 构建 / 首层通道改造 / 类头设定工具。

集中两类"最小侵入"的 YOLO 结构操作，供 基线模型1 /2 /实验模型1 复用：
  1) rebuild_first_conv: 把 YOLO 首层输入通道从 3 扩到 5（RGB+IR+Depth 前期融合）；
  2) ensure_detect_classes: 类头的类别数处理说明 + 便捷封装。
提供权重继承策略（新增两通道从预训练 RGB 通道复制/置零两种保守选择）。

依赖 ultralytics 内建结构，函数内 lazy import；请在 EFYOLO 环境执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import models_config as MC


def build_base_model(weights: str):
    """载入权重/结构得到可训练对象。weights 可为 'yolo11s.pt'(code 根) 或 .yaml。"""
    from ultralytics import YOLO
    return YOLO(weights)


def ensure_detect_classes(model, class_num: int):
    """
    提示/封装：把检测分支类别数设定为 class_num(=12)。

    ultralytics 处理类头通常有两条路径：
      A.（推荐、官方）直接用 YOLO.train(data=你的12类data.yaml)：
          trainer 会按 data['nc'] 重新构建 Detect 头 → 无需手动改。
      B.（自定义前向 / 多模态自定义训练循环时）需显式重建 Detect n=class_num,
         本框架建议通过 .train(nc=class_num) 参数或在生成权重后用:
            model = YOLO(你的12类.pt)   （已含 nc=12 —— 训练后产物）
    因此一般不需要手工动 internals。返回 model 原样便于链式调用。
    """
    model.nc = int(class_num)   # 供自定义建模引用（不给训练器造成负担）
    return model


def rebuild_first_conv(model, in_channels: int,
                       strategy: str = "mean_rgb") -> None:
    """
    [核心] 把 YOLO 首层 Conv2d 的输入从 3 通道扩展为 in_channels 通道。

    做法：仅整体替换该层内部的 .conv（保持 stride/kernel/padding/groups/bias），
    不触碰后续任何层 —— 这就是"前期融合"的最小侵入。

    新增通道初始化策略 strategy：
      'mean_rgb'  新增通道 = 原 RGB 三通道权重的通道均值 × 系数（保守起步，推荐）
      'zero'      新增通道置 0（严格要求网络后续层自己学红外/深度投影）

    说明：预训练首层是 RGB 三通道权重。IR/Depth 与 RGB 统计不同，直接给一个
    很小的起始幅度即可：早期靠已冻结底层能力 + 少量新信号，收敛稳定。
    """
    import torch.nn as nn
    from ultralytics.nn.modules import Conv

    # 兼容两种形态：YOLO 对象(model.model=DetectionModel) 或 DetectionModel(model=Sequential)
    outer = getattr(model, "model", model)
    seq = getattr(outer, "model", outer)
    src = seq[0]
    if not isinstance(src, Conv):
        raise TypeError(f"预期首层为 ultralytics.nn.modules.Conv, 实得 {type(src)}")
    old = src.conv
    if old.in_channels != 3:
        raise ValueError(f"首层应为 3 通道，实为 {old.in_channels}（不适用于本扩展）")

    new = nn.Conv2d(
        in_channels, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation if hasattr(old, "dilation") else 1,
        groups=old.groups,
        bias=(old.bias is not None),
    )
    with torch_no_grad():
        w = old.weight.data                # (O,3,kh,kw)
        nw = new.weight.data
        nw[:, :3] = w                      # RGB 原样
        add = in_channels - 3
        if add > 0:
            if strategy == "mean_rgb":
                nw[:, 3:] = w.mean(dim=1, keepdim=True).expand(-1, add,
                                                               *w.shape[2:]) * 0.05
            elif strategy == "zero":
                nw[:, 3:].zero_()
            else:
                raise ValueError(f"未知 strategy: {strategy}")
        if new.bias is not None and old.bias is not None:
            new.bias.data.copy_(old.bias.data)
    # 整层替换内部 conv（外部 Conv wrapper 的属性仍是 nn.Conv2d，可保留 BN/act 等）
    src.conv = new
    print(f"[model_utils] 首层卷积 3→{in_channels} 通道，权重策略={strategy}")


def torch_no_grad():
    """模式守卫：尽量用真 torch.no_grad。"""
    import torch
    return torch.no_grad()


__all__ = ["build_base_model", "ensure_detect_classes", "rebuild_first_conv"]
