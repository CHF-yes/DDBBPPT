# -*- coding: utf-8 -*-
"""
multimodal_augment —— 面向三模态的"一致性增强"原语。

为什么需要"一致"：
   赛题三模态(RGB+IR+Depth)是同一相机组、空间对齐。训练时若对三张图分别做随机的
   几何变换(各翻各的、各裁各的)，会立刻破坏像素对齐 → 框与图像错位、跨模态互补失效。
   因此所有**像素位置相关的几何操作**(flip / crop / letterbox / scale)必须对三张图
   **共享同一份参数**执行。

策略细分：
   - 几何(改像素位置)：对 RGB/IR/Depth 三图使用同一 warp 参数 —— 保证一一对应。
   - 颜色/光度(不改位置)：只对 RGB 做 HSV 等抖动；IR/Depth 不应用，避免伪造温度/距离语义。
   - Depth 特殊性：
       * 用 INTER_NEAREST 缩放/插值，防止在遮挡边界(无效0与有效 mm 交界)引入插值伪值；
       * 不参与翻转外的"颜色扰动"。
   - 标签框同步：翻转/letterbox 后按同一映射更新 bbox。

本模块只提供"镜头级"一致性透明操作与 bbox 同步，不接管训练器主循环(见 README 注释、
进阶阶段把这些镜头接进自定义 DataLoader)。未在顶层 import cv2，函数内 lazy import。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 默认 letterbox 填充色(与 ultralytics 一致, 灰 114；Depth 用 0 表示无效)
_RGB_PAD = 114
_DEPTH_PAD_FLOAT = 0.0
_IR_PAD = 114


# ------------------------------------------------------------
# 像素几何：同步 warp
# ------------------------------------------------------------

def _cv():
    import cv2
    return cv2


def flip_lr_consistent(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
                       boxes_norm: Optional[np.ndarray] = None, p: float = 0.5,
                       rng=None):
    """
    (可选)水平翻转, 三图用**同一** p 决策 — 要么一起翻要么都不翻。

    rgb  : (H,W,3)
    ir   : (H,W) uint8
    depth: (H,W) float/uint16 (缩放值域无所谓, 只是镜像)
    boxes_norm: (N,5) cls,cx,cy,w,h 归一化(以原始未翻转图像为基准)。
                翻转后 cx -> 1-cx，w/h 不变。

    返回 (rgb,ir,depth, boxes_out) boxes_out 与输入同参考系(未 letterbox)。
    """
    roll = (rng if rng is not None else random).random()
    boxes = np.array(boxes_norm, dtype=np.float32).reshape(-1, 5).copy() if boxes_norm is not None else None
    if roll >= p:
        return rgb, ir, depth, boxes

    # 保证单通道与 3 通道各自镜像, 避免 numpy 翻转对深度的语义混乱
    def _flip(img):
        if img.ndim == 3:
            return img[:, ::-1, :]
        return img[:, ::-1]

    if boxes is not None and boxes.size:
        boxes[:, 1] = 1.0 - boxes[:, 1]      # cx
        # (若参考系是 letterbox 画布, 会因奇数 pad 有半个像素偏差, 一般可忽略)
    return _flip(rgb), _flip(ir), _flip(depth), boxes


def letterbox_consistent(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
                         new_size=(1024, 1024)):
    """
    三模态**同步 letterbox**：对三图取**同一组 (scale, pad dx/dy)**，等比缩放到
    new_size 后居中填充——保证三图 pad 位置一致、像素仍逐位对齐。
    - rgb  : (H,W,3) INTER_LINEAR, 填 114 (灰)
    - ir   : (H,W)   单通道, 填 114
    - depth: (H,W)   INTER_NEAREST + 填 0, 防止深度边界插值伪值
    返回 (rgb,ir,depth[, info]) info=dict(scale,dx,dy,old_h,old_w,new_size)
    """
    iv = _cv()
    H, W = rgb.shape[:2]
    nh, nw = new_size[0], new_size[1]
    # 三图共享同一个缩放比与 pad —— 关键
    scale = min(nh / H, nw / W)
    th, tw = int(round(H * scale)), int(round(W * scale))
    th, tw = max(th, 1), max(tw, 1)
    dx = (nw - tw) // 2
    dy = (nh - th) // 2

    def letter(img, value, interp):
        resized = iv.resize(img, (tw, th), interpolation=interp)
        if img.ndim == 3:
            canvas = np.full((nh, nw, img.shape[2]), value, img.dtype)
        else:
            canvas = np.full((nh, nw), value, img.dtype)
        canvas[dy:dy + th, dx:dx + tw] = resized
        return canvas

    rgb_o = letter(rgb, _RGB_PAD, iv.INTER_LINEAR)
    ir_o = letter(ir, _IR_PAD, iv.INTER_LINEAR)
    dep_o = letter(depth, _DEPTH_PAD_FLOAT, iv.INTER_NEAREST)
    info = dict(scale=scale, dx=dx, dy=dy, old_h=H, old_w=W, new_size=tuple(new_size))
    return rgb_o, ir_o, dep_o, info


# ------------------------------------------------------------
# Depth 专用平移（对齐 + 随机抖动）
# ------------------------------------------------------------

def effective_depth_shift(align, target_w: int):
    """按 AlignConfig 计算实际平移量（含按分辨率等比缩放）。返回 (sx, sy)。"""
    if align is None:
        return 0, 0
    if getattr(align, "mode", "shift") == "none":
        return 0, 0
    sx, sy = int(align.shift_x), int(align.shift_y)
    if getattr(align, "scale_with_res", True) and int(getattr(align, "ref_size", 0) or 0) > 0:
        k = float(target_w) / float(align.ref_size)
        sx, sy = round(sx * k), round(sy * k)
    return int(sx), int(sy)


def align_depth(depth: np.ndarray, shift_x=0, shift_y=0) -> np.ndarray:
    """
    固定平移对齐：把 depth 内容平移 (shift_x, shift_y) 像素到目标坐标系(如 RGB)。
    - 用 INTER_NEAREST + 填 0(无效)，防止深度边界插值伪值；
    - 标签框不跟随（标签锚定 RGB 坐标系，谁偏谁修）；
    - shift_x < 0 表示内容左移（实测 depth 偏右 ≈ -22px @1920×1080）。
    """
    if int(shift_x) == 0 and int(shift_y) == 0:
        return depth
    iv = _cv()
    h, w = depth.shape[:2]
    M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
    return iv.warpAffine(depth, M, (w, h), flags=iv.INTER_NEAREST,
                         borderMode=iv.BORDER_CONSTANT, borderValue=0)


def random_shift_depth(depth: np.ndarray, jitter_x=(-25, 5), jitter_y=(-5, 5),
                       prob: float = 1.0, rng=None) -> np.ndarray:
    """
    随机平移扰动（**仅训练**）: 在区间内随机平移 depth，模拟未对齐残差，
    让网络学会"深度内容偏一点也能用"（赛题鲁棒性要求）。
    与 align_depth 的区别：这是增强而非配准 —— 标签框**不跟随**（标签绑 RGB）。
    """
    r = rng if rng is not None else random
    if r.random() >= prob:
        return depth
    sx = r.randint(int(jitter_x[0]), int(jitter_x[1]) + 1)
    sy = r.randint(int(jitter_y[0]), int(jitter_y[1]) + 1)
    return align_depth(depth, sx, sy)


def ir_gain_jitter(ir: np.ndarray, gain: float = 0.15, bias: float = 5.0,
                   rng=None) -> np.ndarray:
    """
    IR 光度抖动（**仅训练**）：灰度增益(乘性) + 偏置(加性)，模拟传感器响应差异。
    只改亮度、不改"相对温差"语义；输出 clip 到 [0,255] uint8。
    """
    if gain <= 0 and bias <= 0:
        return ir
    r = rng if rng is not None else random
    g = 1.0 + r.uniform(-gain, gain)
    b = r.uniform(-bias, bias)
    out = np.clip(ir.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    return out


def depth_value_noise(depth: np.ndarray, ratio: float = 0.02, rng=None) -> np.ndarray:
    """
    Depth 值抖动（**仅训练**）：对有效像素(>1)施加乘性噪声，模拟测距抖动。
    无效区(0/1)保持原样，避免伪造"有深度"。
    """
    if ratio <= 0:
        return depth
    r = rng if rng is not None else random
    d = depth.astype(np.float32)
    valid = d > 1
    if not bool(valid.any()):
        return depth
    scale = 1.0 + r.uniform(-ratio, ratio)
    d[valid] = d[valid] * scale
    return d.astype(depth.dtype)


def rgb_dropout(rgb: np.ndarray, prob: float = 0.2, mode: str = "zero",
                rng=None) -> np.ndarray:
    """
    RGB 随机失效（**仅训练**）：按 prob 概率让整张 RGB 失效，
    迫使网络在"RGB 不可用"时依赖 IR/Depth（防主导模态垄断；赛题鲁棒性要求）。

    mode:
      * zero  : 全黑（相机完全失效/遮挡）
      * gray  : 转灰度并复制三通道（低照度/彩色退化，保留结构）
      * noise : 压暗 + 高斯噪声（弱光 / 传感器噪声）
    标签不跟随（标签锚定 RGB 坐标系，与内容无关）。
    """
    r = rng if rng is not None else random
    if r.random() >= prob:
        return rgb
    iv = _cv()
    if mode == "gray":
        g = iv.cvtColor(rgb, iv.COLOR_BGR2GRAY)
        return np.stack([g, g, g], axis=-1)
    if mode == "noise":
        out = rgb.astype(np.float32) * r.uniform(0.2, 0.6)
        noise = np.asarray([r.gauss(0.0, 15.0) for _ in range(3)], dtype=np.float32)
        out = out + noise[None, None, :]
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.zeros_like(rgb)                      # zero（默认）


# ------------------------------------------------------------
# 颜色(仅 RGB)
# ------------------------------------------------------------

def hsv_only_rgb(rgb: np.ndarray, h_gain=0.015, s_gain=0.7, v_gain=0.4,
                 rng=None) -> np.ndarray:
    """只对 RGB(BGR) 做 HSV 扰动；IR/Depth 永不参与（避免伪造温度/距离语义）。"""
    iv = _cv()
    h, s, v = iv.split(iv.cvtColor(rgb, iv.COLOR_RGB2HSV))
    h, s, v = (x.astype(np.float32) for x in (h, s, v))
    r = rng if rng is not None else random
    h = (h + r.uniform(-h_gain, h_gain) * 180) % 180
    s = np.clip(s * (1 + r.uniform(-s_gain, s_gain)), 0, 255)
    v = np.clip(v * (1 + r.uniform(-v_gain, v_gain)), 0, 255)
    hsv = iv.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    return iv.cvtColor(hsv, iv.COLOR_HSV2RGB)


# ------------------------------------------------------------
# bbox 在同步变换后更新
# ------------------------------------------------------------

def transform_boxes_letterbox(boxes_norm: np.ndarray, info: dict) -> np.ndarray:
    """
    把归一化框(以原图为基准) 转成 letterbox 后同画布的归一化框。

    boxes_norm: (N,5) cls, cx,cy,w,h, 分别在旧图[0..old_w]/[0..old_h]内归一。
    info: letterbox_consistent 返回信息（scale/dx/dy/new_size）
    返回同形状的(cx,cy,w,h 以新画布归一)。
    """
    boxes = np.array(boxes_norm, dtype=np.float32).reshape(-1, 5).copy()
    ow, oh = info["old_w"], info["old_h"]
    nh, nw = info["new_size"]
    s = info["scale"]
    dx, dy = info["dx"], info["dy"]
    if boxes.size == 0:
        return boxes
    # 归一中心 -> 厘米像素
    px = boxes[:, 1] * ow
    py = boxes[:, 2] * oh
    pw = boxes[:, 3] * ow
    ph = boxes[:, 4] * oh
    # 缩放+pad
    px = px * s + dx
    py = py * s + dy
    pw = pw * s
    ph = ph * s
    # 转回新画布归一
    boxes[:, 1] = np.clip(px / nw, 0, 1)
    boxes[:, 2] = np.clip(py / nh, 0, 1)
    boxes[:, 3] = np.clip(pw / nw, 0, 1)
    boxes[:, 4] = np.clip(ph / nh, 0, 1)
    return boxes


# ------------------------------------------------------------
# 组装
# ------------------------------------------------------------

def consistent_augment_full(
    rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
    boxes_norm: Optional[np.ndarray],
    new_size=(1024, 1024),
    aug: Optional["AugmentParams"] = None,
    seed: Optional[int] = None,
):
    """
    三模态「一致性 + 全覆盖」增强组装（由统一配置 AugmentParams 驱动）：

      (1) 几何：水平/垂直翻转三图**同步**(同一掷骰)；是否翻转由 aug.flip_p/vflip_p；
      (2) RGB 光度：HSV 抖动（aug.hsv_rgb/hsv_h/s/v）；
      (3) IR 光度：灰度增益/偏置抖动（aug.ir_gain/ir_bias）；
      (4) Depth 光度：有效区乘性噪声（aug.depth_noise）；
      (5) Depth 几何：随机平移模拟对齐残差（aug.depth_jitter_*，标签不动）；
      (6) 几何：同步 LetterBox（同一 scale+pad，Depth 用最近邻防伪值）。

    aug=None 时等于"无光度/无抖动增强，仅同步 letterbox"（验证路径）。
    返回 (rgb_c, ir_c, depth_c, boxes_c)；boxes_c 作用于 letterbox 画布、归一化。
    """
    from models_config import AugmentParams
    if aug is None:
        # aug=None = 验证/推理路径：**无任何随机增强**（关 flip/HSV/IR抖动/Depth噪声/抖动/RGB失效），
        # 仅保留同步 letterbox，确保验证图像与标签不被随机改动。
        aug = AugmentParams(
            flip_p=0.0, vflip_p=0.0, hsv_rgb=False,
            ir_gain=0.0, ir_bias=0.0, depth_noise=0.0,
            depth_jitter_prob=0.0, rgb_drop_prob=0.0,
        )
    rng = random.Random(seed) if seed is not None else random
    # (1) 几何翻转：三图同步（同一掷骰）
    flip_r, flip_i, flip_d, flip_b = flip_lr_consistent(
        rgb, ir, depth, boxes_norm, p=aug.flip_p, rng=rng)
    if aug.vflip_p > 0 and rng.random() < aug.vflip_p:
        def _vf(img):
            return img[::-1, :, :] if img.ndim == 3 else img[::-1, :]
        flip_r, flip_i, flip_d = _vf(flip_r), _vf(flip_i), _vf(flip_d)
        if flip_b is not None and flip_b.size:
            flip_b[:, 2] = 1.0 - flip_b[:, 2]   # cy → 1-cy
    # (2) RGB 随机失效（防主导模态垄断；仅训练，标签不动）
    rgb_dropped = False
    if aug.rgb_drop_prob > 0 and rng.random() < aug.rgb_drop_prob:
        flip_r = rgb_dropout(flip_r, prob=1.0, mode=aug.rgb_drop_mode, rng=rng)
        rgb_dropped = True
    # (3) RGB HSV（RGB 已失效则跳过，颜色增强无意义）
    if aug.hsv_rgb and not rgb_dropped:
        flip_r = hsv_only_rgb(flip_r, aug.hsv_h, aug.hsv_s, aug.hsv_v, rng=rng)
    # (4) IR 增益/偏置
    flip_i = ir_gain_jitter(flip_i, aug.ir_gain, aug.ir_bias, rng=rng)
    # (5) Depth 值噪声（有效区）
    flip_d = depth_value_noise(flip_d, aug.depth_noise, rng=rng)
    # (6) Depth 随机平移（对齐残差模拟）
    flip_d = random_shift_depth(flip_d, aug.depth_jitter_x, aug.depth_jitter_y,
                                aug.depth_jitter_prob, rng=rng)
    # (7) 同步 letterbox
    r_o, i_o, d_o, info = letterbox_consistent(flip_r, flip_i, flip_d, new_size)
    b_o = transform_boxes_letterbox(flip_b, info) if flip_b is not None else None
    return r_o, i_o, d_o, b_o


__all__ = [
    "flip_lr_consistent",
    "letterbox_consistent",
    "hsv_only_rgb",
    "transform_boxes_letterbox",
    "consistent_augment_full",
    "align_depth",
    "random_shift_depth",
    "ir_gain_jitter",
    "depth_value_noise",
    "rgb_dropout",
]
