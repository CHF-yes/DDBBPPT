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
        g = iv.cvtColor(rgb, iv.COLOR_RGB2GRAY)   # RGB 语义（P1-7：与全链路 RGB 序一致）
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
    """只对 RGB（**RGB 序**，调用方须已 BGR2RGB）做 HSV 扰动；
    IR/Depth 永不参与（避免伪造温度/距离语义）。"""
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
# 几何增强（三模态同步）：随机缩放/平移 + 4 图拼接
# ------------------------------------------------------------

def _boxes_to_canvas(boxes_norm, info, min_px: float = 2.0):
    """归一化框 → 画布归一化框（复用 letterbox 变换），并丢弃出画/退化框。"""
    if boxes_norm is None:
        return None
    b = transform_boxes_letterbox(boxes_norm, info)
    if b is None or b.size == 0:
        return b
    nh, nw = info["new_size"]
    keep = (b[:, 3] * nw >= min_px) & (b[:, 4] * nh >= min_px)
    return b[keep]


def random_affine_consistent(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
                             boxes_norm, new_size=(1024, 1024),
                             scale: float = 0.0, translate: float = 0.0, rng=None):
    """
    三模态**同步**的随机缩放 + 随机平移（等比、不旋转/不剪切）。

    缩放比 = 基准 letterbox 缩放 × (1 ± scale)，再叠加 ±translate 的画布比例平移；
    三图共用**同一仿射矩阵**，像素一一对应关系严格保持（对齐不破坏）。
    - rgb  : INTER_LINEAR + 填 114（与 letterbox 一致）
    - ir   : INTER_LINEAR + 填 114
    - depth: **INTER_NEAREST** + 填 0 —— 绝不插值出伪距离，无效区仍为 0
    返回 (rgb_o, ir_o, dep_o, boxes_o, info)。
    """
    iv = _cv()
    rng = rng if rng is not None else random
    h0, w0 = rgb.shape[:2]
    nh, nw = int(new_size[0]), int(new_size[1])
    base = min(nh / h0, nw / w0)
    s = base * (1.0 + rng.uniform(-scale, scale)) if scale > 0 else base
    dx = (nw - w0 * s) / 2.0 + (rng.uniform(-translate, translate) * nw if translate > 0 else 0.0)
    dy = (nh - h0 * s) / 2.0 + (rng.uniform(-translate, translate) * nh if translate > 0 else 0.0)
    M = np.float32([[s, 0.0, dx], [0.0, s, dy]])
    rgb_o = iv.warpAffine(rgb, M, (nw, nh), flags=iv.INTER_LINEAR,
                          borderMode=iv.BORDER_CONSTANT, borderValue=(_RGB_PAD,) * 3)
    ir_o = iv.warpAffine(ir, M, (nw, nh), flags=iv.INTER_LINEAR,
                         borderMode=iv.BORDER_CONSTANT, borderValue=_IR_PAD)
    dep_o = iv.warpAffine(depth, M, (nw, nh), flags=iv.INTER_NEAREST,
                          borderMode=iv.BORDER_CONSTANT, borderValue=_DEPTH_PAD_FLOAT)
    info = dict(scale=s, dx=dx, dy=dy, old_h=h0, old_w=w0, new_size=(nh, nw))
    return rgb_o, ir_o, dep_o, _boxes_to_canvas(boxes_norm, info), info


def mosaic_consistent(items, new_size=(1024, 1024), rng=None, jitter: float = 0.15):
    """
    4 图 2×2 拼接（三模态共用同一布局）。

    items: [(rgb, ir, depth, boxes_norm), ...]（各源均为原始图坐标、已过翻转/光度）
    返回 (rgb_o, ir_o, dep_o, boxes_o)：画布 new_size，框为画布归一化。
    - 每个源等比 letterbox 进各自象限（居中 + ±jitter 的位置抖动），象限顺序随机打乱；
      **位置抖动很重要**：固定象限会让"物体永远不出现在画布中心/接缝附近"成为强位置先验，
      验证集（物体在任意位置）会因此掉点。
    - rgb/ir 用 INTER_LINEAR 填 114，depth 用 INTER_NEAREST 填 0（保持"0=无效"）；
    - 框按各自仿射变换到画布，越界裁剪、退化框丢弃；
    - 注意：拼接缝在深度图上不是真实几何边界（物理上不存在），靠 close_mosaic
      在训练末段关闭 mosaic 来消除该偏差。
    - **尺度**：本函数把每个源固定缩到约 0.5 倍；必须再叠一层随机缩放（见
      consistent_augment_full 中 mosaic 之后的 random_affine），否则模型会被
      单一尺度锁死（实测：只在 0.5 倍尺度上训练，1.0 倍验证集 mAP 只有 0.16，
      同一权重在 0.5 倍验证集上却有 0.60）。
    """
    iv = _cv()
    rng = rng if rng is not None else random
    nh, nw = int(new_size[0]), int(new_size[1])
    hh, hw = nh // 2, nw // 2
    items = list(items)[:4]
    order = list(range(len(items)))
    rng.shuffle(order)
    rgb_o = np.full((nh, nw, items[0][0].shape[2]), _RGB_PAD, items[0][0].dtype)
    ir_o = np.full((nh, nw), _IR_PAD, items[0][1].dtype)
    dep_o = np.zeros((nh, nw), items[0][2].dtype)
    boxes_all = []
    for slot, idx in enumerate(order):
        rgb, ir, dep, boxes = items[idx]
        h0, w0 = rgb.shape[:2]
        s = min(hh / h0, hw / w0)
        col, row = slot % 2, slot // 2
        dx = col * hw + (hw - w0 * s) / 2.0 + (rng.uniform(-jitter, jitter) * hw if jitter else 0.0)
        dy = row * hh + (hh - h0 * s) / 2.0 + (rng.uniform(-jitter, jitter) * hh if jitter else 0.0)
        # 只 warp 该 tile 的目标矩形（而不是整张画布 + 掩码复制）——后者在 640×640 上
        # 每样本要多做 12 次全尺寸 warp，是整个数据管线的主要开销。
        tw, th = int(round(w0 * s)), int(round(h0 * s))
        x0, y0 = int(round(dx)), int(round(dy))
        rx0, ry0 = max(0, x0), max(0, y0)
        rx1, ry1 = min(nw, x0 + tw), min(nh, y0 + th)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        M = np.float32([[s, 0.0, dx - rx0], [0.0, s, dy - ry0]])
        rgb_o[ry0:ry1, rx0:rx1] = iv.warpAffine(
            rgb, M, (rx1 - rx0, ry1 - ry0), flags=iv.INTER_LINEAR,
            borderMode=iv.BORDER_CONSTANT, borderValue=(_RGB_PAD,) * 3)
        ir_o[ry0:ry1, rx0:rx1] = iv.warpAffine(
            ir, M, (rx1 - rx0, ry1 - ry0), flags=iv.INTER_LINEAR,
            borderMode=iv.BORDER_CONSTANT, borderValue=_IR_PAD)
        dep_o[ry0:ry1, rx0:rx1] = iv.warpAffine(
            dep, M, (rx1 - rx0, ry1 - ry0), flags=iv.INTER_NEAREST,
            borderMode=iv.BORDER_CONSTANT, borderValue=_DEPTH_PAD_FLOAT)
        b = _boxes_to_canvas(boxes, dict(scale=s, dx=dx, dy=dy, old_h=h0, old_w=w0,
                                         new_size=(nh, nw)))
        if b is not None and len(b):
            boxes_all.append(b)
    boxes_o = np.concatenate(boxes_all, axis=0) if boxes_all else None
    return rgb_o, ir_o, dep_o, boxes_o


# ------------------------------------------------------------
# 组装
# ------------------------------------------------------------

def consistent_augment_full(
    rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
    boxes_norm: Optional[np.ndarray],
    new_size=(1024, 1024),
    aug: Optional["AugmentParams"] = None,
    seed: Optional[int] = None,
    extras: Optional[Sequence[tuple]] = None,
    rng=None,
):
    """
    三模态「一致性 + 全覆盖」增强组装（由统一配置 AugmentParams 驱动）：

      (1) 几何：水平/垂直翻转三图**同步**（同一掷骰）；
      (2) RGB 光度：HSV 抖动；IR 光度：灰度增益/偏置；Depth 光度：有效区乘性噪声；
      (3) Depth 几何：随机平移模拟对齐残差（标签不动）；
      (4) 几何（三图同步，二选一）：
            * mosaic：与 extras 的 3 个源做 4 图拼接（aug.mosaic_p 命中时）；
            * 随机缩放+平移（aug.scale / aug.translate > 0 时）；
            * 否则退回原**确定性 letterbox**（验证/推理路径行为完全不变）。
      (5) 拼接缝在深度图上非真实几何，训练末段用 close_mosaic_epochs 关闭 mosaic。

    aug=None 时等于"无光度/无抖动增强，仅同步 letterbox"（验证路径）。
    extras: 可选的 3 组额外源 [(rgb, ir, depth, boxes_norm), ...]（原图坐标）。
    rng   : 传入时复用该随机源（与上层采样共享种子）；否则由 seed 构造。
    返回 (rgb_c, ir_c, depth_c, boxes_c)；boxes_c 作用于画布、归一化。
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
    rng = rng if rng is not None else (random.Random(seed) if seed is not None else random)

    # (1) 几何翻转：**所有源共用同一掷骰**（各源内部三模态因此严格同步）
    do_flip = aug.flip_p > 0 and rng.random() < aug.flip_p
    do_vflip = aug.vflip_p > 0 and rng.random() < aug.vflip_p

    def _prep(src):
        r, i, d, b = src
        if do_flip:
            r, i, d, b = flip_lr_consistent(r, i, d, b, p=1.0, rng=rng)
        if do_vflip:
            def _vf(img):
                return img[::-1, :, :] if img.ndim == 3 else img[::-1, :]
            r, i, d = _vf(r), _vf(i), _vf(d)
            if b is not None and np.asarray(b).size:
                b = np.array(b, dtype=np.float32).reshape(-1, 5).copy()
                b[:, 2] = 1.0 - b[:, 2]
        # (2) RGB 随机失效（防主导模态垄断；仅训练，标签不动）
        dropped = False
        if aug.rgb_drop_prob > 0 and rng.random() < aug.rgb_drop_prob:
            r = rgb_dropout(r, prob=1.0, mode=aug.rgb_drop_mode, rng=rng)
            dropped = True
        # (3) 光度：HSV 只作用 RGB；IR 增益/偏置；Depth 有效区噪声
        if aug.hsv_rgb and not dropped:
            r = hsv_only_rgb(r, aug.hsv_h, aug.hsv_s, aug.hsv_v, rng=rng)
        i = ir_gain_jitter(i, aug.ir_gain, aug.ir_bias, rng=rng)
        d = depth_value_noise(d, aug.depth_noise, rng=rng)
        d = random_shift_depth(d, aug.depth_jitter_x, aug.depth_jitter_y,
                               aug.depth_jitter_prob, rng=rng)
        return r, i, d, b

    main_src = _prep((rgb, ir, depth, boxes_norm))
    use_mosaic = (bool(extras) and getattr(aug, "mosaic_p", 0.0) > 0
                  and rng.random() < float(aug.mosaic_p))
    if use_mosaic:
        srcs = [main_src] + [_prep(e) for e in list(extras)[:3]]
        r_o, i_o, d_o, b_o = mosaic_consistent(srcs, new_size, rng)
        # mosaic 把每个源固定缩到约 0.5 倍 → 必须再叠随机缩放/平移，否则尺度退化为单一值
        # （与基线1 的 ultralytics 流程一致：mosaic + RandomPerspective(scale/translate)）。
        if getattr(aug, "scale", 0.0) > 0 or getattr(aug, "translate", 0.0) > 0:
            r_o, i_o, d_o, b_o, _ = random_affine_consistent(
                r_o, i_o, d_o, b_o, new_size, float(aug.scale), float(aug.translate), rng)
    elif getattr(aug, "scale", 0.0) > 0 or getattr(aug, "translate", 0.0) > 0:
        r_o, i_o, d_o, b_o, _ = random_affine_consistent(
            main_src[0], main_src[1], main_src[2], main_src[3],
            new_size, float(aug.scale), float(aug.translate), rng)
    else:
        r_o, i_o, d_o, info = letterbox_consistent(
            main_src[0], main_src[1], main_src[2], new_size)
        b_o = (transform_boxes_letterbox(main_src[3], info)
               if main_src[3] is not None else None)
    return r_o, i_o, d_o, b_o


__all__ = [
    "flip_lr_consistent",
    "letterbox_consistent",
    "hsv_only_rgb",
    "transform_boxes_letterbox",
    "consistent_augment_full",
    "random_affine_consistent",
    "mosaic_consistent",
    "align_depth",
    "random_shift_depth",
    "ir_gain_jitter",
    "depth_value_noise",
    "rgb_dropout",
]
