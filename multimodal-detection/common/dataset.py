# -*- coding: utf-8 -*-
"""
dataset —— 三/五通道数据读取与拼接核心。

说明 / 定位
----------
ultralytics 从 data.yaml 读取单张图像目录训练，其内建 Dataset 只吃"一图 = 固定通道"。
而我们 5 通道融合需要"一个样本 = RGB + IR + Depth 三张对齐图合成一个 NCHW 张量"，
无法直接喂给内建训练器。因此本模块提供两种封装：

  A. 面向推理（本框架首要用例）：
      依次把某样本的三模态读成 nparray，再 merge_to_tensor() 合成 (C,H,W)；
      numpy 布局为 CHW, C = cfg.in_channels(3 只留 RGB / 5 拼 RGB+IR+D)。

  B. 面向训练：提供一个标准 PyTorch Dataset `MultimodalDetectionDataset`，
      训练主循环需要自定义（实验模型1 阶段）、或字段供 ultralytics 前处理/评估。

赛题标签格式（每行）：class_id, cx, cy, w, h（0-1 归一化）
预测输出：同 class_id, cx, cy, w, h, confidence
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

import models_config as MC

# Depth: 16bit 毫米。ZED 有效距离约 30cm~20m → [0,19999] 约占 [0,65535]
_DEPTH_MM_SCALE: int = 20000
# IR: 三通道视觉一致的灰阶图 → 取单通道即信息完整
_IR_CHANNEL: int = 0
# 无参读图依赖 opencv，这里延迟 import 避免本模块被于无 opencv 环境 import
_imread_codec = {"cv2": None}


def _read(path, flags: int):
    import cv2
    img = cv2.imread(str(path), flags)
    return img


# ------------------------------------------------------------
# 模态读取与归一化
# ------------------------------------------------------------

def read_rgb_bgr(path) -> np.ndarray:
    """读可见光三通道, 返回 BGR ndarray (H,W,3) uint8。后续需 BGR2RGB。
    用 fromfile+imdecode 兼容中文路径。"""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB 读取失败: {path}")
    return img


def read_ir_gray(path) -> np.ndarray:
    """
    读红外图。文件存为三通道灰阶(PNG/JPG)，视觉一致(实为 3x 单通道堆叠)。
    因此返回单通道 uint8 (H,W)，取第 0 通道。兼容中文路径。
    """
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)  # 保留原深度、不转彩
    if img is None:
        raise FileNotFoundError(f"IR 读取失败: {path}")
    if img.ndim == 3:
        img = img[:, :, _IR_CHANNEL]  # 取单通道堆叠中一个即可
    return img  # uint8 单通道


def read_depth_mm(path) -> np.ndarray:
    """
    读深度图。16bit 单通道、单位毫米、未归一；缺失/过小像素=0。
    返回 uint16 (H,W) 毫米值。兼容中文路径。
    """
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Depth 读取失败: {path}")
    if img.dtype != np.uint16:
        # 部分 depth 被存成 uint8/float png —— 尽力转，保留注释
        img = img.astype(np.uint16)
    return img


def norm_array(x: np.ndarray, mode: str = "div255") -> np.ndarray:
    """8bit 数组按模式归一化：div255 → [0,1]；none → 原值(float32)。"""
    x = np.asarray(x)
    if mode == "none":
        return x.astype(np.float32)
    return x.astype(np.float32) / 255.0


def depth_mm_to_scaled(depth_mm: np.ndarray, mode: str = "mm_unit",
                       scale_mm: float = 20000.0,
                       invalid_zero: bool = True) -> np.ndarray:
    """
    16bit 毫米深度 → 网络输入值域（模式可配置，见 models_config.PreprocessParams）。
      * mm_unit : 毫米 / scale_mm(默认 20000，赛题有效约 20m) → [0,1]；无效区按 invalid_zero 处理
      * raw     : 保留毫米原值（float32）
      * div255  : 按 8bit 尺度缩放（仅当 depth 已 8bit 化时用）
    """
    d = depth_mm.astype(np.float32)
    if mode == "raw":
        return d
    if mode == "div255":
        return np.clip(d, 0, 255.0) / 255.0
    # mm_unit（默认）
    valid = d > 1
    if invalid_zero or not bool(valid.any()):
        d_out = np.clip(d, 0, float(scale_mm)) / float(scale_mm)
        d_out[~valid] = 0.0
        return d_out
    m = float(d[valid].mean())
    d_out = np.clip(d, 0, float(scale_mm)) / float(scale_mm)
    d_out[~valid] = m / float(scale_mm)
    return d_out


def build_input_channels(sample_img_paths: dict, in_channels: int,
                         target_size=None, depth_shift=(0, 0), preprocess=None):
    """
    把一个样本的 {rgb,ir,depth} 文件路径组装成 (C,H,W) float32 输入给网络做前向。
    - in_channels==3 : 只用 RGB(可选缩小) crop BGR 三通道, 顺序 BGR(与 torchpretrained 一致)
    - in_channels==5 : RGB + IR 单通道 + Depth 归一化单通道
    - depth_shift: (dx,dy) 固定平移,把 depth warp 到 RGB 坐标系(对齐, 标签不动)
    - preprocess: PreprocessParams（赛题数据未归一化；三种模态值域策略，默认 div255/mm_unit）
    返回 (np.ndarray chw) 颜色通道数=3时即 [3,H,W]，5即 [5,H,W]。
    图像统一 resize 到 target_size(H,W)；未给则用三者中的最大。
    """
    from .multimodal_augment import align_depth
    from models_config import PreprocessParams
    pp = preprocess if preprocess is not None else PreprocessParams()
    rgb = read_rgb_bgr(sample_img_paths["rgb"])  # (H,W,3) BGR
    H, W = rgb.shape[:2]
    if in_channels == 3:
        ch = norm_array(rgb, pp.rgb_mode).transpose(2, 0, 1)
    else:
        assert in_channels == 5, in_channels
        ir = read_ir_gray(sample_img_paths["ir"])   # (H,W)  像素即温度亮暗
        dep = read_depth_mm(sample_img_paths["depth"])  # (H,W) mm
        dep = align_depth(dep, *depth_shift)          # 固定平移对齐到 RGB 坐标系
        dep_sc = depth_mm_to_scaled(dep, pp.depth_mode,
                                    pp.depth_scale_mm, pp.depth_invalid_zero)
        # 统一尺寸(resize 三书记一致既保证空间对齐保留——题目已经对齐，此处只顺引)
        if (ir.shape[0], ir.shape[1]) != (H, W):
            import cv2
            ir = cv2.resize(ir, (W, H), interpolation=cv2.INTER_LINEAR)
            dep_sc = cv2.resize(dep_sc, (W, H), interpolation=cv2.INTER_LINEAR)

        rf = norm_array(rgb, pp.rgb_mode)
        r, g, b = rf[:, :, 0], rf[:, :, 1], rf[:, :, 2]
        irn = norm_array(ir, pp.ir_mode)
        # 组装 [B, G, R, IR, D]
        ch = np.stack([b, g, r, irn, dep_sc], axis=0)  # (5,H,W)
    if target_size is not None:
        import cv2
        Ht, Wt = target_size
        ch = np.array([cv2.resize(c, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
                       for c in ch])
    return ch


# ------------------------------------------------------------
# 标签(TXT) / 预测(TXT) 读写 —— 赛题指定格式
# ------------------------------------------------------------

def read_label_txt(path) -> np.ndarray:
    """读赛题标签 txt → (N,5) float。列: cls,cx,cy,w,h (均 0-1 归一)。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = [float(x) for x in line.split()]
            if len(v) >= 5:
                rows.append(v[:5])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def write_pred_txt(path, boxes_n5, confs=None, class_of_line_first=True) -> None:
    """
    写预测结果 txt。形如每行: cls cx cy w h conf。
    boxes_n5: (N,5) cls,cx,cy,w,h(0-1)
    confs  : (N,) 或 None(缺失则跳过 conf)
    每张测试图一个同名 txt，无目标也须写空后缀文件(自动 create)。
    """
    if boxes_n5.size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    lines = []
    has_conf = confs is not None
    for i, row in enumerate(boxes_n5):
        cls, cx, cy, w, h = row
        if has_conf and i < len(confs):
            lines.append(
                f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {confs[i]:.6f}")
        else:
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------
# 训练用 PyTorch Dataset（供 5 通道自定义训练；3 通道可回归 ultralytics 内建）
# ------------------------------------------------------------

def to_torch_tensor(chw: npt.NDArray):
    """(C,H,W) float numpy → torch tensor (C,H,W)，供直接 forward 5 通道网络。"""
    import torch
    return torch.from_numpy(chw).float()


class MultimodalDetectionDataset:
    """
    简单迭代器包装：把一组 Sample(见 scan_data) 逐组产出
    (tensor_chw(C,H,W), label_txt(N,5))。

    用于在 baseline2 / 实验模型 阶段做自定义推理 or 训练采样。
    """

    def __init__(self, samples, cfg: MC.ModelConfig, imgsz=(1024, 1024), transform=None):
        self.samples = samples
        self.cfg = cfg
        self.imgsz = imgsz
        self.transform = transform
        # 从统一配置读取 depth 固定对齐（默认不偏移，基线2/实验1 各自配置可覆写）
        self.depth_shift = (getattr(cfg.hyper, "depth_shift_x", 0),
                            getattr(cfg.hyper, "depth_shift_y", 0))
        # 数据预处理开关（赛题数据未归一化；值域策略可配置）
        self.preprocess = getattr(cfg.hyper, "preprocess", None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        chw = build_input_channels(s.img, self.cfg.in_channels,
                                   target_size=self.imgsz,
                                   depth_shift=self.depth_shift,
                                   preprocess=self.preprocess)
        lab = read_label_txt(s.label) if s.label else np.empty((0, 5), np.float32)
        if self.transform is not None:
            chw = self.transform(chw)
        return chw, lab, s.stem


# data.yaml 拓扑占位：ultralytics 训练走内建，此函数仅供文档/生成骨架
def yaml_image_dir_hint(split) -> str:
    return f"（按 common.scan_data.build_data_yaml 填 split 图像目录）"


def load_dataset_cfg(cfg: "MC.ModelConfig", split: str = "train"):
    """按版本的输入规格返回数据集描述 dict，供各实例组装数据路径/样章。"""
    return {
        "cfg_key": cfg.key,
        "split": split,
        "in_channels": cfg.in_channels,
        "modality": cfg.modality.value,
        "fusion": cfg.fusion.value,
        "imgsz_hint": cfg.hyper.imgsz,
        "sample_slots": ["rgb", "ir", "depth"],
        "note": ("5 通道建议用 common.dataset.MultimodalDetectionDataset；"
                 "3 通道可直接走 ultralytics data.yaml 图片目录。"),
    }


def build_consistent_aug_5ch(
    sample,                        # scan_data.Sample: img={rgb,ir,depth}, label
    target_size=(1024, 1024),
    aug=None,                      # AugmentParams 统一增强配置；None=仅同步 letterbox(验证)
    depth_shift=(0, 0),            # 固定平移对齐(只动 depth；标签锚定 RGB)
    preprocess=None,               # PreprocessParams 值域开关（赛题数据未归一化）
    seed: Optional[int] = None,
    to_tensor: bool = False,
):
    """
    三模态 5 通道「一致性增强 + 通道拼装」的单一入口，供自定义训练/验证循环使用。

    步骤：读 RGB/IR/Depth 三张对齐原图 → [Depth 固定平移对齐] → 按 aug 做
    **三模态全覆盖一致性增强**(三图同步几何 + RGB-HSV/IR 增益/Depth 噪声与平移)
    → 拼成 (5,H,W) float32 张量。

    返回 (chw, boxes_out, stem)：
      chw     : (5,H,W) float32，通道序 [B,G,R,IR_norm,Depth_scaled]
      boxes_out: (N,5) cls,cx,cy,w,h(归一，参考 letterbox 后画布)；无目标为 None
      stem    : 样本基名
    """
    from .multimodal_augment import consistent_augment_full, align_depth
    from models_config import PreprocessParams
    import numpy as np
    pp = preprocess if preprocess is not None else PreprocessParams()

    rgb = read_rgb_bgr(sample.img["rgb"])                  # (H,W,3) BGR
    # 校验三模态齐全
    for mod, fn in (("ir", read_ir_gray), ("depth", read_depth_mm)):
        if sample.img.get(mod) is None:
            raise ValueError(
                f"build_consistent_aug_5ch 需要 ir 与 depth；样本 {sample.stem} 缺 {mod}")
    ir = read_ir_gray(sample.img["ir"])                     # (H,W) uint8
    dep_mm = read_depth_mm(sample.img["depth"])             # (H,W) uint16 mm
    dep_mm = align_depth(dep_mm, *depth_shift)              # 固定平移对齐到 RGB 坐标系
    boxes_norm = _read_boxes_5(sample)

    rgb_o, ir_o, dep_o, boxes_out = consistent_augment_full(
        rgb, ir, dep_mm, boxes_norm,
        new_size=target_size, aug=aug, seed=seed)

    rf = norm_array(rgb_o, pp.rgb_mode)                     # letterboxed B,G,R 按模式
    b = rf[:, :, 0]
    g = rf[:, :, 1]
    r = rf[:, :, 2]
    irn = norm_array(ir_o, pp.ir_mode)                       # 单通道亮度(温度归一)
    dep_sc = depth_mm_to_scaled(dep_o, pp.depth_mode,
                                pp.depth_scale_mm, pp.depth_invalid_zero)  # mm→[0,1]/原值
    chw = np.stack([b, g, r, irn, dep_sc], axis=0)           # (5,H,W)
    if to_tensor:
        chw = to_torch_tensor(chw)
    return chw, boxes_out, sample.stem


def _read_boxes_5(sample):
    """读样本(赛题格式) label → (N,5) cls,cx,cy,w,h；无标签/空标签返回 None。"""
    if sample.label is None or not Path(sample.label).exists():
        return None
    lab = read_label_txt(sample.label)
    return lab if lab.size else None



__all__ = [
    "read_rgb_bgr", "read_ir_gray", "read_depth_mm", "depth_mm_to_scaled",
    "build_input_channels", "read_label_txt", "write_pred_txt",
    "to_torch_tensor", "MultimodalDetectionDataset", "_DEPTH_MM_SCALE",
    "build_consistent_aug_5ch",
]
