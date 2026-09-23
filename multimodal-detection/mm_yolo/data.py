# -*- coding: utf-8 -*-
"""数据管线：赛题三模态读取 → 几何同步增强 → 显式先验/质量描述子 → 组感知划分。

关键设计（对应 D0 实测与"显式注入模态直觉"）：
* **unicode 读图**：数据根目录含中文，cv2.imread 在 Windows 上全失败 → 走 imdecode；
* **混 dtype/分辨率**：FHD 组 depth 是 uint16 毫米制、VGA/JPG 组是 uint8 灰度 → 输出
  4ch=[相对深度, 0–20m 绝对深度, 有效掩码, 米制可用标志]，不再把 8bit 伪深度当毫米；
* **depth 先验**（4ch）：逆相对深度 / 有效性 / 法线 nx / 法线 ny —— 把"远近与朝向"显式喂给模型；
* **质量描述子**（每模态 3ch）：亮度/对比度/清晰度类统计 —— 把"哪个模态现在可信"显式喂进门控；
* **组感知划分**：文件名前缀（视频序列）+ 256bit dHash 近重复簇；无来源信息的相邻帧仍可能跨界，
  并按**类别覆盖**贪心（稀有类优先塞进 val，否则 class 11 在 val 里只有 1 个框，选模方差极大）；
* **modality dropout**：训练时整路丢 RGB/IR/depth（承重配方），且**图像/质量描述子/几何先验同步清零**
  （只清图像会让先验泄漏"被丢掉的那个模态"的信息）。

审计修复记录（本轮，全部有单测/实测支撑）：
1. `enabled` 控制数据侧是否生成某模态：RGB-only 锚点不再白白丢掉 25% 的 RGB
   （旧版 `rgb_drop_p=0.25` 在 `--modalities rgb` 下仍然生效 → 1598 张里 406 张永远黑图）；
2. `set_epoch` + `random.Random(seed, epoch, idx)`：随机增强**每轮不同**
   （旧版 seed 只依赖 `seed+idx`，同一张图每个 epoch 的"随机"结果完全一样）；
3. depth 随机错位与 letterbox **合成一个仿射矩阵只 warp 一次**（旧版对已变换的 depth 又 `Mj@M` 了一次，
   实测有效像素只剩正确的 2.46%、7/30 张全空）；先验/质量图与输入图严格同源；
4. 增强框 **clip 到画布 + 过滤**（旧版不裁剪：12,261 个框里 937 个完全在画布外仍进损失）；
5. modality dropout 同步清零 quality/prior；
6. 索引缓存带 `ver`/`hash_bits` 版本戳（旧缓存字段语义变化后自动失效重建，不静默用错哈希）。
"""
from __future__ import annotations

import json
import hashlib
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from io_utils import imread_unicode                        # noqa: E402

MODS = ("visible", "infrared", "depth")
MODALITY_KEY = {"visible": "rgb", "infrared": "ir", "depth": "dep"}
PRIOR_CH = 4            # [逆相对深度, 有效, nx, ny]
QUALITY_CH = 3          # [亮度/强度, 局部对比度, 清晰度/梯度能量]
DEPTH_CH = 4            # [相对深度, 绝对深度/20m, 有效掩码, 米制可用标志]
DEPTH_VALID_CH = 2
EXTS = (".png", ".jpg", ".jpeg", ".bmp")
HASH_BITS = 256         # dHash 位数（17×16 网格 → 16 列 × 16 行 = 256 bit，**不再截断**）
INDEX_VER = 5           # 索引缓存格式版本；加入源文件签名


# ---------------------------------------------------------------- 读图

def read_rgb(path) -> Optional[np.ndarray]:
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def read_ir_bundle(path, mode: str = "legacy_first_channel"):
    """Read thermal intensity plus chroma residue from the stored IR image.

    The median channel is the detector input.  The per-pixel channel spread is
    retained only as a quality/processing-chain signal; it is never subtracted
    from the thermal image.
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if mode not in ("legacy_first_channel", "median_channel"):
        raise ValueError(f"unknown IR read mode: {mode}")
    source_dtype = img.dtype
    if img.ndim == 2:
        channels = np.repeat(img[..., None], 3, axis=2)
    else:
        channels = img[:, :, :3]
    channels = channels.astype(np.float32)
    if source_dtype == np.uint16:
        channels /= 256.0
    channels = np.clip(channels, 0.0, 255.0)
    if mode == "median_channel":
        thermal = np.median(channels, axis=2)
    else:
        thermal = channels[:, :, 0]
    # Magnitude of C_IR = IR - mean_c(IR), normalized for quality maps.
    chroma = np.mean(np.abs(channels - channels.mean(axis=2, keepdims=True)), axis=2) / 255.0
    return np.clip(thermal, 0, 255).astype(np.uint8), chroma.astype(np.float32)


def read_ir(path, mode: str = "legacy_first_channel") -> Optional[np.ndarray]:
    """Read IR without treating a weak channel tint as the thermal signal.

    ``legacy_first_channel`` is kept for exact compatibility with existing
    checkpoints.  ``median_channel`` uses the per-pixel median of the three
    stored channels, which is robust to the small RGB-like chroma residue seen
    in a subset of the PNG files.
    """
    thermal, _ = read_ir_bundle(path, mode=mode)
    return thermal


def read_depth(path, return_metric: bool = False, legacy_valid: bool = False):
    """读取 Depth，默认返回 ``(float32, valid)`` 以兼容旧调用。

    ``return_metric=True`` 时额外返回 ``metric_available``。PNG uint16 是赛题定义的
    毫米数据；JPG/uint8 只保留相对灰度，不伪造绝对距离。对米制图，<0.3m 或 >20m
    按传感器规格视为无效；8bit 图仅以 0 为无效。
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return (None, None, False) if return_metric else (None, None)
    metric_available = img.dtype == np.uint16
    if img.ndim == 3:                       # VGA 组：三通道完全相同的 uint8
        img = img[:, :, 0]
    d = img.astype(np.float32)
    valid = (((d >= 300.0) & (d <= 20000.0)) if metric_available else (d > 0))
    if legacy_valid:
        valid = d > 0
    if return_metric:
        return d, valid, bool(metric_available)
    return d, valid


# ---------------------------------------------------------------- 表示与先验

def relative_depth(dep: np.ndarray, valid: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """每图分位归一 → [0,1]（0=近，1=远）；无效与饱和区置 0。抗未知尺度/裁剪。"""
    out = np.zeros_like(dep, dtype=np.float32)
    if valid.sum() < 16:
        return out
    v = dep[valid]
    q_lo, q_hi = np.percentile(v, [lo, hi])
    if q_hi - q_lo < 1e-6:
        return out
    out[valid] = np.clip((dep[valid] - q_lo) / (q_hi - q_lo), 0.0, 1.0)
    return out


def absolute_metric_depth(dep: np.ndarray, valid: np.ndarray, metric_available: bool) -> np.ndarray:
    """毫米 Depth → [0,1]，1 对应 20m；非米制图和无效像素严格为 0。"""
    out = np.zeros_like(dep, dtype=np.float32)
    if metric_available and valid.any():
        out[valid] = np.clip(dep[valid] / 20000.0, 0.0, 1.0)
    return out


def _fill_invalid(rel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """把无效区填成有效区中位数，避免 Sobel/blur 在 0 洞上产生假边缘。"""
    filled = rel.copy()
    if not valid.all():
        filled[~valid] = float(np.median(rel[valid])) if valid.any() else 0.0
    return filled


def depth_prior(dep: np.ndarray, valid: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """depth 先验 (4,h,w)：[逆相对深度, 有效性, 法线 nx, 法线 ny]。

    逆深度 1/(1+rel) 让近处分辨率更高（深度噪声 ∝ z²，远距离本就不准，也更贴近遮挡次序）；
    法线由局部梯度估得，显式给"朝向/平面/边界"。
    """
    rel = relative_depth(dep, valid)
    inv = 1.0 / (1.0 + 3.0 * rel)                       # 近处→1，远处→0.25
    if inv.max() - inv.min() > 1e-6:
        inv = (inv - inv.min()) / (inv.max() - inv.min())
    if not valid.any():
        inv = np.zeros_like(inv)
    smooth = cv2.GaussianBlur(_fill_invalid(rel, valid), (5, 5), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx * gx + gy * gy + 1e-6)
    nx = (-gx / norm)
    ny = (-gy / norm)
    m = valid & (norm > 1e-4)
    nx = np.where(m, nx, 0.0).astype(np.float32)
    ny = np.where(m, ny, 0.0).astype(np.float32)
    stack = np.stack([inv.astype(np.float32), valid.astype(np.float32), nx, ny], 0)
    h, w = out_size
    if stack.shape[-2:] != (h, w):
        stack = np.stack([cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA) for s in stack], 0)
    return np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _local_stats(gray: np.ndarray, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    x = gray.astype(np.float32)
    mean = cv2.blur(x, (k, k))
    sq = cv2.blur(x * x, (k, k))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return mean, std


def quality_maps(rgb: Optional[np.ndarray], ir: Optional[np.ndarray],
                 dep: Optional[np.ndarray], valid: Optional[np.ndarray],
                 out_size: Tuple[int, int], ir_chroma: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """低分辨率质量描述子；IR 额外保留处理链和边缘可靠性信息。

    ⚠️ **只对"本模态真的可用"的输入生成**：不生成质量图 → 融合块对该槽插零，
    与"整路丢弃该模态"语义一致（旧版无条件生成 RGB 质量图，即使 RGB 已被丢掉）。
    """
    out: Dict[str, np.ndarray] = {}
    h, w = out_size

    def _down(chan: Sequence[np.ndarray]) -> np.ndarray:
        arrs = [cv2.resize(c.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for c in chan]
        return np.nan_to_num(np.stack(arrs, 0).astype(np.float32), nan=0.0,
                             posinf=0.0, neginf=0.0)

    # ---- RGB：亮度 / 局部对比度 / 清晰度（Laplacian 能量）----
    if rgb is not None and rgb.size:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        mean, std = _local_stats(g)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        lap = lap / (np.percentile(lap, 95) + 1e-6)
        out["rgb"] = _down([mean / 255.0, np.clip(std / 64.0, 0, 1), np.clip(lap, 0, 1)])
    # ---- IR：强度 / 局部对比度 / 梯度能量 ----
    if ir is not None and ir.size:
        mean, std = _local_stats(ir)
        gx = cv2.Sobel(ir.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ir.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = grad / (np.percentile(grad, 95) + 1e-6)
        edge = np.clip(grad, 0, 1)
        dark = ((mean < 6.0) | (mean > 249.0)).astype(np.float32)
        dark_edge = cv2.blur(dark, (9, 9))
        saturation = cv2.blur(((mean < 2.0) | (mean > 253.0)).astype(np.float32), (7, 7))
        # Low normalized Laplacian energy is a blur cue; paired edge energy
        # marks thick/double contours produced by interpolation or ghosting.
        blur = 1.0 - np.clip(np.abs(cv2.Laplacian(ir.astype(np.float32), cv2.CV_32F, ksize=3)) /
                             (np.percentile(np.abs(cv2.Laplacian(ir.astype(np.float32), cv2.CV_32F, ksize=3)), 95) + 1e-6), 0, 1)
        double_edge = np.clip(edge * cv2.blur((edge > .35).astype(np.float32), (5, 5)) * 3.0, 0, 1)
        chroma = np.zeros_like(mean, dtype=np.float32) if ir_chroma is None else np.clip(ir_chroma, 0, 1)
        if rgb is not None and rgb.size:
            rgb_float = rgb.astype(np.float32)
            rgb_chroma = np.mean(np.abs(rgb_float - rgb_float.mean(axis=2, keepdims=True)), axis=2) / 255.0
            # Estimate a per-image RGB-like residue coefficient, then retain
            # only its fit confidence.  The fitted residue never enters T.
            rgb_chroma = np.clip(rgb_chroma, 0, 1)
            coeff = float((chroma * rgb_chroma).sum() /
                          (np.square(rgb_chroma).sum() + 1e-6))
            coeff = float(np.clip(coeff, 0.0, 1.0))
            chroma_fit = np.exp(-np.abs(chroma - coeff * rgb_chroma) / .08)
        else:
            chroma_fit = np.zeros_like(chroma)
        out["ir"] = _down([mean / 255.0, np.clip(std / 64.0, 0, 1), edge,
                           np.clip(dark_edge, 0, 1), np.clip(saturation, 0, 1),
                           np.clip(blur, 0, 1), np.clip(double_edge, 0, 1),
                           chroma, np.clip(chroma_fit, 0, 1)])
    # ---- depth：局部有效比例 / 深度梯度 / 局部对比度 ----
    if dep is not None and valid is not None and dep.size:
        rel = relative_depth(dep, valid)
        filled = _fill_invalid(rel, valid)
        gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = grad / (np.percentile(grad, 95) + 1e-6)
        _, std = _local_stats(filled)
        vr = cv2.blur(valid.astype(np.float32), (15, 15))
        out["dep"] = _down([vr, np.clip(grad, 0, 1), np.clip(std * 4, 0, 1)])
    return out


# ---------------------------------------------------------------- 几何增强（三模态同步）

@dataclass
class AugCfg:
    """几何与退化增强配置。

    ⚠️ `imgsz` 支持 (H, W) 元组：赛题原图是 16:9（1920×1080），**用正方形画布会浪费 ~44% 的
    算力在 letterbox 黑边上**（960×960 里图像只占 960×540）。用 16:9 画布（如 544×960）可以
    同算力下把有效分辨率提高约 1.3×，或同分辨率下省 ~1.8× 时间。
    """
    imgsz: object = 960                                  # int（正方形）或 (H, W)
    hflip_p: float = 0.5
    scale_range: Tuple[float, float] = (0.75, 1.4)      # 随机缩放（短边填充 letterbox）
    translate: float = 0.1                              # 相对平移
    rotate_deg: float = 0.0                             # 三模态同步小角度旋转
    misalign_px: float = 5.0                            # depth 轻微错位；不用大幅幻影教坏对齐
    degrade_p: float = 0.3                              # RGB 连续谱退化（低照/噪声/模糊）
    rgb_color_p: float = 0.0                            # RGB 轻量颜色/对比度扰动（默认关闭）
    ir_noise_p: float = 0.0                             # IR 轻微读出噪声（B2 配方显式开）
    ir_gain_p: float = 0.0                              # IR 正增益/偏置，保持热强度次序
    ir_read_mode: str = "legacy_first_channel"          # legacy 首通道 / median 三通道中位数
    depth_hole_p: float = 0.0                           # Depth 小块失效（同步更新 valid）
    target_crop_p: float = 0.0                          # 三模态同步、目标感知裁剪
    target_occlusion_p: float = 0.0                     # 目标内非对称栏杆/块遮挡；标签保持完整框
    ir_affine_p: float = 0.0                            # IR-only 已知小仿射，用于粗对齐监督
    ir_affine_deg: float = 0.0                          # IR 旋转上限（度）
    ir_affine_shift: float = 0.0                        # IR 平移上限（画布像素）
    ir_affine_scale: float = 0.0                        # IR 比例变化上限（fraction）
    legacy_lowlight: bool = False                       # 仅用于 B1 旧实验的严格续训
    # modality dropout：**只在多模态训练时生效**（enabled 少于 2 路时自动跳过）
    rgb_drop_p: float = 0.05                            # 小概率整路丢 RGB
    aux_drop_p: float = 0.05                            # 小概率整路丢 IR / depth
    dropout_start_epoch: int = 0                        # 训练入口默认 warmup 后才启用
    drop_same_sample: bool = True                       # 同一张图整路全丢？
    box_min_size: float = 2.0                           # 过滤：短边小于该像素数的框丢弃
    box_require_center: bool = True                     # 过滤：中心必须落在画布内
    depth_resampling: str = "legacy_bilinear_v1"
    total_epochs: int = 0
    close_aug_frac: float = 0.0
    mosaic_p: float = 0.0


def scheduled_aug(base: AugCfg, epoch: int) -> AugCfg:
    """Pure worker-side schedule; never mutate the parent's/shared base config."""
    if base.total_epochs < 2 or base.close_aug_frac <= 0:
        return base
    start = base.total_epochs * (1.0 - base.close_aug_frac)
    t = float(np.clip((epoch - start) / max(1.0, base.total_epochs - 1 - start), 0, 1))
    mix = lambda a, b: a + (b - a) * t
    return replace(base, mosaic_p=0.0 if epoch >= start else base.mosaic_p,
                   scale_range=(mix(base.scale_range[0], .95), mix(base.scale_range[1], 1.05)),
                   translate=mix(base.translate, .02), rotate_deg=base.rotate_deg * (1-t),
                   target_crop_p=base.target_crop_p * (1-t),
                   misalign_px=base.misalign_px * (1-t),
                   degrade_p=base.degrade_p * (1-.8*t), rgb_color_p=base.rgb_color_p * (1-.5*t),
                   ir_noise_p=base.ir_noise_p * (1-.8*t), ir_gain_p=base.ir_gain_p * (1-.5*t),
                   depth_hole_p=base.depth_hole_p * (1-t),
                   target_occlusion_p=base.target_occlusion_p * (1-t),
                   ir_affine_p=base.ir_affine_p * (1-t),
                   rgb_drop_p=base.rgb_drop_p * (1-t), aux_drop_p=base.aux_drop_p * (1-t))


class CoverageRareSampler(Sampler):
    """First cover every source once, then add bounded, rare-class-biased draws.

    Indices carry (source, occurrence, epoch), so persistent workers, prefetch and
    epoch-boundary resume use identical augmentation identities without IPC races.
    """
    def __init__(self, samples, seed=42, extra_frac=.1, max_extra_per_image=3):
        if not 0 <= extra_frac <= 1 or max_extra_per_image < 1:
            raise ValueError("invalid coverage sampler budget")
        self.seed, self.epoch = int(seed), 0
        self.n = len(samples)
        self.max_extra = int(max_extra_per_image)
        classes = [set(int(b[0]) for b in s.get("boxes", [])) for s in samples]
        counts = {c: sum(c in cs for cs in classes) for cs in classes for c in cs}
        # Total class mass ~ 1/sqrt(frequency), rather than favoring classes
        # that merely co-occur with the many person/animal images.
        self.weights = [sum(counts[c] ** -1.5 for c in cs) for cs in classes]
        self.extra = min(round(self.n * extra_frac), self.max_extra * sum(w > 0 for w in self.weights))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n + self.extra

    def __iter__(self):
        rng = random.Random(self.seed * 1000003 + self.epoch * 7919)
        order = list(range(self.n))
        rng.shuffle(order)
        for i in order:
            yield (i, 0, self.epoch)
        weights, seen = list(self.weights), [0] * self.n
        for _ in range(self.extra):
            i = rng.choices(range(self.n), weights=weights, k=1)[0]
            seen[i] += 1
            yield (i, seen[i], self.epoch)
            if seen[i] >= self.max_extra:
                weights[i] = 0.0


def canvas_of(imgsz) -> Tuple[int, int]:
    """把 imgsz（int 或 (H,W) / 'HxW'）统一成画布 (H, W)，并对齐到 32 的倍数。"""
    if isinstance(imgsz, str):
        s = imgsz.strip().lower()
        if "x" in s:
            h_s, w_s = s.split("x")[:2]
            h, w = int(h_s), int(w_s)
        else:
            h = w = int(s)
    elif isinstance(imgsz, (tuple, list)):
        h, w = int(imgsz[0]), int(imgsz[1])
    else:
        h = w = int(imgsz)
    return (max(64, (h // 32) * 32), max(64, (w // 32) * 32))


def parse_imgsz(value) -> object:
    """CLI 值 → dataset 用的 imgsz：'544x960' → (544, 960)；'960' → 960；(H,W) 原样通过。

    ⚠️ 必须接受 tuple/list：`eval.py`/`submit.py` 会把 `resolve_infer_canvas()` 的结果
    （ckpt 里记录的 (H,W) 画布）再喂回本函数，不认元组就会 `int(tuple)` 直接崩。
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return (int(value[0]), int(value[1]))
    if isinstance(value, str):
        s = value.strip().lower()
        if "x" in s:
            h, w = s.split("x")[:2]
            return (int(h), int(w))
        return int(s)
    return int(value)


def _worker_init(_worker_id: int) -> None:
    """DataLoader worker 初始化：**限制每个 worker 的线程数**。

    Windows 下 DataLoader 用 spawn，父进程的 `cv2.setNumThreads(1)` 不会传进来；
    不设的话每个 worker 内部 cv2/torch 又会各开满线程 → N workers × N threads 过度订阅，
    实测 12 workers 反而比 6 workers 慢（2.5 → 1.6 样本/秒）。
    """
    try:
        cv2.setNumThreads(1)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        torch.set_num_threads(1)
    except Exception:                                    # noqa: BLE001
        pass


def _mat3(M: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(M, np.float32), [0, 0, 1]])


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (N,2) 像素 → 经 2×3 仿射变换（含齐次项）。"""
    return np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))] @ np.asarray(M, np.float32).T[:, :2]


def letterbox_M(H: int, W: int, canvas: Tuple[int, int], scale: float, dx: float,
                dy: float, flip: bool) -> np.ndarray:
    """2×3 仿射：等比缩放 + 居中 + 平移 + 水平翻转。三模态共用同一套几何。"""
    Hc, Wc = canvas
    s = min(Wc / max(1, W), Hc / max(1, H)) * scale
    ox = (Wc - W * s) / 2 + dx * Wc
    oy = (Hc - H * s) / 2 + dy * Hc
    M = np.array([[s, 0, ox], [0, s, oy]], dtype=np.float32)
    if flip:
        F = np.array([[-1, 0, Wc - 1], [0, 1, 0]], dtype=np.float32)
        M = (F @ _mat3(M))[:2]
    return M


def target_crop_M(H: int, W: int, canvas: Tuple[int, int], boxes: np.ndarray,
                  rng: random.Random, flip: bool) -> np.ndarray:
    """围绕一个真实目标构造裁剪仿射，且保持画布宽高比。

    该矩阵与普通 letterbox 使用同一个几何入口，因此 RGB/IR/Depth/标签严格
    同步；Depth 的小残差平移仍在其后叠加。
    """
    b = np.asarray(boxes if boxes is not None else [], dtype=np.float32).reshape(-1, 5)
    if not len(b):
        return letterbox_M(H, W, canvas, 1.0, 0.0, 0.0, flip)
    row = b[rng.randrange(len(b))]
    cx, cy = float(row[1] * W), float(row[2] * H)
    bw, bh = max(1.0, float(row[3] * W)), max(1.0, float(row[4] * H))
    aspect = float(canvas[1]) / float(canvas[0])
    frac = rng.uniform(0.58, 0.88)
    crop_w = max(W * frac, bw * 1.6, bh * 1.6 * aspect)
    crop_h = crop_w / aspect
    if crop_h > H:
        crop_h = float(H)
        crop_w = crop_h * aspect
    if crop_w > W:
        crop_w = float(W)
        crop_h = crop_w / aspect
    # 小幅抖动裁剪中心，但选中目标中心始终保留在裁剪内。
    ccx = cx + rng.uniform(-0.10, 0.10) * crop_w
    ccy = cy + rng.uniform(-0.10, 0.10) * crop_h
    x0 = min(max(ccx - crop_w / 2.0, 0.0), max(0.0, W - crop_w))
    y0 = min(max(ccy - crop_h / 2.0, 0.0), max(0.0, H - crop_h))
    x0 = min(x0, cx)
    x0 = max(x0, cx - crop_w)
    y0 = min(y0, cy)
    y0 = max(y0, cy - crop_h)
    sx, sy = canvas[1] / crop_w, canvas[0] / crop_h
    M = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], dtype=np.float32)
    if flip:
        Fm = np.array([[-1, 0, canvas[1] - 1], [0, 1, 0]], dtype=np.float32)
        M = (Fm @ _mat3(M))[:2]
    return M


def shift_M(dx_px: float, dy_px: float) -> np.ndarray:
    """纯平移 2×3（像素）。"""
    return np.array([[1, 0, float(dx_px)], [0, 1, float(dy_px)]], dtype=np.float32)


def centered_affine_M(canvas: Tuple[int, int], angle_deg: float, scale: float,
                      dx_px: float, dy_px: float) -> np.ndarray:
    """Source-to-destination affine around the canvas centre (OpenCV convention)."""
    H, W = int(canvas[0]), int(canvas[1])
    M = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0),
                               float(angle_deg), float(scale)).astype(np.float32)
    M[:, 2] += np.asarray([dx_px, dy_px], np.float32)
    return M


def _warp(img: np.ndarray, M: np.ndarray, canvas: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (canvas[1], canvas[0]), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _degrade_rgb(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """真正的低照退化：压暗 + gamma>1，再叠加弱噪声/可选运动模糊。

    旧实现用 gamma∈[0.35,1]，对 [0,1] 图像实际是**变亮**，与“低照”语义相反。
    """
    x = rgb.astype(np.float32)
    gamma = rng.uniform(1.25, 2.20)
    gain = rng.uniform(0.35, 0.78)
    x = 255.0 * gain * (x / 255.0) ** gamma
    if rng.random() < 0.35:
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        ang = rng.uniform(0, 180)
        Mr = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        kern = cv2.warpAffine(kern, Mr, (k, k))
        kern /= kern.sum() + 1e-6
        x = cv2.filter2D(x, -1, kern)
    if rng.random() < 0.7:
        # 暗光读出噪声：幅度保守，避免把增强变成纯噪声训练。
        x = x + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1.5, 6.0), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def _degrade_rgb_b1(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """B1 旧增强的原样实现，仅供旧 checkpoint 严格续训时复现。"""
    x = rgb.astype(np.float32)
    x = 255.0 * (x / 255.0) ** rng.uniform(0.35, 1.0)
    if rng.random() < 0.5:
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        Mr = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kern = cv2.warpAffine(kern, Mr, (k, k))
        kern /= kern.sum() + 1e-6
        x = cv2.filter2D(x, -1, kern)
    if rng.random() < 0.5:
        x += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 12), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def _noise_ir(ir: np.ndarray, rng: random.Random) -> np.ndarray:
    """IR 的轻微高斯读出噪声，不改变其热辐射强度语义。"""
    x = ir.astype(np.float32)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.8, 3.0), x.shape)
    return np.clip(x + noise, 0, 255).astype(np.uint8)


def _jitter_rgb(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """轻度颜色/对比度扰动，不模拟完全不同的照明或变更几何。"""
    x = rgb.astype(np.float32)
    contrast = rng.uniform(0.9, 1.1)
    gains = np.array([rng.uniform(0.92, 1.08) for _ in range(3)], np.float32)
    x = (x - 127.5) * contrast + 127.5
    return np.clip(x * gains[None, None, :], 0, 255).astype(np.uint8)


def _gain_ir(ir: np.ndarray, rng: random.Random) -> np.ndarray:
    """传感器增益/零点漂移；正斜率保证未截断像素的冷暖顺序不颠倒。"""
    gain = rng.uniform(0.85, 1.15)
    offset = rng.uniform(-6.0, 6.0)
    return np.clip(ir.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


def _drop_depth_blocks(dep: np.ndarray, valid: np.ndarray, rng: random.Random
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """在 Depth 上制造 1–3 个小型无效块，并同步清除 valid。"""
    if not valid.any():
        return dep, valid
    out, vm = dep.copy(), valid.copy()
    H, W = vm.shape
    ys, xs = np.where(vm)
    for _ in range(rng.randint(1, 3)):
        j = rng.randrange(len(xs))
        cy, cx = int(ys[j]), int(xs[j])
        hh = max(2, int(H * rng.uniform(0.015, 0.045)))
        ww = max(2, int(W * rng.uniform(0.015, 0.045)))
        y0, y1 = max(0, cy - hh // 2), min(H, cy + (hh + 1) // 2)
        x0, x1 = max(0, cx - ww // 2), min(W, cx + (ww + 1) // 2)
        out[y0:y1, x0:x1] = 0
        vm[y0:y1, x0:x1] = False
    return out, vm


def _target_occlusion(rgb: np.ndarray, ir: np.ndarray, boxes: np.ndarray,
                      rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Occlude one sensor inside one target without changing the full-box label.

    Thin bars approximate fences/railings; an occasional compact patch covers a
    contiguous visible part.  Only RGB *or* IR is modified, ensuring another
    appearance sensor remains available to teach complementary routing.
    """
    b = np.asarray(boxes if boxes is not None else [], np.float32).reshape(-1, 5)
    if not len(b):
        return rgb, ir
    H, W = rgb.shape[:2]
    row = b[rng.randrange(len(b))]
    cx, cy, bw, bh = row[1] * W, row[2] * H, row[3] * W, row[4] * H
    if bw < 8 or bh < 8:
        return rgb, ir
    x0, x1 = int(max(0, cx - bw / 2)), int(min(W - 1, cx + bw / 2))
    y0, y1 = int(max(0, cy - bh / 2)), int(min(H - 1, cy + bh / 2))
    if x1 <= x0 or y1 <= y0:
        return rgb, ir
    use_rgb = rng.random() < .55
    out = rgb.copy() if use_rgb else ir.copy()
    roi = out[y0:y1 + 1, x0:x1 + 1]
    median = (np.median(roi.reshape(-1, roi.shape[-1]), axis=0)
              if roi.ndim == 3 else np.asarray(float(np.median(roi))))
    # A median-colour occluder becomes a no-op on exactly the low-texture
    # objects this augmentation is meant to protect.  Keep it plausible but
    # enforce visible contrast in either bright or dark regions.
    offset = 48.0 if float(np.mean(median)) < 128.0 else -48.0
    fill = np.clip(median.astype(np.float32) + offset, 0, 255).astype(out.dtype)
    if rng.random() < .75:
        vertical = rng.random() < .65
        count = rng.randint(2, 5)
        thickness = max(2, int(min(bw, bh) * rng.uniform(.025, .07)))
        slope = rng.uniform(-.18, .18)
        for j in range(count):
            frac = (j + rng.uniform(.35, .65)) / count
            if vertical:
                x = int(x0 + frac * max(1, x1 - x0))
                delta = int(slope * (y1 - y0))
                p0, p1 = (x - delta // 2, y0), (x + delta // 2, y1)
            else:
                y = int(y0 + frac * max(1, y1 - y0))
                delta = int(slope * (x1 - x0))
                p0, p1 = (x0, y - delta // 2), (x1, y + delta // 2)
            cv2.line(out, p0, p1, fill.tolist() if hasattr(fill, "tolist") else float(fill), thickness)
    else:
        rw = max(3, int(bw * rng.uniform(.15, .35)))
        rh = max(3, int(bh * rng.uniform(.15, .35)))
        px = rng.randint(x0, max(x0, x1 - rw))
        py = rng.randint(y0, max(y0, y1 - rh))
        out[py:min(y1 + 1, py + rh), px:min(x1 + 1, px + rw)] = fill
    return (out, ir) if use_rgb else (rgb, out)


# ---------------------------------------------------------------- 组感知划分

def source_group(stem: str) -> str:
    """文件名 → 来源组（同一视频序列的帧不许跨 train/val）。

    实测命名规律（2000 张）：
      * `000032_003_00000001` → 前两段 = (采集批次, 相机/子序列)；**只用第一段太粗**
        （1052 张都变成 "000032" 之外的 "PLAIN"），只用整名太细 → 取前两段；
      * `shuming_100_000001_049_...` / `hehe_100_...` → 第一段就是来源（370 / 89 张）；
      * 单段名 → 无来源信息，靠 dHash 近重复兜底。
    """
    parts = stem.split("_")
    if len(parts) == 1:
        return "PLAIN"
    if len(parts) >= 2 and (parts[0].isdigit() or parts[1].isdigit()):
        return "_".join(parts[:2])
    return parts[0]


def dhash(gray: np.ndarray, gw: int = 17, gh: int = 16) -> int:
    """dHash：灰度图缩到 **(gw)×(gh)**（17 列 16 行）→ 每行 gw−1 个相邻比较 → 256 bit。

    ⚠️ 旧版两个 bug 都在这里：
      ① 把 32×33（**32 行**，每行 32 个比较 = 1024 bit）的哈希 `& (1<<64)-1`，
         只留下**最后两行**（第 30–31 行）→ "近重复检测"实际只比较图像底部两行；
      ② 9×8 网格对 16:9 图会把整幅画面压成 8 列 → 不同画面撞哈希（实测 387 张撞成一个簇）。
    现在用 17×16 = 256 bit，先等比缩到统一宽高比再算，缓存里按十进制大整数存取。
    """
    g = cv2.resize(np.asarray(gray, np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


def hamming(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def _thumb_hash_bits(gray: np.ndarray, size: int = 16) -> int:
    """一个额外的 256bit 粗哈希（16×16 缩略图阈值化），用于缓存自检/可选的强判重。"""
    t = cv2.resize(np.asarray(gray, np.float32), (size, size), interpolation=cv2.INTER_AREA)
    bits = (t > float(np.median(t))).flatten()[:256]
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


def _cluster_ids(n: int, hashes: np.ndarray, groups: Sequence[str], hamming_thr: int):
    """并查集：dHash 汉明距离 ≤ 阈值，或同一文件名前缀（视频序列）→ 同簇。

    ⚠️ `hashes` 现在可能是 256bit 大整数，不能用 np.uint64 装 → 统一走 Python int。
    """
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:                 # 按秩合并 → 树形状与迭代顺序无关（可复现）
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    by_group: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        if g != "PLAIN":
            by_group.setdefault(g, []).append(i)
    for g, idxs in by_group.items():
        for j in idxs[1:]:
            union(idxs[0], j)
    H = [int(x) for x in hashes]
    for i in range(n):
        hi = H[i]
        # 只有哈希完全不同的才可能近重复；向量化算汉明距离
        d = np.array([(hi ^ h).bit_count() for h in H], dtype=np.uint8)
        for j in np.where(d[1:] <= hamming_thr)[0] + 1:
            union(i, int(j))
    return [find(i) for i in range(n)]


def _image_classes(s: dict) -> set:
    b = s.get("boxes")
    if b is None or len(b) == 0:
        return set()
    return {int(c) for c in np.asarray(b)[:, 0]}


def group_split(samples: List[dict], val_ratio: float = 0.2, hamming_thr: int = 3,
                seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """按"来源前缀 + dHash 近重复簇"分组后再切分，避免同源帧泄漏进验证集。

    **类别感知**（审计修复）：稀有类（class 11 全局仅 27 框）如果只按组大小贪心，
    很容易整体落进训练集 → 验证集某个类一个框都没有，mAP 与选模都失效。
    做法：把"含稀有类"的组**优先**放进验证集，直到达到目标比例。
    """
    n = len(samples)
    if n == 0:
        return [], []
    hashes = [int(s["_hash_int"]) if "_hash_int" in s else 0 for s in samples]
    groups = [source_group(s["stem"]) for s in samples]
    roots = _cluster_ids(n, hashes, groups, hamming_thr)
    gid: Dict[int, List[int]] = {}
    for i, r in enumerate(roots):
        gid.setdefault(r, []).append(i)

    # 类别频率 → 稀有度权重
    cls_count: Dict[int, int] = {}
    for s in samples:
        for c in _image_classes(s):
            cls_count[c] = cls_count.get(c, 0) + 1
    max_cnt = max(cls_count.values()) if cls_count else 1

    def rarity(group_idx: List[int]) -> float:
        """组内最稀有类别越稀有，分数越高（0..1）。"""
        best = 0.0
        for i in group_idx:
            for c in _image_classes(samples[i]):
                best = max(best, 1.0 - cls_count.get(c, 1) / max_cnt)
        return best

    keys = sorted(gid.keys())
    rng = random.Random(seed)
    order = keys[:]
    rng.shuffle(order)
    order.sort(key=lambda k: len(gid[k]))               # 小簇先放 → 更容易精确贴合 val_ratio
    want = max(1, int(round(n * val_ratio)))
    chosen: List[int] = []
    taken = set()

    def _cls_of_group(gidx: List[int]) -> set:
        cs: set = set()
        for i in gidx:
            cs |= _image_classes(samples[i])
        return cs

    # ---- 阶段 1：**类别覆盖**：每个类别至少有一组进 val（稀有类优先）----
    covered: set = set()
    for c in sorted(cls_count, key=lambda c: cls_count[c]):
        if c in covered:
            continue
        best = None
        for k in order:
            if k in taken or c not in _cls_of_group(gid[k]):
                continue
            r = rarity(gid[k])
            if best is None or (r, len(gid[k])) > (rarity(gid[best]), len(gid[best])):
                best = k
        if best is not None:
            chosen.append(best)
            taken.add(best)
            covered |= _cls_of_group(gid[best])

    # ---- 阶段 2：按组大小贪心填到目标比例（小簇优先 → 超冲很小）----
    got = sum(len(gid[k]) for k in chosen)
    for k in order:
        if k in taken or got >= want:
            continue
        chosen.append(k)
        taken.add(k)
        got += len(gid[k])
    val_idx: List[int] = []
    for k in chosen:
        val_idx += gid[k]
    val_set = set(val_idx)
    tr = [s for i, s in enumerate(samples) if i not in val_set]
    va = [s for i, s in enumerate(samples) if i in val_set]
    return tr, va


def pick_val_subset(samples: List[dict], limit: int, seed: int = 42) -> List[dict]:
    """在组感知验证集内按**类别均匀**挑 limit 张（训练期快速验证用）。

    审计修复：旧版直接取前 N 张（`va[:limit]`），排序靠前的样本完全没有 class 1/7/11，
    于是"日志里的 best"既不是 12 类 mAP，也不可比。这里改成：
    ① 每张图按其**最稀有类别**排序（稀有类优先被覆盖）；② 同类内按确定性打散。
    """
    if limit <= 0 or limit >= len(samples):
        return list(samples)
    cls_count: Dict[int, int] = {}
    for s in samples:
        for c in _image_classes(s):
            cls_count[c] = cls_count.get(c, 0) + 1
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        cs = _image_classes(s)
        key = min(cls_count.get(c, 0) for c in cs) if cs else 10 ** 9     # 无框图排最后
        buckets.setdefault(key, []).append(i)
    out: List[int] = []
    for key in sorted(buckets):
        idxs = buckets[key][:]
        rng.shuffle(idxs)
        out += idxs
    return [samples[i] for i in out[:limit]]


# ---------------------------------------------------------------- Dataset

class MMDataset(Dataset):
    """三模态数据集：返回 {rgb, ir, depth, quality{...}, prior, boxes, stem, keep, enabled}。"""

    def __init__(self, root: Path, samples: List[dict], imgsz=960, train: bool = True,
                 aug: Optional[AugCfg] = None, prior_stride: int = 8, seed: int = 0,
                 enabled: Sequence[str] = ("rgb", "ir", "dep"), epoch: int = 0,
                 epoch_file: Optional[Path] = None):
        self.root = Path(root)
        self.samples = samples
        self.imgsz = imgsz
        self.canvas = canvas_of(imgsz)                 # (H, W)
        self.train = bool(train)
        self.aug = aug or AugCfg(imgsz=imgsz)
        self.prior_stride = int(prior_stride)
        self.seed = int(seed)
        self.epoch = int(epoch)
        # ⚠️ DataLoader 的 worker 进程（尤其 persistent_workers=True 时）**不会**看到父进程
        #    对 dataset.epoch 的修改；worker 只在启动那一刻拷贝一次 dataset。
        #    没有这条通道，set_epoch 就是空操作 → "随机增强每轮相同"的 P0 复现。
        #    这里用一个单行文本文件做 epoch 通道，__getitem__ 里按 mtime 惰性刷新（开销可忽略）。
        self.epoch_file = Path(epoch_file) if epoch_file else None
        self._epoch_stat = None
        self._load_epoch_file()
        self.enabled = tuple(m for m in ("rgb", "ir", "dep") if m in set(enabled))
        if not self.enabled:
            raise ValueError("enabled 不能为空")
        self._use_dropout = bool(self.train and len(self.enabled) > 1)

    def __len__(self):
        return len(self.samples)

    def _load_epoch_file(self) -> None:
        if self.epoch_file is None:
            return
        try:
            st = self.epoch_file.stat()
            if self._epoch_stat != st.st_mtime_ns:
                self.epoch = int(self.epoch_file.read_text(encoding="utf-8").strip())
                self._epoch_stat = st.st_mtime_ns
        except Exception:                                    # noqa: BLE001
            pass

    def set_epoch(self, epoch: int) -> None:
        """切换 epoch → 随机增强每轮不同（可复现，且对 worker/续训都成立）。"""
        self.epoch = int(epoch)
        if self.epoch_file is not None:
            try:
                self.epoch_file.parent.mkdir(parents=True, exist_ok=True)
                self.epoch_file.write_text(str(int(epoch)), encoding="utf-8")
                self._epoch_stat = self.epoch_file.stat().st_mtime_ns
            except Exception:                                # noqa: BLE001
                pass

    def _paths(self, s: dict) -> Dict[str, Path]:
        return {m: self.root / m / s["files"][m] for m in MODS if m in s["files"]}

    def _rng(self, idx: int, draw: int = 0) -> random.Random:
        return random.Random((self.seed * 1000003) ^ (self.epoch * 7919) ^ (idx * 104729)
                             ^ (draw * 15485863))

    @staticmethod
    def _blank_like(img: np.ndarray, kind: str):
        """某个模态文件缺失时的占位：**该路整路置零 + keep=0**（融合门控会屏蔽它）。

        提交阶段必须保证"每张图都有 TXT"，所以缺模态不能抛异常/跳过。
        """
        h, w = img.shape[:2]
        if kind == "ir":
            return np.zeros((h, w), np.uint8 if img.dtype == np.uint8 else img.dtype)
        return np.zeros((h, w), np.float32), np.zeros((h, w), bool)

    def __getitem__(self, idx: int) -> Optional[dict]:
        self._load_epoch_file()
        if isinstance(idx, tuple):
            index, draw, epoch = idx[:3]
        else:
            index, draw, epoch = idx, 0, self.epoch
        self.epoch = epoch
        aug = scheduled_aug(self.aug, epoch) if self.train else self.aug
        rng = self._rng(index, draw)
        if self.train and aug.mosaic_p > 0 and rng.random() < aug.mosaic_p:
            return self._mosaic_item(index, draw, epoch, rng)
        return self._single_item(idx)

    def _mosaic_item(self, index, draw, epoch, rng):
        """Four synchronized source scenes; no distance blending or cross-scene matches."""
        import torch.nn.functional as tf
        h, w = self.canvas
        cy, cx = int(h*rng.uniform(.4,.6)), int(w*rng.uniform(.4,.6))
        rects = [(0,0,cy,cx),(0,cx,cy,w-cx),(cy,0,h-cy,cx),(cy,cx,h-cy,w-cx)]
        sources = [index] + [rng.randrange(len(self)) for _ in range(3)]
        original_canvas, original_aug = self.canvas, self.aug
        children = []
        try:
            self.aug = replace(self.aug, mosaic_p=0, rgb_drop_p=0, aux_drop_p=0)
            for j,(source,(_,_,hh,ww)) in enumerate(zip(sources,rects)):
                self.canvas = (hh,ww)
                children.append(self._single_item((source,draw*5+j+1,epoch,False)))
        finally:
            self.canvas, self.aug = original_canvas, original_aug
        if any(x is None for x in children):
            raise OSError("unreadable Mosaic source")
        out = dict(children[0])
        for key in ("rgb","ir","depth"):
            out[key] = torch.cat((torch.cat((children[0][key],children[1][key]),2),
                                  torch.cat((children[2][key],children[3][key]),2)),1)
        boxes = []
        for item,(y,x,hh,ww) in zip(children,rects):
            bb = item["boxes"].clone()
            bb[:,1] = (bb[:,1]*ww+x)/w
            bb[:,2] = (bb[:,2]*hh+y)/h
            bb[:,3] *= ww/w
            bb[:,4] *= hh/h
            boxes.append(bb)
        out["boxes"] = torch.cat(boxes)
        out["quality"] = {}
        for key in set().union(*(c["quality"] for c in children)):
            tiles = []
            for item,(_,_,hh,ww) in zip(children,rects):
                if key in item["quality"]:
                    value = item["quality"][key]
                else:
                    # V5 IR quality maps have nine channels; preserve the
                    # channel contract when a dropped tile has no descriptor.
                    channels = 9 if key == "ir" else (1 if key == "scene_id" else 3)
                    value = torch.zeros(channels, 1, 1)
                tiles.append(tf.interpolate(value[None],size=(hh,ww),mode="nearest")[0])
            value = torch.cat((torch.cat(tiles[:2],2),torch.cat(tiles[2:],2)),1)
            out["quality"][key] = value if key == "availability" else tf.interpolate(value[None],size=(max(4,h//8),max(4,w//8)),mode="area")[0]
        scene = torch.zeros(1,h,w)
        for j,(y,x,hh,ww) in enumerate(rects):
            scene[:,y:y+hh,x:x+ww] = j+1
        out["quality"]["scene_id"] = scene
        out["prior"] = torch.zeros(4,max(4,h//self.prior_stride),max(4,w//self.prior_stride))
        # A Mosaic contains four independently shifted Depth tiles, so there is
        # no single global displacement target for flow supervision.
        out["alignment_shift"] = torch.zeros(2, dtype=torch.float32)
        out["alignment_supervised"] = torch.tensor(0.0, dtype=torch.float32)
        out["ir_affine_target"] = torch.zeros(4, dtype=torch.float32)
        out["ir_affine_supervised"] = torch.tensor(0.0, dtype=torch.float32)
        out["keep"] = {m: float(any(c["keep"][m] for c in children)) for m in ("rgb","ir","dep")}
        # Whole-modality dropout applies to the completed Mosaic, never ambiguous individual tiles.
        aug = scheduled_aug(self.aug,epoch)
        if self._use_dropout and epoch >= aug.dropout_start_epoch:
            original = dict(out["keep"])
            for m,p in (("rgb",aug.rgb_drop_p),("ir",aug.aux_drop_p),("dep",aug.aux_drop_p)):
                if rng.random() < p:
                    out["keep"][m] = 0.
            if not any(out["keep"].values()):
                m = rng.choice([m for m in original if original[m]])
                out["keep"][m] = 1.
        for j,m in enumerate(("rgb","ir","dep")):
            if not out["keep"][m]:
                out["quality"]["availability"][j].zero_()
                out["quality"].pop(m,None)
        out["stem"] = self.samples[index]["stem"]
        return out

    def _single_item(self, idx: int) -> Optional[dict]:
        self._load_epoch_file()                # worker 侧惰性同步 epoch（见 __init__ 注释）
        draw = 0
        allow_special = True
        if isinstance(idx, tuple):
            if len(idx) == 4:
                idx, draw, self.epoch, allow_special = idx
            else:
                idx, draw, self.epoch = idx
        s = self.samples[idx]
        rng = self._rng(idx, draw)
        aug = scheduled_aug(self.aug, self.epoch) if self.train else self.aug
        paths = self._paths(s)
        want_rgb, want_ir, want_dep = ("rgb" in self.enabled, "ir" in self.enabled,
                                       "dep" in self.enabled)
        # 单模态消融仍需 visible 的尺寸/标签坐标作为参考，但像素会在下面清零，
        # 绝不把 RGB 视觉信息送给 IR-only / Depth-only 模型。
        rgb = read_rgb(paths["visible"])
        ir, ir_chroma = (read_ir_bundle(paths["infrared"], mode=aug.ir_read_mode)
                         if want_ir and "infrared" in paths else (None, None))
        dep, valid, metric_available = (read_depth(paths["depth"], return_metric=True,
                                                   legacy_valid=aug.legacy_lowlight)
                                        if want_dep and "depth" in paths
                                        else (None, None, False))
        if rgb is None:
            if self.train:
                raise OSError(f"Unreadable training RGB: {paths['visible']}")
            return None                                     # 没有 RGB 就没法定位（也不该提交）
        missing = set()
        if ir is None:
            ir = self._blank_like(rgb, "ir")
            ir_chroma = np.zeros(ir.shape[:2], np.float32)
            if want_ir:
                missing.add("ir")
        if dep is None:
            dep, valid = self._blank_like(rgb, "dep")
            metric_available = False
            if want_dep:
                missing.add("dep")
        H, W = rgb.shape[:2]
        if ir.shape[:2] != (H, W):
            ir = cv2.resize(ir, (W, H), interpolation=cv2.INTER_AREA)
        if ir_chroma is None:
            ir_chroma = np.zeros((H, W), np.float32)
        elif ir_chroma.shape[:2] != (H, W):
            ir_chroma = cv2.resize(ir_chroma, (W, H), interpolation=cv2.INTER_AREA)
        metric_map = np.full(dep.shape[:2], 1 if metric_available else 0, dtype=np.uint8)
        if dep.shape[:2] != (H, W):
            dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(valid.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            metric_map = cv2.resize(metric_map, (W, H), interpolation=cv2.INTER_NEAREST)

        # ---- 几何：三模态同一套仿射（错位与 letterbox **合成一次**）----
        if self.train:
            scale = rng.uniform(*aug.scale_range)
            dx = rng.uniform(-aug.translate, aug.translate)
            dy = rng.uniform(-aug.translate, aug.translate)
            flip = rng.random() < aug.hflip_p
        else:
            scale, dx, dy, flip = 1.0, 0.0, 0.0, False
        do_target_crop = (self.train and aug.target_crop_p > 0 and
                          s.get("boxes") is not None and len(s["boxes"]) > 0 and
                          rng.random() < aug.target_crop_p)
        M = (target_crop_M(H, W, self.canvas, s["boxes"], rng, flip) if do_target_crop
             else letterbox_M(H, W, self.canvas, scale, dx, dy, flip))
        if self.train and aug.rotate_deg > 0:
            angle = rng.uniform(-aug.rotate_deg, aug.rotate_deg)
            M = (centered_affine_M(self.canvas, angle, 1.0, 0.0, 0.0) @ _mat3(M))[:2]
        jx = jy = 0.0
        if self.train and aug.misalign_px > 0:
            jx = rng.uniform(-1, 1) * aug.misalign_px
            jy = rng.uniform(-1, 1) * aug.misalign_px
            if abs(jx) <= 0.5 and abs(jy) <= 0.5:
                jx = jy = 0.0
        Md = (shift_M(jx, jy) @ _mat3(M))[:2] if (jx or jy) else M

        rgb_w = _warp(rgb, M, self.canvas)
        if not want_rgb:
            rgb_w.fill(0)
        ir_w = _warp(ir, M, self.canvas)
        ir_chroma_w = _warp(ir_chroma, M, self.canvas)
        ir_spatial = _warp(np.ones((H, W), np.uint8), M, self.canvas, nearest=True).astype(np.float32)
        ir_affine_target = np.zeros(4, np.float32)
        ir_affine_supervised = 0.0
        if (self.train and allow_special and want_ir and aug.ir_affine_p > 0 and
                rng.random() < aug.ir_affine_p):
            angle = rng.uniform(-aug.ir_affine_deg, aug.ir_affine_deg)
            sx = rng.uniform(-aug.ir_affine_shift, aug.ir_affine_shift)
            sy = rng.uniform(-aug.ir_affine_shift, aug.ir_affine_shift)
            ds = rng.uniform(-aug.ir_affine_scale, aug.ir_affine_scale)
            A = centered_affine_M(self.canvas, angle, 1.0 + ds, sx, sy)
            ir_w = _warp(ir_w, A, self.canvas)
            ir_chroma_w = _warp(ir_chroma_w, A, self.canvas)
            ir_spatial = _warp(ir_spatial, A, self.canvas, nearest=True).astype(np.float32)
            # ``warp`` predicts output-reference -> distorted-input sampling.
            # OpenCV rendered the distorted image with source->destination A,
            # so an RGB reference coordinate must sample the distorted input at
            # A(x): the supervised parameters have the same sign as A.
            ir_affine_target[:] = (
                angle / max(1e-6, aug.ir_affine_deg),
                sx / max(1e-6, aug.ir_affine_shift),
                sy / max(1e-6, aug.ir_affine_shift),
                ds / max(1e-6, aug.ir_affine_scale),
            )
            ir_affine_supervised = 1.0
        # ⚠️ 审计修复：depth/掩码只用 Md **warp 一次**。旧版先 _warp(dep, M) 再 _warp(dep_w, Mj@M)，
        #    等于把 M 应用了两遍 → 有效像素只剩正确的 2.46%、7/30 张整幅变空。
        if aug.depth_resampling not in ("legacy_bilinear_v1", "nearest_valid_v2"):
            raise ValueError(f"Unknown Depth preprocessing: {aug.depth_resampling}")
        dep_w = _warp(dep, Md, self.canvas, nearest=aug.depth_resampling == "nearest_valid_v2")
        valid_w = _warp(valid.astype(np.uint8), Md, self.canvas, nearest=True) > 0
        if aug.depth_resampling == "nearest_valid_v2":
            dep_w = np.where(valid_w, dep_w, 0).astype(np.float32)
        metric_w = _warp(metric_map, Md, self.canvas, nearest=True).astype(np.float32)
        if want_dep and not valid_w.any():
            missing.add("dep")

        # ---- 各模态的轻量质量退化（不改标签）----
        if self.train and want_rgb and rng.random() < aug.degrade_p:
            rgb_w = (_degrade_rgb_b1(rgb_w, rng) if aug.legacy_lowlight
                     else _degrade_rgb(rgb_w, rng))
        if self.train and want_rgb and aug.rgb_color_p > 0 and rng.random() < aug.rgb_color_p:
            rgb_w = _jitter_rgb(rgb_w, rng)
        if self.train and want_ir and aug.ir_noise_p > 0 and rng.random() < aug.ir_noise_p:
            ir_w = _noise_ir(ir_w, rng)
        if self.train and want_ir and aug.ir_gain_p > 0 and rng.random() < aug.ir_gain_p:
            ir_w = _gain_ir(ir_w, rng)
        if self.train and want_dep and aug.depth_hole_p > 0 and rng.random() < aug.depth_hole_p:
            dep_w, valid_w = _drop_depth_blocks(dep_w, valid_w, rng)
        metric_w *= valid_w.astype(np.float32)

        # ---- 标签：原图归一化框 → 画布（letterbox 后），**裁剪到画布并过滤**----
        Hc, Wc = self.canvas
        out_boxes = canvas_boxes_from_norm(s.get("boxes"), M, (H, W), self.canvas,
                                           min_size=aug.box_min_size,
                                           require_center=aug.box_require_center)
        if (self.train and allow_special and aug.target_occlusion_p > 0 and
                len(out_boxes) and rng.random() < aug.target_occlusion_p):
            rgb_w, ir_w = _target_occlusion(rgb_w, ir_w, out_boxes, rng)

        # ---- 显式先验与质量描述子：**在 1/4 降采样图上算**（它们本来就是低分辨率量），
        #      比在全画布上做多次 blur/Sobel 快十几倍 ----
        Hq, Wq = max(32, Hc // 4), max(32, Wc // 4)
        rgb_q = cv2.resize(rgb_w, (Wq, Hq), interpolation=cv2.INTER_AREA)
        ir_q = cv2.resize(ir_w, (Wq, Hq), interpolation=cv2.INTER_AREA)
        ir_chroma_q = cv2.resize(ir_chroma_w, (Wq, Hq), interpolation=cv2.INTER_AREA)
        dep_q = cv2.resize(dep_w, (Wq, Hq), interpolation=cv2.INTER_NEAREST)
        val_q = cv2.resize(valid_w.astype(np.uint8), (Wq, Hq),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        ps = (max(4, Hc // self.prior_stride), max(4, Wc // self.prior_stride))
        prior = depth_prior(dep_q, val_q, ps)
        quality = quality_maps(rgb_q if want_rgb else None, ir_q if want_ir else None,
                               dep_q if want_dep else None, val_q if want_dep else None, ps,
                               ir_chroma=ir_chroma_q if want_ir else None)

        # ---- modality dropout（整路置零 + 记录 keep）----
        keep = {"rgb": 1.0 if want_rgb else 0.0,
                "ir": 1.0 if want_ir and "ir" not in missing else 0.0,
                "dep": 1.0 if want_dep and "dep" not in missing else 0.0}
        if self._use_dropout and self.epoch >= int(aug.dropout_start_epoch):
            for key, p in (("rgb", aug.rgb_drop_p), ("ir", aug.aux_drop_p),
                           ("dep", aug.aux_drop_p)):
                if keep[key] > 0 and rng.random() < p:
                    keep[key] = 0.0
            if not any(keep.values()):
                # 只能恢复原本存在的模态；绝不能把缺文件的占位图恢复为有效输入。
                available = [m for m in self.enabled if m not in missing]
                keep[available[rng.randrange(len(available))]] = 1.0
        # ⚠️ 审计修复：丢掉的模态要**连质量描述子/几何先验一起清零**，
        #    否则 depth 被丢掉、prior 却还在喂"距离/朝向" → 信息泄漏。
        if not keep["dep"]:
            prior = np.zeros_like(prior)
        for key in ("rgb", "ir", "dep"):
            if not keep[key]:
                quality.pop(key, None)

        rel_depth = relative_depth(dep_w, valid_w)
        abs_depth = absolute_metric_depth(dep_w, valid_w, metric_available)
        spatial = _warp(np.ones((H,W),np.uint8), M, self.canvas, nearest=True).astype(np.float32)
        quality["availability"] = np.stack((spatial*keep["rgb"], ir_spatial*keep["ir"],
                                             valid_w*keep["dep"])).astype(np.float32)
        quality["scene_id"] = np.ones((1,*self.canvas),np.float32)
        return {
            "rgb": torch.from_numpy(rgb_w.transpose(2, 0, 1).copy()).float() / 255.0,
            "ir": torch.from_numpy(ir_w[None].copy()).float() / 255.0,
            "depth": torch.from_numpy(np.stack([rel_depth, abs_depth,
                                                  valid_w.astype(np.float32), metric_w], 0)).float(),
            "quality": {k: torch.from_numpy(v) for k, v in quality.items()},
            "prior": torch.from_numpy(prior),
            "boxes": torch.from_numpy(out_boxes),          # 画布归一化 [cls,cx,cy,w,h]
            "M": torch.from_numpy(M.copy()),               # letterbox 仿射（2×3）
            # jx/jy are CANVAS pixels: shift_M is left-multiplied after M.
            # The model converts them to feature pixels using each actual map size.
            "alignment_shift": torch.tensor([jx, jy], dtype=torch.float32),
            "alignment_supervised": torch.tensor(
                float(self.train and aug.misalign_px > 0), dtype=torch.float32),
            "ir_affine_target": torch.from_numpy(ir_affine_target),
            "ir_affine_supervised": torch.tensor(ir_affine_supervised, dtype=torch.float32),
            "orig_hw": torch.tensor([H, W], dtype=torch.float32),
            "stem": s["stem"],
            "keep": keep,
            "enabled": list(self.enabled),
        }


# ---------------------------------------------------------------- 框变换（唯一实现，训练/评测/提交共用）

def canvas_boxes_from_norm(boxes_norm: Optional[np.ndarray], M: np.ndarray,
                           orig_hw: Tuple[float, float], canvas: Tuple[int, int],
                           min_size: float = 2.0, require_center: bool = True
                           ) -> np.ndarray:
    """原图归一化 (cls,cx,cy,w,h) → **画布归一化** (cls,cx,cy,w,h)，带裁剪与过滤。

    审计修复：旧版变换后不裁剪也不过滤，实测 12,261 个框里 2,159 个越过画布、
    1,332 个中心已在画布外、937 个完全在画布外仍进入损失（把"背景"当成正样本教）。
    现在：① 裁剪到 [0,Wc]×[0,Hc]；② 丢掉短边 < min_size 像素的框；
    ③ 丢掉中心落在画布外的框（半截框在外面的也丢，避免教模型预测画布外的目标）。
    """
    H, W = float(orig_hw[0]), float(orig_hw[1])
    Hc, Wc = int(canvas[0]), int(canvas[1])
    if boxes_norm is None or len(boxes_norm) == 0:
        return np.zeros((0, 5), np.float32)
    b = np.asarray(boxes_norm, np.float32).reshape(-1, 5)
    cx, cy, bw, bh = b[:, 1] * W, b[:, 2] * H, b[:, 3] * W, b[:, 4] * H
    x1 = _apply(M, np.c_[cx - bw / 2, cy - bh / 2])
    x2 = _apply(M, np.c_[cx + bw / 2, cy + bh / 2])
    xa = np.minimum(x1[:, 0], x2[:, 0])
    xb = np.maximum(x1[:, 0], x2[:, 0])
    ya = np.minimum(x1[:, 1], x2[:, 1])
    yb = np.maximum(x1[:, 1], x2[:, 1])
    # ① 裁剪到画布
    xa = np.clip(xa, 0.0, Wc)
    xb = np.clip(xb, 0.0, Wc)
    ya = np.clip(ya, 0.0, Hc)
    yb = np.clip(yb, 0.0, Hc)
    w = xb - xa
    h = yb - ya
    keep = (w >= float(min_size)) & (h >= float(min_size))
    ccx = (xa + xb) / 2.0
    ccy = (ya + yb) / 2.0
    if require_center:
        keep &= (ccx > 0) & (ccx < Wc) & (ccy > 0) & (ccy < Hc)
    return np.stack([b[keep, 0], ccx[keep] / Wc, ccy[keep] / Hc, w[keep] / Wc, h[keep] / Hc],
                    1).astype(np.float32)


def canvas_to_orig_norm(boxes_xyxy: np.ndarray, M: np.ndarray, orig_hw, imgsz: int = 0) -> np.ndarray:
    """画布 xyxy 像素 → **原图**归一化 (cx,cy,w,h)。

    ⚠️ 关键：letterbox 会把 16:9 的图放进其他比例的画布并留黑边，所以**不能**只除以画布边长，
    必须用 letterbox 仿射的逆变换再除以原图宽高。否则提交坐标会整体偏移/缩放。
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 4), np.float32)
    H, W = float(orig_hw[0]), float(orig_hw[1])
    Minv = cv2.invertAffineTransform(np.asarray(M, np.float32))
    b = np.asarray(boxes_xyxy, np.float32)
    x1y1 = _apply(Minv, b[:, :2])
    x2y2 = _apply(Minv, b[:, 2:4])
    xa, xb = np.minimum(x1y1[:, 0], x2y2[:, 0]), np.maximum(x1y1[:, 0], x2y2[:, 0])
    ya, yb = np.minimum(x1y1[:, 1], x2y2[:, 1]), np.maximum(x1y1[:, 1], x2y2[:, 1])
    xa = np.clip(xa, 0, W); xb = np.clip(xb, 0, W)
    ya = np.clip(ya, 0, H); yb = np.clip(yb, 0, H)
    return np.stack([(xa + xb) / 2 / W, (ya + yb) / 2 / H,
                     (xb - xa) / W, (yb - ya) / H], 1).astype(np.float32)


def boxes_norm_to_canvas(boxes_norm: np.ndarray, M: np.ndarray, orig_hw, imgsz: int = 0) -> np.ndarray:
    """原图归一化 (cx,cy,w,h) → 画布 xyxy 像素（与 dataset 内部一致的映射，不裁剪）。"""
    if boxes_norm is None or len(boxes_norm) == 0:
        return np.zeros((0, 4), np.float32)
    H, W = float(orig_hw[0]), float(orig_hw[1])
    b = np.asarray(boxes_norm, np.float32).reshape(-1, 4)
    x1 = _apply(M, np.c_[b[:, 0] * W - b[:, 2] * W / 2, b[:, 1] * H - b[:, 3] * H / 2])
    x2 = _apply(M, np.c_[b[:, 0] * W + b[:, 2] * W / 2, b[:, 1] * H + b[:, 3] * H / 2])
    xa, xb = np.minimum(x1[:, 0], x2[:, 0]), np.maximum(x1[:, 0], x2[:, 0])
    ya, yb = np.minimum(x1[:, 1], x2[:, 1]), np.maximum(x1[:, 1], x2[:, 1])
    return np.stack([xa, ya, xb, yb], 1).astype(np.float32)


def collate(batch: List[Optional[dict]]) -> Optional[dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    enabled = list(batch[0]["enabled"])
    # 每张图分别填零；交集会让一张图 dropout 导致整批都失去质量描述子。
    qkeys = [k for k in ("rgb", "ir", "dep") if k in enabled]
    qshape = (3, *batch[0]["prior"].shape[-2:])
    qchannels = {}
    for key in qkeys:
        for item in batch:
            if key in item["quality"]:
                qchannels[key] = int(item["quality"][key].shape[0])
                break
        qchannels.setdefault(key, 9 if key == "ir" else 3)
    out = {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "ir": torch.stack([b["ir"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "prior": torch.stack([b["prior"] for b in batch]),
        "quality": {k: torch.stack([b["quality"].get(
                                    k, b["prior"].new_zeros((qchannels[k], *b["prior"].shape[-2:])))
                                    for b in batch]) for k in qkeys},
        "boxes": [b["boxes"] for b in batch],
        "M": torch.stack([b["M"] for b in batch]),
        "alignment_shift": torch.stack([b.get("alignment_shift", torch.zeros(2)) for b in batch]),
        "alignment_supervised": torch.stack([b.get("alignment_supervised", torch.tensor(0.0)) for b in batch]),
        "ir_affine_target": torch.stack([b.get("ir_affine_target", torch.zeros(4)) for b in batch]),
        "ir_affine_supervised": torch.stack([b.get("ir_affine_supervised", torch.tensor(0.0)) for b in batch]),
        "orig_hw": torch.stack([b["orig_hw"] for b in batch]),
        "stems": [b["stem"] for b in batch],
        "keep": {k: torch.tensor([b["keep"][k] for b in batch]) for k in ("rgb", "ir", "dep")},
        "enabled": enabled,
    }
    for key, channels in (("availability",3),("scene_id",1)):
        if any(key in b["quality"] for b in batch):
            shape = (channels,*batch[0]["rgb"].shape[-2:])
            out["quality"][key] = torch.stack([b["quality"].get(key,torch.ones(shape)) for b in batch])
    # 缺失模态整路置零（承重配方）。⚠️ 这里只清图像；质量描述子/先验在 __getitem__ 里就已按 keep 清除。
    for m in ("rgb", "ir"):
        out[m] = out[m] * out["keep"][m].view(-1, 1, 1, 1)
    out["depth"] = out["depth"] * out["keep"]["dep"].view(-1, 1, 1, 1)
    return out


# ---------------------------------------------------------------- 索引构建

def _parse_label_file(lp: Path) -> np.ndarray:
    """读标签：校验并裁剪到 [0,1]（官方修正版有 5 个坐标轻微越界）。"""
    rows = [l.split() for l in lp.read_text().strip().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 5), np.float32)
    arr = np.array([[int(r[0]), *map(float, r[1:5])] for r in rows], np.float32).reshape(-1, 5)
    arr[:, 1:5] = np.clip(arr[:, 1:5], 0.0, 1.0)
    return arr


def _index_source_signature(root: Path, label_dir: Optional[Path]) -> str:
    """缓存命中前核对文件清单与元数据，避免同目录标签修改后仍用旧框。"""
    digest = hashlib.blake2b(digest_size=16)
    for directory in ([root / m for m in MODS] + ([label_dir] if label_dir else [])):
        if not directory.is_dir():
            raise FileNotFoundError(f"索引源目录不存在：{directory}")
        digest.update(str(directory.resolve()).encode("utf-8"))
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.suffix.lower() not in EXTS + (".txt",):
                continue
            stat = path.stat()
            digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def build_index(root: Path, label_dir: Optional[Path] = None, limit: int = 0,
                hash_size: int = 32, cache_dir: Optional[Path] = None,
                use_cache: bool = True) -> List[dict]:
    """扫描三模态 + 标签，返回样本列表（含 dHash 供分组用）。

    ⚠️ dHash 要对每张 visible **全分辨率解码**，2000 张单线程要 5–10 分钟。
    结果缓存成 JSON（目录名 + 数量 + 版本）；命中时还核对源目录及文件清单、大小和 mtime。
    版本/文件元数据不匹配时自动重建，不沿用过期标签。
    """
    root = Path(root)
    label_dir = Path(label_dir) if label_dir else None
    signature = _index_source_signature(root, label_dir) if use_cache else None
    cache_dir = Path(cache_dir) if cache_dir else (root.parent / "_mm_index_cache")
    cache = cache_dir / (f"index_{root.name}__{label_dir.name if label_dir else 'nolab'}"
                         f"__{limit or 'all'}__v{INDEX_VER}.json")
    if use_cache and cache.exists():
        try:
            obj = json.loads(cache.read_text(encoding="utf-8"))
            if int(obj.get("ver", -1)) != INDEX_VER or int(obj.get("hash_bits", -1)) != HASH_BITS:
                raise ValueError("缓存版本/哈希位数不匹配")
            if obj.get("source_signature") != signature:
                raise ValueError("图像/标签清单或文件元数据已变化")
            idx = obj["samples"]
            for s in idx:
                s["boxes"] = (np.array(s["boxes"], np.float32).reshape(-1, 5) if s["boxes"]
                              else np.zeros((0, 5), np.float32))
                s["_hash_int"] = int(s["_hash"])
            print(f"[index] 命中缓存 {cache.name}（{len(idx)} 样本）")
            return idx
        except Exception as exc:                                     # noqa: BLE001
            print(f"[index] 缓存不可用，重建：{exc}")
    per_mod: Dict[str, Dict[str, str]] = {}
    for m in MODS:
        d = root / m
        if not d.is_dir():
            raise SystemExit(f"缺少模态目录: {d}")
        per_mod[m] = {p.stem: p.name for p in d.iterdir() if p.suffix.lower() in EXTS}
    stems = sorted(set.intersection(*[set(v) for v in per_mod.values()]))
    missing = {m: len(set(per_mod[m]) - set(stems)) for m in MODS}
    if any(missing.values()):
        print(f"[index] 警告：三模态未完全配对，各模态独有文件数 {missing}")
    if limit:
        stems = stems[:limit]
    samples = []
    for st in stems:
        boxes = None
        if label_dir is not None:
            lp = label_dir / f"{st}.txt"
            if lp.exists():
                boxes = _parse_label_file(lp)
        thumb = imread_unicode(root / "visible" / per_mod["visible"][st], cv2.IMREAD_GRAYSCALE)
        h_int = dhash(thumb) if thumb is not None else 0
        samples.append({"stem": st, "files": {m: per_mod[m][st] for m in MODS},
                        "boxes": boxes, "_hash_int": h_int,
                        "_thumb_hash": _thumb_hash_bits(thumb) if thumb is not None else 0})
    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "ver": INDEX_VER, "hash_bits": HASH_BITS, "n": len(samples),
                "source_signature": signature,
                "samples": [{"stem": s["stem"], "files": s["files"],
                             "boxes": (s["boxes"].tolist() if s["boxes"] is not None else []),
                             "_hash": int(s["_hash_int"]),
                             "_thumb_hash": int(s["_thumb_hash"])} for s in samples]},
                ensure_ascii=False), encoding="utf-8")
            print(f"[index] 已写入缓存 {cache}")
        except Exception as exc:                                     # noqa: BLE001
            print(f"[index] 缓存写入失败（不影响训练）：{exc}")
    return samples


def load_split(path: Path, idx: List[dict]) -> Tuple[List[dict], List[dict]]:
    """复用已保存的 split.json（保证"验证集"跨次运行完全一致，可比）。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    by = {s["stem"]: s for s in idx}
    tr = [by[s] for s in obj.get("train", []) if s in by]
    va = [by[s] for s in obj.get("val", []) if s in by]
    return tr, va


def save_split(path: Path, train: List[dict], val: List[dict], extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"train": [s["stem"] for s in train], "val": [s["stem"] for s in val]}
    if extra:
        obj.update(extra)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def val_class_stats(samples: List[dict], nc: int = 12) -> Dict[int, int]:
    """验证集逐类框数（选模可信度自检：某类为 0 就说明该类的 AP 不可信）。"""
    cnt = {c: 0 for c in range(nc)}
    for s in samples:
        b = s.get("boxes")
        if b is None or len(b) == 0:
            continue
        for c in np.asarray(b)[:, 0].astype(int):
            cnt[int(c)] = cnt.get(int(c), 0) + 1
    return cnt


def balanced_sample_weights(samples: Sequence[dict], nc: int = 12,
                            max_weight: float = 3.0) -> Tuple[np.ndarray, Dict[int, int], Dict[int, float]]:
    """按“包含某类的图像数”计算温和稀有类采样权重。

    图像权重取其所含类别权重的最大值；类权重用频次比的平方根并截断到
    ``max_weight``，避免极稀有类把整个 epoch 变成重复记忆。
    """
    max_weight = max(1.0, float(max_weight))
    counts = np.zeros(int(nc), dtype=np.int64)
    per_image = []
    for s in samples:
        cs = sorted(c for c in _image_classes(s) if 0 <= c < nc)
        per_image.append(cs)
        for c in cs:
            counts[c] += 1
    present = counts[counts > 0]
    ref = float(present.max()) if len(present) else 1.0
    factors = np.ones(int(nc), dtype=np.float64)
    nz = counts > 0
    factors[nz] = np.minimum(max_weight, np.sqrt(ref / counts[nz]))
    weights = np.array([max([1.0] + [float(factors[c]) for c in cs])
                        for cs in per_image], dtype=np.float64)
    return (weights, {i: int(counts[i]) for i in range(int(nc))},
            {i: float(factors[i]) for i in range(int(nc))})
