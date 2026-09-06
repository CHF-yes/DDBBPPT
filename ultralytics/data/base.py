# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        use_simotm="RGB",
        pairs_rgb_ir=['visible', 'infrared', 'depth']
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.use_simotm = use_simotm
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        self.pairs_rgb_ir=pairs_rgb_ir
        # 支持二目录(RGBT/RGBRGB6C)或三目录(RGBTD)；长度非法则重置为默认三目录
        # Support 2-dir (RGBT/RGBRGB6C) or 3-dir (RGBTD) modal folders; reset to default on invalid length.
        if not (isinstance(self.pairs_rgb_ir, list) and
                len(self.pairs_rgb_ir) in (2, 3) and
                all(isinstance(x, str) for x in self.pairs_rgb_ir)):
            self.pairs_rgb_ir = ['visible', 'infrared', 'depth']

        # ---- RGBTD 三模态对齐与增强参数(从 hyp/args 读取, 带默认值兜底) ----
        # RGBTD modality alignment & augmentation params (read from hyp/args with defaults)
        self.depth_shift_x = int(getattr(hyp, "depth_shift_x", -22))
        self.depth_shift_y = int(getattr(hyp, "depth_shift_y", 0))
        self.rgb_drop_prob = float(getattr(hyp, "rgb_drop_prob", 0.2))
        self.rgb_drop_mode = str(getattr(hyp, "rgb_drop_mode", "zero"))
        self.ir_gain = float(getattr(hyp, "ir_gain", 0.15))
        self.ir_bias = float(getattr(hyp, "ir_bias", 5.0))
        self.depth_noise = float(getattr(hyp, "depth_noise", 0.02))
        self.depth_jitter_x = tuple(int(v) for v in getattr(hyp, "depth_jitter_x", (-25, 5)))
        self.depth_jitter_y = tuple(int(v) for v in getattr(hyp, "depth_jitter_y", (-5, 5)))
        self.depth_jitter_prob = float(getattr(hyp, "depth_jitter_prob", 1.0))

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        # self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.npy_files = self.generate_npy_files()
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def generate_npy_files(self):
        npy_files = []
        for f in self.im_files:
            file_path = Path(f)
            pre_fix_mode= ""
            if self.use_simotm in {"RGBT","RGBRGB6C","RGBTD"}:
                pre_fix_mode="_"+self.use_simotm
            file_stem = file_path.stem  # 提取文件名主体，比如 "image1"
            new_file_name = file_stem + pre_fix_mode +".npy"
            new_file_path = file_path.parent / new_file_name  # 路径和新文件名拼接
            npy_files.append(Path(str(new_file_path)))
        return npy_files

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_and_preprocess_image(self, file_path, use_simotm=None, pairs_rgb=None, pairs_ir=None, pairs_depth=None):
        if use_simotm is None:
            use_simotm = self.use_simotm

        if use_simotm == 'Gray2BGR':
            im = imread(file_path)  # BGR
        elif use_simotm == 'SimOTM':
            im = imread(file_path, cv2.IMREAD_GRAYSCALE)  # GRAY
            im = SimOTM(im)
        elif use_simotm == 'SimOTMBBS':
            im = imread(file_path, cv2.IMREAD_GRAYSCALE)  # GRAY
            im = SimOTMBBS(im)
        elif use_simotm == 'Gray':
            im = imread(file_path, cv2.IMREAD_GRAYSCALE)  # GRAY
        elif use_simotm == 'Gray16bit':
            im = imread(file_path, cv2.IMREAD_UNCHANGED)  # GRAY
            im = im.astype(np.float32)
        elif use_simotm == 'Multispectral':
            im = imread(file_path, cv2.IMREAD_COLOR)  # Multispectral
        elif use_simotm == 'Multispectral_16bit':
            im = imread(file_path, cv2.IMREAD_UNCHANGED)  # Multispectral  16bit
        elif use_simotm == 'SimOTMSSS':
            im = imread(file_path, cv2.IMREAD_UNCHANGED)  # TIF 16bit
            im = im.astype(np.float32)
            im = SimOTMSSS(im)
        elif use_simotm == 'RGBT':
            im_visible = imread(file_path)  # BGR
            im_infrared = imread(file_path.replace(pairs_rgb, pairs_ir), cv2.IMREAD_GRAYSCALE)  # GRAY

            if im_visible is None or im_infrared is None:
                raise FileNotFoundError(f"Image Not Found {file_path}")

            im_visible, im_infrared = self._resize_images(im_visible, im_infrared)
            im = self._merge_channels(im_visible, im_infrared)
        elif use_simotm == 'RGBRGB6C':
            im_visible = imread(file_path)  # BGR
            im_infrared = imread(file_path.replace(pairs_rgb, pairs_ir))  # BGR

            if im_visible is None or im_infrared is None:
                raise FileNotFoundError(f"Image Not Found {file_path}")

            im_visible, im_infrared = self._resize_images(im_visible, im_infrared)
            im = self._merge_channels_rgb(im_visible, im_infrared)
        elif use_simotm == 'RGBTD':
            # 三模态：可见光(BGR 3ch) + 红外(3ch) + 深度(16bit 单通道→归一化→3ch)
            im_visible = imread(file_path)  # BGR
            im_infrared = imread(file_path.replace(pairs_rgb, pairs_ir))  # 3ch 或 1ch
            im_depth = imread(file_path.replace(pairs_rgb, pairs_depth), cv2.IMREAD_UNCHANGED)  # 16bit 单通道

            if im_visible is None or im_infrared is None or im_depth is None:
                raise FileNotFoundError(f"Image Not Found {file_path}")

            # 红外统一成 3 通道（赛题红外为单通道灰度堆叠 3 份，单通道文件则复制成 3 通道）
            if im_infrared.ndim == 2:
                im_infrared = cv2.cvtColor(im_infrared, cv2.COLOR_GRAY2BGR)
            elif im_infrared.ndim == 3 and im_infrared.shape[2] == 4:
                im_infrared = im_infrared[:, :, :3]

            # 深度对齐修正(配准, 训练/验证都做): depth 相对 RGB 系统性偏移的固定平移
            im_depth = self._align_depth(im_depth)

            # 训练期三模态一致性增强(仅 self.augment): RGB dropout/HSV, IR 抖动, Depth 噪声+随机平移
            if self.augment:
                im_visible, im_infrared, im_depth = self._rgbtd_augment(im_visible, im_infrared, im_depth)

            # 深度：16bit → 归一化 → 8bit 单通道 → 复制成 3 通道
            im_depth = self._preprocess_depth(im_depth)
            im_depth = cv2.cvtColor(im_depth, cv2.COLOR_GRAY2BGR)

            im_visible, im_infrared, im_depth = self._resize_images_3(im_visible, im_infrared, im_depth)
            im = self._merge_channels_rgbt_depth(im_visible, im_infrared, im_depth)
        else:
            im = imread(file_path, cv2.IMREAD_COLOR)  # BGR

        if im is None:
            raise FileNotFoundError(f"Image Not Found {file_path}")

        return im

    def _resize_images(self, im_visible, im_infrared):
        h_vis, w_vis = im_visible.shape[:2]  # orig hw
        h_inf, w_inf = im_infrared.shape[:2]  # orig hw

        if h_vis != h_inf or w_vis != w_inf:
            r_vis = self.imgsz / max(h_vis, w_vis)  # ratio
            r_inf = self.imgsz / max(h_inf, w_inf)  # ratio

            if r_vis != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r_vis > 1) else cv2.INTER_AREA
                im_visible = cv2.resize(im_visible, (
                min(math.ceil(w_vis * r_vis), self.imgsz), min(math.ceil(h_vis * r_vis), self.imgsz)),
                                        interpolation=interp)
            if r_inf != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r_inf > 1) else cv2.INTER_AREA
                im_infrared = cv2.resize(im_infrared, (
                min(math.ceil(w_inf * r_inf), self.imgsz), min(math.ceil(h_inf * r_inf), self.imgsz)),
                                         interpolation=interp)
        return im_visible, im_infrared

    def _merge_channels(self, im_visible, im_infrared):
        b, g, r = cv2.split(im_visible)
        im = cv2.merge((b, g, r, im_infrared))
        return im

    def _merge_channels_rgb(self, im_visible, im_infrared):
        b, g, r = cv2.split(im_visible)
        b2, g2, r2 = cv2.split(im_infrared)
        im = cv2.merge((b, g, r, b2, g2, r2))
        return im

    def _preprocess_depth(self, im_depth):
        """深度图预处理：16bit 毫米 → 8bit [0,255]，有效区逐帧 min-max 归一化，无效值保持 0。

        兼容两种深度输入：
        - 16bit 单通道 PNG（正式深度图）
        - 8bit 3 通道 JPG（深度可视化/伪彩色图，样例中混入），先转单通道灰度

        归一化只对有效像素(≥1e-3)统计 min/max：无效空洞(<1e-3)不参与统计并强制保持 0，
        避免"无效区的 0"拉低 d_min 使 min-max 退化成"除以 max"，同时防止有效/无效边界被
        线性拉伸出伪值；最后 clip 到 [0,255] 保证无越界。
        """
        if im_depth.ndim == 3:
            # 3 通道深度可视化图 → 转单通道灰度（近似深度），避免 cvtColor(GRAY2BGR) 崩溃
            im_depth = cv2.cvtColor(im_depth, cv2.COLOR_BGR2GRAY)
        im_depth = im_depth.astype(np.float32)
        mask_invalid = im_depth < 1e-3  # 无效深度空洞
        valid = im_depth[~mask_invalid]
        if valid.size > 0:
            d_min = float(valid.min())
            d_max = float(valid.max())
            if d_max - d_min > 1e-6:
                im_depth = (im_depth - d_min) / (d_max - d_min) * 255.0
            else:
                im_depth = np.zeros_like(im_depth)
        else:
            im_depth = np.zeros_like(im_depth)
        im_depth[mask_invalid] = 0.0  # 无效区保持 0
        return np.clip(im_depth, 0, 255).astype(np.uint8)

    def _resize_images_3(self, im_visible, im_infrared, im_depth):
        """对齐三模态图像尺寸，确保三者最终 (H, W) 完全一致（以可见光目标尺寸为基准）。

        三路各自原始尺寸/宽高比可能不同，若各自按自己的长边独立缩放，缩放后尺寸仍可能
        不一致，导致后续 cv2.merge 报错。这里统一以可见光缩放后的尺寸为准，把红外/深度
        强制 resize 到同一 (W, H)。
        深度图用 INTER_NEAREST(最近邻)缩放，防止在"0=无效"与"有效深度"边界处
        因线性插值引入伪深度值；可见光/红外保持原插值策略。
        """
        h_ref, w_ref = im_visible.shape[:2]

        def _target_size(h, w):
            """按长边缩放到 imgsz，返回 (W, H)；与官方 letterbox 前置逻辑一致。"""
            r = self.imgsz / max(h, w)
            if r == 1:
                return w, h
            return min(math.ceil(w * r), self.imgsz), min(math.ceil(h * r), self.imgsz)

        w_t, h_t = _target_size(h_ref, w_ref)  # 以可见光目标尺寸为基准

        out = []
        for idx, im in enumerate((im_visible, im_infrared, im_depth)):
            h, w = im.shape[:2]
            if idx == 2:  # depth 通道用最近邻，避免边界伪值
                interp = cv2.INTER_NEAREST
            else:
                r = self.imgsz / max(h, w)
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            if (w, h) != (w_t, h_t):
                im = cv2.resize(im, (w_t, h_t), interpolation=interp)
            out.append(im)
        return out[0], out[1], out[2]

    def _merge_channels_rgbt_depth(self, im_visible, im_infrared, im_depth):
        """合并三模态为 9 通道：BGR + IR(3) + Depth(3)。"""
        b, g, r = cv2.split(im_visible)
        ib, ig, ir = cv2.split(im_infrared)
        db, dg, dr = cv2.split(im_depth)
        im = cv2.merge((b, g, r, ib, ig, ir, db, dg, dr))
        return im

    # ------------------------------------------------------------------
    # RGBTD 三模态对齐与增强（借鉴 feature/multimodal-detection-framework 一致性策略）
    # ------------------------------------------------------------------

    def _align_depth(self, im_depth):
        """深度图固定平移对齐(配准, 训练/验证都做)。

        depth 相对 RGB 存在系统性偏移(实测约右偏 22px@1920x1080)，用最近邻 + 填 0(无效)
        平移修正；标签框不跟随(标签锚定 RGB 坐标系，谁偏谁修)。
        """
        if self.depth_shift_x == 0 and self.depth_shift_y == 0:
            return im_depth
        h, w = im_depth.shape[:2]
        M = np.float32([[1, 0, float(self.depth_shift_x)], [0, 1, float(self.depth_shift_y)]])
        return cv2.warpAffine(im_depth, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _rgbtd_augment(self, im_visible, im_infrared, im_depth):
        """训练期三模态一致性增强(仅 self.augment 时调用)。

        只做"模态级"增强，几何操作(flip/letterbox/mosaic)交给 ultralytics 内建 pipeline
        在 9 通道拼接后统一执行(几何天然同步)。HSV 由内建 RandomHSV9C 处理。
        - RGB: 随机整图失效(dropout)，防网络只依赖 RGB
        - IR : 灰度增益/偏置抖动(模拟传感器差异)
        - Depth: 有效区乘性噪声(16bit 域) + 随机平移(模拟对齐残差)
        """
        # (1) RGB 随机失效(防主导模态垄断；仅训练)
        if self.rgb_drop_prob > 0 and random.random() < self.rgb_drop_prob:
            im_visible = self._rgb_dropout(im_visible, self.rgb_drop_mode)
        # (2) IR 增益/偏置抖动
        im_infrared = self._ir_gain_jitter(im_infrared)
        # (3) Depth 有效区乘性噪声(16bit 域，无效区 0 保持)
        im_depth = self._depth_value_noise(im_depth)
        # (4) Depth 随机平移(模拟对齐残差，标签不跟随)
        im_depth = self._depth_random_jitter(im_depth)
        return im_visible, im_infrared, im_depth

    def _rgb_dropout(self, rgb, mode):
        """RGB 整图失效。mode: zero(全黑)/gray(灰度保结构)/noise(压暗+噪声)。"""
        if mode == "gray":
            g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            return np.stack([g, g, g], axis=-1)
        if mode == "noise":
            out = rgb.astype(np.float32) * random.uniform(0.2, 0.6)
            noise = np.asarray([random.gauss(0.0, 15.0) for _ in range(3)], dtype=np.float32)
            out = out + noise[None, None, :]
            return np.clip(out, 0, 255).astype(np.uint8)
        return np.zeros_like(rgb)  # zero（默认）

    def _ir_gain_jitter(self, ir):
        """红外灰度增益(乘性) + 偏置(加性)抖动，模拟传感器响应差异。"""
        if self.ir_gain <= 0 and self.ir_bias <= 0:
            return ir
        g = 1.0 + random.uniform(-self.ir_gain, self.ir_gain)
        b = random.uniform(-self.ir_bias, self.ir_bias)
        return np.clip(ir.astype(np.float32) * g + b, 0, 255).astype(np.uint8)

    def _depth_value_noise(self, depth):
        """深度有效区(>1)乘性噪声，模拟测距抖动；无效区(0)保持原样。"""
        if self.depth_noise <= 0:
            return depth
        d = depth.astype(np.float32)
        valid = d > 1
        if not bool(valid.any()):
            return depth
        scale = 1.0 + random.uniform(-self.depth_noise, self.depth_noise)
        d[valid] = d[valid] * scale
        return d.astype(depth.dtype)

    def _depth_random_jitter(self, depth):
        """深度随机平移(模拟未对齐残差)，仅训练；最近邻 + 填 0。"""
        if random.random() >= self.depth_jitter_prob:
            return depth
        sx = random.randint(self.depth_jitter_x[0], self.depth_jitter_x[1])
        sy = random.randint(self.depth_jitter_y[0], self.depth_jitter_y[1])
        if sx == 0 and sy == 0:
            return depth
        h, w = depth.shape[:2]
        M = np.float32([[1, 0, float(sx)], [0, 1, float(sy)]])
        return cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        pairs_rgb = self.pairs_rgb_ir[0]
        pairs_ir = self.pairs_rgb_ir[1] if len(self.pairs_rgb_ir) > 1 else pairs_rgb
        pairs_depth = self.pairs_rgb_ir[2] if len(self.pairs_rgb_ir) > 2 else None
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    # im = imread(f,cv2.IMREAD_COLOR)  # BGR
                    im = self.load_and_preprocess_image(f, use_simotm=self.use_simotm, pairs_rgb=pairs_rgb, pairs_ir=pairs_ir, pairs_depth=pairs_depth)
            else:  # read image
                im = self.load_and_preprocess_image(f, use_simotm=self.use_simotm, pairs_rgb=pairs_rgb, pairs_ir=pairs_ir, pairs_depth=pairs_depth)

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            pairs_rgb = self.pairs_rgb_ir[0]
            pairs_ir = self.pairs_rgb_ir[1] if len(self.pairs_rgb_ir) > 1 else pairs_rgb
            pairs_depth = self.pairs_rgb_ir[2] if len(self.pairs_rgb_ir) > 2 else None
            im = self.load_and_preprocess_image(self.im_files[i], use_simotm=self.use_simotm, pairs_rgb=pairs_rgb, pairs_ir=pairs_ir, pairs_depth=pairs_depth)
            np.save(f.as_posix(), im, allow_pickle=False)

    def check_cache_disk(self, safety_margin=0.5):
        """Check image caching requirements vs available disk space."""
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue

            ratio_m =1.0
            if self.use_simotm in { 'RGBT', 'RGBRGB6C'}:
                ratio_m=2.0
            elif self.use_simotm == 'RGBTD':
                ratio_m=3.0

            b += im.nbytes * ratio_m
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio

            ratio_m =1.0
            if self.use_simotm in { 'RGBT', 'RGBRGB6C'}:
                ratio_m=2.0
            elif self.use_simotm == 'RGBTD':
                ratio_m=3.0
            b += im.nbytes * ratio**2 *ratio_m

        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
            return False
        return True

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes,  # xywh
                segments=segments,  # xy
                keypoints=keypoints,  # xy
                normalized=True,  # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError


# When reorganizing the code later, they will be considered to be placed in other folders. For now, it is temporarily kept here.
#------------------------------------------------------------------------------ 后续整理代码时会考虑放在其他文件夹,暂时放在此处
def receptiveField(img, R=3, r=1, fac_r=-1, fac_R=6):
    # img1 = np.float32(img)

    x, y = np.meshgrid(np.arange(1, R * 2 + 2), np.arange(1, R * 2 + 2))
    dis = np.sqrt((x - (R + 1)) ** 2 + (y - (R + 1)) ** 2)
    flag1 = (dis <= r)
    flag2 = np.logical_and(dis > r, dis <= R)
    kernal = flag1 * fac_r + flag2 * fac_R
    # kernal /= kernal.sum()
    kernal = kernal / kernal.sum()
    out = cv2.filter2D(img, -1, kernal)
    return out


def SimOTM(img):
    blur = cv2.blur(img, (3, 3))
    rec = receptiveField(img)
    result = cv2.merge([img, blur, rec])
    return result

def SimOTMBBS(img):
    blur = cv2.blur(img, (3, 3))
    result = cv2.merge([img, blur, blur])
    return result

def SimOTMSSS(img):
    #  TIF  16 bit
    result = cv2.merge([img, img, img])
    return result

def enhance_brightness_or_contrast(image, target_gray_value, brightness_alpha=1.5, contrast_alpha=1.0, beta=0):
    gray_value = np.mean(image)
    if gray_value >= target_gray_value:
        enhanced_image = cv2.convertScaleAbs(image, alpha=contrast_alpha, beta=beta)
    else:
        avg_diff = target_gray_value - gray_value
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.0, beta=avg_diff)
    return enhanced_image

def SimOTMBrights(img):
    blur = cv2.blur(img, (3, 3))
    rec = receptiveField(img)
    result = cv2.merge([img, blur, rec])
    return result

#------------------------------------------------------------------------------