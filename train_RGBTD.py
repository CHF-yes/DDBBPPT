# -*- coding: utf-8 -*-
"""
AIC2026 赛题训练脚本 —— RGBTD（可见光 + 红外 + 深度）三模态目标检测
====================================================================
数据集：/root/autodl-tmp/aic2026/train（12 类，全部 2000 张训练集参与训练）
预训练：yolo11x-RGBTD-pretrained.pt（由 transfer_pretrained.py 从官方 COCO 权重迁移）

用法：
    cd /root/rgbtd
    python train_RGBTD.py
"""

import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 复用迁移后的三模态预训练权重（backbone 复用 COCO 参数，收敛更快更准）
    model = YOLO("yolo11x-RGBTD-pretrained.pt")

    model.train(
        data="ultralytics/cfg/datasets/aic2026-rgbtd.yaml",
        cache=False,          # 三模态增强在加载阶段执行，开 cache 会固化增强、丧失随机性
        imgsz=640,
        epochs=300,
        batch=8,              # 9 通道显存占用约为 RGB 的 3 倍；OOM 则降到 6
        close_mosaic=10,
        workers=8,
        device="0",
        optimizer="SGD",
        seed=0,
        # ---- 三模态关键参数 ----
        use_simotm="RGBTD",
        channels=9,           # BGR(3) + IR(3) + Depth(3)
        pairs_rgb_ir=["visible", "infrared", "depth"],
        # ---- 深度对齐修正（本数据集为官方已对齐数据，故平移量为 0）----
        depth_shift_x=0,
        depth_shift_y=0,
        # ---- 三模态鲁棒性增强（仅训练阶段生效）----
        rgb_drop_prob=0.2,    # RGB 整图随机失效概率，防网络只依赖可见光
        rgb_drop_mode="zero",
        ir_gain=0.15,         # 红外增益抖动
        ir_bias=5.0,          # 红外偏置抖动
        depth_noise=0.02,     # 深度有效区乘性噪声比例
        depth_jitter_x=[-25, 5],
        depth_jitter_y=[-5, 5],
        depth_jitter_prob=1.0,
        # ---- 输出 ----
        project="runs/RGBTD",
        name="aic2026-yolo11x-rgbtd",
        exist_ok=True,
    )
