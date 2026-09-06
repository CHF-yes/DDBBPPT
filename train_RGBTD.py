# -*- coding: utf-8 -*-
"""
RGBTD（可见光 + 红外 + 深度）三模态目标检测训练脚本
======================================================
推荐流程（使用现有模型参数简化训练）：
    1) python transfer_pretrained.py --src yolo11x.pt   # 迁移 COCO 预训练权重
    2) python train_RGBTD.py                             # 从迁移后的预训练权重开始训练
"""

import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 方式一（推荐）：从迁移后的预训练权重开始训练，复用 COCO 模型参数，收敛更快更准
    model = YOLO("yolo11x-RGBTD-pretrained.pt")  # 由 transfer_pretrained.py 生成

    # 方式二（不推荐）：从零训练
    # model = YOLO("ultralytics/cfg/models/11-RGBT/yolo11x-RGBTD-midfusion.yaml")

    model.train(
        data=R"ultralytics/cfg/datasets/coco8-rgbtd.yaml",  # TODO: 改成你的数据集 YAML
        cache=False,
        imgsz=640,  # 显存允许可调到 1280（准确度更高）
        epochs=300,
        batch=8,  # 9 通道显存占用约为 RGB 的 3 倍，按 GPU 显存调整
        close_mosaic=10,
        workers=2,
        device="0",
        optimizer="SGD",
        # ---- 三模态关键参数 ----
        use_simotm="RGBTD",
        channels=9,  # BGR(3) + IR(3) + Depth(3)
        pairs_rgb_ir=["visible", "infrared", "depth"],  # 三个同级目录名，按实际数据目录改
        # ---- 深度对齐修正(配准, 训练/验证都做) ----
        depth_shift_x=-22,  # depth 相对 RGB 系统性偏移(<0 左移), 实测约 -22px@1920x1080
        depth_shift_y=0,
        # ---- 三模态鲁棒性增强(仅训练) ----
        rgb_drop_prob=0.2,   # RGB 整图随机失效概率, 防网络只依赖 RGB
        rgb_drop_mode="zero",  # zero | gray | noise
        ir_gain=0.15,        # 红外增益抖动幅度
        ir_bias=5.0,         # 红外偏置抖动幅度
        depth_noise=0.02,    # 深度有效区乘性噪声比例
        depth_jitter_x=[-25, 5],  # 深度水平随机平移区间(模拟对齐残差)
        depth_jitter_y=[-5, 5],
        depth_jitter_prob=1.0,
        # ---- 输出 ----
        project="runs/RGBTD",
        name="RGBTD-yolo11x-midfusion",
    )
