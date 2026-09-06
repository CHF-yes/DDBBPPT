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
        # ---- 输出 ----
        project="runs/RGBTD",
        name="RGBTD-yolo11x-midfusion",
    )
