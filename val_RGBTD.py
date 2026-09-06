# -*- coding: utf-8 -*-
"""
RGBTD 三模态目标检测验证 / 测试脚本（含 TTA）
=============================================
用法：
    python val_RGBTD.py
说明：
    - 常规验证：计算 mAP50 / mAP50-95；
    - TTA 验证：开启测试时增强（多尺度 + 翻转），零训练成本提升精度。
"""

import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 训练产出的最佳权重（或迁移后的预训练权重）
    model = YOLO("runs/RGBTD/RGBTD-yolo11x-midfusion/weights/best.pt")

    data = R"ultralytics/cfg/datasets/coco8-rgbtd.yaml"  # TODO: 改成你的数据集 YAML

    # 1) 常规验证
    metrics = model.val(
        data=data,
        imgsz=640,
        device="0",
        use_simotm="RGBTD",
        channels=9,
        pairs_rgb_ir=["visible", "infrared", "depth"],
    )
    print(f"[常规验证] mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")

    # 2) TTA 验证（测试时增强，白捡 0.5~1.5 个点）
    metrics_tta = model.val(
        data=data,
        imgsz=640,
        device="0",
        augment=True,  # 开启 TTA
        use_simotm="RGBTD",
        channels=9,
        pairs_rgb_ir=["visible", "infrared", "depth"],
    )
    print(f"[TTA 验证] mAP50: {metrics_tta.box.map50:.4f} | mAP50-95: {metrics_tta.box.map:.4f}")
