# -*- coding: utf-8 -*-
"""YOLO11 最小实验：下载权重 + GPU 推理验证"""
import sys
import torch
import torchvision
import ultralytics

from ultralytics import YOLO

print("=" * 50)
print("环境信息")
print("=" * 50)
print(f"Python        : {sys.version.split()[0]}")
print(f"torch         : {torch.__version__}")
print(f"torchvision   : {torchvision.__version__}")
print(f"ultralytics   : {ultralytics.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"显存总量       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print()
print("=" * 50)
print("第 1 步: 加载 YOLO11s 权重 (会自动下载 ~19MB)")
print("=" * 50)
model = YOLO("yolo11s.pt")
print(f"模型已加载, 参数量约 {sum(p.numel() for p in model.model.parameters())/1e6:.1f}M")

print()
print("=" * 50)
print("第 2 步: CPU 推理 bus.jpg (Ultralytics 官方示例图)")
print("=" * 50)
res_cpu = model.predict("bus.jpg", device="cpu", verbose=False)
boxes_cpu = res_cpu[0].boxes
print(f"CPU  检出 {len(boxes_cpu)} 个目标")

print()
print("=" * 50)
print("第 3 步: GPU (CUDA) 推理同一张图")
print("=" * 50)
res_gpu = model.predict("bus.jpg", device="cuda", verbose=False)
boxes_gpu = res_gpu[0].boxes
print(f"GPU  检出 {len(boxes_gpu)} 个目标")
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")

print()
print("=" * 50)
print("验证通过: YOLO11 已在 EFYOLO 环境跑通 (CPU + GPU)")
print("=" * 50)
