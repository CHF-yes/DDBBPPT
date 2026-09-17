# -*- coding: utf-8 -*-
"""
RGBTD 三模态目标检测 —— 验证 / 测试脚本（AIC2026 赛题适配版）
==============================================================
用法：
    python val_RGBTD.py                          # 用默认权重 + 默认数据集
    python val_RGBTD.py --tta                    # 额外跑一次 TTA（多尺度+翻转）
    python val_RGBTD.py --weights <权重路径> --data <数据集YAML>

要点：
    1. 验证 / 推理的三模态参数（use_simotm / channels / pairs_rgb_ir）必须与训练**完全一致**，
       否则模型会把 9 通道输入误判成 1 通道而报错。
    2. depth_shift_x / depth_shift_y 必须与训练时相同（本赛题数据为官方已对齐，值为 0）。
       否则训练看到的是"对齐后的深度"、验证看到的是"未对齐深度"，指标失真。
    3. TTA 默认不开，需显式 --tta（部分比赛规则禁止 TTA，请先确认赛题是否允许）。
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

# ---- 默认路径（可用命令行参数或环境变量覆盖）----
# 服务器上训练产出：runs/RGBTD/aic2026-yolo11x-rgbtd/weights/best.pt
# 本地备份副本：    训练成果备份/AIC2026_12类_yolo11x-rgbtd/best.pt
DEFAULT_WEIGHTS = os.environ.get(
    "RGBTD_WEIGHTS", "runs/RGBTD/aic2026-yolo11x-rgbtd/weights/best.pt"
)
DEFAULT_DATA = os.environ.get(
    "RGBTD_DATA", "ultralytics/cfg/datasets/aic2026-rgbtd.yaml"
)

# ---- 与训练保持一致的三模态参数（改动需同步 train_RGBTD.py）----
MODAL_ARGS = dict(
    use_simotm="RGBTD",                       # 三模态：可见光 + 红外 + 深度
    channels=9,                               # BGR(3) + IR(3) + Depth(3)
    pairs_rgb_ir=["visible", "infrared", "depth"],
    depth_shift_x=0,                          # 官方已对齐数据，平移量为 0
    depth_shift_y=0,
)


def main():
    ap = argparse.ArgumentParser(description="RGBTD 三模态验证 / 测试")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="权重 .pt 路径")
    ap.add_argument("--data", default=DEFAULT_DATA, help="数据集 YAML 路径")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tta", action="store_true", help="额外跑一次 TTA（多尺度+翻转）")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(
            f"[错误] 找不到权重文件：{args.weights}\n"
            f"       服务器路径示例：runs/RGBTD/aic2026-yolo11x-rgbtd/weights/best.pt\n"
            f"       本地备份路径：  训练成果备份/AIC2026_12类_yolo11x-rgbtd/best.pt"
        )

    model = YOLO(args.weights)

    # 1) 常规验证
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        **MODAL_ARGS,
    )
    print(f"[常规验证] mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")

    # 2) TTA 验证（可选；测试时增强，零训练成本）
    if args.tta:
        metrics_tta = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            augment=True,  # 开启 TTA
            **MODAL_ARGS,
        )
        print(
            f"[TTA 验证] mAP50: {metrics_tta.box.map50:.4f} | "
            f"mAP50-95: {metrics_tta.box.map:.4f}"
        )


if __name__ == "__main__":
    main()
