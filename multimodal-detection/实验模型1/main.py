# -*- coding: utf-8 -*-
"""
实验模型1 主入口 —— RGB 主流 + 轻辅助流 + 分级融合。

用法（EFYOLO 环境；自动使用 vendor ultralytics 源码）：
    python 实验模型1/main.py config        # 打印配置
    python 实验模型1/main.py selfcheck     # 构建模型 + dummy 前向验证
    python 实验模型1/main.py train   ...   # 训练占位（自定义循环见 README）
    python 实验模型1/main.py predict ...   # 推理占位（同上）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_CODE_ROOT / "vendor"), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                      # noqa: E402
from common import trainer as TR                # noqa: E402
from common.multimodal_augment import effective_depth_shift  # noqa: E402


def cmd_config(args):
    print(MC.summarize())
    print(TR.train_config_prints(MC.EXPERIMENT1))
    h = MC.EXPERIMENT1.hyper
    a = h.aug
    print(f"  depth_align : ({h.depth_shift_x}, {h.depth_shift_y})")
    print(f"  depth_jitter: x={a.depth_jitter_x} y={a.depth_jitter_y} p={a.depth_jitter_prob}")


def cmd_selfcheck(args):
    import model_builder as MB
    m = MB.build_experiment1(weights=MC.EXPERIMENT1.hyper.pretrained_weights,
                             nc=MC.EXPERIMENT1.class_num,
                             mega=MC.EXPERIMENT1.hyper.mega,
                             aux_heads=MC.EXPERIMENT1.hyper.aux_heads)
    print("selfcheck: model built OK, params(M) =",
          round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
    print("(dummy forward 请在 model_builder.py __main__ 中运行)")


def cmd_train(args):
    import random
    import numpy as np
    import torch
    from common import scan_data as SD
    from common import train_loop as TL

    cfg = MC.EXPERIMENT1
    h = cfg.hyper
    import model_builder as MB
    import dataset_adapter as DA
    model = MB.build_experiment1(weights=args.weights or h.pretrained_weights,
                                 nc=cfg.class_num,
                                 mega=h.mega, aux_heads=h.aux_heads)

    # --- 数据：扫描根目录；无 val 时从 train 抽出 10% 作 val ---
    root = Path(args.data_root) if args.data_root else MC.DATA_ROOT
    if not root.exists():
        raise SystemExit(f"[实验模型1] 数据根不存在: {root}（先设 MULTIMODAL_DATA_ROOT 或 DATA_ROOT）")
    scanned = SD.scan_samples(root)
    train_samples = scanned.get("train", []) or next(iter(scanned.values()))
    val_samples = scanned.get("val", []) or scanned.get("validation", [])
    if not val_samples and train_samples:
        rng = random.Random(h.seed)
        order = list(train_samples); rng.shuffle(order)
        k = max(1, int(len(order) * 0.1))
        val_samples = order[:k]; train_samples = order[k:]
        print(f"[实验模型1] 无现成 val，从 train 抽出 {k} 组作验证集")
    print(f"[实验模型1] train={len(train_samples)} val={len(val_samples)}")

    imgsz = (args.imgsz or h.imgsz, args.imgsz or h.imgsz)
    isz = int(args.imgsz or h.imgsz)

    aug_cfg = h.aug                      # 统一增强配置（models_config.AugmentParams）

    def build_batch(chunk, rng, augment=True):
        rgb_l, ir_l, dep_l, boxes_l, stems = [], [], [], [], []
        for s in chunk:
            rgb, ir, dep, boxes, stem = DA.build_model_inputs(
                s, imgsz=imgsz,
                aug=aug_cfg if augment else None,
                depth_shift=effective_depth_shift(h.align, imgsz[0]),
                preprocess=h.preprocess,
                seed=rng.randrange(1 << 31), to_tensor=False)
            rgb_l.append(rgb); ir_l.append(ir); dep_l.append(dep)
            boxes_l.append(boxes); stems.append(stem)
        batch = TL.make_batch_dict(rgb_l, boxes_l, stems, imgsz)   # img 占位=RGB 通道
        inputs = {
            "rgb": torch.from_numpy(np.stack(rgb_l)).float(),
            "ir": torch.from_numpy(np.stack(ir_l)).float(),
            "depth": torch.from_numpy(np.stack(dep_l)).float(),
        }
        return inputs, batch

    build_val_batch = lambda chunk, rng: build_batch(chunk, rng, augment=False)  # noqa: E731

    def forward_fn(model, inputs):
        return model(inputs["rgb"], inputs["ir"], inputs["depth"])

    TL.train_custom(model, train_samples, val_samples, cfg,
                    build_batch, forward_fn, out_dir=args.out,
                    build_val_batch=build_val_batch, dataset_mode="three",
                    imgsz_override=isz)


def cmd_predict(args):
    raise SystemExit(
        "[实验模型1] 推理需先完成训练；结构见 model_builder.Experiment1Model，"
        "输出后处理（NMS→赛题 txt 格式）参考 common.inference。")


def main():
    ap = argparse.ArgumentParser(prog="实验模型1")
    ap.add_argument("task", choices=["config", "selfcheck", "train", "predict"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-yaml", default=None)
    ap.add_argument("--data-root", default=None, help="数据根目录（默认 MC.DATA_ROOT / $MULTIMODAL_DATA_ROOT）")
    ap.add_argument("--imgsz", type=int, default=None, help="训练分辨率快捷覆盖（如 640/1024）")
    args = ap.parse_args()
    {"config": cmd_config, "selfcheck": cmd_selfcheck,
     "train": cmd_train, "predict": cmd_predict}[args.task](args)


if __name__ == "__main__":
    main()
