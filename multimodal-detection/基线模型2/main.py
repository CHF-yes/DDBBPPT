# -*- coding: utf-8 -*-
"""
基线模型2 —— 三模态「5 通道前期融合」YOLO 主入口。

用法（在 EFYOLO conda 环境运行）:
    python 基线模型2/main.py config
    python 基线模型2/main.py train      (需 DATA_ROOT + data.yaml 就绪)
    python 基线模型2/main.py predict --weights 你的12类权重.pt --data-rgb-gt-of 图像目录

路径引导：自动注入 code 根；模型构建见 model_builder，数据适配见 dataset_adapter。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                  # noqa: E402
from common import trainer as TR            # noqa: E402
from common import inference as INF         # noqa: E402
from common import dataset as DS            # noqa: E402


def cmd_config(args):
    print(MC.summarize())
    print(TR.train_config_prints(MC.BASELINE2_5CH))


def cmd_train(args):
    import random
    import dataclasses
    from common import scan_data as SD
    from common import train_loop as TL

    cfg = MC.BASELINE2_5CH
    # CLI 快捷覆盖（仅本次运行生效，不改 config）：--epochs/--class-num/--batch/--workers
    ov_h = {}
    if args.epochs:
        ov_h["epochs"] = int(args.epochs)
    if args.batch:
        ov_h["batch"] = int(args.batch)
    if args.workers is not None:
        ov_h["workers"] = int(args.workers)
    if ov_h:
        cfg = dataclasses.replace(cfg, hyper=dataclasses.replace(cfg.hyper, **ov_h))
    if args.class_num:
        cfg = dataclasses.replace(cfg, class_num=int(args.class_num))
    h = cfg.hyper
    from model_builder import build_baseline2
    wrapper = build_baseline2(weights=args.weights or h.pretrained_weights,
                              class_num=cfg.class_num)
    net = wrapper.model                    # 内部 DetectionModel（训练用，勿用 YOLO 包装器）

    # --- 数据：自动布局探测扫描；无 val 时从 train 抽出 10% 作 val（固定 seed 可复现）---
    root = Path(args.data_root) if args.data_root else MC.DATA_ROOT
    if not root.exists():
        raise SystemExit(f"[baseline2] 数据根不存在: {root}（先设 MULTIMODAL_DATA_ROOT 或 DATA_ROOT）")
    scanned = SD.scan_samples_auto(root)          # P0-5: 自动回退 V/T/D layout 扫描
    train_samples = scanned.get("train", []) or next(iter(scanned.values()))
    val_samples = scanned.get("val", []) or scanned.get("validation", [])
    if not train_samples:
        raise SystemExit(
            f"[baseline2] 扫描到 0 个训练样本（root={root}，splits={list(scanned)}）。"
            f"请检查数据根路径/布局（支持 Train/V/T/D/labels_multi 布局）。")
    if not val_samples and train_samples:
        rng = random.Random(h.seed)
        order = list(train_samples); rng.shuffle(order)
        k = max(1, int(len(order) * 0.1))
        val_samples = order[:k]; train_samples = order[k:]
        print(f"[baseline2] 无现成 val，从 train 抽出 {k} 组作验证集")
    print(f"[baseline2] train={len(train_samples)} val={len(val_samples)}")

    imgsz = (args.imgsz or h.imgsz, args.imgsz or h.imgsz)
    isz = int(args.imgsz or h.imgsz)

    aug_cfg = h.aug                      # 统一增强配置（models_config.AugmentParams）

    def build_batch(chunk, rng, augment=True):
        chw_list, boxes_list, stems = [], [], []
        for s in chunk:
            chw, boxes, stem = DS.build_consistent_aug_5ch(
                s, target_size=imgsz,
                aug=aug_cfg if augment else None,
                align=h.align,                       # P1-6: 对齐量按原图宽换算
                preprocess=h.preprocess,
                seed=rng.randrange(1 << 31))
            chw_list.append(chw); boxes_list.append(boxes); stems.append(stem)
        batch = TL.make_batch_dict(chw_list, boxes_list, stems, imgsz)
        return {"img": batch["img"]}, batch

    build_val_batch = lambda chunk, rng: build_batch(chunk, rng, augment=False)  # noqa: E731

    forward_fn = lambda net, inputs: net(inputs["img"])   # noqa: E731  内部网络直前向
    TL.train_custom(net, train_samples, val_samples, cfg,
                    build_batch, forward_fn, out_dir=args.out,
                    build_val_batch=build_val_batch, dataset_mode="5ch",
                    imgsz_override=isz)


def cmd_predict(args):
    import model_builder as MB
    from common import train_loop as TL
    from common import scan_data as SD
    from common import inference as INF
    cfg = MC.BASELINE2_5CH
    h = cfg.hyper
    isz = int(args.imgsz or h.imgsz)                     # P2-12: 预测也读 --imgsz

    # P0-2/P0-4: 结构始终从预训练权重构建（保证首层 6ch / 12 类一致），
    # 训练产物（自定义 {"model_state":...} 或 ultralytics 原生 ckpt）由统一协议回填
    wrapper = MB.build_baseline2(weights=h.pretrained_weights, class_num=cfg.class_num)
    net = wrapper.model                                  # 内部 DetectionModel（非 YOLO 包装器）
    if args.weights:
        TL.load_custom_checkpoint(args.weights, net, strict=False)

    root = Path(args.data_root) if args.data_root else MC.DATA_ROOT
    if not root.exists():
        raise SystemExit(f"[baseline2] 数据根不存在: {root}")
    scanned = SD.scan_samples_auto(root)                 # P0-5: 自动布局回退
    samples = (scanned.get("test") or scanned.get("val")
               or scanned.get("train") or next(iter(scanned.values()), []))
    if not samples:
        raise SystemExit(f"[baseline2 predict] 扫描到 0 个样本（root={root}），无法预测")

    # P2-12: 实际输出目录 = 打印路径（函数 out_dir 参数优先，不再各自为政）
    out = args.out or (MC.DATA_ROOT / f"pred_{cfg.key}")
    result = INF.predict_multimodal_custom(net, samples, cfg, out_dir=out,
                                           imgsz=(isz, isz))
    print(f"[baseline2 predict] 共 {len(samples)} 组 -> {result}")


def main():
    ap = argparse.ArgumentParser(prog="基线模型2")
    ap.add_argument("task", choices=["config", "train", "predict"],
                    help="config=打印配置; train=训练入口; predict=预测")
    ap.add_argument("--weights", default=None, help="模型权重")
    ap.add_argument("--out", default=None, help="输出目录(预测)")
    ap.add_argument("--data-yaml", default=None, help="data.yaml 路径(训练)")
    ap.add_argument("--data-root", default=None, help="数据根目录（默认 MC.DATA_ROOT / $MULTIMODAL_DATA_ROOT）")
    ap.add_argument("--imgsz", type=int, default=None, help="训练分辨率快捷覆盖（如 640/1024）")
    ap.add_argument("--epochs", type=int, default=None, help="轮数覆盖（如 1 快速冒烟）")
    ap.add_argument("--class-num", type=int, default=None, help="类别数覆盖（如 VDT=45）")
    ap.add_argument("--batch", type=int, default=None, help="batch size 覆盖")
    ap.add_argument("--workers", type=int, default=None, help="读图 worker 数覆盖（0=单进程，Windows 冒烟推荐）")
    args = ap.parse_args()
    {"config": cmd_config, "train": cmd_train, "predict": cmd_predict}[args.task](args)


if __name__ == "__main__":
    main()
