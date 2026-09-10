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


def cmd_config(args):
    print(MC.summarize())
    print(TR.train_config_prints(MC.EXPERIMENT1))
    h = MC.EXPERIMENT1.hyper
    a = h.aug
    print(f"  depth_align : ({h.depth_shift_x}, {h.depth_shift_y})")
    print(f"  depth_jitter: x={a.depth_jitter_x} y={a.depth_jitter_y} p={a.depth_jitter_prob}")


def _model_kwargs(h) -> dict:
    """实验模型1 的结构开关（train/selfcheck/predict 必须一致，否则 state_dict 对不上）。"""
    return dict(mega=h.mega, aux_heads=h.aux_heads,
                aux_depth_head=h.aux_depth_head,
                dist_head_src=getattr(h, "aux_dist_head_src", "ir"),
                per_modality_gate=h.per_modality_gate,
                gamma_init=h.gamma_init,
                dropout_p=h.modal_dropout_p)


def cmd_selfcheck(args):
    import model_builder as MB
    h = MC.EXPERIMENT1.hyper
    m = MB.build_experiment1(weights=h.pretrained_weights,
                             nc=MC.EXPERIMENT1.class_num, **_model_kwargs(h))
    print("selfcheck: model built OK, params(M) =",
          round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
    print("(dummy forward 请在 model_builder.py __main__ 中运行)")


def cmd_train(args):
    import random
    import dataclasses
    import numpy as np
    import torch
    from common import scan_data as SD
    from common import train_loop as TL

    cfg = MC.EXPERIMENT1
    # CLI 快捷覆盖（仅本次运行生效，不改 config）：--epochs/--class-num/--batch/--workers
    ov_h = {}
    if args.epochs:
        ov_h["epochs"] = int(args.epochs)
    if args.batch:
        ov_h["batch"] = int(args.batch)
    if args.workers is not None:
        ov_h["workers"] = int(args.workers)
    if args.no_depth_align:
        # VDT 本地数据已对齐：本次运行同时关闭固定配准与随机平移增强。
        # dataclasses.replace 只创建运行期副本，正式比赛默认 shift 配置保持不变。
        ov_h["align"] = dataclasses.replace(cfg.hyper.align, mode="none")
        ov_h["aug"] = dataclasses.replace(cfg.hyper.aug, depth_jitter_prob=0.0)
    if ov_h:
        cfg = dataclasses.replace(cfg, hyper=dataclasses.replace(cfg.hyper, **ov_h))
    if args.class_num:
        cfg = dataclasses.replace(cfg, class_num=int(args.class_num))
    if args.fusion_warmup is not None:
        ov_h2 = {"fusion_warmup_epochs": float(args.fusion_warmup)}
        cfg = dataclasses.replace(cfg, hyper=dataclasses.replace(cfg.hyper, **ov_h2))
    if args.mosaic_p is not None:
        cfg = dataclasses.replace(
            cfg, hyper=dataclasses.replace(
                cfg.hyper, aug=dataclasses.replace(cfg.hyper.aug,
                                                   mosaic_p=float(args.mosaic_p))))
    # 结构消融开关（必须与 checkpoint 匹配；默认取 config）
    ov_h3 = {}
    if args.no_per_modality_gate:
        ov_h3["per_modality_gate"] = False
    if args.no_dist_head:
        ov_h3["aux_depth_head"] = False
    if args.no_mega:
        ov_h3["mega"] = False
    if args.patience is not None:
        ov_h3["patience"] = int(args.patience)
    if ov_h3:
        cfg = dataclasses.replace(cfg, hyper=dataclasses.replace(cfg.hyper, **ov_h3))
    h = cfg.hyper
    print(f"[实验模型1] depth_align mode={h.align.mode} shift=({h.align.shift_x},{h.align.shift_y}) "
          f"depth_jitter_p={h.aug.depth_jitter_prob}")
    print(f"[实验模型1] 增强: flip={h.aug.flip_p} scale=±{h.aug.scale} translate={h.aug.translate} "
          f"mosaic_p={h.aug.mosaic_p}(末 {h.aug.close_mosaic_epochs} 轮关) "
          f"rgb_drop={h.aug.rgb_drop_prob} modal_dropout={h.modal_dropout_p} "
          f"融合课程={h.fusion_warmup_epochs}轮 λ下限={h.aux_lambda_final}")
    import model_builder as MB
    import dataset_adapter as DA
    model = MB.build_experiment1(weights=args.weights or h.pretrained_weights,
                                 nc=cfg.class_num, **_model_kwargs(h))

    # --- 数据：自动布局探测扫描；无 val 时从 train 抽出 val_ratio 作 val ---
    root = Path(args.data_root) if args.data_root else MC.DATA_ROOT
    if not root.exists():
        raise SystemExit(f"[实验模型1] 数据根不存在: {root}（先设 MULTIMODAL_DATA_ROOT 或 DATA_ROOT）")
    scanned = SD.scan_samples_auto(root)          # P0-5: 自动回退 V/T/D layout 扫描
    train_samples = scanned.get("train", []) or next(iter(scanned.values()))
    val_samples = scanned.get("val", []) or scanned.get("validation", [])
    if not train_samples:
        raise SystemExit(
            f"[实验模型1] 扫描到 0 个训练样本（root={root}，splits={list(scanned)}）。"
            f"请检查数据根路径/布局（支持 Train/V/T/D/labels_multi 布局）。")
    val_ratio = float(args.val_ratio if args.val_ratio is not None else 0.2)
    if not val_samples and train_samples:
        rng = random.Random(h.seed)
        order = list(train_samples); rng.shuffle(order)
        k = max(1, int(len(order) * val_ratio))
        val_samples = order[:k]; train_samples = order[k:]
        print(f"[实验模型1] 无现成 val，从 train 抽出 {k} 组作验证集（ratio={val_ratio}）")
    print(f"[实验模型1] train={len(train_samples)} val={len(val_samples)}")

    imgsz = (args.imgsz or h.imgsz, args.imgsz or h.imgsz)
    isz = int(args.imgsz or h.imgsz)

    aug_cfg = h.aug                      # 统一增强配置（models_config.AugmentParams）

    def build_batch(chunk, rng, augment=True):
        # train_loop 每轮会写入 model._aug_override（close_mosaic 后的生效增强）；
        # 同步读图路径用它，多进程路径由 MultiSampleDataset(aug=...) 直接接收。
        aug_use = getattr(model, "_aug_override", None) or aug_cfg
        rgb_l, ir_l, dep_l, boxes_l, stems = [], [], [], [], []
        for s in chunk:
            rgb, ir, dep, boxes, stem = DA.build_model_inputs(
                s, imgsz=imgsz,
                aug=aug_use if augment else None,
                align=h.align,                       # P1-6: 对齐量按原图宽换算
                preprocess=h.preprocess,
                seed=rng.randrange(1 << 31), to_tensor=False,
                mosaic_pool=train_samples if augment else None)
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
    """实验模型1 三路推理（P0-3）：扫描测试样本 → rgb/ir/depth 三路前向 → NMS → 赛题 txt。"""
    import numpy as np
    import torch
    import model_builder as MB
    import dataset_adapter as DA
    from common import scan_data as SD
    from common import train_loop as TL
    from common.evaluate import decode_preds
    from common.inference import _pick_out_dir

    cfg = MC.EXPERIMENT1
    if args.no_depth_align:
        import dataclasses
        cfg = dataclasses.replace(
            cfg, hyper=dataclasses.replace(
                cfg.hyper, align=dataclasses.replace(cfg.hyper.align, mode="none")))
    if args.class_num:                                     # 与 train 一致：类别数可覆盖
        import dataclasses
        cfg = dataclasses.replace(cfg, class_num=int(args.class_num))
    h = cfg.hyper
    print(f"[实验模型1 predict] depth_align mode={h.align.mode} class_num={cfg.class_num}")
    isz = int(args.imgsz or h.imgsz)                       # P2-12: 预测也读 --imgsz

    # P0-4: 结构从预训练构建（保证辅助流/MEGA/Detect 结构与训练一致），再回填 checkpoint
    model = MB.build_experiment1(weights=h.pretrained_weights, nc=cfg.class_num,
                                 **_model_kwargs(h))
    if args.weights:
        TL.load_custom_checkpoint(args.weights, model, strict=False)
    model.eval()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    root = Path(args.data_root) if args.data_root else MC.DATA_ROOT
    if not root.exists():
        raise SystemExit(f"[实验模型1] 数据根不存在: {root}")
    scanned = SD.scan_samples_auto(root)                   # P0-5: 自动布局回退
    samples = (scanned.get("test") or scanned.get("val")
               or scanned.get("train") or next(iter(scanned.values()), []))
    if not samples:
        raise SystemExit(f"[实验模型1 predict] 扫描到 0 个样本（root={root}），无法预测")

    out_dir = Path(args.out) if args.out else _pick_out_dir(cfg.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with torch.no_grad():
        for s in samples:
            rgb, ir, dep, _boxes, stem = DA.build_model_inputs(
                s, imgsz=(isz, isz), aug=None,
                align=h.align,                             # P1-6: 对齐按原图宽换算
                preprocess=h.preprocess, to_tensor=False)
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0).to(dev)
            ir_t = torch.from_numpy(np.ascontiguousarray(ir)).float().unsqueeze(0).to(dev)
            dep_t = torch.from_numpy(np.ascontiguousarray(dep)).float().unsqueeze(0).to(dev)
            out = model(rgb_t, ir_t, dep_t)
            det = decode_preds(out, cfg.class_num,
                               conf_thres=0.25, iou_thres=0.7)[0]
            det = det.cpu().numpy() if not isinstance(det, np.ndarray) else det
            lines = []
            if len(det):
                for x1, y1, x2, y2, conf, cls in det[:100]:   # conf 降序已由 NMS 保证
                    cx = (x1 + x2) / 2 / isz
                    cy = (y1 + y2) / 2 / isz
                    w = (x2 - x1) / isz
                    hh = (y2 - y1) / isz
                    lines.append(
                        f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {hh:.6f} {float(conf):.6f}")
            (out_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n" if lines else "", encoding="utf-8")
            n_written += 1
    print(f"[实验模型1 predict] {n_written} 组样本 -> {out_dir}（含空 txt，≤100 框/图）")


def main():
    ap = argparse.ArgumentParser(prog="实验模型1")
    ap.add_argument("task", choices=["config", "selfcheck", "train", "predict"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-yaml", default=None)
    ap.add_argument("--data-root", default=None, help="数据根目录（默认 MC.DATA_ROOT / $MULTIMODAL_DATA_ROOT）")
    ap.add_argument("--imgsz", type=int, default=None, help="训练分辨率快捷覆盖（如 640/1024）")
    ap.add_argument("--epochs", type=int, default=None, help="轮数覆盖（如 1 快速冒烟）")
    ap.add_argument("--class-num", type=int, default=None, help="类别数覆盖（如 VDT=45）")
    ap.add_argument("--batch", type=int, default=None, help="batch size 覆盖")
    ap.add_argument("--workers", type=int, default=None, help="读图 worker 数覆盖（0=单进程，Windows 冒烟推荐）")
    ap.add_argument("--val-ratio", type=float, default=None,
                    help="无现成 val 时从 train 抽出的比例（默认 0.2）")
    ap.add_argument("--fusion-warmup", type=float, default=None,
                    help="融合课程：前 N 轮旁路注入（0=一开始就开融合）")
    ap.add_argument("--mosaic-p", type=float, default=None,
                    help="4 图拼接概率覆盖（0=关闭 mosaic）")
    ap.add_argument("--no-per-modality-gate", action="store_true",
                    help="关闭逐模态逐位置门控（退化为与 v3 等价的 concat 单投影）")
    ap.add_argument("--no-dist-head", action="store_true", help="关闭稀疏距离辅助头")
    ap.add_argument("--no-mega", action="store_true",
                    help="关闭 MEGA 边缘引导门控（隔离实验：其 Sober 会把 mosaic 拼接缝当边缘）")
    ap.add_argument("--patience", type=int, default=None,
                    help="早停耐心（建议 ≥ 0.3×epochs，否则会在余弦退火前被截断）")
    ap.add_argument("--no-depth-align", action="store_true",
                    help="仅本次运行关闭 Depth 固定对齐；训练时也关闭随机 Depth 平移增强")
    args = ap.parse_args()
    {"config": cmd_config, "selfcheck": cmd_selfcheck,
     "train": cmd_train, "predict": cmd_predict}[args.task](args)


if __name__ == "__main__":
    main()
