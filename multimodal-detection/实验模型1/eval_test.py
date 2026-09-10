# -*- coding: utf-8 -*-
"""
实验模型1 —— Test 集裁判入口（阶段0.2）。

同一条数据管线（同一 letterbox 几何、同一 GT、同一评估函数）评测两个模型，
只换模型 —— 保证"实验模型1 vs 基线模型1"是干净对比。

用法（EFYOLO 环境）：
    python 实验模型1/eval_test.py --root "<VDT Test>" --nc 45 --imgsz 640 --no-depth-align \
        --exp1  code/runs/experiment1_v3/weights/best.pt \
        --base1 code/runs/baseline1_3ch/train/weights/best.pt
两者都给则做对比；只给一个则只评一个。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_CODE_ROOT / "vendor"), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import models_config as MC                                  # noqa: E402
from common import evaluate as EV                           # noqa: E402
from common import scan_data as SD                           # noqa: E402
from common import train_loop as TL                          # noqa: E402


def build_inputs(root: Path, imgsz: int, align):
    """扫描 + 返回 (samples, build_batch)。无增强，仅同步 letterbox。"""
    import dataset_adapter as DA
    scanned = SD.scan_samples_auto(root)
    samples = (scanned.get("test") or scanned.get("val")
               or scanned.get("train") or next(iter(scanned.values()), []))
    if not samples:
        raise SystemExit(f"[eval_test] 扫描到 0 个样本: {root}")
    pp = MC.EXPERIMENT1.hyper.preprocess

    def build_batch(chunk, rng):
        rgb_l, ir_l, dep_l, boxes_l, stems = [], [], [], [], []
        for s in chunk:
            rgb, ir, dep, boxes, stem = DA.build_model_inputs(
                s, imgsz=(imgsz, imgsz), aug=None, align=align,
                preprocess=pp, seed=None, to_tensor=False)
            rgb_l.append(rgb); ir_l.append(ir); dep_l.append(dep)
            boxes_l.append(boxes); stems.append(stem)
        batch = TL.make_batch_dict(rgb_l, boxes_l, stems, (imgsz, imgsz))
        inputs = {"rgb": torch.from_numpy(np.stack(rgb_l)).float(),
                  "ir": torch.from_numpy(np.stack(ir_l)).float(),
                  "depth": torch.from_numpy(np.stack(dep_l)).float()}
        return inputs, batch

    return samples, build_batch


def load_exp1(weights: str, nc: int):
    import model_builder as MB
    h = MC.EXPERIMENT1.hyper
    m = MB.build_experiment1(weights=h.pretrained_weights, nc=nc,
                             mega=h.mega, aux_heads=h.aux_heads,
                             aux_depth_head=h.aux_depth_head,
                             dist_head_src=getattr(h, "aux_dist_head_src", "ir"),
                             per_modality_gate=h.per_modality_gate,
                             gamma_init=h.gamma_init,
                             dropout_p=h.modal_dropout_p)
    TL.load_custom_checkpoint(weights, m, strict=False)
    return m, (lambda mm, i: mm(i["rgb"], i["ir"], i["depth"]))


def load_base1(weights: str, nc: int):
    from ultralytics import YOLO
    m = YOLO(weights).model
    got = int(m.model[-1].nc)
    if got != nc:
        raise SystemExit(f"[eval_test] 基线1 的 nc={got} 与 --nc {nc} 不一致")
    return m, (lambda mm, i: mm(i["rgb"]))


def report(tag: str, res: dict):
    print(f"\n===== {tag} =====", flush=True)
    print(f"mAP50-95 {res['map50_95']:.4f}   mAP50 {res['map50']:.4f}   "
          f"P {res['precision']:.4f}   R {res['recall']:.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(prog="实验模型1/eval_test")
    ap.add_argument("--root", required=True, help="测试集根目录（V/T/D/labels_multi 布局）")
    ap.add_argument("--nc", type=int, default=MC.CLASS_NUM)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--no-depth-align", action="store_true",
                    help="关闭 depth 固定对齐（本地无偏移数据集用）")
    ap.add_argument("--exp1", default=None, help="实验模型1 checkpoint（best.pt）")
    ap.add_argument("--base1", default=None, help="基线模型1 checkpoint（best.pt）")
    args = ap.parse_args()

    align = MC.EXPERIMENT1.hyper.align
    if args.no_depth_align:
        align = dataclasses.replace(align, mode="none")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    samples, build_batch = build_inputs(Path(args.root), args.imgsz, align)
    print(f"[eval_test] {len(samples)} 组样本 | imgsz={args.imgsz} | conf={args.conf} "
          f"iou={args.iou} | align={align.mode} | device={dev}", flush=True)

    out = {}
    if args.exp1:
        m, fw = load_exp1(args.exp1, args.nc)
        out["exp1"] = report(f"实验模型1 (RGB+IR+Depth)  {Path(args.exp1).name}",
                             EV.evaluate_mAP(m, samples, args.imgsz, build_batch, fw,
                                             nc=args.nc, conf_thres=args.conf,
                                             iou_nms=args.iou, device=dev))
        del m
        torch.cuda.empty_cache()
    if args.base1:
        m, fw = load_base1(args.base1, args.nc)
        out["base1"] = report(f"基线模型1 (仅 RGB)  {Path(args.base1).name}",
                              EV.evaluate_mAP(m, samples, args.imgsz, build_batch, fw,
                                              nc=args.nc, conf_thres=args.conf,
                                              iou_nms=args.iou, device=dev))

    if len(out) == 2:
        a, b = out["exp1"], out["base1"]
        print("\n===== 对比 =====", flush=True)
        print(f"{'指标':<12}{'实验模型1':>10}{'基线模型1':>12}{'Δ':>10}")
        for k in ("map50_95", "map50", "precision", "recall"):
            print(f"{k:<12}{a[k]:>10.4f}{b[k]:>12.4f}{a[k] - b[k]:>+10.4f}")
    print("EVAL_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
