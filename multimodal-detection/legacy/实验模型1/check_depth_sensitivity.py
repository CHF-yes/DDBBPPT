# -*- coding: utf-8 -*-
"""对齐有效性实验：深度错位对检测 mAP 的敏感度（同一 checkpoint，只改推理时的 depth 平移）。

目的：量化"对齐没做好会损失多少 mAP"，从而判断对齐这条线值不值得投入。
  A) depth 原样（= 已对齐）
  B/C/D) depth 人为平移 +10 / +20 / +40 px（640 画布尺度），模拟未校正的传感器偏移

实测结论（S3 best，Test 前 300 组，赛题口径）：
    +0px 0.6272 | +10px 0.6396 | +20px 0.6379 | +40px 0.6122
→ 20px 错位只掉 <0.013、甚至在噪声内升高：**模型对 depth 几乎不敏感**。
  对齐属于"输入管线正确性"修复（赛题数据实测有 −21px@1920 固定偏移，必须修），
  但它本身不是涨点手段。

用法：python 实验模型1/check_depth_sensitivity.py [抽样数，默认 300]
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\35482\Desktop\人工智能精英\2")
CODE = ROOT / "code"
for p in (str(CODE), str(CODE / "vendor"), str(CODE / "实验模型1")):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                       # noqa: E402
from common import scan_data as SD               # noqa: E402
from common import train_loop as TL              # noqa: E402
from common import evaluate as EV                # noqa: E402
import dataset_adapter as DA                     # noqa: E402
import model_builder as MB                       # noqa: E402

ISZ, NC = 640, 45
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
te = SD.scan_samples_auto(ROOT / "VDT-2048" / "VDT-2048 dataset" / "Test")
te = te.get("test") or next(iter(te.values()))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300      # 抽样评估（默认 300 组，够看趋势）
te = te[:N]
print(f"[exp] Test 抽样 {len(te)} 组 | conf=0.001 max_det=100 | {DEV}", flush=True)

ALIGN_NONE = dataclasses.replace(MC.EXPERIMENT1.hyper.align, mode="none")
PP = MC.EXPERIMENT1.hyper.preprocess
DX_LIST = (0, 10, 20, 40)


def build_batch(chunk, rng):
    rgb_l, ir_l, dep_l, boxes_l, stems = [], [], [], [], []
    for s in chunk:
        rgb, ir, dep, boxes, stem = DA.build_model_inputs(
            s, imgsz=(ISZ, ISZ), aug=None, align=ALIGN_NONE,
            preprocess=PP, seed=None, to_tensor=False)
        rgb_l.append(rgb); ir_l.append(ir); dep_l.append(dep)
        boxes_l.append(boxes); stems.append(stem)
    batch = TL.make_batch_dict(rgb_l, boxes_l, stems, (ISZ, ISZ))
    inputs = {"rgb": torch.from_numpy(np.stack(rgb_l)).float(),
              "ir": torch.from_numpy(np.stack(ir_l)).float(),
              "depth": torch.from_numpy(np.stack(dep_l)).float()}
    return inputs, batch


def shift_depth(t: torch.Tensor, dx: int, dy: int = 0) -> torch.Tensor:
    """把 (B,2,H,W) 的 depth 平移 (dx,dy)，越界填 0（保持"0=无效"语义）。"""
    if dx == 0 and dy == 0:
        return t
    out = torch.zeros_like(t)
    H, W = t.shape[-2:]
    xs0, xs1 = max(0, dx), min(W, W + dx)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    out[..., ys0:ys1, xs0:xs1] = t[..., ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


m, meta = MB.load_experiment1_checkpoint(
    CODE / "runs" / "experiment1_s3_fixed" / "weights" / "best.pt", mega=False)
print(f"[exp] 载入 S3 best (ep{meta.get('epoch')})", flush=True)
base = None
for dx in DX_LIST:
    def fw(mm, i, dx=dx):
        return mm(i["rgb"], i["ir"], shift_depth(i["depth"], dx))
    r = EV.evaluate_mAP(m, te, ISZ, build_batch, fw, nc=NC,
                        conf_thres=0.001, iou_nms=0.7, max_det=100, device=DEV)
    if base is None:
        base = r["map50_95"]
    print(f"  depth 平移 +{dx:>2}px: mAP50-95 {r['map50_95']:.4f}  mAP50 {r['map50']:.4f}  "
          f"Δ vs 对齐 {r['map50_95'] - base:+.4f}", flush=True)
print("ALIGN_SENSITIVITY_DONE", flush=True)
