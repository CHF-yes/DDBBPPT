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


def load_exp1(weights: str, nc: int = None, struct: dict = None):
    """按 checkpoint **自描述的结构**重建实验模型1 并严格加载（missing 非空直接报错）。

    这样 mega / per_modality_gate / aux_depth_head / dist_head_src 等开关与训练时
    严格一致，避免"按默认配置重建 → 缺 MEGA 参数却静默忽略"的误读。
    旧格式 checkpoint（无 structure 字段）需用 --struct 显式给出开关。
    """
    import model_builder as MB
    ov = dict(struct or {})
    if nc is not None:
        ov["nc"] = int(nc)
    m, meta = MB.load_experiment1_checkpoint(weights, strict=True, **ov)
    return m, (lambda mm, i: mm(i["rgb"], i["ir"], i["depth"]))


def _has_structure(weights: str) -> bool:
    """checkpoint 是否自带 structure 字段（旧格式才需要 --struct 覆盖）。"""
    try:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception:
        return False
    ok = isinstance(ck.get("structure"), dict) and bool(ck["structure"])
    del ck
    return ok


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
    ap.add_argument("--nc", type=int, default=None,
                    help="类别数（默认从 checkpoint 自描述结构读取；基线1 用 --nc 指定）")
    ap.add_argument("--imgsz", type=int, default=640)
    # ★ 赛题口径：规则按"提交的全部框按置信度排序"算 AP、每图≤100 框、无置信度阈值
    ap.add_argument("--conf", type=float, default=0.001,
                    help="NMS 置信度（赛题口径 0.001；旧默认 0.25 会低估约 30%%）")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100,
                    help="每图最多保留框数（赛题规则 ≤100）")
    ap.add_argument("--no-depth-align", action="store_true",
                    help="关闭 depth 固定对齐（本地无偏移数据集用）")
    ap.add_argument("--exp1", action="append", default=None,
                    help="实验模型1 checkpoint（可重复给多个，如 S4 的 best/last + S3，"
                         "同一条数据管线顺序评测）")
    ap.add_argument("--base1", default=None, help="基线模型1 checkpoint（best.pt）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只评测前 N 组（仅供省电/调试时的快速读数，非随机抽样，正式报数不要用）")
    ap.add_argument("--struct", default=None,
                    help="旧格式 checkpoint 的结构覆盖，如 'mega=false,per_modality_gate=true'"
                         "（新格式 checkpoint 自带 structure，无需此项）")
    args = ap.parse_args()

    align = MC.EXPERIMENT1.hyper.align
    if args.no_depth_align:
        align = dataclasses.replace(align, mode="none")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    samples, build_batch = build_inputs(Path(args.root), args.imgsz, align)
    if args.limit:
        samples = samples[:args.limit]
        print(f"[eval_test] ★ --limit {args.limit}：只取前 {len(samples)} 组（快速读数用）",
              flush=True)
    print(f"[eval_test] {len(samples)} 组样本 | imgsz={args.imgsz} | conf={args.conf} "
          f"iou={args.iou} max_det={args.max_det}（赛题口径）| align={align.mode} | "
          f"device={dev}", flush=True)

    def _eval(m, fw, nc):
        return EV.evaluate_mAP(m, samples, args.imgsz, build_batch, fw,
                               nc=nc, conf_thres=args.conf, iou_nms=args.iou,
                               max_det=args.max_det, device=dev)

    def _parse_struct(spec):
        if not spec:
            return None
        out = {}
        for kv in spec.split(","):
            k, _, v = kv.partition("=")
            v = v.strip()
            out[k.strip()] = (v.lower() in ("1", "true", "yes") if v.lower() in
                              ("1", "true", "false", "yes", "no", "0") else v)
        return out

    struct = _parse_struct(args.struct)
    out = {}
    for w in (args.exp1 or []):
        path = Path(w)
        if not path.exists():
            raise SystemExit(f"[eval_test] checkpoint 不存在: {w}")
        ov = struct
        if struct is not None and _has_structure(w):
            print(f"[eval_test] {path.name} 自带 structure，忽略 --struct", flush=True)
            ov = None
        elif struct is None and not _has_structure(w):
            raise SystemExit(f"[eval_test] {path.name} 是旧格式 checkpoint（无 structure），"
                             f"请用 --struct 显式给出开关，如 --struct \"mega=false\"")
        m, fw = load_exp1(w, args.nc, ov)
        nc_e = int(m.detect.nc)
        # 唯一标签：run 目录/文件名（只取 path.name 会让多个 best.pt/last.pt 互相覆盖）
        label = f"{path.parent.parent.name}/{path.name}"
        out[label] = report(f"实验模型1 (RGB+IR+Depth)  {label}", _eval(m, fw, nc_e))
        del m
        torch.cuda.empty_cache()
    if args.base1:
        nc_b = int(args.nc if args.nc is not None else MC.CLASS_NUM)
        m, fw = load_base1(args.base1, nc_b)
        out[f"base1:{Path(args.base1).name}"] = report(
            f"基线模型1 (仅 RGB)  {Path(args.base1).name}", _eval(m, fw, nc_b))

    if len(out) >= 2:
        print("\n===== 对比（mAP50-95 / mAP50 / P / R）=====", flush=True)
        print(f"{'模型':<34}{'mAP50-95':>10}{'mAP50':>9}{'P':>9}{'R':>9}")
        for k, r in out.items():
            print(f"{k:<34}{r['map50_95']:>10.4f}{r['map50']:>9.4f}"
                  f"{r['precision']:>9.4f}{r['recall']:>9.4f}")
    print("EVAL_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
