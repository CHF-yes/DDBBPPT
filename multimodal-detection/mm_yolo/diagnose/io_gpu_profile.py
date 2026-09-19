# -*- coding: utf-8 -*-
"""测训练的"取数据 vs 算 GPU"时间占比，用来判断单进程(workers=0)到底慢在哪。

用法：
  python code/mm_yolo/diagnose/io_gpu_profile.py --root <train_extracted> --labels <lab> [--n 30]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MM = _HERE.parent
_CODE = _MM.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_MM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import AugCfg, MMDataset, build_index, collate, group_split      # noqa: E402
from model import MMYOLO                                                   # noqa: E402
from config import default_config                                          # noqa: E402

from ultralytics.utils.loss import v8DetectionLoss                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--modalities", default="all")
    ap.add_argument("--imgsz", default="544x960")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=30, help="测多少个 micro-batch")
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgsz = (544, 960) if args.imgsz == "544x960" else int(args.imgsz)
    canvas = (int(imgsz[0]), int(imgsz[1])) if isinstance(imgsz, tuple) else (imgsz, imgsz)
    enabled = ["rgb"] if args.modalities == "rgb" else ["rgb", "ir", "dep"]

    idx = build_index(Path(args.root), Path(args.labels))
    tr, _ = group_split(idx, 0.2, seed=42)
    ds = MMDataset(Path(args.root), tr[: args.batch * args.n], imgsz=imgsz, train=True,
                   aug=AugCfg(imgsz=imgsz), seed=0, enabled=enabled)

    cfg = default_config()
    cfg.fusion.tier = "L2"
    model = MMYOLO(cfg).to(dev)
    if args.modalities == "rgb":
        model.modality_off = {"ir", "dep"}
    crit = v8DetectionLoss(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    def to_batch(chunk):
        b = collate(chunk)
        rgb = b["rgb"].to(dev)
        ir = None if args.modalities == "rgb" else b["ir"].to(dev)
        dep = None if args.modalities == "rgb" else b["depth"].to(dev)
        qual = {k: v.to(dev) for k, v in b["quality"].items()}
        prior = None if args.modalities == "rgb" else b["prior"].to(dev)
        bi, cl, bx = [], [], []
        for i, bb in enumerate(b["boxes"]):
            if len(bb):
                bi.append(torch.full((len(bb),), i, dtype=torch.float32))
                cl.append(bb[:, 0]); bx.append(bb[:, 1:5])
        tgt = {"batch_idx": (torch.cat(bi).to(dev) if bi else torch.zeros(0, device=dev)),
               "cls": (torch.cat(cl).to(dev).float() if cl else torch.zeros(0, device=dev)),
               "bboxes": (torch.cat(bx).to(dev).float() if bx else torch.zeros((0, 4), device=dev)),
               "imgsz": torch.tensor([canvas[0], canvas[1]], device=dev),
               "batch_size": len(b["boxes"])}
        return rgb, ir, dep, qual, prior, tgt

    def gpu_step(pre):
        rgb, ir, dep, qual, prior, tgt = pre
        with torch.autocast("cuda", enabled=(dev.type == "cuda")):
            preds = model(rgb, ir, dep, quality=qual, prior=prior)
            lv, _li = crit(preds, tgt)
            loss = lv.sum()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    # 预热（第一次含 cudnn 算法选择）
    warm = [ds[i] for i in range(args.batch)]
    warm = [w for w in warm if w is not None]
    if warm:
        rgb, ir, dep, qual, prior, tgt = to_batch(warm)
        with torch.autocast("cuda", enabled=(dev.type == "cuda")):
            lv, _ = crit(model(rgb, ir, dep, quality=qual, prior=prior), tgt)
        scaler.scale(lv.sum()).backward(); opt.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # --- 串行：取数 → GPU ---
    t_fetch = 0.0
    t_gpu = 0.0
    for k in range(args.n):
        t0 = time.time()
        chunk = [ds[k * args.batch + j] for j in range(args.batch)]
        chunk = [c for c in chunk if c is not None]
        pre = to_batch(chunk)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_fetch += time.time() - t0
        t1 = time.time()
        gpu_step(pre)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_gpu += time.time() - t1
    print(f"[profile] 串行 {args.n} step：取数据 {t_fetch:.1f}s（{t_fetch/args.n*1000:.0f} ms/step）"
          f" | GPU {t_gpu:.1f}s（{t_gpu/args.n*1000:.0f} ms/step）")
    print(f"[profile] 理论串行总时间 {t_fetch + t_gpu:.1f}s；"
          f"若完全重叠（后台预取）≈ max({t_fetch:.1f}, {t_gpu:.1f}) = {max(t_fetch, t_gpu):.1f}s"
          f"，加速 {((t_fetch + t_gpu) / max(t_fetch, t_gpu, 1e-6)):.2f}x")

    # --- 重叠：后台线程预取 ---
    import queue as _q
    import threading
    q: "_q.Queue" = _q.Queue(maxsize=2)

    def producer():
        for k in range(args.n):
            chunk = [ds[k * args.batch + j] for j in range(args.batch)]
            chunk = [c for c in chunk if c is not None]
            q.put(to_batch(chunk) if chunk else None)
        q.put(None)
    th = threading.Thread(target=producer, daemon=True)
    t0 = time.time()
    th.start()
    while True:
        pre = q.get()
        if pre is None:
            break
        gpu_step(pre)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_ov = time.time() - t0
    print(f"[profile] 后台预取总时间 {t_ov:.1f}s（{t_ov/args.n*1000:.0f} ms/step）"
          f" → 相比串行加速 {(t_fetch + t_gpu) / max(t_ov, 1e-6):.2f}x")
    print("[profile] 说明：workers=0 时这层重叠是唯一能抢回来的算力；"
          "多进程 worker 已在进程级并行，此项收益更小。")


if __name__ == "__main__":
    main()
