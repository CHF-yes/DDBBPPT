# -*- coding: utf-8 -*-
"""提交生成 + 本地校验。

赛题输出格式（规则原文）：每行 `[class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]`，
归一化到原图；全部框按置信度排序、**每图 ≤100 框、无置信度阈值**；
**每张测试图必须有一个同名 TXT（检不到目标也要交空文件，不允许缺失）**。

用法：
  python code/mm_yolo/submit.py --ckpt <best.pt> --root "<test_extracted>" \
      --out code/runs/submit_phase1 --zip

审计修复记录：
* P0-7 推理模态/画布以 checkpoint 记录为准（RGB-only 权重不再被误按三模态推理 → 实测 0 框）；
* P1 测试图片清单以 **visible 目录**为准（旧版取三模态文件名**交集**：任何一张缺 IR/depth
  就会静默少交一个 TXT，直接违反"每张测试图必须有同名 TXT"的规则）；缺的模态补零并计数告警；
* P1 输出目录**先清空**再写（旧版旧 TXT 会残留并被打进 zip）；
* P1 校验器升级为**全量严格校验**：文件数 == 图片数、允许空文件、逐行字段/范围/类别/降序全查，
  并把问题文件清单写盘（旧版只报前 20 条、且不检查"是否每张图都交了"）。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import (AugCfg, EXTS, MODS, MMDataset, canvas_of, canvas_to_orig_norm,
                  collate, parse_imgsz)                              # noqa: E402
from eval import _decode                                          # noqa: E402
from model import load_mm_checkpoint, resolve_infer_canvas, resolve_infer_modalities   # noqa: E402


def scan_test(root: Path, limit: int = 0) -> tuple:
    """测试集没有标签。**清单以 visible 目录为准**（每张图都必须交 TXT）。

    返回 (samples, 统计)。缺 IR/depth 的样本会保留（推理时该路给零 + keep=0），
    但会在统计里明确计数，绝不静默跳过。
    """
    root = Path(root)
    vis_dir = root / "visible"
    if not vis_dir.is_dir():
        raise SystemExit(f"缺少模态目录: {vis_dir}")
    vis = {p.stem: p.name for p in vis_dir.iterdir() if p.suffix.lower() in EXTS}
    other: Dict[str, Dict[str, str]] = {}
    for m in ("infrared", "depth"):
        d = root / m
        other[m] = ({p.stem: p.name for p in d.iterdir() if p.suffix.lower() in EXTS}
                    if d.is_dir() else {})
    stems = sorted(vis)
    if limit:
        stems = stems[:limit]
    samples = []
    miss = {"infrared": 0, "depth": 0}
    for st in stems:
        files = {"visible": vis[st]}
        for m in ("infrared", "depth"):
            if st in other[m]:
                files[m] = other[m][st]
            else:
                miss[m] += 1
        samples.append({"stem": st, "files": files, "boxes": None})
    return samples, miss


def write_txt(path: Path, dets: np.ndarray, M: np.ndarray, orig_hw, imgsz: int = 0,
              max_det: int = 100) -> int:
    """dets: (n,6) [x1,y1,x2,y2,conf,cls]（**画布像素**）→ 写**原图归一化**行。

    ⚠️ 必须用 letterbox 仿射的逆变换回到原图再归一化；只除以画布边长会整体错位
    （16:9 图放进其他比例画布时有黑边与缩放）。
    """
    rows: List[str] = []
    if dets is not None and len(dets):
        order = np.argsort(-dets[:, 4])[:max_det]                 # 按置信度排序取前 100
        d = dets[order]
        norm = canvas_to_orig_norm(d[:, :4], M, orig_hw, imgsz)   # (n,4) cx,cy,w,h 原图归一化
        conf = np.clip(d[:, 4], 0, 1)
        cls = d[:, 5].astype(int)
        ok = (norm[:, 2] > 0) & (norm[:, 3] > 0)
        rows = [f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f} {p:.6f}"
                for c, (a, b, w, h), p in zip(cls[ok], norm[ok], conf[ok])]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def validate(out_dir: Path, expect_stems: List[str], nc: int = 12, max_det: int = 100,
             expect_box: bool = False) -> dict:
    """严格校验：文件齐全 + 逐行格式/范围/类别/降序 + 每图框数。

    `expect_box=True` 时额外把"一张图 0 框"记为问题（正常测试应几乎不会全空）。
    """
    issues: List[tuple] = []
    got = {p.stem: p for p in sorted(out_dir.glob("*.txt"))}
    missing = [s for s in expect_stems if s not in got]
    extra = [s for s in got if s not in set(expect_stems)]
    n_boxes = over = empty = 0
    for st in expect_stems:
        p = got.get(st)
        if p is None:
            continue
        raw = p.read_text(encoding="utf-8")
        rows = [l.split() for l in raw.strip().splitlines() if l.strip()]
        n_boxes += len(rows)
        if len(rows) == 0:
            empty += 1
            if expect_box:
                issues.append((p.name, "空文件（该图应有目标）"))
            continue
        if len(rows) > max_det:
            over += 1
            issues.append((p.name, f"框数 {len(rows)} > {max_det}"))
        prev = 1.1
        for i, r in enumerate(rows):
            if len(r) != 6:
                issues.append((p.name, f"第 {i+1} 行字段数 {len(r)}"))
                break
            try:
                c = int(r[0])
                cx, cy, w, h, cf = (float(x) for x in r[1:])
            except ValueError:
                issues.append((p.name, f"第 {i+1} 行无法解析：{r}"))
                break
            if not (0 <= c < nc):
                issues.append((p.name, f"类别越界 {c}"))
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h, cf)):
                issues.append((p.name, f"数值越界 {r}"))
            if w <= 0 or h <= 0:
                issues.append((p.name, f"非正宽高 {r}"))
            if cf > prev + 1e-9:
                issues.append((p.name, "置信度未降序"))
                break
            prev = cf
    return {"files": len(got), "expect": len(expect_stems), "missing": missing[:20],
            "n_missing": len(missing), "extra": extra[:20], "n_extra": len(extra),
            "boxes": n_boxes, "over_max_det": over, "empty_files": empty,
            "issues": issues[:50], "n_issues": len(issues)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-weights", default="",
                    help="服务器上本地预训练权重路径；用于重建 checkpoint 结构")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    ap.add_argument("--root", required=True, help="测试集目录（含 visible/infrared/depth）")
    ap.add_argument("--imgsz", default="", help="空=用 checkpoint 记录的训练画布；支持 'HxW'")
    ap.add_argument("--modalities", default="",
                    choices=["", "all", "rgb", "rgb_ir", "rgb_dep", "ir", "dep"],
                    help="空=按 checkpoint 记录的模态")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--tiled", action="store_true",
                    help="同一模型做全图+三模态同步切片推理")
    ap.add_argument("--tile-fraction", type=float, default=0.6,
                    help="切片宽高占原图宽高的比例；默认 0.6（通常 2x2）")
    ap.add_argument("--tile-overlap", type=float, default=0.2,
                    help="切片之间按切片尺寸计算的重叠比例")
    ap.add_argument("--tile-merge-iou", type=float, default=0.6,
                    help="全图与切片结果的逐类去重 IoU")
    ap.add_argument("--tile-batch", type=int, default=2)
    ap.add_argument("--mask", default="", help="屏蔽模态，如 'ir' 或 'dep'（诊断用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clean", action="store_true", default=True,
                    help="写之前清空输出目录（默认开，避免旧 TXT 混入 zip）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    if args.tiled:
        from tiled_inference import tile_windows
        tile_windows(100, 100, args.tile_fraction, args.tile_overlap)
        if not (0 < args.tile_merge_iou < 1) or args.tile_batch < 1:
            ap.error("tile-merge-iou 必须在 (0,1)，tile-batch 必须 >= 1")

    device_arg = ("cuda:0" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else ("cuda:0" if args.device == "cuda" else args.device)
    dev = torch.device(device_arg)
    overrides = {"weights": args.base_weights} if args.base_weights else {}
    model, _ck = load_mm_checkpoint(args.ckpt, device=dev, **overrides)
    model.eval()
    modalities = resolve_infer_modalities(model, args.modalities or None)
    imgsz = parse_imgsz(resolve_infer_canvas(model, parse_imgsz(args.imgsz) if args.imgsz
                                             else None))
    if not args.imgsz:
        print(f"[submit] 使用 checkpoint 记录的训练画布 {canvas_of(imgsz)}")
    root = Path(args.root)
    samples, miss = scan_test(root, args.limit)
    if not samples:
        raise SystemExit(f"{root/'visible'} 下没有图片")
    print(f"[submit] 测试图 {len(samples)} 张；缺 IR {miss['infrared']} 张、缺 depth {miss['depth']} 张"
          + ("（缺的模态按整路缺失处理：置零 + 融合门控屏蔽）" if any(miss.values()) else ""))
    ds = MMDataset(root, samples, imgsz=imgsz, train=False,
                   aug=AugCfg(imgsz=imgsz,
                              depth_resampling=getattr(model.cfg, "depth_resampling", "legacy_bilinear_v1"),
                              ir_read_mode=getattr(model.cfg, "ir_read_mode", "legacy_first_channel"),
                              legacy_lowlight=int(model.cfg.encoder.depth_input_channels) == 2),
                   enabled={"rgb": ("rgb",), "ir": ("ir",), "dep": ("dep",),
                            "rgb_ir": ("rgb", "ir"),
                            "rgb_dep": ("rgb", "dep"),
                            "all": ("rgb", "ir", "dep")}[modalities])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        old = list(out_dir.glob("*.txt"))
        for p in old:
            p.unlink()
        if old:
            print(f"[submit] 已清空输出目录里 {len(old)} 个旧 TXT")
    off = [args.mask] if args.mask else None
    total = wrote = skipped = 0
    for i in range(0, len(samples), args.batch):
        chunk = [ds[j] for j in range(i, min(i + args.batch, len(samples)))]
        if args.tiled and any(item is None for item in chunk):
            raise RuntimeError("切片推理遇到无法读取的测试图；拒绝生成不完整结果")
        batch = collate(chunk)
        if batch is None:
            continue
        dec = _decode(model, batch, dev, args.conf, args.iou, args.max_det, modalities, off, imgsz)
        if args.tiled:
            from tiled_inference import decode_full_and_tiles
            dec = decode_full_and_tiles(model, ds, samples[i:i + len(chunk)], batch, dec,
                                        dev, args.conf, args.iou, args.max_det, modalities,
                                        off, imgsz, args.tile_fraction, args.tile_overlap,
                                        args.tile_merge_iou, args.tile_batch)
        for k, (det, s) in enumerate(zip(dec, chunk)):
            total += write_txt(out_dir / f"{s['stem']}.txt", det, batch["M"][k].numpy(),
                               batch["orig_hw"][k].numpy(), imgsz, args.max_det)
            wrote += 1
        skipped += len(chunk) - len(dec)
        if (i // args.batch) % 20 == 0:
            print(f"[submit] {i + len(chunk)}/{len(samples)} 图，累计 {total} 框", flush=True)
    # 兜底：任何没写出的图**必须**补一个空 TXT（规则不允许缺失）
    for s in samples:
        p = out_dir / f"{s['stem']}.txt"
        if not p.exists():
            p.write_text("", encoding="utf-8")
    print(f"[submit] 写出 {wrote}/{len(samples)} 个 txt，共 {total} 框"
          + (f"（{skipped} 张读图失败，已补空文件）" if skipped else "") + f" → {out_dir}")
    rep = validate(out_dir, [s["stem"] for s in samples], nc=model.nc, max_det=args.max_det)
    print(f"[submit] 校验：文件 {rep['files']}/{rep['expect']}，框 {rep['boxes']}，"
          f"缺失 {rep['n_missing']}，多余 {rep['n_extra']}，超限 {rep['over_max_det']}，"
          f"空文件 {rep['empty_files']}，问题 {rep['n_issues']}")
    if rep["n_missing"]:
        print(f"   [ERROR] 缺少 TXT：{rep['missing']}")
    for name, msg in rep["issues"][:10]:
        print(f"   - {name}: {msg}")
    rep_path = out_dir.parent / f"{out_dir.name}_validate.json"
    import json
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[submit] 校验详情 {rep_path}")
    if rep["n_missing"] or rep["n_issues"]:
        raise SystemExit("[submit] 校验未通过：先修问题再打包（规则：非法提交无效）")
    if args.zip:
        zpath = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out_dir.glob("*.txt")):
                z.write(p, p.name)
        with zipfile.ZipFile(zpath) as z:
            n = len([x for x in z.namelist() if x.endswith(".txt")])
        print(f"[submit] 已打包 {zpath}（{n} 个 txt）")


if __name__ == "__main__":
    main()
