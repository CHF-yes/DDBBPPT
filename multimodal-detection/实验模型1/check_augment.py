# -*- coding: utf-8 -*-
"""
实验模型1 —— 增强正确性检查（阶段1.2′）。

三条硬检查（每次改增强后都该跑）：
  A. 深度只用最近邻：增强后画布上的非零深度值必须**全部来自原图取值**（无插值伪值）；
  B. 框与图同步：纯几何增强（翻转/缩放/平移）下，GT 框内有效深度的中位数**不变**
     —— 若框变换与图像变换不一致，框会盖到别的区域，中位数立刻漂移；
  C. 掩码自洽：depth 第 2 通道（有效性掩码）与第 1 通道严格对应（mask=1 ⟺ 距离>1mm）。

用法：python 实验模型1/check_augment.py --root "<VDT Train>" --no-depth-align
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_CODE_ROOT / "vendor"), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402

import models_config as MC                                  # noqa: E402
from common import scan_data as SD                           # noqa: E402


def box_dep_medians(dep_mm: np.ndarray, boxes_norm, min_valid: int = 5):
    """每个 GT 框内有效深度(>1mm)的中位数列表。"""
    out = []
    H, W = dep_mm.shape[:2]
    if boxes_norm is None:
        return out
    for b in np.asarray(boxes_norm).reshape(-1, 5):
        m = _med_one(dep_mm, b, W, H, min_valid)
        if m is not None:
            out.append(m)
    return out


def _med_one(dep_mm: np.ndarray, box, W: int, H: int, min_valid: int = 5):
    """单个归一化框内有效深度(>1mm)的中位数；样本不足返回 None。"""
    cx, cy, w, h = box[1] * W, box[2] * H, box[3] * W, box[4] * H
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    if x1 <= x0 or y1 <= y0:
        return None
    v = dep_mm[y0:y1, x0:x1]
    v = v[v > 1]
    return float(np.median(v)) if v.size >= min_valid else None


def check_nn_synthetic():
    """A. 合成深度图（取值稀疏可辨）验证：几何增强只做最近邻，绝不插值出伪值。"""
    from common import multimodal_augment as MA
    H, W = 64, 80
    yy, xx = np.mgrid[0:H, 0:W]
    dep = np.where((yy // 4 + xx // 6) % 2 == 0, 5000, 1000).astype(np.uint16)  # 仅两个取值
    rgb = np.zeros((H, W, 3), np.uint8)
    ir = np.zeros((H, W), np.uint8)
    boxes = np.array([[0.0, 0.5, 0.5, 0.4, 0.4]], np.float32)
    src = set(np.unique(dep).tolist())
    ok = {}
    _, _, d1, _, _ = MA.random_affine_consistent(rgb, ir, dep, boxes, (96, 96), 0.5, 0.1,
                                                 random.Random(1))
    ok["affine"] = set(np.unique(d1[d1 > 0]).tolist()) <= src
    _, _, d2, _ = MA.mosaic_consistent(
        [(rgb, ir, dep, boxes), (rgb, ir, dep + 1, boxes),
         (rgb, ir, dep + 2, boxes), (rgb, ir, dep + 3, boxes)], (96, 96), random.Random(2))
    src4 = set()
    for k in range(4):
        src4 |= set(np.unique(dep + k).tolist())
    ok["mosaic"] = set(np.unique(d2[d2 > 0]).tolist()) <= src4
    # A2. mosaic 四象限都必须有内容（回归检查：曾出现"只写最后一个象限"的合成 bug）
    hh, hw = 48, 48
    cov = [float((d2[r0:r0 + hh, c0:c0 + hw] > 0).mean())
           for r0 in (0, hh) for c0 in (0, hw)]
    ok["mosaic 四象限覆盖"] = all(c > 0.3 for c in cov) and (d2 > 0).mean() > 0.5
    print(f"   mosaic 象限覆盖: {[round(c, 3) for c in cov]} 总覆盖 {float((d2 > 0).mean()):.3f}",
          flush=True)
    # 负对照：双线性插值一定产生"新值"，证明该检查本身有效
    import cv2
    lin = cv2.warpAffine(dep, np.float32([[0.7, 0, 3], [0, 0.7, 5]]), (96, 96),
                         flags=cv2.INTER_LINEAR, borderValue=0)
    ok["负对照(双线性会引入新值)"] = not set(np.unique(lin[lin > 0]).tolist()) <= src
    return ok


def main():
    ap = argparse.ArgumentParser(prog="实验模型1/check_augment")
    ap.add_argument("--root", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--no-depth-align", action="store_true")
    args = ap.parse_args()

    import dataset_adapter as DA
    import numpy as np

    h = MC.EXPERIMENT1.hyper
    align = h.align
    if args.no_depth_align:
        align = dataclasses.replace(align, mode="none")
    aug_geo = dataclasses.replace(h.aug, depth_jitter_prob=0.0, rgb_drop_prob=0.0,
                                  mosaic_p=0.0, scale=0.5, translate=0.1)
    aug_mosaic = dataclasses.replace(aug_geo, mosaic_p=1.0, scale=0.0, translate=0.0)

    scanned = SD.scan_samples_auto(Path(args.root))
    samples = scanned.get("train") or next(iter(scanned.values()))
    print(f"[check] {len(samples)} 组样本；检查前 {args.trials} 次随机增强", flush=True)

    print("\nA. 最近邻（合成深度图单元测试）:", flush=True)
    nn_res = check_nn_synthetic()
    for k, v in nn_res.items():
        print(f"   {k}: {'通过' if v else '失败'}", flush=True)
    a_ok = all(nn_res.values())

    rng = random.Random(0)
    nB = nC = n_tot = 0
    worst_med_rel = 0.0
    for t in range(args.trials):
        s = samples[t % len(samples)]
        rgb, ir, dep_mm, boxes = DA._read_source(s, align)
        src_meds = box_dep_medians(dep_mm, boxes)
        use_mosaic = t % 3 == 2
        aug = aug_mosaic if use_mosaic else aug_geo
        pool = samples if use_mosaic else None
        r_o, i_o, d_o, b_o, _ = DA.build_model_inputs(
            s, imgsz=(args.imgsz, args.imgsz), aug=aug, align=align,
            preprocess=h.preprocess, seed=1000 + t, to_tensor=False,
            mosaic_pool=pool)

        # B. 框内深度中位数不变（mosaic 不检查：拼贴会改变框内内容）
        ok_b = True
        if not use_mosaic and src_meds:
            dep_out_mm = d_o[0] * h.preprocess.depth_scale_mm
            dep_out_mm = np.where(d_o[1] > 0, dep_out_mm, 0.0)
            out_meds = box_dep_medians(dep_out_mm, b_o)
            if len(out_meds) >= 1:
                m = min(len(src_meds), len(out_meds))
                rel = [abs(out_meds[k] - src_meds[k]) / max(1.0, src_meds[k]) for k in range(m)]
                worst_med_rel = max(worst_med_rel, max(rel))
                ok_b = max(rel) < 0.10
        nB += int(ok_b)

        # C. 掩码自洽：mask=1 ⟺ 距离 > 1mm
        dep_mm_out = d_o[0] * h.preprocess.depth_scale_mm
        ok_c = bool(np.array_equal(d_o[1] > 0, dep_mm_out > 1.0))
        nC += int(ok_c)

        # 框合法性
        box_ok = True
        if b_o is not None and len(b_o):
            box_ok = bool(((b_o[:, 1:] >= 0) & (b_o[:, 1:] <= 1)).all() and
                          (b_o[:, 3] > 0).all() and (b_o[:, 4] > 0).all())
        n_tot += 1
        print(f"  trial {t:>2} {'mosaic' if use_mosaic else 'affine'}: "
              f"B(框内深度中位数不变)={ok_b} C(掩码自洽)={ok_c} "
              f"框合法={box_ok} | 输出框数={0 if b_o is None else len(b_o)}", flush=True)

    # D. 尺度方差：训练增强后的框像素宽度必须有明显方差（回归检查：曾出现"永远 0.5 倍"）
    widths = []
    for t in range(args.trials * 3):
        s = samples[t % len(samples)]
        r_o, i_o, d_o, b_o, _ = DA.build_model_inputs(
            s, imgsz=(args.imgsz, args.imgsz), aug=h.aug, align=align,
            preprocess=h.preprocess, seed=5000 + t, to_tensor=False,
            mosaic_pool=samples)
        if b_o is not None and len(b_o):
            widths.extend((b_o[:, 3] * args.imgsz).tolist())
    w = np.asarray(widths, dtype=float)
    cv = float(w.std() / max(1e-6, w.mean()))
    print(f"\nD. 增强后框像素宽度: n={len(w)} 均值 {w.mean():.1f} 标准差 {w.std():.1f} "
          f"变异系数 {cv:.3f}（阈值 >0.25）", flush=True)
    d_ok = cv > 0.25
    print(f"   尺度方差检查: {'通过' if d_ok else '失败'}", flush=True)

    # E. Mosaic 跨模态对齐（回归检查：曾出现"辅助源 RGB/IR 半分辨率、Depth 全分辨率"
    #    共用一个仿射矩阵 → Depth 被放大 2 倍、与 RGB/IR 错位，实测框内深度中位数偏差 70%）
    print("\nE. Mosaic 三模态对齐:", flush=True)
    from common import multimodal_augment as MA
    bad_shape = bad_align = n_box = 0
    for t in range(4):
        base = samples[t * 4:t * 4 + 4]
        if len(base) < 4:
            break
        srcs = [DA._read_source(s, align) for s in base]
        for si, (r, ir, d, b) in enumerate(srcs):
            if not (r.shape[:2] == ir.shape[:2] == d.shape[:2]):
                bad_shape += 1
                print(f"   ❌ 源{si} 三模态分辨率不一致: rgb{r.shape[:2]} "
                      f"ir{ir.shape[:2]} depth{d.shape[:2]}", flush=True)
        r_o, i_o, d_o, b_o = MA.mosaic_consistent(srcs, (args.imgsz, args.imgsz),
                                                  random.Random(100 + t))
        src_meds = []
        for (r, ir, d, b) in srcs:
            for bb in (np.asarray(b).reshape(-1, 5) if b is not None else []):
                m0 = _med_one(d, bb, d.shape[1], d.shape[0])
                if m0:
                    src_meds.append(m0)
        for bb in (b_o if b_o is not None else []):
            n_box += 1
            mo = _med_one(d_o, bb, args.imgsz, args.imgsz)
            if mo is None or not src_meds:
                continue
            near = min(src_meds, key=lambda x: abs(x - mo))
            if abs(mo - near) / max(1.0, near) >= 0.10:
                bad_align += 1
                print(f"   ❌ 框错位: 画布中位 {mo:.0f}mm vs 最近源 {near:.0f}mm", flush=True)
    e_ok = (bad_shape == 0 and bad_align == 0)
    print(f"   三模态分辨率不一致的源 {bad_shape} 个；深度错位框 {bad_align}/{n_box} 个 -> "
          f"{'通过' if e_ok else '失败'}", flush=True)

    print(f"\n汇总: A(最近邻)={'通过' if a_ok else '失败'}，"
          f"B 通过 {nB}/{n_tot}，C 通过 {nC}/{n_tot}，D(尺度方差)={'通过' if d_ok else '失败'}，"
          f"E(Mosaic 跨模态对齐)={'通过' if e_ok else '失败'}；"
          f"框内深度中位数最大相对偏差 {worst_med_rel:.4f}", flush=True)
    print("AUGCHECK_OK" if (a_ok and d_ok and e_ok and nB == n_tot and nC == n_tot)
          else "AUGCHECK_FAIL", flush=True)


if __name__ == "__main__":
    main()
