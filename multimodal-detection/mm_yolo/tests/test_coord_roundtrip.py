# -*- coding: utf-8 -*-
"""坐标往返 / 提交格式一致性测试（用**真实图片与真实标签**）。

验证三件事：
1. `canvas_boxes_from_norm`（训练用）与 `canvas_to_orig_norm`（提交用）互为逆变换
   —— 报告里"坐标必须走同一套 letterbox 映射"的说法必须可验证，否则提交坐标会系统性偏移；
2. 非正方形画布（544×960）下也成立（旧版 eval/submit 只接受正方形整数尺寸）；
3. `write_txt` 的输出能被 `submit.validate` 全量通过（格式/范围/降序/每图≤100）。

用法：python code/mm_yolo/tests/test_coord_roundtrip.py --root <train_extracted> --labels <new_labels_2000>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MM = _HERE.parent
_CODE = _MM.parent
_RUNS = _CODE / "runs"
for _p in (str(_CODE), str(_CODE / "vendor"), str(_MM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import (build_index, canvas_boxes_from_norm, canvas_of, letterbox_M,
                  canvas_to_orig_norm, boxes_norm_to_canvas)      # noqa: E402
import submit as S                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--canvas", default="544x960")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    root, labels = Path(args.root), Path(args.labels)
    canvas = canvas_of(args.canvas)
    idx = [s for s in build_index(root, labels) if s["boxes"] is not None and len(s["boxes"])]
    print(f"=== 坐标往返测试（canvas={canvas}，{args.n} 张有标注图）===")
    rng = np.random.default_rng(0)
    worst = 0.0
    worst_clip = 0.0
    n_box = 0
    for s in idx[: args.n]:
        # 复现数据集内部的几何：训练态随机缩放/平移/翻转
        H, W = 1080, 1920
        scale = float(rng.uniform(0.75, 1.4))
        dx, dy = float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1))
        flip = bool(rng.random() < 0.5)
        M = letterbox_M(H, W, canvas, scale, dx, dy, flip)
        gt = np.asarray(s["boxes"], np.float32)
        # 原图归一化 → 画布归一化（含裁剪/过滤）
        cv = canvas_boxes_from_norm(gt, M, (H, W), canvas, min_size=0.0, require_center=False)
        if not len(cv):
            continue
        # 画布归一化 → 画布 xyxy 像素
        xyxy = boxes_norm_to_canvas(cv[:, 1:5], M, (H, W))
        # 画布像素 → 原图归一化（提交路径）
        back = canvas_to_orig_norm(xyxy, M, (H, W))
        # 与"直接由原图归一化经裁剪"的期望值比较
        exp = cv[:, 1:5].copy()
        worst = max(worst, float(np.abs(back - exp).max()))
        # 未裁剪版本用于测量"裁剪造成的偏差"（裁剪只应发生在出界框上）
        raw = boxes_norm_to_canvas(gt[:, 1:5], M, (H, W))
        clipped = np.clip(raw, [0, 0, 0, 0], [canvas[1], canvas[0], canvas[1], canvas[0]])
        moved = float(np.abs(clipped - raw).max()) if len(raw) else 0.0
        worst_clip = max(worst_clip, moved)
        n_box += len(cv)
    print(f"  往返最大误差 {worst:.2e}（归一化尺度，应 < 1e-5）")
    print(f"  参考：本次抽样中被裁剪框的最大像素位移 {worst_clip:.1f}px"
          f"（说明出界框确实被裁剪了；未出界框位移为 0）")
    ok1 = worst < 1e-5
    print(f"  {'OK  ' if ok1 else 'FAIL'} 坐标往返一致（训练映射 <-> 提交映射互逆）")

    # ---- 提交写盘 + 严格校验 ----
    out = _RUNS / "_coord_test"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.txt"):
        p.unlink()
    stems = []
    for i, s in enumerate(idx[: args.n]):
        H, W = 1080, 1920
        M = letterbox_M(H, W, canvas, 1.0, 0.0, 0.0, False)
        det = np.zeros((0, 6), np.float32)
        # 造 5 个假检测：以真实框为中心 + 一个越界框（测裁剪到原图）
        gt = np.asarray(s["boxes"], np.float32)[:5]
        if len(gt):
            cv = canvas_boxes_from_norm(gt, M, (H, W), canvas, min_size=0.0, require_center=False)
            xy = boxes_norm_to_canvas(cv[:, 1:5], M, (H, W))
            det = np.c_[xy, np.linspace(0.9, 0.5, len(xy)), cv[:, 0]].astype(np.float32)
            det = np.vstack([det, [[-50, -50, canvas[1] + 80, canvas[0] + 80, 0.3, 0.0]]])
        S.write_txt(out / f"{s['stem']}.txt", det, M, (H, W), canvas[0], max_det=100)
        stems.append(s["stem"])
    rep = S.validate(out, stems, nc=12, max_det=100)
    ok2 = rep["n_missing"] == 0 and rep["n_issues"] == 0 and rep["files"] == len(stems)
    print(f"  {'OK  ' if ok2 else 'FAIL'} 提交校验器全量通过：文件 {rep['files']}/{rep['expect']}，"
          f"框 {rep['boxes']}，缺失 {rep['n_missing']}，问题 {rep['n_issues']}")
    for n, m in rep["issues"][:5]:
        print(f"       - {n}: {m}")

    # ---- 越界框必须被反向裁剪到原图内（不产生 >1 的归一化坐标）----
    bad = 0
    for p in sorted(out.glob("*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = [float(x) for x in line.split()[1:]]
                if not all(0.0 <= x <= 1.0 for x in v):
                    bad += 1
    ok3 = bad == 0
    print(f"  {'OK  ' if ok3 else 'FAIL'} 越界检测框被裁剪回 [0,1]（越界行数 {bad}）")

    hard = ok1 and ok2 and ok3
    print("COORD_ROUNDTRIP_OK" if hard else "COORD_ROUNDTRIP_FAILED")
    return 0 if hard else 1


if __name__ == "__main__":
    sys.exit(main())
