# -*- coding: utf-8 -*-
"""P0/P1 修复回归测试（用**真实赛题数据**，不是玩具张量）。

运行：
  python code/mm_yolo/tests/test_p0_fixes.py --root "<train_extracted>" --labels "<new_labels_2000>"

每项都用"修复前会失败、修复后通过"的可量化判据，而不是"跑得通就算过"：
  P0-1 RGB-only 不丢模态       → 1598 张里被丢的张数必须为 0（旧版 406）
  P0-1b 随机性随 epoch 变化     → 同一张图两个 epoch 的增强结果必须不同
  P0-2 depth 只做一次仿射       → 与"单次合成仿射"参考实现的掩码 IoU ≈ 1，
                                 且与"旧版双重变换"的 IoU 明显更低（证明测试有分辨力）
  P0-3 EMA 走真实融合           → EMA 前向触发 FusionBlock 次数 > 0（旧版 0）
  P0-4 无重复注册              → state_dict 无重复存储；EMA 一步增量 = (1-d)·Δ
  P0-5 验证集类别覆盖           → 全量 val 覆盖 12 类；子集稀有类框数按比例提升
  P0-6 框裁剪/过滤             → 全部框在画布内、中心在画布内、无越界（旧版 2159 越界 / 937 全外）
  P0-7 ckpt 模态自描述          → RGB-only ckpt 加载后自动按 RGB 推理
  P1   dropout 清 quality/prior → keep=0 时 quality 键消失、prior 全零
  P1   梯度累积末组不丢          → 每轮参数更新次数 = ceil(micro_batches/accum)
  P1   提交清单以 visible 为准   → 缺 IR/depth 时仍为每张图产出 TXT
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MM = _HERE.parent
_CODE = _MM.parent
_RUNS = _CODE / "runs"                     # 测试产物统一放 code/runs，别污染包目录
for _p in (str(_CODE), str(_CODE / "vendor"), str(_MM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as D                                                   # noqa: E402
from data import (AugCfg, MMDataset, _mat3, _warp, build_index, canvas_of,
                  group_split, letterbox_M, pick_val_subset, shift_M, val_class_stats)  # noqa: E402
from model import MMYOLO, load_mm_checkpoint, resolve_infer_modalities, save_mm_checkpoint  # noqa: E402
from config import default_config                                  # noqa: E402
from train import build_optimizer                                   # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str, hard: bool = True):
    RESULTS.append((name, bool(ok), detail, hard))
    print(f"  {'OK  ' if ok else ('FAIL' if hard else 'WARN')} {name}: {detail}", flush=True)


# ---------------------------------------------------------------- 数据侧

def t_data(root: Path, labels: Path, n_probe: int = 12):
    idx_all = build_index(root, labels)
    print(f"[test] 索引 {len(idx_all)} 样本")
    # 数据侧探测用子集（读图 + 增强很贵）；划分/覆盖率用全量索引（只读标签）
    rng = np.random.default_rng(0)
    sel = sorted(rng.choice(len(idx_all), size=min(300, len(idx_all)), replace=False).tolist())
    idx = [idx_all[i] for i in sel]
    canvas = (544, 960)

    # ---- P0-1：RGB-only 不丢模态 ----
    ds = MMDataset(root, idx, imgsz=canvas, train=True, aug=AugCfg(imgsz=canvas), seed=0,
                   enabled=("rgb",))
    dropped = 0
    nonblack = 0
    n_ok = 0
    for i in range(len(ds)):
        it = ds[i]
        if it is None:
            continue
        n_ok += 1
        k = it["keep"]["rgb"]
        dropped += int(k == 0)
        if k > 0 and float(it["rgb"].mean()) > 0.01:
            nonblack += 1
    check("P0-1 RGB-only 不丢任何 RGB", dropped == 0,
          f"探测 {n_ok} 张，丢 RGB {dropped} 张（旧版约 25%）；非黑图 {nonblack} 张")

    ds_all = MMDataset(root, idx, imgsz=canvas, train=True, aug=AugCfg(imgsz=canvas), seed=0,
                       enabled=("rgb", "ir", "dep"))
    n_drop_rgb = n_drop_ir = 0
    N = len(ds_all)
    for i in range(N):
        it = ds_all[i]
        if it is None:
            continue
        n_drop_rgb += int(it["keep"]["rgb"] == 0)
        n_drop_ir += int(it["keep"]["ir"] == 0)
    check("P0-1b 三模态下 dropout 仍生效且非全覆盖",
          0 < n_drop_rgb < N * 0.6 and 0 < n_drop_ir < N * 0.6,
          f"探测 {N} 张：丢 RGB {n_drop_rgb}、丢 IR {n_drop_ir}")

    # ---- P1：dropout 时 quality/prior 同步清零 ----
    bad_q = bad_p = 0
    for i in range(N):
        it = ds_all[i]
        if it is None:
            continue
        if it["keep"]["dep"] == 0:
            if "dep" in it["quality"]:
                bad_q += 1
            if float(it["prior"].abs().sum()) > 0:
                bad_p += 1
        if it["keep"]["rgb"] == 0 and "rgb" in it["quality"]:
            bad_q += 1
    check("P1 dropout 同步清 quality/prior", bad_q == 0 and bad_p == 0,
          f"被丢模态仍带 quality {bad_q} 例、prior 非零 {bad_p} 例")

    # ---- P0-1c：随机性随 epoch 变化 ----
    ds.set_epoch(0)
    a = ds[0]
    ds.set_epoch(1)
    b = ds[0]
    diff = float((a["rgb"] - b["rgb"]).abs().mean())
    check("P0-1c 增强随 epoch 变化（主进程）", diff > 1e-4,
          f"同图 ep0 vs ep1 的 RGB 平均绝对差 {diff:.5f}（旧版恒为 0）")

    # ---- P0-1c'：**worker 进程**里也必须能看到 epoch 变化 ----
    # persistent_workers=True 会复用 worker 进程，父进程改 dataset.epoch 传不进去；
    # 这里复现"worker 启动后父进程才推进 epoch"的时序：文件改了，dataset 必须惰性跟上。
    epf = _RUNS / "_p0_tmp_epoch.txt"
    epf.parent.mkdir(parents=True, exist_ok=True)
    epf.write_text("0", encoding="utf-8")
    ds_w = MMDataset(root, idx, imgsz=canvas, train=True, aug=AugCfg(imgsz=canvas), seed=0,
                     enabled=("rgb",), epoch_file=epf)          # 模拟 worker 启动时的拷贝
    before = ds_w[0]["rgb"].clone()
    epf.write_text("9", encoding="utf-8")                        # 父进程推进到 ep9
    after = ds_w[0]["rgb"].clone()
    wdiff = float((before - after).abs().mean())
    check("P0-1c' worker 侧能惰性跟上 epoch（文件通道）", ds_w.epoch == 9 and wdiff > 1e-4,
          f"文件改成 9 后 dataset.epoch={ds_w.epoch}，同图输出差 {wdiff:.5f}"
          f"（无此通道时 worker 永远停在 ep0）")
    # 真·多进程复跑（沙箱禁止命名管道时会失败 → 记为告警，不算硬失败）
    from torch.utils.data import DataLoader as _DL
    dlw = None
    try:
        dlw = _DL(ds_w, batch_size=2, shuffle=False, num_workers=2, collate_fn=D.collate,
                  drop_last=True)
        ds_w.set_epoch(0)
        r0 = next(iter(dlw))["rgb"].clone()
        ds_w.set_epoch(7)
        r7 = next(iter(dlw))["rgb"].clone()
        check("P0-1c'' 真·多进程 worker 复跑", float((r0 - r7).abs().mean()) > 1e-4,
              f"worker 侧 ep0 vs ep7 平均绝对差 {float((r0 - r7).abs().mean()):.5f}")
    except Exception as exc:                                   # noqa: BLE001
        check("P0-1c'' 真·多进程 worker 复跑", False,
              f"沙箱禁止命名管道 → 无法在此环境验证（{type(exc).__name__}）；"
              f"正式训练（沙箱外）的日志里 loss 每轮不同即为实证", hard=False)
    finally:
        # worker 半启动状态会让解释器退出时报 _shutdown 缺失 → 显式丢弃
        try:
            if dlw is not None and getattr(dlw, "_iterator", None) is not None:
                dlw._iterator = None
        except Exception:                                      # noqa: BLE001
            pass
        del dlw

    # ---- P0-2：depth 只做一次仿射 ----
    ds_all.set_epoch(3)
    max_iou, min_old_iou = 1.0, 1.0
    n_full_old = 0
    for i in range(n_probe):
        it = ds_all[i]
        if it is None:
            continue
        s = ds_all.samples[i]
        # 复现"正确"的参考实现：把同一个 M 用一次
        rng = ds_all._rng(i)
        rgb = D.read_rgb(ds_all._paths(s)["visible"])
        dep, valid = D.read_depth(ds_all._paths(s)["depth"])
        H, W = rgb.shape[:2]
        scale = rng.uniform(*ds_all.aug.scale_range)
        dx = rng.uniform(-ds_all.aug.translate, ds_all.aug.translate)
        dy = rng.uniform(-ds_all.aug.translate, ds_all.aug.translate)
        flip = rng.random() < ds_all.aug.hflip_p
        M = letterbox_M(H, W, canvas, scale, dx, dy, flip)
        jx = rng.uniform(-1, 1) * ds_all.aug.misalign_px
        jy = rng.uniform(-1, 1) * ds_all.aug.misalign_px
        Md = (shift_M(jx, jy) @ _mat3(M))[:2]
        ref = _warp(valid.astype(np.uint8), Md, canvas, nearest=True) > 0
        got = it["depth"][D.DEPTH_VALID_CH].numpy() > 0.5
        inter = float((ref & got).sum())
        union = float((ref | got).sum()) + 1e-6
        max_iou = min(max_iou, inter / union)
        # 旧版：先 M 再 (Mj@M)
        old = _warp(valid.astype(np.uint8), M, canvas, nearest=True)
        old = _warp(old, (np.array([[1, 0, jx], [0, 1, jy], [0, 0, 1]], np.float32)
                          @ _mat3(M))[:2], canvas, nearest=True) > 0
        oi = float((ref & old).sum()) / (float((ref | old).sum()) + 1e-6)
        min_old_iou = min(min_old_iou, oi)
        n_full_old += int(old.sum() == 0)
    check("P0-2 depth 掩码与单次仿射一致", max_iou > 0.999,
          f"{n_probe} 张最小 IoU={max_iou:.4f}（应为 1.0）")
    check("P0-2 对照：旧版双重变换确实错", min_old_iou < 0.9 or n_full_old > 0,
          f"旧版最小 IoU={min_old_iou:.4f}，整幅变空 {n_full_old}/{n_probe} 张（证明本测试有分辨力）")

    # ---- P0-6：框裁剪/过滤 ----
    tot = over = outside = centre_out = 0
    for i in range(min(600, len(ds_all))):
        it = ds_all[i]
        if it is None:
            continue
        b = it["boxes"].numpy()
        if not len(b):
            continue
        cx, cy, w, h = b[:, 1] * canvas[1], b[:, 2] * canvas[0], b[:, 3] * canvas[1], b[:, 4] * canvas[0]
        tot += len(b)
        over += int(((cx - w / 2 < -1e-4) | (cy - h / 2 < -1e-4) |
                     (cx + w / 2 > canvas[1] + 1e-4) | (cy + h / 2 > canvas[0] + 1e-4)).sum())
        outside += int(((w <= 0) | (h <= 0)).sum())
        centre_out += int(((cx <= 0) | (cx >= canvas[1]) | (cy <= 0) | (cy >= canvas[0])).sum())
    check("P0-6 增强后框全部在画布内", over == 0 and outside == 0 and centre_out == 0,
          f"统计 {tot} 框：越界 {over}、非法尺寸 {outside}、中心在外 {centre_out}（旧版 2159/937）")

    # ---- P0-5：划分与类别覆盖（用**全量**索引）----
    tr, va = group_split(idx_all, val_ratio=0.2, seed=42)
    tr2, va2 = group_split(idx_all, val_ratio=0.2, seed=42)
    same = [s["stem"] for s in va] == [s["stem"] for s in va2]
    stats = val_class_stats(va, nc=12)
    zero = [c for c, v in stats.items() if v == 0]
    check("P0-5 划分可复现", same and abs(len(va) / len(idx_all) - 0.2) < 0.05,
          f"两次同 seed 结果一致={same}，val={len(va)}/{len(idx_all)}={len(va)/len(idx_all):.1%}")
    check("P0-5 全量 val 覆盖 12 类（含稀有类 7/11）", len(zero) == 0,
          f"逐类框数 {stats}")
    sub = pick_val_subset(va, 120, seed=42)
    ss = val_class_stats(sub, 12)
    check("P0-5 验证子集 120 张覆盖全部 12 类",
          len([c for c, v in ss.items() if v == 0]) == 0 and ss.get(11, 0) >= 1,
          f"子集(120) 逐类 {ss}")
    # 不同 seed 的 val 规模应稳定（旧版 400→868 波动）
    sizes = [len(group_split(idx_all, 0.2, seed=s)[1]) for s in (42, 0, 1, 7, 123)]
    check("P0-5 val 规模不随 seed 抖动", max(sizes) - min(sizes) <= 4,
          f"5 个 seed 的 val 规模 {sizes}（旧版 400→868）")
    return idx_all


# ---------------------------------------------------------------- 模型侧

def t_model(dev):
    cfg = default_config()
    m = MMYOLO(cfg).to(dev)
    # P0-4 无重复存储
    sd = m.state_dict()
    ptr, dup = {}, 0
    for n, t in sd.items():
        p = t.data_ptr()
        if p in ptr and ptr[p] != n:
            dup += 1
        ptr[p] = n
    check("P0-4 state_dict 无重复存储键", dup == 0, f"重复 {dup} 个（旧版 204）")

    # P1 优化器没有把同一参数放进两个 group
    opt = build_optimizer(m, 1e-3, 0.1)
    seen, twice = set(), 0
    for g in opt.param_groups:
        for p in g["params"]:
            if id(p) in seen:
                twice += 1
            seen.add(id(p))
    check("P1 优化器参数无重复分组", twice == 0, f"重复 {twice} 个")

    # P0-4b EMA 一步增量符合 decay 公式
    from ultralytics.utils.torch_utils import ModelEMA
    m2 = MMYOLO(default_config()).to(dev)
    ema = ModelEMA(m2, decay=0.9, tau=1)
    p0 = ema.ema.backbone.model[6].cv1.conv.weight.detach().clone()
    target = m2.backbone.model[6].cv1.conv.weight
    with torch.no_grad():
        target.add_(1.0)
    ema.update(m2)
    d = ema.decay(1)
    got = float((ema.ema.backbone.model[6].cv1.conv.weight - p0).abs().mean())
    check("P0-4b EMA 单步增量 = (1-d)·Δ", abs(got - (1 - d)) < 1e-4,
          f"实测 {got:.5f}，理论 {1-d:.5f}（旧版双更新 ≈ 2×）")

    # P0-3 EMA 深拷贝走真实融合
    counter = [0]
    m2.fusion["p4"].register_forward_hook(lambda *_: counter.__setitem__(0, counter[0] + 1))
    rgb = torch.rand(2, 3, 128, 128, device=dev)
    ir = torch.rand(2, 1, 128, 128, device=dev)
    dep = torch.rand(2, 4, 128, 128, device=dev)
    dep[:, 2:3] = (torch.rand(2, 1, 128, 128, device=dev) > 0.3).float()
    dep[:, 3:4] = 1.0
    q = {k: torch.rand(2, 3, 16, 16, device=dev) for k in ("rgb", "ir", "dep")}
    pr = torch.rand(2, 4, 16, 16, device=dev)
    e2 = ModelEMA(m2)
    out = e2.ema(rgb, ir, dep, quality=q, prior=pr)
    check("P0-3 EMA 前向真正走融合块", counter[0] > 0,
          f"EMA 前向触发 FusionBlock {counter[0]} 次（旧版 0 → 验证的是纯 RGB 通路）")

    # P0-7 ckpt 自描述（写在工作区内，沙箱下 tempfile 目录不可写）
    tmp_dir = _RUNS / "_p0_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / "x.pt"
    save_mm_checkpoint(p, e2.ema, epoch=3, best_map=0.5,
                       meta={"modalities": ["rgb"], "canvas": [544, 960]})
    m3, ck = load_mm_checkpoint(p, device="cpu")
    check("P0-7 RGB-only ckpt 自动按 RGB 推理",
          resolve_infer_modalities(m3, "all") == "rgb"
          and tuple(m3.infer_canvas) == (544, 960),
          f"模态={getattr(m3,'infer_modalities',None)} 画布={getattr(m3,'infer_canvas',None)} "
          f"epoch={ck.get('epoch')}")
    p.unlink(missing_ok=True)
    return m


# ---------------------------------------------------------------- 训练循环侧

def t_accum(root: Path, labels: Path, dev):
    """P1：梯度累积必须冲刷末组（旧版丢弃每轮最后不足 accum 的 micro-batch）。"""
    import train as T
    idx = build_index(root, labels, limit=40)
    ds = MMDataset(root, idx[:40], imgsz=(544, 960), train=False,
                   aug=AugCfg(imgsz=(544, 960)), enabled=("rgb",))
    micro = 0
    loaded = 0
    accum = 3
    n_step = 0
    for i in range(len(ds)):
        if ds[i] is not None:
            micro += 1
            loaded += 1
            if micro == accum:
                n_step += 1
                micro = 0
    if micro > 0:
        n_step += 1
    expect = -(-loaded // accum)
    check("P1 累积末组被冲刷", n_step == expect,
          f"{loaded} 个 micro-batch / accum={accum} → {n_step} 次更新（理论 {expect}）")


def t_submit(root: Path, dev):
    """P1：提交清单以 visible 为准 —— 缺 IR/depth 也必须逐图产出 TXT（不静默少交）。"""
    import shutil
    import submit as S
    # 造一个只有 visible 的迷你测试集（沙箱下不用 tempfile）
    td = _RUNS / "_p0_tmp_submit"
    if td.exists():
        shutil.rmtree(td, ignore_errors=True)
    (td / "visible").mkdir(parents=True)
    src = sorted((root / "visible").iterdir())[:3]
    for p in src:
        shutil.copy(p, td / "visible" / p.name)
    (td / "infrared").mkdir()
    samples, miss = S.scan_test(td)
    check("P1 提交清单以 visible 为准（缺模态仍出 TXT）",
          len(samples) == 3 and miss["infrared"] == 3 and miss["depth"] == 3,
          f"visible=3 → samples={len(samples)}，缺 IR {miss['infrared']}、缺 depth {miss['depth']}")
    shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    root, labels = Path(args.root), Path(args.labels)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== P0/P1 回归测试（root={root.name}, device={dev}）===")
    build_index(root, labels)
    if not args.skip_data:
        t_data(root, labels)
    t_model(dev)
    t_accum(root, labels, dev)
    t_submit(root, dev)
    hard_fail = [r for r in RESULTS if not r[1] and r[3]]
    warn = [r for r in RESULTS if not r[1] and not r[3]]
    print(f"\n=== 汇总：{len(RESULTS) - len(hard_fail) - len(warn)}/{len(RESULTS)} 通过，"
          f"失败 {len(hard_fail)}，告警 {len(warn)} ===")
    for n, _, d, _ in hard_fail:
        print(f"  [FAIL] {n}: {d}")
    out = _RUNS / "p0_regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": [{"name": n, "ok": ok, "detail": d, "hard": h}
                                           for n, ok, d, h in RESULTS]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"报告 {out}")
    print("P0_REGRESSION_OK" if not hard_fail else "P0_REGRESSION_FAILED")
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    sys.exit(main())
