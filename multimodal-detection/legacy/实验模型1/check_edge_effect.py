# -*- coding: utf-8 -*-
"""边缘头影响力测量：把"预测边缘"置零，看网络输出改变多少。

逻辑：S4 与 S3 唯一的架构差异是 EdgeHead（预测边缘作为融合门控的额外输入通道）。
若把该通道强制置零（sigmoid 输入 -20 → 边缘图 ≡ 0），网络就退化成"没有边缘信息"的版本。
两次前向的输出差 = **边缘头对检测结果的实际影响力上限**（用最训练充分的权重测）。

同时报告每个注入点（层 4/6/10）融合残差相对主干的幅度 ‖ΔF‖/‖F‖，以及 tanh(γ)。

CPU 运行（不占 GPU，可与训练并行），只为拿几个样本的放大倍数：
    python 实验模型1/check_edge_effect.py [checkpoint] [样本数=4]
"""
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for p in (str(_CODE), str(_CODE / "vendor"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                        # noqa: E402
from common import scan_data as SD                # noqa: E402
import dataset_adapter as DA                      # noqa: E402
import model_builder as MB                        # noqa: E402

ROOT = _CODE.parent
NC, ISZ = 45, 640


def load_model(ckpt: Path):
    m, meta = MB.load_experiment1_checkpoint(ckpt)
    m.eval()
    return m, meta


def build_inputs(n: int):
    te = SD.scan_samples_auto(ROOT / "VDT-2048" / "VDT-2048 dataset" / "Test")
    te = te.get("test") or next(iter(te.values()))
    import dataclasses
    align = dataclasses.replace(MC.EXPERIMENT1.hyper.align, mode="none")
    rgb_l, ir_l, dep_l = [], [], []
    for s in te[:n]:
        rgb, ir, dep, _boxes, _stem = DA.build_model_inputs(
            s, imgsz=(ISZ, ISZ), aug=None, align=align,
            preprocess=MC.EXPERIMENT1.hyper.preprocess, seed=None, to_tensor=False)
        rgb_l.append(rgb); ir_l.append(ir); dep_l.append(dep)
    return (torch.from_numpy(np.stack(rgb_l)).float(),
            torch.from_numpy(np.stack(ir_l)).float(),
            torch.from_numpy(np.stack(dep_l)).float())


def _flatten_tensors(obj):
    """把任意嵌套的 dict/list/tuple 输出拍平成 tensor 列表（不同版本 Detect 返回结构不同）。"""
    out = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _flatten_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _flatten_tensors(v)
    return out


def capture(model, rgb, ir, dep):
    """返回 {层索引: 可微输出} 与 Detect 原始输出（拍平后的 tensor 列表）。"""
    feats, det = {}, {}
    handles = []

    def mk(idx):
        def hook(_m, _i, out):
            feats[idx] = out.detach().clone() if torch.is_tensor(out) else None
        return hook

    for idx in (4, 6, 10):
        handles.append(model.model[idx].register_forward_hook(mk(idx)))

    def det_hook(_m, _i, out):
        det["out"] = [t.detach().clone() for t in _flatten_tensors(out)]

    handles.append(model.model[23].register_forward_hook(det_hook))
    try:
        with torch.no_grad():
            y = model(rgb, ir, dep)
    finally:
        for h in handles:
            h.remove()
    return feats, det, _flatten_tensors(y)


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        _CODE / "runs" / "experiment1_s4_edge" / "weights" / "best.pt")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if not ckpt.exists():
        print(f"找不到 checkpoint: {ckpt}")
        return
    model, meta = load_model(ckpt)
    print(f"checkpoint: {ckpt.name} (epoch={meta.get('epoch')}) | 样本 {n} | CPU")
    if not getattr(model, "edge_enabled", False):
        print("该 checkpoint 未启用边缘头，无法做本测量")
        return

    rgb, ir, dep = build_inputs(n)
    print("前向 #1：正常（边缘头激活）...", flush=True)
    f_on, d_on, y_on = capture(model, rgb, ir, dep)

    # 前向 #2：把边缘头输出强制为"极负 logits" → sigmoid≈0 → 门控的边缘通道 ≡ 0
    head = model.aux_ir_edge_head
    orig_forward = head.forward

    def zero_edge(x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -20.0,
                          device=x.device, dtype=x.dtype)

    head.forward = zero_edge
    print("前向 #2：边缘通道置零...", flush=True)
    try:
        f_off, d_off, y_off = capture(model, rgb, ir, dep)
    finally:
        head.forward = orig_forward

    print("\n【各注入点特征变化】‖F_edge0 − F_edge1‖ / ‖F_edge1‖")
    for idx, name in ((4, "P3"), (6, "P4"), (10, "P5")):
        a, b = f_on[idx].float(), f_off[idx].float()
        rel = float((a - b).norm() / a.norm().clamp_min(1e-9))
        print(f"  {name}(层{idx}): {rel * 100:.4f}%")

    # 融合强度（tanh(γ)）与"融合残差到底有多大"的对照
    print("\n【融合强度】")
    for inj, name in zip((model.inject3, model.inject4, model.inject5), ("P3", "P4", "P5")):
        g = torch.tanh(inj.gamma.detach().float())
        print(f"  {name} tanh(γ)={float(g):+.4f}")

    print("\n【Detect 原始输出变化】")
    o1, o2 = d_on["out"], d_off["out"]
    tot1 = tot_d = 0.0
    for i, (t1, t2) in enumerate(zip(o1, o2)):
        if t1.shape != t2.shape:
            continue
        t1f, t2f = t1.float(), t2.float()
        n1 = float(t1f.norm())
        nd = float((t1f - t2f).norm())
        tot1 += n1 ** 2
        tot_d += nd ** 2
        print(f"  张量{i} shape={tuple(t1.shape)} 相对差={nd / max(n1, 1e-9) * 100:.4f}%")
    print(f"  合计 ‖ΔDetect‖ / ‖Detect‖ = {tot_d ** 0.5 / max(tot1 ** 0.5, 1e-9) * 100:.4f}%")
    print("\nEDGE_EFFECT_DONE")


if __name__ == "__main__":
    main()
