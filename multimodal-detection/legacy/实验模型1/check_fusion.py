# -*- coding: utf-8 -*-
"""融合强度诊断：读 checkpoint 的 state_dict，打印 γ（融合强度）与门控权重规模。

用途：判断"辅助/融合支路到底有没有在起作用"。`tanh(γ)` 就是融合残差相对主干特征的
上限比例——它只有几个百分点时，任何对**门控输入特征**的改动（换边缘头、加辅助头）
都不可能显著改变检测输出。

只读 checkpoint（CPU，不建模型、不吃显存），因此可以在训练进行中安全运行。

用法：
    python 实验模型1/check_fusion.py                       # 默认扫 runs/ 下的 checkpoint
    python 实验模型1/check_fusion.py path1.pt path2.pt
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

RUNS = _HERE.parent / "runs"


def _default_ckpts():
    out = []
    for pat in ("experiment1*/weights/*.pt", "*_edge/weights/*.pt"):
        out += sorted(RUNS.glob(pat))
    return out


def report(path: Path) -> None:
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:                     # 训练可能正在写文件 → 读到半个文件
        print(f"[skip] {path.name}: 读取失败（{exc}）")
        return
    sd = ck.get("model_state") or ck.get("model") or ck.get("state_dict")
    if not isinstance(sd, dict):
        print(f"[skip] {path.name}: 无 state_dict")
        return

    gamma = sorted((k, v) for k, v in sd.items() if "gamma" in k)
    gates = {k: v for k, v in sd.items() if "gate" in k and getattr(v, "ndim", 0) >= 2}
    print(f"\n=== {path.parent.parent.name}/{path.name} "
          f"(epoch={ck.get('epoch')}, best_map={ck.get('best_map')}) ===")
    print(f"  张量总数 {len(sd)} | γ {len(gamma)} 个 | 门控卷积 {len(gates)} 个")
    for k, v in gamma:
        t = torch.tanh(v.detach().float()).flatten()
        print(f"  {k:<18} raw={[round(x, 4) for x in v.detach().float().flatten().tolist()]}"
              f"  tanh={[round(x, 4) for x in t.tolist()]}")
    if gamma:
        mags = [abs(float(torch.tanh(v.detach().float()).mean())) for _, v in gamma]
        print(f"  → 融合残差上限 ≈ {min(mags):.3f} ~ {max(mags):.3f} × ‖F‖")
    if gates:
        tot = sum(float(v.detach().float().norm()) ** 2 for v in gates.values()) ** 0.5
        print(f"  门控卷积权重总范数 = {tot:.3f}")
    print(f"  structure = {ck.get('structure')}")


def main():
    args = [Path(a) for a in sys.argv[1:]]
    ckpts = args or _default_ckpts()
    if not ckpts:
        print("未找到 checkpoint，请显式传入路径")
        return
    for p in ckpts:
        report(p)


if __name__ == "__main__":
    main()
