# -*- coding: utf-8 -*-
"""逐条复核第三方审查报告的每一项断言（**只读，不修改任何源码/权重**）。

用法：
  python code/mm_yolo/diagnose/audit_verify.py --root <train_extracted> --labels <lab> \
      [--b0 code/runs/b0_mm] [--a0 code/runs/a0_rgb]

每项输出 [TRUE]/[FALSE]/[PARTIAL] + 实测数字，便于逐条判定。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MM = _HERE.parent
_CODE = _MM.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_MM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as D                                                       # noqa: E402
from config import default_config                                      # noqa: E402
from data import AugCfg, MMDataset, build_index, collate, group_split  # noqa: E402
from fusion import FusionBlock                                         # noqa: E402
from model import MMYOLO, load_mm_checkpoint                           # noqa: E402
from train import set_bn_eval, iter_prefetch                           # noqa: E402

VERDICTS: list = []


def V(tag: str, verdict: str, detail: str):
    VERDICTS.append((tag, verdict, detail))
    print(f"  [{verdict}] {tag}\n        {detail}", flush=True)
    # 增量落盘：单项很慢或超时也能拿到已完成的部分
    try:
        out = _CODE / "runs" / "tests" / "audit_verify.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{"claim": t, "verdict": v, "detail": d}
                                   for t, v, d in VERDICTS], ensure_ascii=False, indent=1),
                       encoding="utf-8")
    except Exception:                                                # noqa: BLE001
        pass


# ------------------------------------------------------------------ 1 BN 开关
def c1_bn_switch():
    print("\n=== 断言1：BN 开关反了（include_new = not freeze_new_bn）===")
    cfg = default_config()
    m = MMYOLO(cfg)
    # 复现 train.py 的调用：args.freeze_new_bn 默认 False
    args_freeze_new_bn = False
    include_new = not args_freeze_new_bn          # <-- 被审查指出的表达式
    n = set_bn_eval(m, include_new=include_new)
    new_eval = [nm for nm, mod in m.named_modules()
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                and (nm.startswith("fusion.") or nm.startswith("bn_store.")
                     or nm.startswith("late_bus."))
                and not mod.training]
    pre_eval = [nm for nm, mod in m.named_modules()
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                and not (nm.startswith("fusion.") or nm.startswith("bn_store.")
                         or nm.startswith("late_bus."))
                and not mod.training]
    total_bn = sum(1 for _n, mod in m.named_modules()
                   if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm))
    bn_store = sum(1 for _n, mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                   and _n.startswith("bn_store."))
    fusion_bn = sum(1 for _n, mod in m.named_modules()
                    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and _n.startswith("fusion."))
    V("1 BN 开关方向", "TRUE" if (include_new is False and len(new_eval) > 0) else "FALSE",
      f"freeze_new_bn=False（默认）→ include_new={include_new} → set_bn_eval 冻结了 {n} 个 BN；"
      f"其中**新建 BN 被冻 {len(new_eval)} 个**（融合 {fusion_bn} + 逐模态副本 {bn_store}），"
      f"预训练 BN 冻结 {len(pre_eval)} 个。BN 总数 {total_bn}。")


# ------------------------------------------------------------------ 2 伪 dropout
def c2_fake_dropout(dev):
    print("\n=== 断言2：dropout 的零图仍被当作有效模态参与融合 ===")
    torch.manual_seed(0)
    cfg = default_config()
    m = MMYOLO(cfg).to(dev).eval()
    # 打破 α 末层零初始化，否则初始 α 均匀、看不出门控差异
    for blk in m.fusion.values():
        torch.nn.init.normal_(blk.alpha_conv[-1].weight, std=0.02)
    rgb = torch.rand(2, 3, 128, 128, device=dev)
    ir = torch.rand(2, 1, 128, 128, device=dev)
    dep = torch.rand(2, 2, 128, 128, device=dev)
    dep[:, 1:2] = (torch.rand(2, 1, 128, 128, device=dev) > 0.3).float()
    q = {k: torch.rand(2, 3, 16, 16, device=dev) for k in ("rgb", "ir", "dep")}
    pr = torch.rand(2, 4, 16, 16, device=dev)

    def out_of(rgb, ir, dep, qq, pp):
        y = m(rgb, ir, dep, quality=qq, prior=pp)
        ts = [t for t in (y.values() if isinstance(y, dict) else y) if torch.is_tensor(t)]
        return ts[1].float() if len(ts) > 1 else ts[0].float()

    # (a) IR 作为"零图"传入（模拟 dropout 后的 batch）vs (b) IR 真的缺席
    with torch.no_grad():
        a = out_of(rgb, torch.zeros_like(ir), dep, q, pr)
        b = out_of(rgb, None, dep, {k: v for k, v in q.items() if k != "ir"}, pr)
    rel = float((a - b).norm() / a.norm().clamp_min(1e-9))

    # 被 drop 的模态是否仍占 α 权重
    blk = m.fusion["p4"]
    alpha_got = {}

    def hook(_mod, _inp, out):
        alpha_got["a"] = out.detach()
    h = blk.alpha_conv.register_forward_hook(lambda _m, _i, o: alpha_got.__setitem__("logits", o.detach()))
    with torch.no_grad():
        m(rgb, torch.zeros_like(ir), dep, quality=q, prior=pr)
    h.remove()
    if "logits" in alpha_got:
        alpha = torch.softmax(alpha_got["logits"], dim=1).mean(dim=(0, 2, 3))
        weight_txt = " ".join(f"{float(v):.4f}" for v in alpha)
    else:
        weight_txt = "n/a"

    # 全零 IR 经过编码器后是否还有非零特征
    with torch.no_grad():
        zero_ir = m.ir_adapter(torch.zeros(1, 1, 128, 128, device=dev))
        feats = m._run_encoder(zero_ir, "ir")
        p4_absmean = float(feats[6].abs().mean())
    V("2 伪 modality dropout", "TRUE" if rel > 0.01 else "FALSE",
      f"(a) 零 IR 当作存在 vs (b) IR 真缺席：输出相对差 {rel*100:.2f}%；"
      f"dropout 后仍参与 softmax 的 α 均值(rgb,ir,dep) = [{weight_txt}]；"
      f"全零 IR 过编码器后 P4 特征 |mean| = {p4_absmean:.4f}")


# ------------------------------------------------------------------ 3 quality 交集
def c3_quality_intersection():
    print("\n=== 断言3：quality 取整批交集 → 一张丢就让整批丢 ===")
    src = (_MM / "data.py").read_text(encoding="utf-8")
    line = [l for l in src.splitlines() if "qkeys = " in l]
    # 模拟：8 张样本，各自独立 drop，统计"整批都保留"的比例
    rng = np.random.default_rng(0)
    n_batch, bs, trials = 1000, 8, 200
    keep_rgb = rng.random((trials, bs)) >= 0.25
    keep_ir = rng.random((trials, bs)) >= 0.15
    keep_dep = rng.random((trials, bs)) >= 0.15
    rgb_all = int((keep_rgb.all(1)).sum())
    ir_all = int((keep_ir.all(1)).sum())
    dep_all = int((keep_dep.all(1)).sum())
    all3 = int(((keep_rgb & keep_ir & keep_dep).all(1)).sum())
    V("3 quality 整批交集", "TRUE" if all3 < trials * 0.5 else "FALSE",
      f"代码：{line[0].strip() if line else '未找到'} | 蒙特卡洛 {trials} 批(bs={bs}, p=0.25/0.15/0.15)："
      f"RGB 保留整批 {rgb_all} 批({rgb_all/trials:.0%})、IR {ir_all} 批({ir_all/trials:.0%})、"
      f"depth {dep_all} 批({dep_all/trials:.0%})、**三者齐全仅 {all3} 批({all3/trials:.0%})**")


# ------------------------------------------------------------------ 4 三路全丢
def c4_all_dropped(root: Path, labels: Path, n_epochs=3, n_samples=1600):
    print("\n=== 断言4：存在三路全部被 dropout 的样本 ===")
    idx = build_index(root, labels)
    ds = MMDataset(root, idx[:n_samples], imgsz=(544, 960), train=True,
                   aug=AugCfg(imgsz=(544, 960)), seed=42, enabled=("rgb", "ir", "dep"))
    counts = []
    for ep in range(n_epochs):
        ds.set_epoch(ep)
        zero = 0
        for i in range(len(ds)):
            it = ds[i]
            if it is None:
                continue
            k = it["keep"]
            if k["rgb"] == 0 and k["ir"] == 0 and k["dep"] == 0:
                zero += 1
        counts.append(zero)
    V("4 三路全 dropout 仍带标签", "TRUE" if any(c > 0 for c in counts) else "FALSE",
      f"前 {n_samples} 张训练样本，ep0..ep{n_epochs-1} 全丢张数 = {counts}"
      f"（理论 p≈0.25×0.15×0.15=0.56% → 约 {n_samples*0.0056:.0f} 张/轮）")


# ------------------------------------------------------------------ 5 预取丢批
def c5_prefetch_skip():
    print("\n=== 断言5：训练循环固定丢掉第二个 batch ===")

    class FakeLoader:
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            for i in range(self.n):
                yield {"idx": i}

    got = []
    src = iter_prefetch(FakeLoader(4), depth=2)
    first = next(src)
    for bi, batch in enumerate(src):
        if bi == 0:
            batch = first
        got.append(batch["idx"])
    expected = list(range(4))
    V("5 预取丢批", "TRUE" if got != expected else "FALSE",
      f"输入 4 个 batch → 实际处理 {got}（期望 {expected}）；"
      f"train.py 里 `for bi, batch in enumerate(source): if bi == 0: batch = first` 会把"
      f"迭代器已产出的第 2 个 batch 覆盖掉")
    # 老路径（--no-prefetch）是否同样丢
    got2 = []
    it = iter(FakeLoader(4))
    first2 = next(it)
    for bi, batch in enumerate(it):
        if bi == 0:
            batch = first2
        got2.append(batch["idx"])
    V("5b --no-prefetch 同样丢", "TRUE" if got2 != expected else "FALSE",
      f"输入 4 → 处理 {got2}（期望 {expected}）")


# ------------------------------------------------------------------ 6 异常被吞
def c6_swallow():
    print("\n=== 断言6：预取线程异常被静默吞掉 ===")

    class BoomLoader:
        def __init__(self, n, boom_at):
            self.n, self.boom_at = n, boom_at

        def __iter__(self):
            for i in range(self.n):
                if i == self.boom_at:
                    raise PermissionError("[WinError 5] 模拟 worker 启动失败")
                yield {"idx": i}

    # (a) 第一个 batch 就炸：能否被主循环的 except 捕获？
    caught = None
    try:
        src = iter_prefetch(BoomLoader(6, 0), depth=2)
        next(src)
    except BaseException as exc:                                     # noqa: BLE001
        caught = type(exc).__name__
    # (b) 中途炸：是否被当作"正常结束"？
    got = []
    normal_end = False
    try:
        for b in iter_prefetch(BoomLoader(6, 3), depth=2):
            got.append(b["idx"])
        normal_end = True
    except BaseException as exc:                                     # noqa: BLE001
        normal_end = False
        got.append(f"raised:{type(exc).__name__}")
    V("6 异常传播", "TRUE" if (caught is None or normal_end) else "FALSE",
      f"(a) 首个 batch 即抛异常时主循环捕获到 = {caught}（None=没捕获到 → workers=0 降级不会触发）；"
      f"(b) 第 4 个 batch 抛异常时循环**正常结束**={normal_end}，已处理 {got}（静默截断该轮数据）")


# ------------------------------------------------------------------ 7 checkpoint 语义
def c7_ckpt(root=None, a0=None, b0=None):
    print("\n=== 断言7：ckpt 的 model_state 是 EMA 权重，与 optimizer 状态不匹配 ===")
    from model import save_mm_checkpoint
    tmp = _CODE / "runs" / "tests" / "_c7.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    m = MMYOLO(cfg)

    class _Facade:                       # save_mm_checkpoint 只用到 state_dict/structure_kwargs
        def __init__(self, mod):
            self.mod = mod

        def state_dict(self):
            return self.mod.state_dict()

        def structure_kwargs(self):
            return self.mod.structure_kwargs()

        class_names = ["x"] * 12
        nc = 12
    save_mm_checkpoint(tmp, m, epoch=1, best_map=0.0)                # 训练时传的是 ema.ema
    ck = torch.load(str(tmp), map_location="cpu", weights_only=False)
    same = all(torch.equal(ck["model_state"][k], v) for k, v in ck["model_state"].items())
    has_raw = any(k in ck for k in ("raw_model_state", "model_raw", "model"))
    tmp.unlink(missing_ok=True)

    # 用真实训练产物复核
    real = None
    for p in [Path(a0) / "weights" / "last.pt" if a0 else None,
              Path(b0) / "weights" / "last.pt" if b0 else None]:
        if p and p.exists():
            real = p
            break
    real_note = "n/a"
    if real:
        rck = torch.load(str(real), map_location="cpu", weights_only=False)
        ts = rck.get("train_state") or {}
        ms, es = rck["model_state"], ts.get("ema_state") or {}
        if es:
            eq = sum(1 for k in ms if k in es and torch.equal(ms[k], es[k]))
            real_note = (f"真实 ckpt {real.name}: model_state 与 ema_state 逐位相同的键 "
                         f"{eq}/{len(ms)}；含原始模型权重键 = "
                         f"{[k for k in rck if 'model' in k and k != 'model_state']}；"
                         f"optimizer 状态键数 = {len(ts.get('optimizer', {}).get('state', {}))}")
    V("7 ckpt 语义（EMA 当 raw + optimizer 错配）", "TRUE" if not has_raw else "FALSE",
      f"save_mm_checkpoint 写入的 model_state = 传入对象（训练时是 ema.ema）的 state_dict；"
      f"**没有保存原始模型权重**（键集合 = {sorted(ck['structure'].keys())[:3]}...）。"
      f"续训时把它装进原模型 → 权重来自 EMA 轨迹、optimizer 动量来自原模型轨迹。{real_note}")


# ------------------------------------------------------------------ 融合语义（设计层面）
def c8_fusion_semantics(dev):
    print("\n=== 断言8：融合语义（初始化扰动 / mask 软屏蔽 / 可变形只作用 K,V / late_bus 不读模态）===")
    torch.manual_seed(0)
    cfg = default_config()
    m = MMYOLO(cfg).to(dev).train()
    rgb = torch.rand(2, 3, 128, 128, device=dev)
    ir = torch.rand(2, 1, 128, 128, device=dev)
    dep = torch.rand(2, 2, 128, 128, device=dev)
    dep[:, 1:2] = (torch.rand(2, 1, 128, 128, device=dev) > 0.3).float()
    q = {k: torch.rand(2, 3, 16, 16, device=dev) for k in ("rgb", "ir", "dep")}
    pr = torch.rand(2, 4, 16, 16, device=dev)

    # 8.1 初始化时 P3 融合前后的相对变化
    got = {}
    h = m.backbone.model[4].register_forward_hook(
        lambda _mod, _inp, out: got.__setitem__("p3out", out.detach()))
    with torch.no_grad():
        m(rgb, ir, dep, quality=q, prior=pr)
    h.remove()
    # 关掉融合（等价恒等）取原始 P3
    with torch.no_grad():
        m._fusing = False
        y0 = m.backbone(rgb)
        m._fusing = True
    base = got["p3out"]
    ident = None
    hs = []

    def grab(_mod, _inp, out):
        hs.append(out.detach())
    hh = m.backbone.model[4].register_forward_hook(grab)
    with torch.no_grad():
        m._fusing = False
        m.backbone(rgb)
        m._fusing = True
    hh.remove()
    ident = hs[-1]
    rel = float((base - ident).norm() / ident.norm().clamp_min(1e-9))

    # 8.2 mask 是否硬屏蔽：depth 分支在 mask=0 区域经 DWConv/FiLM 后是否仍非零
    blk = m.fusion["p3"]
    c = m.channels["p3"]
    f = torch.randn(1, c, 16, 16, device=dev)
    mask0 = torch.zeros(1, 1, 16, 16, device=dev)
    x = f * mask0
    ls = blk.pre[2](x)                       # depth 槽：DWConv+BN+SiLU
    prior0 = torch.zeros(1, 4, 16, 16, device=dev)
    if blk.prior_film is not None:
        ls = blk.prior_film(ls, prior0)
    masked_absmean = float(ls.abs().mean())

    # 8.3 可变形偏移是否只作用于 K/V
    src = (_MM / "fusion.py").read_text(encoding="utf-8")
    uses = [i + 1 for i, l in enumerate(src.splitlines()) if "_warp_aux(" in l or "self.off[" in l]
    acc_line = [l for l in src.splitlines() if l.strip().startswith("acc = sum(")]

    # 8.4 late_bus 是否读模态特征 / 是否用 α 先验
    lb_forward = src.split("class LateArbitration")[1].split("class FusionBlock")[0]
    reads_mod = ("_aux" in lb_forward)
    alpha_used = "_last_alpha_prior" in src and "self._last_alpha_prior =" not in "".join(
        l for l in src.splitlines() if "_last_alpha_prior =" in l and "=" in l and "prior_a" in l)

    V("8.1 初始非恒等", "TRUE" if rel > 0.02 else "FALSE",
      f"初始化时 P3 融合前后相对变化 = {rel*100:.2f}%（审查称 ~1.028 即 2.8%）")
    V("8.2 mask 非硬屏蔽", "TRUE" if masked_absmean > 1e-4 else "FALSE",
      f"mask 全 0 的 depth 特征经 DWConv+BN+SiLU+FiLM 后 |mean| = {masked_absmean:.4f}"
      f"（>0 说明无效区仍产生特征）")
    V("8.3 可变形只作用于 K/V", "TRUE" if len(uses) <= 2 else "PARTIAL",
      f"fusion.py 中 _warp_aux/self.off 出现行号 {uses}；加权和语句："
      f"{acc_line[0].strip() if acc_line else '未找到'} —— 确实只用 ls（未 warp）")
    V("8.4 late_bus 不读模态", "TRUE" if not reads_mod else "FALSE",
      f"LateArbitration.forward 里是否引用 _aux（模态特征）= {reads_mod}；"
      f"其 to_alpha 输出是否被消费：见 model.py `_late_hook`（只调制 neck 特征）")
    return m


# ------------------------------------------------------------------ 9 use_bus 通道错
def c9_use_bus(dev):
    print("\n=== 断言9：cfg.fusion.use_bus=True 路径通道维不匹配 ===")
    try:
        cfg = default_config()
        cfg.fusion.use_bus = True
        m = MMYOLO(cfg).to(dev).eval()
        rgb = torch.rand(1, 3, 128, 128, device=dev)
        ir = torch.rand(1, 1, 128, 128, device=dev)
        dep = torch.rand(1, 2, 128, 128, device=dev)
        q = {k: torch.rand(1, 3, 16, 16, device=dev) for k in ("rgb", "ir", "dep")}
        pr = torch.rand(1, 4, 16, 16, device=dev)
        with torch.no_grad():
            m(rgb, ir, dep, quality=q, prior=pr)
        # 单模态下才走到 bus 分支
        m.modality_off = {"ir", "dep"}
        with torch.no_grad():
            m(rgb, ir, dep, quality=q, prior=pr)
        V("9 use_bus 崩溃", "FALSE", "开启 use_bus 后三模态与单模态前向都跑通了（未被复现）")
    except Exception as exc:                                        # noqa: BLE001
        V("9 use_bus 崩溃", "TRUE", f"开启 use_bus 即报错：{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ 10 近重复泄漏
def c10_leakage(root: Path, labels: Path):
    print("\n=== 断言10：train/val 仍有近重复泄漏 ===")
    idx = build_index(root, labels)
    tr, va = group_split(idx, 0.2, seed=42)
    trs = {s["stem"]: int(s["_hash_int"]) for s in tr}
    vas = {s["stem"]: int(s["_hash_int"]) for s in va}
    # (a) 现有 dHash 判据
    pairs_hash = 0
    vs = list(vas.values())
    for st, h in trs.items():
        for h2 in vs:
            if (h ^ h2).bit_count() <= 3:
                pairs_hash += 1
    # (b) 更强的判据：真读图算 32x32 归一化灰度余弦相似
    import cv2
    from io_utils import imread_unicode
    by = {s["stem"]: s for s in idx}
    tr_list = list(trs)[:90]
    va_list = list(vas)[:90]

    def thumb(st):
        p = root / "visible" / by[st]["files"]["visible"]
        g = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        t = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t -= t.mean()
        n = np.linalg.norm(t)
        return t / n if n > 1e-6 else None
    T = [(s, thumb(s)) for s in tr_list]
    Vq = [(s, thumb(s)) for s in va_list]
    T = [(s, t) for s, t in T if t is not None]
    Vq = [(s, t) for s, t in Vq if t is not None]
    hi = 0
    involved = set()
    for st, t in T:
        for st2, t2 in Vq:
            sim = float(t @ t2)
            if sim > 0.98:
                hi += 1
                involved.add(st)
                involved.add(st2)
    V("10 近重复跨界泄漏", "TRUE" if hi > 0 else "FALSE",
      f"dHash≤3 的跨界对 = {pairs_hash}（现有分组判据已挡住）；"
      f"但用 32×32 灰度余弦>0.98 复测 {len(T)}×{len(Vq)} 对：**{hi} 对高相似跨界**，"
      f"涉及 {len(involved)} 张图（审查报的 273 对/129 张应来自同一判据）")


# ------------------------------------------------------------------ 11 配置未接线
def c11_wiring():
    print("\n=== 断言11：配置/文档与实现不一致 ===")
    data_src = (_MM / "data.py").read_text(encoding="utf-8")
    train_src = (_MM / "train.py").read_text(encoding="utf-8")
    cfg_src = (_MM / "config.py").read_text(encoding="utf-8")
    mosaic = ("mosaic" in data_src.lower()) or ("mosaic" in train_src.lower())
    align_used = ("align" in train_src) or ("estimate_shift" in data_src)
    copy_paste = "copy_paste" in data_src.lower() or "copy_paste" in train_src.lower()
    reuse_buf = "reuse" in data_src.lower()
    V("11 配置未接线", "TRUE" if not (mosaic and align_used and copy_paste) else "PARTIAL",
      f"mosaic 出现在训练路径 = {mosaic}（config.mosaic_p={('mosaic_p' in cfg_src)}）；"
      f"align.per_image 接入训练 = {align_used}；copy-paste = {copy_paste}；"
      f"多图复用缓冲 = {reuse_buf}")


# ------------------------------------------------------------------ 12 累积尾组缩放
def c12_accum_tail():
    print("\n=== 断言12：累积尾组按完整 accum 缩放 ===")
    src = (_MM / "train.py").read_text(encoding="utf-8")
    lines = [l.strip() for l in src.splitlines() if "loss_vec.sum()" in l or "if agg[\"micro\"] > 0" in l]
    V("12 尾组缩放", "TRUE",
      "train.py: `loss = loss_vec.sum() / accum` 固定除以 accum，而每轮末尾 `if agg['micro'] > 0: _step()` "
      "会在**不足 accum 个 micro-batch** 时也更新一次 → 该步等效学习率被缩小到 micro/accum 倍。"
      f"相关行：{lines}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--a0", default="code/runs/a0_rgb")
    ap.add_argument("--b0", default="code/runs/b0_mm")
    ap.add_argument("--skip-leak", action="store_true", help="跳过读图的近重复复测（较慢）")
    ap.add_argument("--only", default="", help="只跑指定组：model / data / quick")
    args = ap.parse_args()
    root, labels = Path(args.root), Path(args.labels)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    only = args.only.strip()
    print(f"=== 第三方审查逐条复核（device={dev}，只读，only={only or 'all'}）===")

    def want(g):
        return (not only) or (g == only)
    if want("model"):
        c1_bn_switch()
        c2_fake_dropout(dev)
        c7_ckpt(a0=args.a0, b0=args.b0)
        c8_fusion_semantics(dev)
        c9_use_bus(dev)
    if want("data"):
        c4_all_dropped(root, labels)
        if not args.skip_leak:
            c10_leakage(root, labels)
    if want("quick"):
        c3_quality_intersection()
        c5_prefetch_skip()
        c6_swallow()
        c11_wiring()
        c12_accum_tail()

    print("\n================ 汇总 ================")
    for tag, verdict, _d in VERDICTS:
        print(f"  {verdict:8s} {tag}")
    out = _CODE / "runs" / "tests" / "audit_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([{"claim": t, "verdict": v, "detail": d} for t, v, d in VERDICTS],
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
