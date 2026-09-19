# -*- coding: utf-8 -*-
"""Stage C：融合块（方案 v4 §3 Stage C 的实现）。

四档阶梯（同一实现按 cfg.tier 开关，便于 A/B）：
  L0  逐像素门控加权和（基线，必须能被打败）
  L1  + 模态内 3×3 DWConv 上下文
  L2  + 窗口化跨模态注意力（瓶颈投影 C→C/heads/bottleneck）
  L3  + 可变形小偏移（学习式残差对齐，Δ 初始化为 0 → L3 起步等价于 L2）

通用约束（四档都适用）：
  * **RGB 锚定小残差**：候选融合仍强制通电，但用小非零比例写回 RGB 主路，
    避免随机模块在第 0 步破坏 COCO 预训练特征；无 RGB 时自动使用融合候选；
  * α 的 logits 末层零初始化 → 初始 α 均匀（三模态等权通电）；
  * 输出用 **BN**（不是 LN）：预训练 neck 的 running stats 是在 COCO 分布上估的，
    换成 LN 会让 neck 带过期统计跑；
  * 权重共享迭代 cfg.iters 次（深度换参数），并监控特征范数比（自检里做）；
  * 可靠性掩码：对 depth 特征做乘性门控（mask=0 的区域不参与融合）。

L4（register 总线）另见 PersistentRegisterBus：同一动态状态贯穿 P3→P4→P5。
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv

from config import FusionCfg


def dwconv(c: int, k: int = 3) -> nn.Sequential:
    """深度可分离 3×3（保通道）。"""
    return nn.Sequential(nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False),
                         nn.BatchNorm2d(c), nn.SiLU(inplace=True))


def _base_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * 2.0 / H - 1.0
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * 2.0 / W - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()


def _prob_logit(p: float) -> float:
    p = min(1.0 - 1e-4, max(1e-4, float(p)))
    return float(torch.logit(torch.tensor(p)))


class PersistentRegisterBus(nn.Module):
    """贯穿 P3→P4→P5 的**每图动态** register workspace。

    `seed` 只是全数据共享的初始状态 R0；每张图在 P3/P4/P5 依次得到 R1/R2/R3。
    Register 先从带模态/尺度类型嵌入的低分辨率 token 读取信息，随后空间特征再从
    register 读回。状态由调用方显式传递，不存成跨 batch 的全局缓存，兼容 DataParallel
    之外的普通单 GPU/CPU 服务器执行，也不会把上一张图的信息泄漏到下一张图。
    """

    def __init__(self, channels: Mapping[str, int], n_mod: int = 3, n_tokens: int = 8,
                 dim: int = 64, heads: int = 4, pool: int = 4,
                 residual_init: float = 0.02):
        super().__init__()
        if dim % heads:
            raise ValueError(f"register dim={dim} 必须能被 heads={heads} 整除")
        self.scale_names = tuple(channels)
        self.n_mod = int(n_mod)
        self.n_tokens = int(n_tokens)
        self.dim = int(dim)
        self.pool = max(1, int(pool))
        self.seed = nn.Parameter(torch.empty(1, self.n_tokens, self.dim))
        self.modality_embed = nn.Parameter(torch.empty(1, self.n_mod, 1, self.dim))
        self.scale_embed = nn.Parameter(torch.empty(1, len(self.scale_names), 1, self.dim))
        nn.init.normal_(self.seed, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)
        nn.init.normal_(self.scale_embed, std=0.02)

        self.read_proj = nn.ModuleDict({
            s: nn.ModuleList(nn.Conv2d(int(c), self.dim, 1, bias=False)
                             for _ in range(self.n_mod))
            for s, c in channels.items()})
        self.write_query = nn.ModuleDict({s: nn.Conv2d(int(c), self.dim, 1, bias=False)
                                          for s, c in channels.items()})
        self.write_out = nn.ModuleDict({s: nn.Conv2d(self.dim, int(c), 1, bias=False)
                                        for s, c in channels.items()})
        self.register_norm = nn.LayerNorm(self.dim)
        self.feature_norm = nn.LayerNorm(self.dim)
        self.read_attn = nn.MultiheadAttention(self.dim, heads, batch_first=True)
        self.write_attn = nn.MultiheadAttention(self.dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim * 2), nn.GELU(),
                                 nn.Linear(self.dim * 2, self.dim))
        self.read_gain = nn.Parameter(torch.tensor(0.1))
        self.mlp_gain = nn.Parameter(torch.tensor(0.1))
        self.write_logit = nn.ParameterDict({
            s: nn.Parameter(torch.tensor(_prob_logit(residual_init))) for s in channels})

    def initial_state(self, batch: int, ref: torch.Tensor) -> torch.Tensor:
        return self.seed.to(device=ref.device, dtype=ref.dtype).expand(batch, -1, -1)

    def forward(self, scale: str, modal_feats: Sequence[torch.Tensor],
                valid: Sequence[torch.Tensor], fused: torch.Tensor,
                registers: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if scale not in self.read_proj:
            raise KeyError(f"未知 register 尺度 {scale}")
        B = fused.shape[0]
        if registers is None:
            registers = self.initial_state(B, fused)
        if registers.shape != (B, self.n_tokens, self.dim):
            raise ValueError(f"register 形状错误：{tuple(registers.shape)}")

        tokens, padding = [], []
        si = self.scale_names.index(scale)
        for mi, (feat, vm) in enumerate(zip(modal_feats, valid)):
            vm_f = vm.to(dtype=feat.dtype)
            z = self.read_proj[scale][mi](feat * vm_f)
            z = F.adaptive_avg_pool2d(z, self.pool).flatten(2).transpose(1, 2)
            z = z + self.modality_embed[:, mi].to(z.dtype) \
                + self.scale_embed[:, si].to(z.dtype)
            pv = F.adaptive_max_pool2d(vm_f, self.pool).flatten(1) > 0.5
            tokens.append(z)
            padding.append(~pv)
        image_tokens = torch.cat(tokens, dim=1)
        key_padding = torch.cat(padding, dim=1)
        # 某张图可能在当前尺度暂时没有可用模态：例如 modality dropout 只保留
        # Depth，而配置规定 Depth 从 P4 才接入。该样本应让 register 原样穿过 P3，
        # 不能让整批报错，也不能把一个全 mask 的序列送进 MHA（会产生 NaN）。
        active = ~key_padding.all(dim=1)
        if not active.all():
            key_padding = key_padding.clone()
            image_tokens = image_tokens.clone()
            key_padding[~active, 0] = False       # 安全 dummy token
            image_tokens[~active, 0] = 0

        r = self.register_norm(registers)
        read, _ = self.read_attn(r, self.feature_norm(image_tokens),
                                 self.feature_norm(image_tokens),
                                 key_padding_mask=key_padding, need_weights=False)
        active_f = active[:, None, None].to(registers.dtype)
        registers = registers + active_f * torch.tanh(self.read_gain) * read
        registers = registers + active_f * torch.tanh(self.mlp_gain) \
            * self.mlp(self.mlp_norm(registers))

        q = self.write_query[scale](fused).flatten(2).transpose(1, 2)
        write, _ = self.write_attn(self.feature_norm(q), self.register_norm(registers),
                                   self.register_norm(registers), need_weights=False)
        write = write.transpose(1, 2).reshape(B, self.dim, *fused.shape[-2:])
        delta = self.write_out[scale](write)
        # 只有确有辅助模态时才写回空间特征；纯 RGB 样本保持严格恒等。
        aux_present = torch.stack([v.flatten(1).any(1) for v in valid[1:]], dim=1).any(1)
        gain = torch.sigmoid(self.write_logit[scale]).to(fused.dtype)
        fused = fused + gain * delta * aux_present[:, None, None, None].to(fused.dtype)
        return fused, registers

    @torch.no_grad()
    def state_stats(self, registers: Optional[torch.Tensor]) -> Dict[str, float]:
        if registers is None:
            return {"register_norm": 0.0, "register_token_std": 0.0}
        return {"register_norm": float(registers.norm(dim=-1).mean()),
                "register_token_std": float(registers.std(dim=1).mean())}


class PriorFiLM(nn.Module):
    """用**模态先验**调制该模态分支的特征（显式注入"直觉"）。

    以 depth 为例：先验通道 = [逆相对深度, 有效性, 法线 nx, 法线 ny]
    → 小卷积 → (γ, β) → `f * (1 + tanh(γ)) + β`。
    这样"这里是近处平面 / 远处背景 / 无效区"是**显式写进特征**的，
    而不是让网络从 2000 张图里自己发现。
    """

    def __init__(self, c: int, p_ch: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(p_ch, hidden, 1), nn.SiLU(inplace=True),
                                  nn.Conv2d(hidden, 2 * c, 1))
        nn.init.zeros_(self.conv[-1].weight)          # 初始为恒等（不破坏预训练特征）
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, feat: torch.Tensor, prior: Optional[torch.Tensor]) -> torch.Tensor:
        if prior is None:
            return feat
        if prior.shape[-2:] != feat.shape[-2:]:
            prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        g, b = self.conv(prior.to(dtype=feat.dtype)).chunk(2, dim=1)
        return feat * (1.0 + torch.tanh(g)) + b


class LateArbitration(nn.Module):
    """Detect 之前的"片后"总线读：对颈部输出的三张特征图做一次**全局仲裁调制**。

    动机：融合点（主干 4/6/10）之后还有 12 层颈部，最终送进 Detect 的特征
    （层 16/19/22 输出）才是决定预测的那一份。这里用少量 register token 读它，
    输出逐尺度 FiLM 与模态权重全局先验 —— 这是"整张图该信谁"的唯一显式通道。
    """

    def __init__(self, channels: Sequence[int], n_mod: int = 3, n_tokens: int = 8,
                 dim: int = 64, heads: int = 4):
        super().__init__()
        self.n_mod = int(n_mod)
        c_top = int(channels[-1])
        self.inp = nn.Conv2d(c_top, dim, 1)
        self.tokens = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.normal_(self.tokens, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.ModuleList(nn.Conv2d(dim, 2 * int(c), 1) for c in channels)
        # **小随机初始化而不是零初始化**：零初始化时 ∂out/∂ctx = W = 0，
        # 梯度到不了 token/注意力（支路变"可选"→ 重演旧方案被饿死的病）。
        # 用 std=0.01 让它初始接近恒等（tanh(g)≈g≈0）但**从第 0 步就通电**。
        for f in self.film:
            nn.init.normal_(f.weight, std=0.01)
            nn.init.zeros_(f.bias)

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        """诊断：各尺度调制强度的均值（|tanh γ|），用于判断总线是否真在起作用。"""
        return {f"film_scale{i}": float(torch.tanh(f.weight).abs().mean())
                for i, f in enumerate(self.film)}

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """返回被全局仲裁调制后的三张颈部特征（初始接近恒等，随训练增长）。"""
        B = feats[0].shape[0]
        z = self.inp(feats[-1]).flatten(2).transpose(1, 2)          # (B,L,dim)
        tok = self.tokens.expand(B, -1, -1)
        seq = torch.cat([tok, z], 1)
        n = self.norm(seq)
        out, _ = self.attn(n, n, n)
        tok_out = out[:, : self.tokens.shape[1]].mean(1)            # (B,dim)
        ctx = tok_out.view(B, -1, 1, 1)
        outs = []
        for f, film in zip(feats, self.film):
            g, b = film(ctx).chunk(2, dim=1)
            outs.append(f * (1.0 + torch.tanh(g)) + b)
        return outs


class FusionBlock(nn.Module):
    """单尺度融合块。feats = [main, aux1, aux2?]（同通道、同分辨率）。"""

    def __init__(self, c: int, cfg: FusionCfg, n_mod: int = 3, slot_names: Sequence[str] = ()):
        super().__init__()
        self.cfg = cfg
        self.n_mod = int(n_mod)
        self.tier = str(cfg.tier).upper()
        assert self.tier in ("L0", "L1", "L2", "L3"), f"未知融合档: {cfg.tier}"
        self.slot_names = tuple(slot_names) or tuple(cfg.slot_names[: self.n_mod])
        k = int(cfg.window) | 1
        self.k = k
        self.heads = max(1, int(cfg.heads))
        self.cb = max(8, c // max(1, int(cfg.bottleneck)))
        self.cb -= self.cb % self.heads
        self.scale_attn = (self.cb // self.heads) ** -0.5
        # 每个辅助槽是否启用**学习式残差对齐**：tier=L3 时全开，否则看 deformable 名单
        self.use_off = [False] * self.n_mod
        for i in range(1, self.n_mod):
            nm = self.slot_names[i] if i < len(self.slot_names) else f"aux{i}"
            self.use_off[i] = (self.tier == "L3") or (nm in tuple(cfg.deformable))
        self.any_off = any(self.use_off)

        # 模态内上下文（L1+）
        self.pre = nn.ModuleList(dwconv(c) if self.tier != "L0" else nn.Identity()
                                 for _ in range(self.n_mod))
        # 逐区域模态权重：瓶颈化（3C→C/8→n_mod），末层零初始化 → 初始 α 均匀
        hid = max(16, c // 8)
        # 门控输入 = 每槽特征 + 每槽质量描述子（每个槽一份，共 n_mod 份）
        q_in = c * self.n_mod + (int(cfg.quality_channels) * self.n_mod
                                 if cfg.quality_gate else 0)
        self.quality_gate = bool(cfg.quality_gate)
        self.alpha_conv = nn.Sequential(nn.Conv2d(q_in, hid, 1), nn.SiLU(inplace=True),
                                        nn.Conv2d(hid, self.n_mod, 1))
        nn.init.zeros_(self.alpha_conv[-1].weight)
        nn.init.zeros_(self.alpha_conv[-1].bias)
        # 强制调制强度：γ = floor + (1−floor)·sigmoid(raw)，raw=0 → γ=(1+floor)/2（通电）
        self.gamma_raw = nn.Parameter(torch.zeros(self.n_mod))
        # β：交互项的强度。**不能零初始化**（否则交互支路从第 0 步就没有梯度 = 重演"可选支路饿死"）
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.local_mix_logit = nn.Parameter(torch.tensor(_prob_logit(cfg.residual_init)))
        # 模态先验 FiLM（显式注入"直觉"，默认只给 depth 槽）
        self.prior_film = PriorFiLM(c, int(cfg.prior_channels)) if cfg.prior_film else None
        # 交互（L2+）
        self.q = nn.Conv2d(c, self.cb, 1)
        self.k_proj = nn.ModuleList(nn.Conv2d(c, self.cb, 1) for _ in range(self.n_mod))
        self.v_proj = nn.ModuleList(nn.Conv2d(c, self.cb, 1) for _ in range(self.n_mod))
        self.o_proj = nn.Conv2d(self.cb, c, 1)
        if self.any_off:                                        # 可变形偏移：零初始化 → 起步等价 L2
            self.off = nn.ModuleList(nn.Conv2d(c, 2, 1) for _ in range(self.n_mod))
            for m in self.off:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            self.off = None
        self.bn = nn.BatchNorm2d(c)

    # ------------------------------------------------------------------ 交互
    def _offset_grid(self, src: torch.Tensor, max_off: float, slot: int) -> torch.Tensor:
        """由主路预测受限偏移；同一网格同时用于特征、有效掩码和质量图。"""
        off = torch.tanh(self.off[slot](src)) * float(max_off)             # (B,2,H,W)
        B, _, H, W = src.shape
        off = off.permute(0, 2, 3, 1)                                      # (B,H,W,2) 单位=像素
        off = off * torch.tensor([2.0 / W, 2.0 / H], device=src.device, dtype=src.dtype)
        return _base_grid(B, H, W, src.device, src.dtype) + off

    @staticmethod
    def _sample_grid(x: torch.Tensor, grid: torch.Tensor, nearest: bool = False) -> torch.Tensor:
        return F.grid_sample(x, grid, mode="nearest" if nearest else "bilinear",
                             padding_mode="zeros", align_corners=False)

    def _interaction(self, feats: Sequence[torch.Tensor], present: Sequence[int],
                     valid: Sequence[torch.Tensor]) -> torch.Tensor:
        """窗口化跨模态注意力：以 feat[0] 为 query，其余**在场**模态为 key/value。

        辅助特征已经在进入本函数前统一对齐，门控加权与 K/V 使用同一份结果。
        """
        query = feats[0]
        chosen = valid[0]
        for m in present[1:]:
            query = torch.where(chosen, query, feats[m])
            chosen = chosen | valid[m]
        q = self.q(query)
        B, Cb, H, W = q.shape
        L = H * W
        q = q.view(B, self.heads, Cb // self.heads, L)
        out = None
        n_aux = torch.zeros(B, 1, 1, 1, device=q.device, dtype=q.dtype)
        for m in present:
            if m == 0:
                continue
            kv = feats[m]
            vm_valid = valid[m]
            km = self.k_proj[m](kv)
            vm = self.v_proj[m](kv)
            vm = vm * vm_valid.to(vm.dtype)
            ku = F.unfold(km, self.k, padding=self.k // 2).view(B, self.heads, Cb // self.heads,
                                                                self.k * self.k, L)
            vu = F.unfold(vm, self.k, padding=self.k // 2).view(B, self.heads, Cb // self.heads,
                                                                self.k * self.k, L)
            att = torch.einsum("bhdl,bhdkl->bhkl", q, ku) * self.scale_attn
            vu_valid = F.unfold(vm_valid.to(vm.dtype), self.k, padding=self.k // 2)
            vu_valid = vu_valid.view(B, 1, self.k * self.k, L) > 0.5
            att = att.masked_fill(~vu_valid, -1e4).softmax(dim=2)
            att = att * vu_valid.to(att.dtype)
            att = att / att.sum(dim=2, keepdim=True).clamp_min(1e-6)
            o = torch.einsum("bhkl,bhdkl->bhdl", att, vu).reshape(B, Cb, H, W)
            out = o if out is None else out + o
            n_aux += valid[m].any(dim=(2, 3), keepdim=True).to(q.dtype)
        if out is None:
            return torch.zeros_like(feats[0])
        return (self.o_proj(out / n_aux.clamp_min(1))
                - self.o_proj.bias.view(1, -1, 1, 1)) * (n_aux > 0).to(q.dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, feats: List[Optional[torch.Tensor]], mask: Optional[torch.Tensor] = None,
                quality: Optional[Sequence[Optional[torch.Tensor]]] = None,
                prior: Optional[torch.Tensor] = None,
                keep: Optional[torch.Tensor] = None,
                return_context: bool = False):
        """feats: 长度 = n_mod 的列表，None 表示该模态本次不可用（softmax 里被屏蔽）。

        quality: 每个槽的**模态质量描述子**（亮度/对比度/清晰度…），显式喂进门控；
        prior : 主模态之外的**几何先验**（depth 的逆深度/法线），做 FiLM 显式注入。
        只有一个模态可用时**直接恒等返回**（保证 RGB-only 与基线可比）。
        """
        present = [i for i, f in enumerate(feats) if f is not None]
        assert present and feats[0] is not None, "主路特征张量必须存在"
        ref = feats[0]
        assert ref is not None
        if keep is None:
            keep = ref.new_tensor([[float(i in present) for i in range(len(feats))]]).expand(ref.shape[0], -1)
        keep = keep.to(device=ref.device, dtype=torch.bool)
        if keep.shape != (ref.shape[0], len(feats)):
            raise ValueError(f"keep 形状应为 {(ref.shape[0], len(feats))}，实际 {tuple(keep.shape)}")
        if not keep.any(dim=1).all():
            raise ValueError("每张图至少须保留一路有效模态")
        valid = keep[:, :, None, None].expand(-1, -1, *ref.shape[-2:]).clone()
        for i in range(len(feats)):
            if i not in present:
                valid[:, i] = False
        if mask is not None and len(feats) > 2 and feats[-1] is not None:
            depth_valid = F.interpolate(mask.float(), size=ref.shape[-2:], mode="nearest") > 0.5
            valid[:, -1] &= depth_valid[:, 0]

        # 对齐只算一次，同一结果供 α 加权、局部 attention K/V、register 读取共同使用。
        grids: List[Optional[torch.Tensor]] = [None] * len(feats)
        xs: List[torch.Tensor] = []
        align_src = ref * valid[:, 0:1].to(ref.dtype)
        for i, f in enumerate(feats):
            x = torch.zeros_like(ref) if f is None else f
            vm = valid[:, i:i + 1]
            x = x * vm.to(x.dtype)
            if self.prior_film is not None and i == len(feats) - 1 and prior is not None:
                x = self.prior_film(x, prior) * vm.to(x.dtype)
            if f is not None and i > 0 and self.off is not None and self.use_off[i]:
                grid = self._offset_grid(align_src, self.cfg.max_offset, i)
                grids[i] = grid
                x = self._sample_grid(x, grid)
                vm = self._sample_grid(vm.to(x.dtype), grid, nearest=True) > 0.5
                valid[:, i:i + 1] = vm
                x = x * vm.to(x.dtype)
            xs.append(x)
        spatial = [valid[:, i:i + 1] for i in range(len(feats))]

        # 质量图使用与对应模态相同的残差对齐网格。
        qs: List[Optional[torch.Tensor]] = [None] * len(feats)
        if self.quality_gate:
            for i in range(len(feats)):
                qi = quality[i] if quality is not None and i < len(quality) else None
                if qi is None:
                    qv = torch.zeros(ref.shape[0], int(self.cfg.quality_channels),
                                     *ref.shape[-2:], device=ref.device, dtype=ref.dtype)
                else:
                    qv = F.interpolate(qi, size=ref.shape[-2:], mode="bilinear",
                                       align_corners=False).to(dtype=ref.dtype)
                if grids[i] is not None:
                    qv = self._sample_grid(qv, grids[i])
                qs[i] = qv * spatial[i].to(qv.dtype)

        anchor = xs[0]
        out = anchor
        last_ls = xs
        for _ in range(max(1, int(self.cfg.iters))):
            x = [out] + xs[1:]
            ls = [self.pre[i](x[i]) * spatial[i].to(x[i].dtype) for i in range(len(x))]
            last_ls = ls
            if self.quality_gate:
                logits = self.alpha_conv(torch.cat(ls + list(qs), 1))  # type: ignore[arg-type]
            else:
                logits = self.alpha_conv(torch.cat(ls, 1))
            logits = (logits / max(1e-6, float(self.cfg.tau))).masked_fill(~valid, -1e4)
            alpha = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
            gamma = float(self.cfg.gamma_floor) + (1.0 - float(self.cfg.gamma_floor)) \
                * torch.sigmoid(self.gamma_raw[: len(ls)]).view(1, -1, 1, 1)
            acc = sum(alpha[:, i: i + 1] * (1.0 + gamma[:, i: i + 1]) * ls[i]
                      for i in range(len(ls)))
            inter = self._interaction(ls, present, spatial) if self.tier in ("L2", "L3") else 0.0
            candidate = self.bn(acc + self.beta * inter)
            mix = torch.sigmoid(self.local_mix_logit).to(candidate.dtype)
            rgb_valid = spatial[0]
            out = torch.where(rgb_valid, anchor + mix * (candidate - anchor), candidate)
            # 对没有任何辅助信息的空间位置保持 RGB 严格恒等。
            aux_valid = torch.stack(spatial[1:], dim=0).any(dim=0)
            out = torch.where(rgb_valid & ~aux_valid, ref, out)
        if return_context:
            return out, last_ls, spatial
        return out
