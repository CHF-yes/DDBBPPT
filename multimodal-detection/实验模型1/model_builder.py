# -*- coding: utf-8 -*-
"""
实验模型1 —— 三模态「官方 YOLO11s 全量主干 + 零初始化残差注入」模型。

设计原则（本次修复的核心）
==========================
1. **完整保留官方 YOLO11s 的 Sequential**（`YOLO(yolo11s.pt).model`，24 层）：
       model[0:10]  backbone（P1..P5，含 SPPF/C2PSA）
       model[11:22] 官方预训练 neck（FPN+PAN，Concat 依赖 4/6/10/13）
       model[23]    Detect（cv2 回归分支 + dfl + cv3 分类分支）
   不再自建 SimplePAN、不再重造 Detect —— COCO 权重（backbone + neck + 回归分支）
   全部继承。注意 Sequential 内部含 `from: [-1, 4/6/10/13]` 的 Concat，
   因此必须通过 DetectionModel 的 forward 跑（它会按 `m.f` 取早层输出），
   不能把 `model.model` 当普通 Sequential 直接调用。

2. **只在第 4、6、10 层（= P3/P4/P5）输出之后做零初始化残差注入**：
       F' = F + tanh(gamma) · proj([IR, Depth])          # gamma 初始 0
       F' = F' · (1 + alpha · Σ_m W_m·A_m)               # MEGA，alpha 初始 0（仅 P4/P5）
   gamma=0、alpha=0 时两条式子都是**严格恒等**（0·x 与 x·1.0 在浮点下逐位不变），
   所以 RGB 三通道输入的完整前向（P3/P4/P5 → 预训练 neck → Detect）与官方
   YOLO11s **逐位相同**（`python model_builder.py` 的 __main__ 自检会断言这一点）。

3. **Detect 只适配类别输出层**：仅把 cv3 每个尺度的末层 1×1 卷积换成 nc 类输出
   （默认按类别名从 COCO 同行迁移，其余随机初始化 + 官方类别先验偏置）；
   cv3 的共享层（DWConv/Conv）、回归分支 cv2、dfl 以及 neck/backbone 权重全部保留。

辅助流（IR/Depth）与 Step2-4（模态 dropout / MEGA 边缘注意力 / 每模态辅助头）保持不变。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _CODE_ROOT / "vendor"
for _p in (str(_CODE_ROOT), str(_VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

import models_config as MC                                        # noqa: E402
from ultralytics import YOLO                                     # vendor 版
from ultralytics.nn.modules import Conv, C2f                     # 复用官方组件
from ultralytics.nn.modules.head import Detect                   # 复用官方检测头

# 注入点：backbone 中输出 P3/P4/P5 的层索引（yolo11.yaml：3-P3/8、5-P4/16、10-P5/32 之后）
# 官方 neck 的 Concat 正好从 4/6/10 取这三层输出（from: [-1,6] / [-1,4] / [-1,10]），
# 因此在这三处做残差注入 = 融合信息参与全部 neck/head 计算。
_IDX_P3, _IDX_P4, _IDX_P5 = 4, 6, 10


# ============================================================
# 辅助流 / 模态 dropout / MEGA / 辅助头（保持原设计）
# ============================================================

class AuxStream(nn.Module):
    """轻量单通道辅助流：IR/Depth 各一条，输出与主流 P3(1/8)/P4(1/16)/P5(1/32) 同分辨率。
    每级恰好下采样 1 次：1/2 → 1/4 → 1/8(f3) → 1/16(f4) → 1/32(f5)。
    ch = (c_f3, c_f4, c_f5) 为三个输出通道数。
    in_ch：输入通道（IR=1；Depth=2：距离+有效掩码）。"""

    def __init__(self, ch=(32, 48, 64), in_ch: int = 1):
        super().__init__()
        c1, c2, c3 = ch
        self.s1 = nn.Sequential(Conv(in_ch, 16, 3, 2), C2f(16, 16, n=1))
        self.s2 = nn.Sequential(Conv(16, 24, 3, 2), C2f(24, 24, n=1))
        self.s3 = nn.Sequential(Conv(24, c1, 3, 2), C2f(c1, c1, n=1))
        self.s4 = nn.Sequential(Conv(c1, c2, 3, 2), C2f(c2, c2, n=1))
        self.s5 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, n=1))

    def forward(self, x: torch.Tensor):
        """x: (B,in_ch,H,W) → (f3(B,c1,H/8,W/8), f4(B,c2,H/16,W/16), f5(B,c3,H/32,W/32))"""
        f3 = self.s3(self.s2(self.s1(x)))
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        return f3, f4, f5


class ModalDropout(nn.Module):
    """逐模态独立 dropout（Step2）：IR 与 Depth **各自独立**按概率置零（注入前），
    训练覆盖「仅 IR 失效 / 仅 Depth 失效 / 双失效 / 齐全」四种态——
    对应赛题"单模态质量差时性能不崩"的鲁棒性要求。
    只作用于辅助流（RGB 主流不丢）；仅 self.training 时生效。"""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def _sample_keep(self, feats):
        """每个样本独立采样一次，并让同一样本在 P3/P4/P5 共用该掩码。"""
        ref = feats[0]
        if not self.training or self.p <= 0.0:
            return torch.ones((ref.shape[0], 1, 1, 1), device=ref.device, dtype=ref.dtype)
        return (torch.rand((ref.shape[0], 1, 1, 1), device=ref.device) >= self.p).to(ref.dtype)

    @staticmethod
    def _apply_mask(feats, keep):
        return tuple(f * keep for f in feats)

    def forward(self, feats_ir, feats_dep):
        keep_ir = self._sample_keep(feats_ir)
        keep_dep = self._sample_keep(feats_dep)
        return (self._apply_mask(feats_ir, keep_ir), self._apply_mask(feats_dep, keep_dep),
                keep_ir, keep_dep)


class EdgeAttnGate(nn.Module):
    """Step4 MEGA —— 多模态边缘引导注意力。

    E_m：模型内**固定 Sobel** 生成（零可训练参数）——RGB/IR/Depth 各自边缘幅值；
    A_m = σ(Conv([F, E_rgb, E_ir, E_dep]))：位置级注意力门（每模态一个）；
    残差保底门控：F' = F * (1 + α·Σ_m W_m·A_m)
      α：可学标量（初始 0 → 起步等价于纯 F，防无边缘区被抑制）
      W_m：每模态可学权重（初始均分）—— 位置级 × 模态级
    只放 P4/P5 注入后的特征上；边缘图在特征分辨率最近邻插值（不抹平边缘）。
    """

    def __init__(self, c_in: int, n_mod: int = 3, alpha_max: float = 1.0):
        super().__init__()
        self.att = nn.Conv2d(c_in + n_mod, n_mod, 1)
        self.alpha_raw = nn.Parameter(torch.zeros(1))        # tanh 后有界，初始严格恒等
        self.w_logits = nn.Parameter(torch.zeros(n_mod))     # softmax 后非负且和为1
        self.alpha_max = float(alpha_max)
        k = torch.tensor([[-1.0, 0, 1], [-2.0, 0, 2], [-1.0, 0, 1]]).view(1, 1, 3, 3)
        self.register_buffer("_sobel_x", k)
        self.register_buffer("_sobel_y", k.transpose(2, 3))

    def edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → 边缘幅值图 (B,1,H,W)。固定 Sobel，无参数。"""
        g = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(g, self._sobel_x.to(g.dtype), padding=1)
        ey = F.conv2d(g, self._sobel_y.to(g.dtype), padding=1)
        return (ex.abs() + ey.abs()).clamp_max(1.0)

    def forward(self, feat: torch.Tensor, edge_imgs):
        """feat: (B,C,H,W)；edge_imgs: 输入分辨率边缘源（rgb/ir/depth 各一）。"""
        es = [F.interpolate(self.edge_map(e), size=feat.shape[-2:], mode="nearest")
              for e in edge_imgs]
        A = torch.sigmoid(self.att(torch.cat([feat, *es], 1)))       # (B,3,H,W)
        # 门控系数统一到 feat 的 dtype/device：AMP 下不把主流通路抬成 fp32，
        # 同时保证 alpha=0 时 1+0=1、feat*1.0 逐位恒等。
        weights = torch.softmax(self.w_logits, dim=0).to(feat.dtype).view(1, -1, 1, 1)
        alpha = (self.alpha_max * torch.tanh(self.alpha_raw)).to(feat.dtype)
        return feat * (1.0 + alpha * (A * weights).sum(dim=1, keepdim=True))


class AuxHead(nn.Module):
    """Step3 每模态辅助头（P4 分辨率中心分类）：
    从辅助流特征预测中心 heatmap（nc 类），GT 框中心像素为 1（one-hot），
    用 BCE 训练 —— 逼迫"该模态被真正用上"（防止被 RGB 主流淹没）。
    输入只来自 AuxStream → 梯度天然不经过主流 backbone（隔离）。"""

    def __init__(self, c_in: int, nc: int):
        super().__init__()
        # BCEWithLogits/CenterNet focal loss 要求末层是未经 BN/激活的原始 logits。
        self.conv = nn.Sequential(Conv(c_in, 64, 3), nn.Conv2d(64, nc, 1))
        # 稀疏中心热图的先验正样本率约 1%，避免初始背景损失淹没中心监督。
        nn.init.constant_(self.conv[-1].bias, -4.59511985013459)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)          # (B,nc,H,W) logits


# ============================================================
# 核心：零初始化残差注入器（挂在第 4/6/10 层的 forward hook 上）
# ============================================================

class FusionInjector(nn.Module):
    """零初始化残差注入器 —— 注册为预训练层的 forward hook。

        out = F + tanh(gamma) · Σ_m g_m · proj_m(aux_m)   # gamma 初始 0 → 严格恒等
        g_m = 1 + tanh(conv([F, aux_m, mask_m]))          # 逐模态 × 逐位置门（初始 ≡ 1）
        out = mega(out, edges)                            # MEGA alpha 初始 0 → 恒等（可选）

    * **逐模态**：IR 与 Depth 各自一条 1×1 投影 + 独立门（不再 concat 后共用一条通路）；
    * **逐位置**：门是空间分辨率输出 (B,1,H,W)，能表达"这块可信、那块不可信"，
      初始恒等于 1 —— 开启门控不改变初始行为；
    * **可信度先验**：Depth 有效性掩码（输入第 2 通道）直接作为门的输入之一，
      无效区自然被压低，不必让网络自己猜；
    * `gamma`：全局强度，初始 0 → 与官方 YOLO11s 逐位一致；`set_enabled(False)` 可整体旁路；
    * hook 挂在**预训练层本身**上，Sequential 结构与 state_dict 键名保持不变
      （因此 COCO 权重可原样继承）。
    """

    def __init__(self, c_main: int, c_aux: int, n_mod: int = 2, mega: bool = False,
                 hidden: int = 32, gamma_init: float = 0.0, per_modality_gate: bool = True,
                 alpha_max: float = 1.0):
        super().__init__()
        self.n_mod = int(n_mod)
        self.per_modality_gate = bool(per_modality_gate)
        self.proj = nn.ModuleList(Conv(c_aux, c_main, 1, 1) for _ in range(self.n_mod))
        # 注：1×1 Conv 在 concat 输入上等价于两个分块投影之和，因此
        # per_modality_gate=False 时本注入器与"concat + 单投影"数学等价（便于消融）。
        if self.per_modality_gate:
            self.gate = nn.ModuleList(
                nn.Sequential(Conv(c_main + c_aux + 1, hidden, 1), nn.Conv2d(hidden, 1, 1))
                for _ in range(self.n_mod))
            for g in self.gate:                  # 末层零初始化 → 门初始 ≡ 1（恒等）
                nn.init.zeros_(g[-1].weight)
                nn.init.zeros_(g[-1].bias)
        self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))
        self.mega = EdgeAttnGate(c_main, alpha_max=alpha_max) if mega else None
        self._aux = None
        self._mask = None
        self._edges = None
        self._enabled = True                     # 融合课程：训练早期整体旁路

    def set_context(self, aux_list, edges=None, mask=None):
        """aux_list: 每模态辅助特征 [aux_ir, aux_dep]；mask: Depth 有效性掩码 (B,1,H,W)。"""
        self._aux = aux_list
        self._edges = edges
        self._mask = mask

    def set_enabled(self, flag: bool):
        """融合课程开关：False 时 hook 直接返回原输出（等价 γ=0，但连投影都不算）。"""
        self._enabled = bool(flag)

    def hook(self, _module, _inputs, output):
        """forward hook：返回改写后的输出（nn.Module 会用返回值替换该层输出）。"""
        if not self._enabled or self._aux is None:    # 旁路 / 纯 RGB → 恒等
            return output
        gate = torch.tanh(self.gamma).to(dtype=output.dtype, device=output.device)
        add = None
        for m, aux in enumerate(self._aux):
            if aux is None:
                continue
            p = self.proj[m](aux)
            if self.per_modality_gate:
                mask = self._mask
                if mask is None:
                    mask = torch.ones_like(p[:, :1])
                elif mask.shape[-2:] != p.shape[-2:]:
                    mask = F.interpolate(mask, size=p.shape[-2:], mode="nearest")
                g = 1.0 + torch.tanh(self.gate[m](torch.cat([output, aux, mask], 1)))
                p = p * g
            add = p if add is None else add + p
        if add is None:
            return output
        out = output + gate * add
        if self.mega is not None and self._edges is not None:
            out = self.mega(out, self._edges)
        return out


def _cls_name_index(tgt_names, src_names):
    """按类别名（小写去空白）把目标类映射到源类下标；无匹配为 -1。"""
    if not tgt_names or not src_names:
        return None
    norm = lambda s: str(s).strip().lower()                      # noqa: E731
    src = {norm(v): int(k) for k, v in dict(src_names).items()}
    return [src.get(norm(n), -1) for n in tgt_names]


# ============================================================
# 主模型
# ============================================================

class Experiment1Model(nn.Module):
    """
    三模态检测模型：官方 YOLO11s 全量主干 + 第 4/6/10 层零初始化残差注入。

    weights    : COCO 预训练权重（默认 yolo11s.pt，整网迁移）
    nc         : 检测类别数（赛题 12；本地 VDT 用 --class-num 45）
    aux_ch     : 辅助流各层通道 (32,48,64)
    dropout_p  : 模态 dropout 概率（训练时随机置零 IR/Depth 辅助特征）
    mega       : Step4 MEGA 边缘引导注意力（P4/P5；alpha 初始 0 → 恒等）
    aux_heads  : Step3 每模态辅助头（中心分类）
    cls_remap  : Detect 类别输出层按类别名从 COCO 迁移同名行（默认开）
    class_names: 类别名列表（None → nc==12 时用赛题 12 类，否则不迁移）

    结构说明（与 `common/train_loop._build_optimizer` 的约定）：
      * `self.backbone`  = 官方 DetectionModel（0..23 完整 Sequential，命名沿用
        `backbone.` 前缀，使 `backbone_lr_mult` 分组学习率继续生效）；
      * `model.model`    = `self.backbone.model`（Sequential 本体，供 v8DetectionLoss
        取 `model.model[-1]` = Detect）；
      * 注入器/辅助流/辅助头在 `backbone.` 之外 → 新模块用全量学习率。
    """

    def __init__(self, weights: str = "yolo11s.pt", nc: int = 12,
                 aux_ch=(32, 48, 64), dropout_p: float = 0.0,
                 mega: bool = True,      # Step4 MEGA 边缘引导注意力(P4/P5)
                 aux_heads: bool = True,  # Step3 每模态辅助头(中心分类)
                 aux_depth_head: bool = True,   # 稀疏距离辅助头（深度自监督）
                 dist_head_src: str = "ir",     # 距离头挂在哪个流：ir（默认，跨模态）| dep
                 per_modality_gate: bool = True,  # 逐模态 × 逐位置门控 + 掩码先验
                 gamma_init: float = 0.0,         # γ 初始值（0 = 严格恒等）
                 cls_remap: bool = True,
                 class_names=None,
                 ):
        super().__init__()
        # P2-16: 权重解析为绝对路径（离线环境禁止联网下载兜底）
        weights = str(MC.resolve_pretrained_weights(weights))
        base = YOLO(weights).model                       # vendor 加载（含全部 COCO 权重）
        # ---- 完整保留官方 Sequential：backbone(0-10) + 预训练 neck(11-22) + Detect(23) ----
        self.backbone = base
        for p in self.backbone.parameters():
            p.requires_grad_(True)                       # 预训练参数可能被冻结，显式解冻

        # ---- Detect：只适配类别输出层（回归分支/共享层/neck/backbone 权重全保留）----
        if class_names is None and int(nc) == MC.CLASS_NUM:
            class_names = list(MC.CLASS_NAMES)
        self.class_names = list(class_names) if class_names else [str(i) for i in range(int(nc))]
        if len(self.class_names) != int(nc):
            raise ValueError(f"class_names 长度({len(self.class_names)}) != nc({nc})")
        detect = self.backbone.model[-1]
        if not isinstance(detect, Detect):
            raise TypeError(f"最后一个模块应为 Detect，实际为 {type(detect).__name__}")
        self._adapt_detect_cls(detect, int(nc), cls_remap=cls_remap)
        self.backbone.names = {i: n for i, n in enumerate(self.class_names)}
        if hasattr(self.backbone, "nc"):
            self.backbone.nc = int(nc)

        # ---- 探测 backbone P3/P4/P5 通道（与输入尺寸无关；避免硬编码随规格漂移）----
        c3, c4, c5 = self._probe_channels()

        # ---- 辅助流 + 模态 dropout（默认关：门控本就小，dropout 只会更小）----
        self.aux_ir = AuxStream(aux_ch)                  # IR 辅助流 (1ch 输入)
        self.aux_dep = AuxStream(aux_ch, in_ch=2)        # Depth 辅助流 (Step1: 2ch 输入)
        self.modal_drop = ModalDropout(dropout_p)

        # ---- 零初始化残差注入：第 4/6/10 层输出后（gamma=0 → 恒等）----
        ca1, ca2, ca3 = aux_ch
        inj_kw = dict(n_mod=2, hidden=32, gamma_init=float(gamma_init),
                      per_modality_gate=bool(per_modality_gate))
        self.inject3 = FusionInjector(c3, ca1, mega=False, **inj_kw)
        self.inject4 = FusionInjector(c4, ca2, mega=bool(mega), **inj_kw)
        self.inject5 = FusionInjector(c5, ca3, mega=bool(mega), **inj_kw)
        self.mega_enabled = bool(mega)
        for idx, inj in ((_IDX_P3, self.inject3), (_IDX_P4, self.inject4), (_IDX_P5, self.inject5)):
            self.backbone.model[idx].register_forward_hook(inj.hook)

        # ---- Step3: 每模态辅助头（P4 分辨率中心分类；梯度只回传辅助流）----
        self.aux_enabled = bool(aux_heads)
        self._aux_logits: dict = {}
        self._aux_keep: dict = {}
        if self.aux_enabled:
            self.aux_ir_head = AuxHead(ca2, int(nc))
            self.aux_dep_head = AuxHead(ca2, int(nc))

        # ---- Step3b: 稀疏距离头（GT 来自深度图，免费监督；P4 分辨率）----
        # 挂在 **IR 流**（dist_head_src="ir"）而非 depth 流：depth 是模型输入，
        # 头若挂在 depth 流上只需"抄输入"就能把 loss 压到 log 空间 ~2%（实测），
        # 任务过于 trivial，学不到对检测有用的东西；挂在 IR 上则必须跨模态推断距离，
        # 监督才会真正塑造 IR 流特征（再经融合注入主流）。
        self.dist_enabled = bool(aux_depth_head)
        self.dist_head_src = str(dist_head_src)
        self._aux_dist_logits = None
        if self.dist_enabled:
            self.aux_dep_dist_head = nn.Sequential(Conv(ca2, 64, 3), nn.Conv2d(64, 1, 1))
            nn.init.constant_(self.aux_dep_dist_head[-1].bias, math.log(1.5))  # 初始 ~1.5m

        # v8DetectionLoss 兼容面（自定义模型复用 ultralytics 损失所需）
        from ultralytics.cfg import DEFAULT_CFG          # hyp: box/cls/dfl 等
        self.args = DEFAULT_CFG
        # 官方权重经 load_checkpoint 载入后处于 eval 态；统一为 train 态，
        # 避免"模型 self.training=True 而 backbone 在 eval"的半态（train/predict 入口会各自切换）。
        self.train()

    def set_fusion_enabled(self, flag: bool) -> None:
        """融合课程：False = 三个注入点整体旁路（等价纯 RGB 前向，连投影都不计算）。"""
        for inj in (self.inject3, self.inject4, self.inject5):
            inj.set_enabled(flag)

    # ---------------- 结构适配 ----------------

    def _adapt_detect_cls(self, detect: Detect, nc: int, cls_remap: bool = True) -> None:
        """只替换 Detect 的**类别输出层**（cv3 末层 1×1 conv），其余权重原样保留。

        * 新卷积：`Conv2d(c3, nc, 1)`；先按官方 `bias_init` 的类别先验置偏置；
        * cls_remap=True 时，对与 COCO **同名**的类别（person/bicycle/car/boat 等）
          从旧 80 类输出层逐行迁移 weight/bias，其余类别保持随机初始化；
        * 不调用官方 `bias_init()`（它会把回归分支 cv2 的 bias 覆写成 2.0，
          违反"回归分支 COCO 权重全部保留"）。
        """
        old_nc = int(detect.nc)
        src_names = getattr(self.backbone, "names", None)
        idx_map = _cls_name_index(self.class_names, src_names) if cls_remap else None
        for attr in ("cv3", "one2one_cv3"):              # one2one 仅 end2end 模型才有
            head = getattr(detect, attr, None)
            if head is None:
                continue
            for i, seq in enumerate(head):
                old = seq[-1]
                if not isinstance(old, nn.Conv2d):
                    raise TypeError(f"{attr}[{i}] 末层应为 Conv2d，实际为 {type(old).__name__}")
                new = nn.Conv2d(old.in_channels, nc, 1, 1, bias=True)
                # 类别先验偏置（与官方 Detect.bias_init 的 cls 分支同式）
                prior = math.log(5 / nc / (640 / float(detect.stride[i])) ** 2)
                new.bias.data.fill_(prior)
                if idx_map is not None:
                    for j, k in enumerate(idx_map):
                        if k < 0 or k >= old.out_channels:
                            continue
                        new.weight.data[j].copy_(old.weight.data[k])
                        new.bias.data[j].copy_(old.bias.data[k])
                seq[-1] = new
        detect.nc = nc
        detect.no = nc + detect.reg_max * 4
        matched = [n for n, k in zip(self.class_names, idx_map) if k >= 0] if idx_map else []
        if matched:
            print(f"[实验模型1] Detect 类别输出层适配: {old_nc} → {nc} 类；"
                  f"COCO 同名类逐行迁移 {len(matched)}/{nc}: {matched}")
        else:
            print(f"[实验模型1] Detect 类别输出层适配: {old_nc} → {nc} 类"
                  f"（无同名类可迁移，随机初始化 + 官方类别先验偏置）")

    def _probe_channels(self):
        """探测第 4/6/10 层输出通道（64×64 dummy 前向；eval 模式避免污染 BN 统计量）。"""
        store: dict = {}
        trunk = self.backbone.model[: _IDX_P5 + 1]        # 0..10 全为单输入层，可当 Sequential
        handles = []
        for idx in (_IDX_P3, _IDX_P4, _IDX_P5):
            handles.append(self.backbone.model[idx].register_forward_hook(
                lambda _m, _i, out, idx=idx: store.__setitem__(idx, out.shape[1])))
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                _ = trunk(torch.zeros(1, 3, 64, 64))
        finally:
            for h in handles:
                h.remove()
            self.backbone.train(was_training)
        return store[_IDX_P3], store[_IDX_P4], store[_IDX_P5]

    # ---------------- 访问器 ----------------

    @property
    def model(self) -> nn.Sequential:
        """官方 Sequential 本体（v8DetectionLoss 依赖 `model.model[-1]` = Detect）。"""
        return self.backbone.model

    @property
    def detect(self) -> Detect:
        """官方 Detect（类别输出层已适配 nc）。"""
        return self.backbone.model[-1]

    # ---------------- 前向 ----------------

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor):
        """rgb(B,3,H,W); ir(B,1,H,W); depth(B,2,H,W) → Detect 输出（训练态 3 尺度 dict）。

        主流路径与官方 YOLO11s 完全一致：本方法只负责准备辅助特征并把它们交给
        第 4/6/10 层的注入 hook，随后调用官方 DetectionModel.forward(rgb)。
        """
        f_ir = self.aux_ir(ir)
        f_dep = self.aux_dep(depth)
        f_ir, f_dep, keep_ir, keep_dep = self.modal_drop(f_ir, f_dep)

        # Step4: MEGA 边缘源；Dropout 掩码同时作用于边缘输入，杜绝被丢弃模态旁路泄漏。
        edges = [rgb, ir * keep_ir, depth * keep_dep] if self.mega_enabled else None
        dep_mask = depth[:, 1:2]                              # 深度有效性掩码（可信度先验）
        self.inject3.set_context([f_ir[0], f_dep[0]], None, dep_mask)
        self.inject4.set_context([f_ir[1], f_dep[1]], edges, dep_mask)
        self.inject5.set_context([f_ir[2], f_dep[2]], edges, dep_mask)

        # Step3: 辅助头（P4 尺度 = f_ir[1]/f_dep[1]，通道 ca2）
        if self.aux_enabled and self.training:
            self._aux_logits = {"ir": self.aux_ir_head(f_ir[1]),
                                "dep": self.aux_dep_head(f_dep[1])}
            self._aux_keep = {"ir": keep_ir, "dep": keep_dep}
        # Step3b: 稀疏距离头（P4 分辨率，log(米)；GT 来自深度图自身）
        if self.dist_enabled and self.training:
            src_feat = f_ir[1] if self.dist_head_src == "ir" else f_dep[1]
            self._aux_dist_logits = self.aux_dep_dist_head(src_feat)

        try:
            return self.backbone(rgb)                     # 官方完整前向 0..23
        finally:
            for inj in (self.inject3, self.inject4, self.inject5):
                inj.set_context(None, None, None)         # 不跨前向持有运行期张量


def build_experiment1(**kwargs) -> Experiment1Model:
    """便捷构建入口（参数见 Experiment1Model）。"""
    return Experiment1Model(**kwargs)


# ============================================================
# 自检（dummy 前向 + 与官方 YOLO11s 的逐位等价性 + 梯度/EMA/优化器分组）
# ============================================================

def _shape_of(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    if isinstance(v, (list, tuple)):
        return [_shape_of(x) for x in v]
    return type(v).__name__


def _selfcheck(imgsz: int = 320, nc: int = None):
    import models_config as MC
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import ModelEMA

    nc = int(nc or MC.CLASS_NUM)
    weights = str(MC.resolve_pretrained_weights("yolo11s.pt"))
    print(f"=== 实验模型1 自检 (weights={weights}, nc={nc}, imgsz={imgsz}) ===")

    model = build_experiment1(weights=weights, nc=nc)
    ref = YOLO(weights).model                             # 官方 YOLO11s（nc=80）

    n_all = sum(p.numel() for p in model.parameters())
    n_pre = sum(p.numel() for p in model.backbone.parameters())
    n_new = n_all - n_pre
    print(f"参数量: 总 {n_all / 1e6:.2f}M = 预训练 {n_pre / 1e6:.2f}M + 新增 {n_new / 1e6:.2f}M")
    print(f"backbone P3/P4/P5 通道: "
          f"{model.inject3.proj[0].conv.out_channels}/{model.inject4.proj[0].conv.out_channels}/"
          f"{model.inject5.proj[0].conv.out_channels}；Detect nc={model.detect.nc} "
          f"stride={model.detect.stride.tolist()}；"
          f"逐模态门控={model.inject3.per_modality_gate}，距离头={model.dist_enabled}")

    # ---- 1) COCO 权重继承情况（应只有 cv3 末层 6 个张量不同）----
    # 本模型的预训练部分挂在 `self.backbone`（DetectionModel）下，
    # 键名 backbone.model.<i>.* == 官方的 model.<i>.*
    ours = {k.replace("backbone.model.", "model.", 1) if k.startswith("backbone.model.") else k: v
            for k, v in model.state_dict().items()}
    theirs = ref.state_dict()
    kept = [k for k in theirs if k in ours and ours[k].shape == theirs[k].shape]
    same = [k for k in kept if torch.equal(ours[k], theirs[k])]
    lost = [k for k in theirs if k not in kept]
    print(f"COCO 权重继承: {len(same)}/{len(theirs)} 个张量逐位相同；"
          f"未继承 {len(lost)} 个 -> {lost[:6]}{' ...' if len(lost) > 6 else ''}")
    assert all("cv3" in k and k.endswith(("weight", "bias")) for k in lost) and len(lost) == 6, \
        f"除 Detect 类别输出层外，不应有任何 COCO 权重被替换: {lost}"

    # ---- 2) 等价性：gamma=0 / alpha=0 时 RGB 前向与官方逐位一致 ----
    x = torch.randn(1, 3, imgsz, imgsz)
    ir = torch.rand(1, 1, imgsz, imgsz)
    dep = torch.rand(1, 2, imgsz, imgsz)
    neck_idx = (16, 19, 22)                               # Detect 的三个输入层

    def capture(m):
        store = {}
        for i in neck_idx:
            m.model[i].register_forward_hook(
                lambda _mo, _in, out, i=i: store.__setitem__(i, out.detach()))
        return store

    ours_feat, ref_feat = capture(model), capture(ref)
    model.eval(), ref.eval()
    with torch.no_grad():
        out, ref_out = model(x, ir, dep), ref(x)
    assert isinstance(out, tuple) and isinstance(ref_out, tuple), "eval 态应返回 (raw, preds)"
    feat_ok = all(torch.equal(ours_feat[i], ref_feat[i]) for i in neck_idx)
    box_ok = torch.equal(out[1]["boxes"], ref_out[1]["boxes"])
    print(f"等价性: neck(16/19/22) 特征逐位相同={feat_ok}；Detect 回归分支输出逐位相同={box_ok}")
    print(f"        raw 形状 ours={tuple(out[0].shape)}（4+{nc}）ref={tuple(ref_out[0].shape)}（4+80）")
    assert feat_ok and box_ok, "gamma=0/alpha=0 时主流前向必须与官方 YOLO11s 逐位一致"

    # ---- 3) 注入是否真的接通（打开 gamma 后主流前向必须变化）----
    with torch.no_grad():
        for inj in (model.inject3, model.inject4, model.inject5):
            inj.gamma.fill_(0.5)
        out_g = model(x, ir, dep)
    changed = not torch.equal(out_g[1]["boxes"], ref_out[1]["boxes"])
    print(f"注入接通: gamma=0.5 后主流输出变化={changed}")
    assert changed, "残差注入未接通主流前向"
    with torch.no_grad():
        for inj in (model.inject3, model.inject4, model.inject5):
            inj.gamma.zero_()

    # ---- 4) 梯度：gamma 起步即有梯度；gamma≠0 后辅助流/proj 才被激活（零初始化残差的正常行为）----
    model.train()
    model.zero_grad(set_to_none=True)
    preds = model(x, ir, dep)
    (preds["boxes"].float().sum() + preds["scores"].float().sum()).backward()
    g0 = {"inject3.gamma": model.inject3.gamma.grad is not None,
          "backbone.0.conv.weight": model.backbone.model[0].conv.weight.grad is not None,
          "backbone.23.cv2.0.2.weight": model.detect.cv2[0][-1].weight.grad is not None,
          "backbone.23.cv3.0.2.weight": model.detect.cv3[0][-1].weight.grad is not None}
    print(f"梯度(gamma=0): {g0}  ← 注入分支 proj 梯度为 0 属预期（γ=0 时 d/dW=γ·…）")
    assert all(g0.values()), f"gamma=0 时主流/检测头/γ 必须都有梯度: {g0}"

    # 模拟第一次优化器更新（γ 离开 0）后，辅助流与 proj 必须被激活
    with torch.no_grad():
        for inj in (model.inject3, model.inject4, model.inject5):
            inj.gamma.fill_(1e-2)
    model.zero_grad(set_to_none=True)
    preds = model(x, ir, dep)
    (preds["boxes"].float().sum() + preds["scores"].float().sum()).backward()
    g1 = {"inject3.proj[0].conv.weight": model.inject3.proj[0].conv.weight.grad is not None,
          "inject3.gate[0][0].conv.weight": model.inject3.gate[0][0].conv.weight.grad is not None,
          "aux_ir.s1.0.conv.weight": model.aux_ir.s1[0].conv.weight.grad is not None,
          "mega4.att.weight": model.inject4.mega.att.weight.grad is not None}
    print(f"梯度(gamma=1e-2): {g1}")
    assert all(g1.values()), f"γ≠0 后辅助流/投影/逐位置门/MEGA 必须都有梯度: {g1}"
    # 辅助头损失梯度（train_loop 的 Step3 λ 项）：需重新前向取新的计算图
    model.zero_grad(set_to_none=True)
    _ = model(x, ir, dep)
    aux_loss = sum(v.float().sum() for v in model._aux_logits.values())
    aux_loss = aux_loss + model._aux_dist_logits.float().sum()
    aux_loss.backward()
    g2 = {"aux_ir_head.conv.0.conv.weight": model.aux_ir_head.conv[0].conv.weight.grad is not None,
          "aux_dep.s4.0.conv.weight": model.aux_dep.s4[0].conv.weight.grad is not None,
          "aux_dep_dist_head.0.conv.weight": model.aux_dep_dist_head[0].conv.weight.grad is not None,
          "aux_ir.s1.0.conv.weight": model.aux_ir.s1[0].conv.weight.grad is not None}
    print(f"梯度(辅助头/距离头损失): {g2}")
    assert all(g2.values()), f"辅助头/距离头损失必须回传辅助流: {g2}"
    with torch.no_grad():
        for inj in (model.inject3, model.inject4, model.inject5):
            inj.gamma.zero_()

    # ---- 5) EMA（common/train_loop 会用 ModelEMA 保存 best/last）----
    # train_loop 在训练前创建 EMA；此处先清掉上一段前向留下的辅助 logits 计算图，
    # 模拟同样的状态（非叶子张量不支持 deepcopy）。
    model._aux_logits, model._aux_keep, model._aux_dist_logits = {}, {}, None
    ema = ModelEMA(model)
    ema.update(model)
    sd = ema.ema.state_dict()
    print(f"EMA: 张量数={len(sd)}，键与训练模型一致={set(sd) == set(model.state_dict())}")
    assert set(sd) == set(model.state_dict()), "EMA 状态字典键必须与训练模型一致"

    # ---- 6) 优化器分组（train_loop 按 `backbone.` 前缀给预训练部分降学习率）----
    from common import train_loop as TL
    opt = TL._build_optimizer(model, MC.EXPERIMENT1.hyper)
    print("优化器分组: " + ", ".join(
        f"{gp.get('group_name')}={len(gp['params'])}params@{gp['target_lr']:.1e}"
        for gp in opt.param_groups))

    # ---- 7) 训练态输出形状 + 辅助头 + 注入幅度体检（2.5）----
    with torch.no_grad():
        tr_out = model(x, ir, dep)
    print(f"训练态 Detect 输出: { {k: _shape_of(v) for k, v in tr_out.items()} }")
    print(f"辅助头 logits: { {k: _shape_of(v) for k, v in model._aux_logits.items()} }")
    print(f"距离头 logits: {_shape_of(model._aux_dist_logits)}")

    # 注入幅度体检：γ=1 时 ‖注入‖ / ‖主流特征‖（若远小于 1，说明"温柔"是结构性的）
    model.eval()
    ratios = {}
    for name, idx in (("P3", _IDX_P3), ("P4", _IDX_P4), ("P5", _IDX_P5)):
        store = {}
        h = model.backbone.model[idx].register_forward_hook(
            lambda _m, _i, out, s=store: s.__setitem__("feat", out.detach()))
        with torch.no_grad():
            inj = getattr(model, {"P3": "inject3", "P4": "inject4", "P5": "inject5"}[name])
            inj.set_enabled(False)
            model(x, ir, dep)
            base_feat = store["feat"]
            inj.set_enabled(True)
            inj.gamma.fill_(math.atanh(0.99))            # tanh(γ)=0.99
            model(x, ir, dep)
            with_feat = store["feat"]
        h.remove()
        num = (with_feat - base_feat).float().norm()
        den = base_feat.float().norm().clamp_min(1e-6)
        ratios[name] = float(num / den)
        with torch.no_grad():
            inj.gamma.zero_()
        inj.set_enabled(True)
    print(f"注入幅度体检(γ=0.99 时 ‖ΔF‖/‖F‖): "
          + "  ".join(f"{k}={v:.4f}" for k, v in ratios.items()))
    print("EXP1_SELFCHECK_OK")


if __name__ == "__main__":
    _selfcheck()
