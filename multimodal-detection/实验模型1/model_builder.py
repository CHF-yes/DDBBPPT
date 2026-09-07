# -*- coding: utf-8 -*-
"""
实验模型1 —— 多模态「RGB 主流 + 轻量辅助流 + 分级融合」模型骨架。

架构（见 code/README.md 架构图）：
  RGB(3) ──▶ 主流 backbone(yolo11s, COCO 预训练) ──▶ P3/P4/P5
  IR(1) ─┐
  Depth(1)┴─▶ 轻量辅助流(每模态) ──▶ P3'/P4'/P5' 辅助特征
  融合：P3/P4/P5 各 FusionBlock(concat+1×1, NiN 级) [+ 模态 dropout] ──▶ SimplePAN ──▶ Detect

要点：
  * 使用 **vendor 源码**（code/vendor/ultralytics, 8.4.138）：本模块所有 ultralytics 组件
    均从 vendor 导入（入口最先注入 sys.path）。
  * 主流完整加载 COCO 预训练权重 yolo11s.pt（赛题允许），只加融合点、不解构主干。
  * 辅助流轻量：只提取与 P3/P4/P5 同分辨率的特征；模态 dropout 在训练时随机置零一路。
  * Detect 复用官方实现（nc=12，stride 按 640 输入 [8,16,32]）；训练/验证主循环见 README。
"""

from __future__ import annotations

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

from ultralytics import YOLO                                     # vendor 版
from ultralytics.nn.modules import Conv, C2f, Concat            # 复用官方组件
from ultralytics.nn.modules.head import Detect                  # 复用官方检测头

# yolo11 backbone 输出 P3/P4/P5 的模块索引（yolo11.yaml：0-P1 1-P2 3-P3 5-P4 7-P5 9-SPPF 10-C2PSA）
# 官方 neck 的 P5 输入是 **C2PSA(10)** 而非 SPPF(9)；backbone 切片须含 C2PSA，P5 从 10 取。
_IDX_P3, _IDX_P4, _IDX_P5 = 4, 6, 10


class AuxStream(nn.Module):
    """轻量单通道辅助流：IR/Depth 各一条，输出与主流 P3(1/8)/P4(1/16)/P5(1/32) 同分辨率。
    每级恰好下采样 1 次：1/2 → 1/4 → 1/8(f3) → 1/16(f4) → 1/32(f5)。
    ch = (c_f3, c_f4, c_f5) 为三个输出通道数。
    in_ch：输入通道（IR=1；Step1 后 Depth=2：距离+有效掩码）。"""

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
    """逐模态独立 dropout（Step2）：IR 与 Depth **各自独立**按概率置零（concat 前），
    训练覆盖「仅 IR 失效 / 仅 Depth 失效 / 双失效 / 齐全」四种态——
    对应赛题"单模态质量差时性能不崩"的鲁棒性要求。
    只作用于辅助流（RGB 主流不丢）；仅 self.training 时生效。"""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def _drop(self, feats):
        return [torch.zeros_like(f) if torch.rand(1).item() < self.p else f for f in feats]

    def forward(self, feats_ir, feats_dep):
        if not self.training or self.p <= 0.0:
            return feats_ir, feats_dep
        return self._drop(feats_ir), self._drop(feats_dep)


class FusionBlock(nn.Module):
    """轻量融合（NiN 级）：concat(主流, 辅助流) → 1×1 卷积。
    c_main/c_aux 输入通道；c_out 输出通道（=主流通道，保持 neck 输入不变）。"""

    def __init__(self, c_main: int, c_aux: int, c_out: int):
        super().__init__()
        self.concat = Concat(dimension=1)
        self.conv = Conv(c_main + c_aux, c_out, 1, 1)

    def forward(self, main_feat: torch.Tensor, aux_feat: torch.Tensor):
        return self.conv(self.concat([main_feat, aux_feat]))


class SimplePAN(nn.Module):
    """轻量 PAN-FPN：融合后的 (P3,P4,P5) 先自顶向下(FPN)，再自底向上(PAN)。
    通道自洽：cv5t 保 c5；f4 输入 c4+c5；f3 输入 c3+c4；p4 输入 c4+c3；p5 输入 c5+c4。"""

    def __init__(self, c3: int, c4: int, c5: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # FPN: 深层语义上传
        self.cv5t = Conv(c5, c5, 1, 1)
        self.f4 = C2f(c4 + c5, c4, n=2)
        self.f3 = C2f(c3 + c4, c3, n=2)
        # PAN: 浅层定位下传
        self.down3 = Conv(c3, c3, 3, 2)
        self.down4 = Conv(c4, c4, 3, 2)
        self.p4 = C2f(c4 + c3, c4, n=2)
        self.p5 = C2f(c5 + c4, c5, n=2)

    def forward(self, p3, p4, p5):
        # FPN（自顶向下）
        p5t = self.cv5t(p5)
        p4f = self.f4(torch.cat([p4, self.upsample(p5t)], 1))
        p3f = self.f3(torch.cat([p3, self.upsample(p4f)], 1))
        # PAN（自底向上）
        p4b = self.p4(torch.cat([p4f, self.down3(p3f)], 1))
        p5b = self.p5(torch.cat([p5t, self.down4(p4b)], 1))
        return p3f, p4b, p5b


class EdgeAttnGate(nn.Module):
    """Step4 MEGA —— 多模态边缘引导注意力。

    E_m：模型内**固定 Sobel** 生成（零可训练参数）——RGB/IR/Depth 各自边缘幅值；
    A_m = σ(Conv([F, E_rgb, E_ir, E_dep]))：位置级注意力门（每模态一个）；
    残差保底门控：F' = F * (1 + α·Σ_m W_m·A_m)
      α：可学标量（初始 0 → 起步等价于纯 F，防无边缘区被抑制）
      W_m：每模态可学权重（初始均分）—— 位置级 × 模态级
    只放 P4/P5 融合特征上；边缘图在特征分辨率最近邻插值（不抹平边缘）。
    """

    def __init__(self, c_in: int, n_mod: int = 3):
        super().__init__()
        self.att = nn.Conv2d(c_in + n_mod, n_mod, 1)
        self.alpha = nn.Parameter(torch.zeros(1))            # 残差保底系数
        self.w = nn.Parameter(torch.full((n_mod,), 1.0 / n_mod))
        k = torch.tensor([[-1.0, 0, 1], [-2.0, 0, 2], [-1.0, 0, 1]]).view(1, 1, 3, 3)
        self.register_buffer("_sobel_x", k)
        self.register_buffer("_sobel_y", k.transpose(2, 3))

    def edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) → 边缘幅值图 (B,1,H,W)。固定 Sobel，无参数。"""
        g = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(g, self._sobel_x, padding=1)
        ey = F.conv2d(g, self._sobel_y, padding=1)
        return (ex.abs() + ey.abs()).clamp_max(1.0)

    def forward(self, feat: torch.Tensor, edge_imgs):
        """feat: (B,C,H,W)；edge_imgs: 输入分辨率边缘源（rgb/ir/depth 各一）。"""
        es = [F.interpolate(self.edge_map(e), size=feat.shape[-2:], mode="nearest")
              for e in edge_imgs]
        A = torch.sigmoid(self.att(torch.cat([feat, *es], 1)))       # (B,3,H,W)
        A = A * self.w.view(1, -1, 1, 1)                             # 模态级×位置级
        return feat * (1.0 + self.alpha * A.sum(dim=1, keepdim=True))


class AuxHead(nn.Module):
    """Step3 每模态辅助头（P4 分辨率中心分类）：
    从辅助流特征预测中心 heatmap（nc 类），GT 框中心像素为 1（one-hot），
    用 BCE 训练 —— 逼迫"该模态被真正用上"（防止被 RGB 主流淹没）。
    输入只来自 AuxStream → 梯度天然不经过主流 backbone（隔离）。"""

    def __init__(self, c_in: int, nc: int):
        super().__init__()
        self.conv = nn.Sequential(Conv(c_in, 64, 3), Conv(64, nc, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)          # (B,nc,H,W) logits


class Experiment1Model(nn.Module):
    """
    三模态检测模型骨架。

    weights   : COCO 预训练权重（默认 yolo11s.pt，主流完整迁移）
    nc        : 检测类别数（赛题 12）
    aux_ch    : 辅助流各层通道 (32,48,64)
    fusion_ch : 主流 P3/P4/P5 通道（yolo11s 为 256/512/512；换规格须对应调整）
    dropout_p : 模态 dropout 概率（训练时随机置零 IR/Depth 辅助特征）
    """

    def __init__(self, weights: str = "yolo11s.pt", nc: int = 12,
                 aux_ch=(32, 48, 64), fusion_ch=None,
                 dropout_p: float = 0.2,
                 mega: bool = True,      # Step4 MEGA 边缘引导注意力(P4/P5)
                 aux_heads: bool = True,  # Step3 每模态辅助头(中心分类)
                 ):
        super().__init__()
        base = YOLO(weights).model                       # vendor 加载（含 COCO 权重）
        self.backbone = base.model[:_IDX_P5 + 1]         # 0..10 主干(含 SPPF + C2PSA)

        # backbone 输出 hooks：收集 P3/P4/P5
        self._feats: dict = {}
        self._hooks = [
            self.backbone[_IDX_P3].register_forward_hook(self._hook("p3")),
            self.backbone[_IDX_P4].register_forward_hook(self._hook("p4")),
            self.backbone[_IDX_P5].register_forward_hook(self._hook("p5")),
        ]
        # 探测真实 P3/P4/P5 通道（避免硬编码随规格漂移；通道数与输入尺寸无关）
        with torch.no_grad():
            _ = self.backbone(torch.zeros(1, 3, 64, 64))
        c3, c4, c5 = (self._feats[k].shape[1] for k in ("p3", "p4", "p5"))

        self.aux_ir = AuxStream(aux_ch)                  # IR 辅助流 (1ch 输入)
        self.aux_dep = AuxStream(aux_ch, in_ch=2)        # Depth 辅助流 (Step1: 2ch 输入)
        self.modal_drop = ModalDropout(dropout_p)

        ca1, ca2, ca3 = aux_ch
        self.fuse3 = FusionBlock(c3, ca1 * 2, c3)         # IR+D 两路 concat 后 = aux*2
        self.fuse4 = FusionBlock(c4, ca2 * 2, c4)
        self.fuse5 = FusionBlock(c5, ca3 * 2, c5)
        self.pan = SimplePAN(c3, c4, c5)

        # ---- Step4: MEGA（P4/P5 融合特征；模型内固定 Sobel，零参数边缘）----
        self.mega_enabled = bool(mega)
        self.mega4 = EdgeAttnGate(c4)
        self.mega5 = EdgeAttnGate(c5)

        # ---- Step3: 每模态辅助头（P4 分辨率中心分类；梯度只回传辅助流）----
        self.aux_enabled = bool(aux_heads)
        self._aux_logits: dict = {}
        if self.aux_enabled:
            self.aux_ir_head = AuxHead(ca2, nc)
            self.aux_dep_head = AuxHead(ca2, nc)

        # 官方 Detect（nc=12，非端到端；stride 按 640 输入推断的 [8,16,32]）
        self.detect = Detect(nc, reg_max=16, end2end=False, ch=(c3, c4, c5))
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.inplace = True
        self.detect.bias_init()
        self._feats.clear()

        # v8DetectionLoss 兼容面（自定义模型复用 ultralytics 损失所需）
        from ultralytics.cfg import DEFAULT_CFG          # hyp: box/cls/dfl 等
        self.args = DEFAULT_CFG
        self.model = nn.Sequential(self.detect)        # model.model[-1] 即 Detect

    def _hook(self, name: str):
        def h(_m, _inp, out):
            self._feats[name] = out
        return h

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, depth: torch.Tensor):
        """rgb(B,3,H,W); ir(B,1,H,W); depth(B,2,H,W) → Detect 输出（训练格式 3 个尺度）。
        Step3 辅助 logits 缓存到 self._aux_logits（仅训练态；主检测输出不变）。"""
        self._feats.clear()
        _ = self.backbone(rgb)
        p3, p4, p5 = self._feats["p3"], self._feats["p4"], self._feats["p5"]

        f_ir = self.aux_ir(ir)
        f_dep = self.aux_dep(depth)
        f_ir, f_dep = self.modal_drop(f_ir, f_dep)      # Step2: 逐模态独立置零
        aux = [torch.cat([a, b], 1) for a, b in zip(f_ir, f_dep)]

        f3 = self.fuse3(p3, aux[0])
        f4 = self.fuse4(p4, aux[1])
        f5 = self.fuse5(p5, aux[2])

        # Step4: MEGA 边缘引导（P4/P5；边缘图最近邻插值到特征分辨率）
        if self.mega_enabled:
            f4 = self.mega4(f4, [rgb, ir, depth])
            f5 = self.mega5(f5, [rgb, ir, depth])

        # Step3: 辅助头（P4 尺度 = f_ir[1]/f_dep[1]，通道 ca2）
        if self.aux_enabled and self.training:
            self._aux_logits = {"ir": self.aux_ir_head(f_ir[1]),
                                "dep": self.aux_dep_head(f_dep[1])}

        neck = self.pan(f3, f4, f5)
        return self.detect(list(neck))


def build_experiment1(**kwargs) -> Experiment1Model:
    """便捷构建入口（参数见 Experiment1Model）。"""
    return Experiment1Model(**kwargs)


def _shape_of(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    if isinstance(v, (list, tuple)):
        return [_shape_of(x) for x in v]
    return type(v).__name__


if __name__ == "__main__":
    # 简易自检：dummy 前向（不加载真实数据，仅验证结构/通道/输出形状）
    m = Experiment1Model()
    print("param count(M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 640, 640),
                torch.zeros(1, 1, 640, 640),
                torch.zeros(1, 2, 640, 640))          # Step1: depth 两通道
    print("train-mode detect output:", {k: _shape_of(v) for k, v in out.items()})
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 640, 640),
              torch.zeros(1, 1, 640, 640),
              torch.zeros(1, 2, 640, 640))
    print("eval-mode decode shape:", _shape_of(y))
    print("EXP1_FORWARD_OK")
