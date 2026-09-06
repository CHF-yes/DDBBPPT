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

from ultralytics import YOLO                                     # vendor 版
from ultralytics.nn.modules import Conv, C2f, Concat            # 复用官方组件
from ultralytics.nn.modules.head import Detect                  # 复用官方检测头

# yolo11 backbone 输出 P3/P4/P5 的模块索引（yolo11.yaml 结构，n/s/m/l/x 一致）
_IDX_P3, _IDX_P4, _IDX_P5 = 4, 6, 9


class AuxStream(nn.Module):
    """轻量单通道辅助流：IR/Depth 各一条，输出与主流 P3(1/8)/P4(1/16)/P5(1/32) 同分辨率。
    每级恰好下采样 1 次：1/2 → 1/4 → 1/8(f3) → 1/16(f4) → 1/32(f5)。
    ch = (c_f3, c_f4, c_f5) 为三个输出通道数。"""

    def __init__(self, ch=(32, 48, 64)):
        super().__init__()
        c1, c2, c3 = ch
        self.s1 = nn.Sequential(Conv(1, 16, 3, 2), C2f(16, 16, n=1))
        self.s2 = nn.Sequential(Conv(16, 24, 3, 2), C2f(24, 24, n=1))
        self.s3 = nn.Sequential(Conv(24, c1, 3, 2), C2f(c1, c1, n=1))
        self.s4 = nn.Sequential(Conv(c1, c2, 3, 2), C2f(c2, c2, n=1))
        self.s5 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, n=1))

    def forward(self, x: torch.Tensor):
        """x: (B,1,H,W) → (f3(B,c1,H/8,W/8), f4(B,c2,H/16,W/16), f5(B,c3,H/32,W/32))"""
        f3 = self.s3(self.s2(self.s1(x)))
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        return f3, f4, f5


class ModalDropout(nn.Module):
    """模态 dropout：训练时按概率把某一辅助路特征整体置零（赛题鲁棒性：某模态失效时不崩）。
    只作用于辅助流（RGB 主流不作丢弃），且仅在 self.training 时生效。"""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def forward(self, feats):
        if not self.training or self.p <= 0.0:
            return feats
        return [torch.zeros_like(f) if torch.rand(1).item() < self.p else f for f in feats]


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
                 dropout_p: float = 0.2):
        super().__init__()
        base = YOLO(weights).model                       # vendor 加载（含 COCO 权重）
        self.backbone = base.model[:_IDX_P5 + 1]         # 0..9 主干(含 SPPF)

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

        self.aux_ir = AuxStream(aux_ch)                  # IR 辅助流
        self.aux_dep = AuxStream(aux_ch)                 # Depth 辅助流
        self.modal_drop = ModalDropout(dropout_p)

        ca1, ca2, ca3 = aux_ch
        self.fuse3 = FusionBlock(c3, ca1 * 2, c3)         # IR+D 两路 concat 后 = aux*2
        self.fuse4 = FusionBlock(c4, ca2 * 2, c4)
        self.fuse5 = FusionBlock(c5, ca3 * 2, c5)
        self.pan = SimplePAN(c3, c4, c5)

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
        """rgb(B,3,H,W); ir/depth(B,1,H,W) → Detect 输出（训练格式 3 个尺度）。"""
        self._feats.clear()
        _ = self.backbone(rgb)
        p3, p4, p5 = self._feats["p3"], self._feats["p4"], self._feats["p5"]

        f_ir = self.aux_ir(ir)
        f_dep = self.aux_dep(depth)
        aux = [torch.cat([a, b], 1) for a, b in zip(f_ir, f_dep)]
        aux = self.modal_drop(aux)

        f3 = self.fuse3(p3, aux[0])
        f4 = self.fuse4(p4, aux[1])
        f5 = self.fuse5(p5, aux[2])
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
                torch.zeros(1, 1, 640, 640))
    print("train-mode detect output:", {k: _shape_of(v) for k, v in out.items()})
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 640, 640),
              torch.zeros(1, 1, 640, 640),
              torch.zeros(1, 1, 640, 640))
    print("eval-mode decode shape:", _shape_of(y))
    print("EXP1_FORWARD_OK")
