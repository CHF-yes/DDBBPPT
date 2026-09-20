# -*- coding: utf-8 -*-
"""mm_yolo 模型组装（方案 v4 §3 Stage B/C/D/E 的实现）。

结构（保持 ultralytics 官方层索引 4/6/10 与 neck 的 Concat 依赖不变）：

```
rgb → backbone[0..2]（共享 stem，COCO 权重）
ir  → 1×1 适配 → 同一 stem ┐
dep → 1×1 适配 → 同一 stem ┘
        ↓（每模态各自跑 backbone[3..10]：档 a 独立副本 / 档 b 共享+低秩适配 / 档 c 共享+逐模态 BN）
   P3(layer4) / P4(layer6) / P5(layer10)
        ↓  三个 FusionBlock 通过 **forward hook 替换主路径输出**（不插入新层）
   backbone[11..22] 官方预训练 neck（通道不变 → 权重继承）
        ↓
   Detect（回归分支原样；类别层适配 nc）
```

关键性质：
* **只随机初始化小模块**（适配器、融合块），主干/neck/回归分支全部继承 COCO；
* **主模态只有一次前向**（辅助模态手动跑 3..10），计算量 ≈ 2.0–2.5×；
* 三模态路径**从第 0 步就有梯度**（无零初始化旁路）；
* 结构自描述 + 严格加载（missing/unexpected 非空直接报错）。
"""
from __future__ import annotations

import copy
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for _p in (str(_CODE), str(_CODE / "vendor"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultralytics.nn.modules import Conv, Detect          # noqa: E402
from ultralytics import YOLO                             # noqa: E402

from config import MMConfig, default_config              # noqa: E402
from data import canvas_of                               # noqa: E402
from fusion import FusionBlock, LateArbitration, PersistentRegisterBus  # noqa: E402
from memory_fusion import CrossScaleMemory, SpatialMemoryFusion, NeckMemoryRead, masked_pool
from torch.nn import functional as F

IDX_P3, IDX_P4, IDX_P5 = 4, 6, 10
_SCALE_OF = {IDX_P3: "p3", IDX_P4: "p4", IDX_P5: "p5"}
_SCALE_ORDER = ("p3", "p4", "p5")
MODALITIES = ("ir", "dep")


def _cls_name_index(tgt_names, src_names):
    if not tgt_names or not src_names:
        return None
    norm = lambda s: str(s).strip().lower()               # noqa: E731
    src = {norm(v): int(k) for k, v in dict(src_names).items()}
    return [src.get(norm(n), -1) for n in tgt_names]


class MMYOLO(nn.Module):
    """三模态 YOLO11：共享 stem + 可配置共享度编码器 + 承重式融合。"""

    def __new__(cls, cfg=None, *args, **kwargs):
        if cls is MMYOLO and cfg is not None and cfg.fusion.architecture == "independent_p2_memory_v3":
            from independent_model import IndependentMMYOLO
            return IndependentMMYOLO(cfg)
        return super().__new__(cls)

    def __init__(self, cfg: Optional[MMConfig] = None, class_names=None, nc: Optional[int] = None,
                 weights: Optional[str] = None):
        super().__init__()
        cfg = cfg or default_config()
        self.cfg = cfg
        if class_names is not None:
            cfg.class_names = tuple(class_names)
        if nc is not None:
            cfg.nc = int(nc)
        if weights is not None:
            cfg.weights = str(weights)
        self.class_names = list(cfg.class_names) or [str(i) for i in range(cfg.class_num())]
        cfg.nc = self.nc = len(self.class_names)

        wpath = cfg.resolve_weights()
        base = YOLO(wpath).model                          # DetectionModel（含全部 COCO 权重）
        self.backbone = base
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        # DFL is a fixed integral over bins 0..reg_max-1, not a learned head.
        # train-time loss has its own fixed projection; keep inference identical.
        dfl = getattr(self.backbone.model[-1], "dfl", None)
        if dfl is not None:
            dfl.requires_grad_(False)
        self._adapt_detect_cls(self.backbone.model[-1], self.nc, cls_remap=cfg.cls_remap)
        self.backbone.names = {i: n for i, n in enumerate(self.class_names)}
        if hasattr(self.backbone, "nc"):
            self.backbone.nc = self.nc

        c3, c4, c5 = self._probe_channels()
        self.channels = {"p3": c3, "p4": c4, "p5": c5}
        self.spatial_memory = cfg.fusion.architecture == "spatial_memory_v1"
        if cfg.fusion.architecture not in ("legacy_hook_v1", "spatial_memory_v1"):
            raise ValueError(f"Unknown architecture: {cfg.fusion.architecture}")
        if self.spatial_memory and (not cfg.fusion.use_bus or cfg.fusion.late_bus):
            raise ValueError("spatial_memory_v1 requires its cross-scale memory, without legacy late_bus")

        # ---------------- 输入适配：IR(1→3) / Depth(4→3)，初始化为"复制相对深度"，
        #                  使共享 stem 看到与预训练同分布的 3 通道输入 ----------------
        self.ir_adapter = nn.Conv2d(1, 3, 1, 1, bias=False)
        self.depth_input_channels = int(cfg.encoder.depth_input_channels)
        if self.depth_input_channels not in (2, 4):
            raise ValueError(f"Depth 输入通道只支持 2/4，得到 {self.depth_input_channels}")
        self.depth_view = str(cfg.encoder.depth_view)
        self.depth_init = str(cfg.encoder.depth_init)
        if self.depth_view not in ("both", "relative", "metric_fallback", "metric_log_fallback"):
            raise ValueError(f"未知 depth_view: {self.depth_view}")
        if self.depth_init not in ("relative", "balanced", "metric_fallback"):
            raise ValueError(f"未知 depth_init: {self.depth_init}")
        if self.depth_input_channels == 2 and (self.depth_view != "both" or self.depth_init != "relative"):
            raise ValueError("旧 2ch Depth 只支持默认表示和初始化")
        if self.depth_view == "relative" and self.depth_init != "relative":
            raise ValueError("relative 表示不能初始化被屏蔽的绝对深度通道")
        if self.depth_view in ("metric_fallback", "metric_log_fallback") and self.depth_init != "metric_fallback":
            raise ValueError("metric_fallback 表示须初始化 PNG 绝对深度与 JPG 回退通道")
        self.dep_adapter = nn.Conv2d(self.depth_input_channels, 3, 1, 1, bias=False)
        with torch.no_grad():
            self.ir_adapter.weight.fill_(1.0)
            self.dep_adapter.weight.zero_()
            if self.depth_init == "balanced":
                self.dep_adapter.weight[:, 0].fill_(0.5)
                self.dep_adapter.weight[:, 1].fill_(0.5)
            elif self.depth_init == "metric_fallback":
                self.dep_adapter.weight[:, 0].fill_(1.0)  # JPG 相对深度回退
                self.dep_adapter.weight[:, 1].fill_(1.0)  # PNG 毫米深度
            else:
                self.dep_adapter.weight[:, 0].fill_(1.0)

        # ---------------- Stage B：共享度三档 ----------------
        self.share_tier = str(cfg.encoder.share_tier).lower()
        assert self.share_tier in ("a", "b", "c"), f"未知共享档: {self.share_tier}"
        # ⚠️ 审计修复：**必须是普通 list，不能是 nn.ModuleList / nn.Sequential**。
        #    这些层已经挂在 self.backbone.model[3:11] 上；再注册一次会产生
        #    204 组"共享存储的重复 state_dict 键"，于是 ModelEMA.update 的
        #    state_dict 遍历会把同一参数更新两遍（decay=0.9 实测增量 0.19 vs 正确 0.10）。
        #    list 不参与 _modules 注册 → 参数仍通过 backbone 被 .to()/optimizer 覆盖。
        self.stage_shared = list(self.backbone.model[3:11])   # 3..10（共享时直接复用）
        # depth 独立 stem：D0 实测 depth↔RGB 边缘相关仅 0.05（IR 是 0.19）→
        # 度量不连续 ≠ 纹理边缘，浅层卷积核不该共享给 depth。
        if cfg.encoder.depth_own_stem:
            self.dep_stem = copy.deepcopy(self.backbone.model[:3])
            for p in self.dep_stem.parameters():
                p.requires_grad_(True)
        if self.share_tier == "a":
            self.aux_stages = nn.ModuleDict(
                {m: nn.Sequential(*copy.deepcopy(self.stage_shared)) for m in MODALITIES})
        if self.share_tier == "b":
            r = int(cfg.encoder.adapter_rank)
            self.adapters = nn.ModuleDict({
                m: nn.ModuleDict({s: nn.Sequential(nn.Conv2d(self.channels[s], r, 1),
                                                   nn.Conv2d(r, self.channels[s], 1))
                                  for s in _SCALE_ORDER}) for m in MODALITIES})
            for md in self.adapters.values():
                for seq in md.values():
                    nn.init.zeros_(seq[-1].weight)
                    nn.init.zeros_(seq[-1].bias)
        if self.share_tier == "c" and cfg.encoder.per_modality_bn:
            self._build_bn_replicas()

        # ---------------- Stage C：三个尺度的融合块 ----------------
        if self.spatial_memory:
            self.fusion = nn.ModuleDict({s: SpatialMemoryFusion(c,
                window=1 if s == "p3" else 3, dim=32, memory_dim=cfg.fusion.bus_dim,
                rounds=2, residual=cfg.fusion.residual_init,
                memory_control=cfg.fusion.memory_control) for s, c in self.channels.items()})
            self.register_bus = CrossScaleMemory(self.channels, dim=cfg.fusion.bus_dim,
                heads=cfg.fusion.heads, pool=cfg.fusion.bus_pool)
        else:
            self.fusion = nn.ModuleDict({s: FusionBlock(self.channels[s], cfg.fusion,
                                                    slot_names=cfg.fusion.slot_names)
                                     for s in _SCALE_ORDER})
            self.register_bus = (PersistentRegisterBus(
            self.channels, n_mod=len(cfg.fusion.slot_names),
            n_tokens=cfg.fusion.bus_tokens, dim=cfg.fusion.bus_dim,
            heads=cfg.fusion.heads, pool=cfg.fusion.bus_pool,
            residual_init=cfg.fusion.register_residual_init)
            if cfg.fusion.use_bus else None)
        if cfg.encoder.metric_branch:
            if self.depth_input_channels != 4:
                raise ValueError("metric branch requires 4-channel Depth")
            # Independent nonzero route. No per-image normalization/BN here:
            # the level of absolute distance must survive to the fused features.
            self.metric_encoder = nn.ModuleDict({s: nn.Sequential(
                nn.Conv2d(4, 16, 1), nn.SiLU(), nn.Conv2d(16, c, 1, bias=False))
                for s, c in self.channels.items()})
        # ---------------- Stage C（片后）：Detect 之前的全局仲裁总线 ----------------
        det = self.backbone.model[-1]
        # Detect 的输入通道 = 颈部输出通道（层 16/19/22，yolo11s 实测 128/256/512），
        # 与主干 P3/P4/P5 通道（256/256/512）不同 —— 别混用。
        try:
            neck_ch = [int(m[0].conv.in_channels) for m in det.cv2]
        except Exception:                                        # noqa: BLE001
            neck_ch = [self.channels[s] for s in _SCALE_ORDER]
        self.neck_channels = neck_ch
        if self.spatial_memory:
            self.neck_memory = NeckMemoryRead(neck_ch[0], cfg.fusion.bus_dim, cfg.fusion.heads)
        self.late_bus = (LateArbitration(neck_ch, n_mod=len(cfg.fusion.slot_names),
                                         n_tokens=cfg.fusion.bus_tokens,
                                         dim=cfg.fusion.bus_dim)
                         if cfg.fusion.late_bus else None)
        self._late_on = False
        self._last_alpha_prior = None
        self._aux: Dict[str, Dict[int, torch.Tensor]] = {}
        self._mask: Optional[torch.Tensor] = None
        self._quality: Dict[str, torch.Tensor] = {}
        self._prior: Optional[torch.Tensor] = None
        self._keep: Optional[torch.Tensor] = None
        self._register_state: Optional[torch.Tensor] = None
        self._last_register_state: Optional[torch.Tensor] = None
        self._fusing = False
        self._hook_handles: list = []
        self.modality_off: set = set()                    # 诊断用：屏蔽某模态（'ir'/'dep'/'rgb'）
        self._register_hooks()

        self._struct = cfg.structure()
        # v8DetectionLoss 兼容面：它读 model.args（超参）与 model.model[-1]（Detect）
        from ultralytics.cfg import DEFAULT_CFG
        self.args = DEFAULT_CFG
        self.train()

    # ------------------------------------------------------------------ 深拷贝（EMA 必需）
    def _register_hooks(self) -> None:
        """把融合 hook 注册到**自己**的 backbone 层上（幂等：先摘旧的）。"""
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:                                    # noqa: BLE001
                pass
        self._hook_handles = []
        if self.spatial_memory:
            return  # explicit pure-encoder -> fusion -> original YOLO neck DAG
        for idx in (IDX_P3, IDX_P4, IDX_P5):
            self._hook_handles.append(
                self.backbone.model[idx].register_forward_hook(self._make_hook(idx)))
        if self.late_bus is not None:
            self._hook_handles.append(
                self.backbone.model[-1].register_forward_pre_hook(self._late_hook))

    def _deepcopy_impl(self, memo):
        """深拷贝 = 普通深拷贝 + **在本副本上重建 hook 与共享层引用**。

        ⚠️ 审计 P0：旧版在 `__init__` 里用闭包注册 hook，闭包捕获的是 `self`（原模型）；
        `ModelEMA` 深拷贝出的副本上，这些 hook 仍然写着**原模型**的 `_aux/_quality/_prior`，
        而且**副本自己的 `_aux` 永远是空的** → EMA 前向时 FusionBlock 调用次数 = 0，
        即"验证的不是三模态融合模型，而是纯 RGB 通路"。B0 的 best 选模因此完全失真。
        这里在副本上重新注册，并把共享层/hook 句柄指向副本自身。
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        # 共享层重新指向副本自己的 backbone 层（deepcopy 后是另一批对象）
        new.stage_shared = list(new.backbone.model[3:11])
        new._register_hooks()
        return new

    def __deepcopy__(self, memo):
        return self._deepcopy_impl(memo)

    def __copy__(self):
        return self._deepcopy_impl({})

    # ------------------------------------------------------------------ 结构适配
    def _adapt_detect_cls(self, detect: Detect, nc: int, cls_remap: bool = True) -> None:
        """只替换 Detect 的类别输出层；回归分支/共享层/neck/backbone 全部保留。"""
        old_nc = int(detect.nc)
        src_names = getattr(self.backbone, "names", None)
        idx_map = _cls_name_index(self.class_names, src_names) if cls_remap else None
        for attr in ("cv3", "one2one_cv3"):
            head = getattr(detect, attr, None)
            if head is None:
                continue
            for i, seq in enumerate(head):
                old = seq[-1]
                new = nn.Conv2d(old.in_channels, nc, 1, 1, bias=True)
                new.bias.data.fill_(math.log(5 / nc / (640 / float(detect.stride[i])) ** 2))
                if idx_map is not None:
                    for j, k in enumerate(idx_map):
                        if 0 <= k < old.out_channels:
                            new.weight.data[j].copy_(old.weight.data[k])
                            new.bias.data[j].copy_(old.bias.data[k])
                seq[-1] = new
        detect.nc = nc
        detect.no = nc + detect.reg_max * 4
        matched = [n for n, k in zip(self.class_names, idx_map) if k >= 0] if idx_map else []
        print(f"[mm_yolo] Detect 类别层适配 {old_nc} → {nc} 类；COCO 同名类迁移 "
              f"{len(matched)}/{nc}: {matched}" if matched else
              f"[mm_yolo] Detect 类别层适配 {old_nc} → {nc} 类（无同名类可迁移）")

    def _probe_channels(self):
        store: dict = {}
        handles = [self.backbone.model[i].register_forward_hook(
            lambda _m, _i, out, i=i: store.__setitem__(i, out.shape[1])) for i in (IDX_P3, IDX_P4, IDX_P5)]
        was = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                self.backbone.model[:IDX_P5 + 1](torch.zeros(1, 3, 64, 64))
        finally:
            for h in handles:
                h.remove()
            self.backbone.train(was)
        return store[IDX_P3], store[IDX_P4], store[IDX_P5]

    def _build_bn_replicas(self) -> None:
        """档 c：为 backbone[0..10] 的每个 BN 建一份逐模态副本，前向时临时替换。

        副本注册进 ModuleDict（不是普通 dict），否则 `.to(device)` / 优化器都不认它们。
        """
        self._bn_swaps: List[tuple] = []                 # (parent, attr, index)
        self._bn_orig: List[nn.BatchNorm2d] = []         # 原 BN（保留在 backbone 树里）
        self.bn_store = nn.ModuleDict()
        i = 0
        for name, mod in self.backbone.named_modules():
            if not isinstance(mod, nn.BatchNorm2d) or "." not in name:
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = self.backbone.get_submodule(parent_name)
            self._bn_swaps.append((parent, attr, i))
            self._bn_orig.append(mod)
            for m in MODALITIES:
                self.bn_store[f"{m}_{i}"] = copy.deepcopy(mod)
            i += 1

    @contextmanager
    def _bn_scope(self, modality: str):
        if not hasattr(self, "_bn_swaps") or modality == "rgb":
            yield
            return
        for parent, attr, i in self._bn_swaps:
            setattr(parent, attr, self.bn_store[f"{modality}_{i}"])
        try:
            yield
        finally:
            for parent, attr, i in self._bn_swaps:
                setattr(parent, attr, self._bn_orig[i])

    # ------------------------------------------------------------------ 编码
    def _stages_for(self, modality: str):
        return self.aux_stages[modality] if self.share_tier == "a" else self.stage_shared

    def _metric_features(self, depth, feats):
        if not hasattr(self, "metric_encoder"):
            return feats
        valid = depth[:, 2:3] * depth[:, 3:4]
        absolute = depth[:, 1:2] * valid
        for idx, scale in _SCALE_OF.items():
            avg, fraction = masked_pool(absolute, valid, feats[idx].shape[-2:])
            metric = torch.cat((avg, torch.log1p(avg*20) / math.log(21),
                                fraction, (fraction > 0).to(avg.dtype)), 1)
            feats[idx] = feats[idx] + .1 * self.metric_encoder[scale](metric) * (fraction > 0)
        return feats

    def _run_encoder(self, x: torch.Tensor, modality: str) -> Dict[int, torch.Tensor]:
        """跑 stem + 3..10，返回 {4: P3, 6: P4, 10: P5}。

        主模态（rgb）复用 backbone 自带 stem；IR 共享同一 stem；depth 用独立 stem（可配）。
        """
        with self._bn_scope(modality):
            s = self.stage_shared if modality == "rgb" else self._stages_for(modality)
            if modality == "dep" and hasattr(self, "dep_stem"):
                stem = self.dep_stem
            else:
                stem = self.backbone.model[:3]
            y = stem(x) if modality == "rgb" else stem[0](x)
            if modality != "rgb":
                y = stem[1](y)
                y = stem[2](y)
            y3 = s[0](y)
            p3 = s[1](y3)
            y5 = s[2](p3)
            p4 = s[3](y5)
            y7 = s[4](p4)
            y8 = s[5](y7)
            y9 = s[6](y8)
            p5 = s[7](y9)
        feats = {IDX_P3: p3, IDX_P4: p4, IDX_P5: p5}
        if self.share_tier == "b" and modality != "rgb":
            for idx, s_name in zip((IDX_P3, IDX_P4, IDX_P5), _SCALE_ORDER):
                feats[idx] = feats[idx] + self.adapters[modality][s_name](feats[idx])
        return feats

    def _run_present_aux(self, x: torch.Tensor, modality: str,
                         present: torch.Tensor) -> Dict[int, torch.Tensor]:
        """只编码在场样本，避免零占位图污染逐模态 BN 的批统计。"""
        active = present.bool().nonzero(as_tuple=True)[0]
        adapter = self.ir_adapter if modality == "ir" else self.dep_adapter
        active_x = x.index_select(0, active)
        if modality == "dep" and int(self.dep_adapter.in_channels) == 4:
            # B2 实测：Depth 独立 stem 的输出经 model.3 卷积后幅值约 3.5e4，
            # 仍是有限的 FP32 数，但 FP16 卷积中间累加会溢出 65504。溢出会先
            # 把逐 Depth BN running stats 写成 NaN，GradScaler 只能跳过梯度，
            # 无法回滚 buffer。因此 4ch Depth 编码器强制 FP32；归一化后的
            # P3/P4/P5 进入融合时仍由外层 autocast 使用 AMP。
            with torch.autocast(device_type=x.device.type, enabled=False):
                sub = self._run_encoder(adapter(active_x.float()), modality)
        else:
            sub = self._run_encoder(adapter(active_x), modality)
        if modality == "dep":
            sub = self._metric_features(active_x.float(), sub)
        if active.numel() == x.shape[0]:
            return sub
        return {k: v.new_zeros((x.shape[0], *v.shape[1:])).index_copy(0, active, v)
                for k, v in sub.items()}

    def _depth_view_input(self, depth: torch.Tensor) -> torch.Tensor:
        """实验表示在模型内应用，随 checkpoint 保存，训练/评测/提交完全一致。"""
        if depth.shape[1] != 4 or self.depth_view == "both":
            return depth
        relative, absolute, valid, metric = depth.split(1, dim=1)
        if self.depth_view == "relative":
            absolute = torch.zeros_like(absolute)
        else:  # PNG 只用毫米距离，JPG 没有米制值时才用相对深度
            relative = relative * (1.0 - metric)
            if self.depth_view == "metric_log_fallback":
                # 绝对距离 z[m]∈[0,20] → log(1+z)/log(21)：单调且可逆，放大近距离差异。
                absolute = torch.log1p(20.0 * absolute.clamp_min(0)) / math.log(21.0)
        return torch.cat((relative, absolute, valid, metric), dim=1)

    def _prepare_depth_input(self, depth: torch.Tensor) -> torch.Tensor:
        """在 B1 2ch 与 B2 4ch 之间做无损语义映射。

        旧模型接收新数据时取 [relative, valid]；新模型接收旧 2ch 输入时把
        absolute/metric_available 补 0。这也让历史 B1 checkpoint 仍可直接评测。
        """
        got, need = int(depth.shape[1]), int(self.dep_adapter.in_channels)
        if got == need:
            return self._depth_view_input(depth)
        if got == 4 and need == 2:
            return depth[:, (0, 2)]
        if got == 2 and need == 4:
            out = depth.new_zeros((depth.shape[0], 4, *depth.shape[2:]))
            out[:, 0] = depth[:, 0]
            out[:, 2] = depth[:, 1]
            return self._depth_view_input(out)
        raise ValueError(f"Depth 通道不兼容：输入 {got}ch，模型需要 {need}ch")

    # ------------------------------------------------------------------ 融合 hook
    def _make_hook(self, idx: int):
        s_name = _SCALE_OF[idx]
        s_i = _SCALE_ORDER.index(s_name)

        def hook(_module, _inputs, output):
            if not self._fusing:
                return output
            feats: List[Optional[torch.Tensor]] = [output, None, None]      # [rgb, ir, dep]
            for slot, m in enumerate(MODALITIES, start=1):
                f = self._aux.get(m, {}).get(idx)
                if f is None:
                    continue
                if m == "dep" and not self.cfg.fusion.depth_scales[s_i]:
                    continue
                assert f.shape[1] == output.shape[1], \
                    f"{m} 特征通道 {f.shape[1]} 与主模态 {output.shape[1]} 不一致（{s_name}）"
                feats[slot] = f
            if feats[1] is None and feats[2] is None:                      # 只有主模态 → 恒等
                return output
            mask = self._mask if (feats[2] is not None and self.cfg.fusion.use_mask) else None
            qual = [self._quality.get("rgb"), self._quality.get("ir"), self._quality.get("dep")]
            fused, modal_ctx, valid = self.fusion[s_name](
                feats, mask=mask, quality=qual, prior=self._prior, keep=self._keep,
                return_context=True)
            if self.register_bus is not None:
                fused, self._register_state = self.register_bus(
                    s_name, modal_ctx, valid, fused, self._register_state)
            return fused

        return hook

    # ------------------------------------------------------------------ 片后总线
    def _late_hook(self, _module, args):
        """Detect 的 forward pre-hook：对颈部输出（层 16/19/22）做一次全局仲裁调制。"""
        if not self._late_on or self.late_bus is None or not args:
            return None
        feats = args[0]
        if not isinstance(feats, (list, tuple)) or len(feats) != 3:
            return None
        modulated = self.late_bus(list(feats))
        return (list(modulated), *args[1:])

    # ------------------------------------------------------------------ forward
    def _enabled(self, m: str) -> bool:
        return m not in self.modality_off

    def _forward_spatial_memory(self, rgb):
        active = self._keep[:, 0].bool().nonzero(as_tuple=True)[0]
        if active.numel():
            sub = self._run_encoder(rgb.index_select(0, active), "rgb")
            pure_rgb = (sub if active.numel() == rgb.shape[0] else
                        {k: v.new_zeros((rgb.shape[0], *v.shape[1:])).index_copy(0, active, v)
                         for k, v in sub.items()})
        else:
            pure_rgb = {k: torch.zeros_like(v) for k, v in next(iter(self._aux.values())).items()}
        saved = [None] * len(self.backbone.model)
        state = None
        for si, (idx, scale) in enumerate(_SCALE_OF.items()):
            evidence = [pure_rgb[idx], self._aux.get("ir", {}).get(idx), self._aux.get("dep", {}).get(idx)]
            ref = pure_rgb[idx]
            masks = [self._keep[:, i:i+1, None, None].to(ref.dtype).expand(-1, 1, *ref.shape[-2:])
                     if x is not None else ref.new_zeros(ref.shape[0], 1, *ref.shape[-2:])
                     for i, x in enumerate(evidence)]
            if evidence[2] is not None:
                masks[2] = masks[2] * F.interpolate(self._mask.float(), ref.shape[-2:], mode="nearest")
            # Memory reads only pure evidence. Depth memory may summarize P3,
            # while its direct spatial injection remains restricted to P4/P5.
            state = self.register_bus(scale, evidence, masks, state)
            local_evidence, local_masks = list(evidence), list(masks)
            if not self.cfg.fusion.depth_scales[si]:
                local_evidence[2] = None
                local_masks[2] = torch.zeros_like(masks[2])
            qual = [self._quality.get(m) for m in ("rgb", "ir", "dep")]
            fused = self.fusion[scale](local_evidence, local_masks, state, quality=qual)
            # Exact per-sample RGB-only identity, including dropped/invalid auxiliaries.
            aux_valid = (masks[1].flatten(1).any(1) | masks[2].flatten(1).any(1))[:, None, None, None]
            saved[idx] = torch.where(aux_valid, fused, ref)
        self._register_state = state
        x = saved[IDX_P5]
        for layer in self.backbone.model[11:]:
            source = layer.f
            if source != -1:
                x = (saved[source] if isinstance(source, int)
                     else [x if j == -1 else saved[j] for j in source])
            x = layer(x)
            if layer.i == 16:
                present = self._keep.clone()
                if self._mask is not None:
                    present[:, 2] *= self._mask.flatten(1).any(1)
                changed = self.neck_memory(x, state, present)
                x = torch.where(present[:, 1:].any(1)[:, None, None, None], changed, x)
            saved[layer.i] = x
        return x

    def forward(self, rgb: torch.Tensor, ir: Optional[torch.Tensor] = None,
                depth: Optional[torch.Tensor] = None, quality: Optional[Dict[str, torch.Tensor]] = None,
                prior: Optional[torch.Tensor] = None,
                keep: Optional[Dict[str, torch.Tensor]] = None):
        """rgb(B,3,H,W)；ir(B,1,H,W)；depth(B,4,H,W)=[相对, 绝对/20m, valid, metric]。

        quality: {'rgb'|'ir'|'dep': (B,q,h,w)} 模态质量描述子（低分辨率即可，融合块会插值）；
        prior  : (B,p,h,w) 几何先验（depth：逆深度/有效/法线…），做 FiLM 显式注入。
        """
        self._aux, self._mask = {}, None
        self._register_state = None
        self._quality = dict(quality or {})
        self._prior = prior
        availability = {"rgb": rgb is not None and self._enabled("rgb"),
                        "ir": ir is not None and self._enabled("ir"),
                        "dep": depth is not None and self._enabled("dep")}
        if keep is None:
            self._keep = rgb.new_tensor([[float(availability[m]) for m in ("rgb", "ir", "dep")]]).expand(rgb.shape[0], -1).clone()
        else:
            self._keep = torch.stack([torch.as_tensor(keep[m], device=rgb.device).reshape(-1)
                                      if m in keep else rgb.new_full((rgb.shape[0],), float(availability[m]))
                                      for m in ("rgb", "ir", "dep")], dim=1)
        for i, m in enumerate(("rgb", "ir", "dep")):
            if not availability[m]:
                self._keep[:, i] = 0
        if not self._keep.bool().any(dim=1).all():
            raise ValueError("每张图至少须保留一路有效模态")
        # 去 RGB 诊断/单模态训练时不能只关融合槽：RGB 主干的中间跳连仍可能
        # 把原像素带到 neck。按逐样本 keep 在进入主干前清零，杜绝视觉信息泄漏。
        rgb = rgb * self._keep[:, 0].to(dtype=rgb.dtype).view(-1, 1, 1, 1)
        if ir is not None and self._enabled("ir") and bool(self._keep[:, 1].bool().any()):
            self._aux["ir"] = self._run_present_aux(ir, "ir", self._keep[:, 1])
        if depth is not None and self._enabled("dep") and bool(self._keep[:, 2].bool().any()):
            if depth.shape[1] not in (2, 4):
                raise ValueError(f"Depth 应为 B1 2ch 或 B2 4ch，得到 {depth.shape[1]}ch")
            self._mask = depth[:, 2:3] if depth.shape[1] == 4 else depth[:, 1:2]
            self._aux["dep"] = self._run_present_aux(
                self._prepare_depth_input(depth), "dep", self._keep[:, 2])
        self._fusing = not self.spatial_memory
        self._late_on = True
        try:
            out = self._forward_spatial_memory(rgb) if self.spatial_memory else self.backbone(rgb)
        finally:
            self._last_register_state = (self._register_state.detach()
                                         if self._register_state is not None else None)
            self._register_state = None
            self._fusing = False
            self._late_on = False
        return out

    # ------------------------------------------------------------------ 结构/保存
    @property
    def model(self):
        """给 ultralytics 的 v8DetectionLoss / 工具函数用（它们取 model.model[-1]）。"""
        return self.backbone.model

    @property
    def stride(self):
        return self.backbone.model[-1].stride

    def structure_kwargs(self) -> dict:
        return copy.deepcopy(self._struct)

    def param_report(self) -> Dict[str, int]:
        def _n(mod):
            return sum(p.numel() for p in mod.parameters())
        pretrained = sum(p.numel() for p in self.backbone.parameters())
        new = _n(self.fusion) + _n(self.ir_adapter) + _n(self.dep_adapter)
        if self.register_bus is not None:
            new += _n(self.register_bus)
        if hasattr(self, "metric_encoder"):
            new += _n(self.metric_encoder)
        if hasattr(self, "neck_memory"):
            new += _n(self.neck_memory)
        if self.share_tier == "a":
            new += _n(self.aux_stages)
        if self.share_tier == "b":
            new += _n(self.adapters)
        if hasattr(self, "dep_stem"):
            new += _n(self.dep_stem)
        if self.late_bus is not None:
            new += _n(self.late_bus)
        bn_rep = sum(p.numel() for p in self.bn_store.parameters()) \
            if hasattr(self, "bn_store") else 0
        return {"pretrained": pretrained, "new": new, "fusion": _n(self.fusion),
                "register_bus": _n(self.register_bus) if self.register_bus is not None else 0,
                "late_bus": _n(self.late_bus) if self.late_bus is not None else 0,
                "depth_stem": _n(self.dep_stem) if hasattr(self, "dep_stem") else 0,
                "total": _n(self), "bn_replicas": bn_rep}


# ---------------------------------------------------------------- 加载

def save_mm_checkpoint(path, model: MMYOLO, epoch: int = 0, best_map: float = 0.0,
                       meta: Optional[dict] = None, **extra) -> None:
    """保存 checkpoint。**必须带上"训练时用了哪些模态/多大画布"**。

    审计 P0：旧 checkpoint 只存 structure/model_state，没存训练模态；加载 RGB-only 权重后
    默认开启 IR/Depth，未训练的融合/适配分支直接参与前向 → 实测同一张图 0 框（正确关闭时正常）。
    现在把 `meta`（modalities/canvas/imgsz/nc/class_names…）一起写进 ckpt，
    `load_mm_checkpoint` 会据此恢复，eval/submit 不再靠猜。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "structure": model.structure_kwargs(),
               "epoch": int(epoch), "best_map": float(best_map),
               "class_names": list(model.class_names), "nc": int(model.nc),
               "meta": dict(meta or {})}
    payload.update(extra)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)                      # 原子替换：断电/崩溃不会留下半个 ckpt


def load_mm_checkpoint(path, strict: bool = True, device=None, **overrides) -> tuple:
    """按 checkpoint 自描述的 structure 重建 MMYOLO 并严格加载。

    返回 (model, ck)；`ck["meta"]` 里有训练模态与画布，调用方必须用它来推理
    （`resolve_infer_modalities` / `resolve_infer_canvas`）。
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(f"{path} 不是本框架的 checkpoint（缺 model_state）")
    struct = dict(ck.get("structure") or {})
    if not struct:
        raise ValueError(f"{path} 缺 structure 字段：旧格式无法自动重建，请显式传配置覆盖")
    struct.update(overrides)
    cfg = MMConfig.from_structure(struct)
    model = MMYOLO(cfg)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"结构不匹配：missing={len(missing)} unexpected={len(unexpected)}\n"
                           f"  missing[:8]={missing[:8]}\n  unexpected[:8]={unexpected[:8]}")
    meta = dict(ck.get("meta") or {})
    if meta.get("modalities"):
        model.infer_modalities = tuple(meta["modalities"])
        model.modality_off = {"rgb", "ir", "dep"} - set(model.infer_modalities)
    model.infer_canvas = tuple(meta["canvas"]) if meta.get("canvas") else None
    if device is not None:
        model.to(device)
    print(f"[mm_yolo] 已加载 {Path(path).name}（epoch={ck.get('epoch')} "
          f"best={ck.get('best_map')}）模态={meta.get('modalities', '未记录')} "
          f"画布={meta.get('canvas', '未记录')} missing={len(missing)} unexpected={len(unexpected)}")
    return model, ck


def resolve_infer_modalities(model: MMYOLO, requested: Optional[str] = None) -> str:
    """决定推理时用哪些模态：**以 checkpoint 记录的训练模态为准**，命令行只能收紧不能放开。

    旧 RGB-only checkpoint 仍自动收紧；两路实验权重不能误开第三路。
    """
    modes = {"rgb": {"rgb"}, "ir": {"ir"}, "dep": {"dep"},
             "rgb_ir": {"rgb", "ir"}, "rgb_dep": {"rgb", "dep"},
             "all": {"rgb", "ir", "dep"}}
    if requested is not None and requested not in modes:
        raise ValueError(f"未知推理模态 {requested!r}")
    trained = tuple(getattr(model, "infer_modalities", ()) or ())
    if trained:
        trained_set = set(trained)
        if trained_set not in modes.values():
            raise ValueError(f"checkpoint 记录了不支持的训练模态：{trained}")
        wanted = modes[requested] if requested else trained_set
        if not wanted.issubset(trained_set):
            print(f"[mm_yolo] 请求模态 {requested} 包含未训练分支 → 使用 checkpoint 模态 {trained}")
            wanted = trained_set
        return next(k for k, v in modes.items() if v == wanted)
    return requested or "all"


def resolve_infer_canvas(model: MMYOLO, imgsz) -> object:
    """画布以 checkpoint 记录为准（模型是在那个画布上训的）。"""
    c = getattr(model, "infer_canvas", None)
    if not c:
        return imgsz
    if imgsz and tuple(canvas_of(imgsz)) != tuple(c):
        print(f"[mm_yolo] 画布与 checkpoint 不一致：训练 {tuple(c)} vs 传入 "
              f"{tuple(canvas_of(imgsz))} → 采用训练画布 {tuple(c)}")
    return tuple(c)


# ---------------------------------------------------------------- 自检

def _selfcheck(nc: int = 12, size: int = 256, batch: int = 2, tiers=("L0", "L1", "L2", "L3")):
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== mm_yolo 自检（nc={nc}, size={size}, device={dev}）===")
    rgb = torch.rand(batch, 3, size, size, device=dev)
    ir = torch.rand(batch, 1, size, size, device=dev)
    dep = torch.rand(batch, 4, size, size, device=dev)
    dep[:, 2:3] = (torch.rand(batch, 1, size, size, device=dev) > 0.3).float()
    dep[:, 3:4] = 1.0
    # 低分辨率先验（数据管线按 stride 8 计算即可）
    ps = size // 8
    quality = {m: torch.rand(batch, 3, ps, ps, device=dev) for m in ("rgb", "ir", "dep")}
    prior = torch.rand(batch, 4, ps, ps, device=dev)

    results = {}
    for tier in tiers:
        cfg = default_config(nc=nc)
        cfg.fusion.tier = tier
        model = MMYOLO(cfg).to(dev)
        pr = model.param_report()
        model.train()
        out = model(rgb, ir, dep, quality=quality, prior=prior)
        outs = [t for t in (out.values() if isinstance(out, dict) else out) if torch.is_tensor(t)]
        loss = sum(t.float().mean() for t in outs)
        loss.backward()
        g = lambda mod, name: any(p.grad is not None and float(p.grad.abs().sum()) > 0     # noqa: E731
                                  for n_, p in mod.named_parameters() if name in n_)
        grads = {
            "rgb主干": any(float(p.grad.abs().sum()) > 0 for p in model.backbone.model[6].parameters()
                        if p.grad is not None),
            "ir适配器": bool(model.ir_adapter.weight.grad is not None
                          and float(model.ir_adapter.weight.grad.abs().sum()) > 0),
            "dep适配器": bool(model.dep_adapter.weight.grad is not None
                           and float(model.dep_adapter.weight.grad.abs().sum()) > 0),
            "融合α": g(model.fusion["p4"].alpha_conv, "weight"),
            "融合γ": bool(model.fusion["p4"].gamma_raw.grad is not None
                        and float(model.fusion["p4"].gamma_raw.grad.abs().sum()) > 0),
            "融合β(交互强度)": bool(model.fusion["p4"].beta.grad is not None
                                and float(model.fusion["p4"].beta.grad.abs().sum()) > 0),
        }
        if tier in ("L2", "L3"):
            grads["交互q"] = g(model.fusion["p4"], "q.weight")
        if getattr(model.fusion["p4"], "any_off", False):
            grads["可变形off"] = g(model.fusion["p4"], "off")
        if model.fusion["p4"].prior_film is not None:
            grads["depth先验FiLM"] = g(model.fusion["p4"].prior_film, "conv")
        if hasattr(model, "dep_stem"):
            grads["depth独立stem"] = any(float(p.grad.abs().sum()) > 0
                                      for p in model.dep_stem.parameters() if p.grad is not None)
        # 模态屏蔽影响（E3/E6 的"接线"版本；随机初始化下的量级，只证明通路承重）
        def _outs(y):
            return [t for t in (y.values() if isinstance(y, dict) else y) if torch.is_tensor(t)]
        with torch.no_grad():
            base = _outs(out)
            base = base[1].float() if len(base) > 1 else base[0].float()
            infl = {}
            for m in ("ir", "dep"):
                model.modality_off = {m}
                y = model(rgb, ir, dep, quality=quality, prior=prior)
                model.modality_off = set()
                mm = _outs(y)
                mm = mm[1].float() if len(mm) > 1 else mm[0].float()
                infl[m] = float((base - mm).norm() / base.norm().clamp_min(1e-9))
            # 质量描述子 / 几何先验是否真的接进了决策：
            # 注意 α 末层与 FiLM 末层都是**零初始化**（初始恒等、之后才学），
            # 所以要先人为打破零初始化，否则"打乱输入 → 输出不变"是正确行为而非接线错误。
            for blk in model.fusion.values():
                nn.init.normal_(blk.alpha_conv[-1].weight, std=0.02)
                if blk.prior_film is not None:
                    nn.init.normal_(blk.prior_film.conv[-1].weight, std=0.02)
            y = model(rgb, ir, dep, quality=quality, prior=prior)
            yb = _outs(y)
            base2 = yb[1].float() if len(yb) > 1 else yb[0].float()
            q2 = {k: torch.rand_like(v) for k, v in quality.items()}
            y = model(rgb, ir, dep, quality=q2, prior=prior)
            yy = _outs(y)
            yy = yy[1].float() if len(yy) > 1 else yy[0].float()
            infl["质量描述子打乱"] = float((base2 - yy).norm() / base2.norm().clamp_min(1e-9))
            y = model(rgb, ir, dep, quality=quality, prior=torch.rand_like(prior))
            yp = _outs(y)
            yp = yp[1].float() if len(yp) > 1 else yp[0].float()
            infl["几何先验打乱"] = float((base2 - yp).norm() / base2.norm().clamp_min(1e-9))
            model._fusing = False                       # 关掉融合 hook（等价"融合=恒等"）
            y0 = model.backbone(rgb)
            model._fusing = True
            y0s = _outs(y0)
            y0s = y0s[1].float() if len(y0s) > 1 else y0s[0].float()
            infl["融合整体"] = float((base - y0s).norm() / base.norm().clamp_min(1e-9))
            a = model.fusion["p4"]
            gamma = (float(a.cfg.gamma_floor) + (1 - float(a.cfg.gamma_floor))
                     * torch.sigmoid(a.gamma_raw.detach()))
        results[tier] = dict(params=pr, grads=grads, infl=infl,
                             gamma=[round(float(v), 3) for v in gamma])
        print(f"\n--- 档 {tier} ---")
        print(f"  参数量：总 {pr['total']/1e6:.2f}M（预训练 {pr['pretrained']/1e6:.2f}M + "
              f"新增 {pr['new']/1e6:.3f}M，其中融合 {pr['fusion']/1e6:.3f}M）")
        print(f"  梯度到达：{grads}")
        print(f"  初始 γ（P4）：{results[tier]['gamma']}")
        print("  输出相对变化（随机初始化下的**接线**证据，非训练后效果）："
              + "  ".join(f"{k}={v*100:.3f}%" for k, v in infl.items()))
        del model
        if dev == "cuda:0":
            torch.cuda.empty_cache()

    # 结构自描述 + 严格加载往返（CPU 上比对，避免设备不一致）
    cfg = default_config(nc=nc)
    m1 = MMYOLO(cfg)
    tmp = Path(_HERE) / "_selfcheck_ckpt.pt"
    save_mm_checkpoint(tmp, m1, epoch=0, best_map=0.0)
    m2, _ = load_mm_checkpoint(tmp)
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    same_keys = list(sd1.keys()) == list(sd2.keys())
    ok = same_keys and all(torch.equal(a, b) for a, b in zip(sd1.values(), sd2.values()))
    tmp.unlink(missing_ok=True)
    print(f"\n结构往返：键集合一致={same_keys}，state_dict 逐位一致 = {ok}")

    # 梯度要求按档位区分：L0/L1 没有交互支路，β/q 无梯度是**正确**的
    need = {"ir适配器", "dep适配器", "融合α", "融合γ", "rgb主干", "depth独立stem", "depth先验FiLM"}
    all_ok = True
    for tier, r in results.items():
        req = set(need)
        if tier in ("L2", "L3"):
            req |= {"融合β(交互强度)", "交互q"}
        if tier == "L3" or r["grads"].get("可变形off"):
            req |= {"可变形off"}
        missing = [k for k in req if not r["grads"].get(k, False)]
        if missing:
            all_ok = False
            print(f"  [FAIL] 档 {tier} 以下通路没有梯度：{missing}")

    # 片后总线（Detect 之前的全局仲裁）：单独构型验证能跑通且有梯度
    cfg_late = default_config(nc=nc)
    cfg_late.fusion.late_bus = True
    m_late = MMYOLO(cfg_late).to(dev)
    m_late.train()
    out_l = m_late(rgb, ir, dep, quality=quality, prior=prior)
    outs_l = [t for t in (out_l.values() if isinstance(out_l, dict) else out_l) if torch.is_tensor(t)]
    sum(t.float().mean() for t in outs_l).backward()
    late_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                    for p in m_late.late_bus.parameters())
    with torch.no_grad():
        base_l = outs_l[1].float() if len(outs_l) > 1 else outs_l[0].float()
        m_late._late_on = False
        y_nolate = m_late(rgb, ir, dep, quality=quality, prior=prior)
        y_nolate = [t for t in (y_nolate.values() if isinstance(y_nolate, dict) else y_nolate)
                    if torch.is_tensor(t)]
        y_nolate = y_nolate[1].float() if len(y_nolate) > 1 else y_nolate[0].float()
        infl_late = float((base_l - y_nolate).norm() / base_l.norm().clamp_min(1e-9))
        m_late._late_on = True
    print("\n--- 片后总线（late_bus）---")
    n_late = sum(p.numel() for p in m_late.late_bus.parameters())
    print(f"  参数 {n_late/1e3:.1f}K | 梯度到达={late_grad} | "
          f"初始调制影响={infl_late*100:.3f}% | 调制统计={m_late.late_bus.stats()}")
    print("  （近似恒等起步、但第 0 步就有梯度 → 不会变成'可选支路'）")
    if not late_grad:
        all_ok = False
    del m_late
    if dev == "cuda:0":
        torch.cuda.empty_cache()

    # ---------------- 回归自检（审计 P0 对应项："修好才通过"的硬断言）----------------
    reg = _regression_selfcheck(dev, nc=nc, size=max(64, size // 2), batch=batch)
    print("\n--- 回归自检（P0 修复项）---")
    for k, v in reg.items():
        print(f"  {'OK  ' if v[0] else 'FAIL'} {k}: {v[1]}")
    reg_ok = all(v[0] for v in reg.values())

    print("MM_SELFCHECK_OK" if (ok and all_ok and reg_ok) else "MM_SELFCHECK_FAILED")
    return results


def _regression_selfcheck(dev, nc: int = 12, size: int = 64, batch: int = 2) -> Dict[str, tuple]:
    """针对审计 7 个 P0 中**可自动化验证**的项。每项返回 (是否通过, 说明)。"""
    out: Dict[str, tuple] = {}
    # ---- P0-4：state_dict 无重复存储（EMA 不会双更新）----
    cfg = default_config(nc=nc)
    m = MMYOLO(cfg)
    sd = m.state_dict()
    names = list(sd.keys())
    seen: Dict[int, str] = {}
    dup = 0
    for n, t in sd.items():
        p = t.data_ptr()
        if p in seen and seen[p] != n:
            dup += 1
        seen[p] = n
    out["P0-4 共享层不再重复注册"] = (dup == 0 and len(names) == len(set(names)),
                                 f"重复存储键 {dup} 个（应为 0）")
    # ---- P0-3：EMA（深拷贝）走真实融合 ----
    from ultralytics.utils.torch_utils import ModelEMA
    counter = [0]

    def _bump(*_a):
        counter[0] += 1
    h = m.fusion["p4"].register_forward_hook(_bump)
    rgb = torch.rand(batch, 3, size, size, device=dev)
    ir = torch.rand(batch, 1, size, size, device=dev)
    dep = torch.rand(batch, 4, size, size, device=dev)
    dep[:, 2:3] = (torch.rand(batch, 1, size, size, device=dev) > 0.3).float()
    dep[:, 3:4] = 1.0
    q = {k: torch.rand(batch, 3, size // 8, size // 8, device=dev) for k in ("rgb", "ir", "dep")}
    pr = torch.rand(batch, 4, size // 8, size // 8, device=dev)
    m = m.to(dev).eval()
    with torch.no_grad():
        m(rgb, ir, dep, quality=q, prior=pr)
    n_call = counter[0]
    ema = ModelEMA(m)
    # EMA 副本上的 hook 必须指向 EMA 自己：手动给它注入 aux 特征，看融合块是否被调用
    with torch.no_grad():
        raw = ema.ema.fusion["p4"]
        raw_h = raw.register_forward_hook(_bump)
        base = counter[0]
        ema.ema.eval()

        def _spy(module, inputs, output, _m=ema.ema):
            _m._aux = {"ir": {IDX_P3: None, IDX_P4: torch.zeros_like(output),
                              IDX_P5: None},
                       "dep": {IDX_P3: None, IDX_P4: torch.zeros_like(output),
                               IDX_P5: None}}
            _m._fusing = True
            _m._quality = q
            _m._prior = None
            return output
        hh = ema.ema.backbone.model[IDX_P4].register_forward_hook(_spy)
        ema.ema(rgb, ir, dep, quality=q, prior=None)
        hh.remove()
        raw_h.remove()
    h.remove()
    out["P0-3 EMA 深拷贝后融合仍生效"] = (counter[0] > base,
                                     f"EMA 前向触发 FusionBlock {counter[0]-base} 次（应为 >0；"
                                     f"修复前为 0）")
    # ---- P0-7：checkpoint 记录训练模态与画布 ----
    tmp = Path(_HERE) / "_selfcheck_meta.pt"
    meta = {"modalities": ["rgb"], "canvas": [544, 960], "imgsz": [544, 960]}
    save_mm_checkpoint(tmp, m, epoch=1, best_map=0.1, meta=meta)
    m2, ck2 = load_mm_checkpoint(tmp, device="cpu")
    got_mod = tuple(getattr(m2, "infer_modalities", ()))
    got_can = tuple(getattr(m2, "infer_canvas", ()) or ())
    resolved = resolve_infer_modalities(m2, "all")
    tmp.unlink(missing_ok=True)
    out["P0-7 ckpt 自描述模态/画布"] = (got_mod == ("rgb",) and got_can == (544, 960)
                                   and resolved == "rgb",
                                   f"模态={got_mod} 画布={got_can} 推理自动解析={resolved}")
    # ---- P0-1 辅助：modality_off 真的关掉分支（RGB-only 权重不再误开未训练分支）----
    m3 = MMYOLO(default_config(nc=nc)).to(dev).eval()
    m3.modality_off = {"ir", "dep"}
    with torch.no_grad():
        y3 = m3(rgb, ir, dep, quality=q, prior=pr)
    outs3 = [t for t in (y3.values() if isinstance(y3, dict) else y3) if torch.is_tensor(t)]
    finite = all(torch.isfinite(t).all().item() for t in outs3)
    out["P0-1 modality_off 安全关闭分支"] = (finite and len(outs3) > 0,
                                        f"关闭 IR/Depth 后输出 {len(outs3)} 张、全有限={finite}")
    del m, m3, ema
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    _selfcheck()
