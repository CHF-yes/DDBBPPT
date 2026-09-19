# -*- coding: utf-8 -*-
"""mm_yolo 配置层 —— 换数据集只改这里（数据根/类名/类别数/分辨率/模态布局/对齐/融合档位）。

设计要点（对应方案 v4）：
* 结构开关（决定 state_dict 键集合）集中到 `structure()`，随 checkpoint 一起保存，
  加载时按它重建 → 杜绝"结构对不上却静默忽略"。
* 融合档位 fusion.tier ∈ {L0,L1,L2,L3}、编码器共享档 encoder.share_tier ∈ {a,b,c}，
  都是**可 A/B 的配置**，不写死在模型里。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent

# 赛题 12 类（顺序 = 标签里的 class id；来源：组织方类别表 / 既有 models_config，
# 训练不受影响，仅用于报告与同名类权重迁移）
COMPETITION_CLASS_NAMES = [
    "person",       # 0
    "boat",         # 1
    "animal",       # 2
    "seat",         # 3
    "sign",         # 4  路牌/标语/标志
    "bicycle",      # 5  双轮车（自行车/双轮电动车）
    "car",          # 6  四轮汽车
    "ball",         # 7
    "light",        # 8  路灯/室内照明灯
    "garbage_can",  # 9
    "uav",          # 10 无人机
    "tricycle",     # 11 三轮车
]


# ---------------------------------------------------------------- 子配置

@dataclass
class AlignCfg:
    """Stage A 对齐。mode: none | shift（固定全局）| per_image（逐图估计+先验收缩）。"""
    mode: str = "per_image"
    shift_x: float = 0.0
    shift_y: float = 0.0
    ref_size: int = 1920          # 固定偏移的参考原图宽（按比例换算到实际宽）
    max_shift: float = 40.0       # 逐图估计搜索范围（工作分辨率下的像素）
    work_width: int = 480         # 估计用的工作分辨率宽
    conf_min: float = 0.25        # 置信度低于此值 → 完全用全局估计
    trim_frac: float = 0.08       # 抗残影：裁掉梯度最强的 top 比例再估一次
    cache: str = ""               # 逐图参数缓存（npz/json），D0 一次算完


@dataclass
class EncoderCfg:
    """Stage B 编码器共享档：a=完全分开 / b=共享+低秩适配 / c=共享+逐模态 BN。

    浅层共享 / 深层分化的落地方式（D0 实测依据）：
      * RGB 与 IR 的边缘相关 0.19 → **共享 stem + 早期阶段**（都是强度/边缘类信号）；
      * depth 与 RGB 的边缘相关仅 0.05 → **depth 用自己的 stem**（度量不连续 ≠ 纹理边缘）。
    """
    share_tier: str = "c"
    adapter_rank: int = 16            # 档 b 的低秩适配秩
    per_modality_bn: bool = True      # 档 c 用；档 a 自带独立 BN
    share_stem_modalities: Tuple[str, ...] = ("rgb", "ir")   # 共享 stem 的模态
    depth_own_stem: bool = True       # depth 是否独立 stem（True 时不受上面共享影响）
    depth_input_channels: int = 4     # [相对, 绝对/20m, valid, metric_available]
    depth_view: str = "both"           # both / relative / metric_fallback / metric_log_fallback
    depth_init: str = "relative"       # relative / balanced / metric_fallback；只改变初始化
    metric_branch: bool = False         # independent absolute-distance feature path (v1)


@dataclass
class FusionCfg:
    """Stage C 融合档：L0 逐像素门控 → L1 +邻域 → L2 窗口跨模态注意力 → L3 +可变形偏移。"""
    tier: str = "L2"
    architecture: str = "legacy_hook_v1"  # legacy_hook_v1 / spatial_memory_v1
    memory_control: str = "unbounded_v1"  # v1 checkpoints retain exact forward semantics
    window: int = 3               # 窗口尺寸（奇数）
    heads: int = 4
    bottleneck: int = 8           # C' = C // bottleneck（8 → 参数量落在 0.3–0.6M）
    gamma_floor: float = 0.2      # 强制调制下限（初始化即通电）
    tau: float = 1.0              # 模态权重温度
    iters: int = 2                # 权重共享迭代次数（深度换参数）
    # 同一组动态 register 在一次前向内沿 P3→P4→P5 传递；不是每层独立 token。
    use_bus: bool = True
    late_bus: bool = False        # 旧的一次性片后 FiLM，仅保留作消融
    bus_tokens: int = 8
    bus_dim: int = 64
    bus_pool: int = 4             # 每模态读入 register 的池化网格边长
    residual_init: float = 0.05   # RGB 锚定融合的初始残差比例（小非零，保证梯度）
    register_residual_init: float = 0.02
    depth_scales: Tuple[bool, bool, bool] = (False, True, True)  # 幻影 depth 默认只进 P4/P5
    use_mask: bool = True         # 可靠性掩码门控 depth 特征
    max_offset: float = 3.0       # 可变形偏移上限（像素）
    deformable: Tuple[str, ...] = ("dep",)   # 哪些模态启用**学习式残差对齐**（L3 的按模态版本）
    quality_gate: bool = True     # 把"模态质量描述子"显式喂进门控（亮度/对比度/清晰度…）
    quality_channels: int = 3     # 每个模态的质量描述子通道数
    prior_film: bool = True       # 用 depth 几何先验做 FiLM（逆深度/法线）
    prior_channels: int = 4       # depth 先验通道数（逆深度/有效/法线x/法线y）
    slot_names: Tuple[str, ...] = ("rgb", "ir", "dep")


@dataclass
class MMConfig:
    """总配置。"""
    # 数据
    data_root: str = ""
    class_names: Tuple[str, ...] = ()
    imgsz: int = 960
    batch: int = 8
    workers: int = 4
    # 训练
    epochs: int = 300
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    freeze_encoder_epochs: int = 5        # 两阶段：先冻编码器
    patience: int = 80
    amp: bool = True
    # 模型
    weights: str = "yolo11s.pt"           # COCO 预训练（离线需自备）
    depth_resampling: str = "legacy_bilinear_v1"
    nc: int = 12
    cls_remap: bool = True
    # 子配置
    align: AlignCfg = field(default_factory=AlignCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    # 增强
    mosaic_p: float = 0.5
    close_mosaic_frac: float = 0.15
    degrade_p: float = 0.3                # 连续谱退化增强概率
    misalign_px: float = 5.0              # 轻微残余错位；幻影数据不宜再用±15px 强化
    rgb_dropout_p: float = 0.05           # 小概率防单模态塌缩；不是训练地基
    aux_dropout_p: float = 0.05

    def structure(self) -> dict:
        """决定 state_dict 键集合的全部开关（checkpoint 自描述）。"""
        return {
            "weights": str(self.weights),
            "nc": int(self.nc),
            "class_names": list(self.class_names),
            "encoder": asdict(self.encoder),
            "fusion": {**asdict(self.fusion),
                       "depth_scales": list(self.fusion.depth_scales)},
            "align_mode": self.align.mode,
            "depth_resampling": self.depth_resampling,
        }

    @staticmethod
    def from_structure(struct: dict) -> "MMConfig":
        cfg = MMConfig()
        cfg.weights = struct.get("weights", cfg.weights)
        cfg.nc = int(struct.get("nc", cfg.nc))
        cfg.class_names = tuple(struct.get("class_names", cfg.class_names))
        if "encoder" in struct:
            enc = dict(struct["encoder"])
            # B1 及更旧 checkpoint 的 Depth 是 [relative, valid] 两通道。
            # 缺字段时必须按 2 重建，否则旧权重无法严格加载。
            enc.setdefault("depth_input_channels", 2)
            cfg.encoder = EncoderCfg(**enc)
        if "fusion" in struct:
            f = dict(struct["fusion"])
            f["depth_scales"] = tuple(f.get("depth_scales", (True, True, True)))
            cfg.fusion = FusionCfg(**f)
        cfg.align.mode = struct.get("align_mode", cfg.align.mode)
        cfg.depth_resampling = struct.get("depth_resampling", "legacy_bilinear_v1")
        return cfg

    def resolve_weights(self) -> str:
        """把权重名解析成绝对路径（离线环境禁止联网兜底）。

        自包含实现：不依赖 legacy 的 models_config，避免新框架被旧配置拖住。
        """
        p = Path(str(self.weights)).expanduser()
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"[mm_yolo] 预训练权重不存在: {p}（离线环境禁止联网下载）")
            return str(p)
        for cand in (_CODE / p, _CODE.parent / p, Path.cwd() / p):
            if cand.exists():
                return str(cand.resolve())
        raise FileNotFoundError(
            f"[mm_yolo] 找不到预训练权重 {self.weights!r}（已查 {_CODE} / {_CODE.parent} / 当前目录）")

    def class_num(self) -> int:
        return len(self.class_names) if self.class_names else int(self.nc)


def default_config(**overrides) -> MMConfig:
    """比赛默认配置：12 类、960、融合 L2、逐模态 BN（档 c）。"""
    cfg = MMConfig(class_names=tuple(COMPETITION_CLASS_NAMES),
                   nc=len(COMPETITION_CLASS_NAMES))
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"未知配置项: {k}")
        setattr(cfg, k, v)
    return cfg
