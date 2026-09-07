# -*- coding: utf-8 -*-
"""
统一配置层 —— 管理「面向城市场景的视觉多模态目标检测」的多个模型版本。

本文件是三个模型版本（基线模型1 / 基线模型2 / 实验模型1）的唯一"版本注册表"。
所有子文件夹的入口通过 `import models_config as MC` 读取各自的配置，保证：
  1) 三版本字段结构一致、便于消融对照；
  2) 改动任何版本的超参/通道数只需改这一个文件；
  3) 后续新增版本（实验模型1 系列）只需在此追加一条注册即可。

本模块使用纯标准库 (dataclasses / enum / typing)，无第三方依赖，
确保可被任意实例的入口模块在任何阶段安全 import。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# 一、竞赛常量
# ============================================================

# 赛题官方 12 个类别（id 从 0 开始，顺序严格对应训练 txt 数值）
CLASS_NAMES: List[str] = [
    "person",       # 0
    "boat",         # 1
    "animal",       # 2
    "seat",         # 3
    "sign",         # 4  (路牌/标语/标志)
    "bicycle",      # 5  (双轮车：自行车/双轮电动车)
    "car",          # 6  (四轮汽车)
    "ball",         # 7
    "light",        # 8  (路灯/室内照明灯)
    "garbage_can",  # 9
    "uav",          # 10 (无人机)
    "tricycle",     # 11 (三轮车)
]
CLASS_NUM: int = len(CLASS_NAMES)

# 数据根目录占位 —— 拿到赛题正式数据后，改成你的数据根实际路径。
# 也可以用环境变量 DATA_ROOT 覆盖（例如在服务器上不修改本文件）。
_DATA_ROOT_ENV = "MULTIMODAL_DATA_ROOT"
DATA_ROOT: Path = Path(
    __import__("os").environ.get(_DATA_ROOT_ENV, r"D:\datasets\multimodal_det")
)

# 预训练权重目录：模型注册表所在目录（code 根）。训练/推理一律解析为绝对路径，
# 离线竞赛环境禁止 ultralytics 联网下载兜底。
_CODE_ROOT: Path = Path(__file__).resolve().parent


def resolve_pretrained_weights(name) -> Path:
    """
    把权重名/路径解析为绝对路径（P2-16）：
      1) 绝对路径：存在则直接返回，否则报错；
      2) 相对名（如 "yolo11s.pt"）：优先 code 根（与 models_config.py 同目录），
         其次当前工作目录；找不到直接 FileNotFoundError —— 禁止触发联网下载。
    """
    p = Path(str(name)).expanduser()
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(
                f"[cfg] 预训练权重不存在: {p}（离线环境禁止联网下载，请检查路径）")
        return p
    for cand in (_CODE_ROOT / p, Path.cwd() / p):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        f"[cfg] 找不到预训练权重 {name!r}（已查 code 根 {_CODE_ROOT} 与当前目录）。"
        f"请把权重放到 code 根或传绝对路径。")

# 每个样本应出现的通道文件名关键字（用于 scan_data 自动配对，可追加别名）
MODALITY_KEYS = {
    "rgb": ("rgb", "visible", "color"),
    "ir": ("ir", "infrared", "thermal"),
    "depth": ("depth", "d",),
}


# ============================================================
# 二、模型模态类型 / 融合类型
# ============================================================

class Modality(Enum):
    """版本使用的输入模态集合。"""
    RGB_ONLY = "rgb_only"          # 基线模型1：3 通道，仅可见光
    RGB_IR_DEPTH = "rgb_ir_depth"  # 基线模型2：5 通道，RGB+IR+Depth 前期融合


class FusionScheme(Enum):
    """融合阶段标注（用于文档/说明，代码上体现在首层卷积与数据加载通道数）。"""
    NONE_EARLY = "none_early"        # 无融合（单模态）
    EARLY = "early"                  # 前期融合：通道拼接一并进网络
    MIDFUSION = "midfusion"          # 中间融合：主干特征层（P3/P4/P5）分级融合


# ============================================================
# 三、训练 / 推理统一超参
# ============================================================

@dataclass
class AugmentParams:
    """
    统一数据增强配置 —— 三模态全覆盖（配置化，所有实例共用一份）。

    原则（与 multimodal_augment 实现一致）：
      * 几何（flip/letterbox 等）：RGB/IR/Depth 三图**同步**（同一参数），保证对齐不破坏；
      * RGB 光度：HSV 抖动（颜色语义）；
      * IR 光度：灰度增益/偏置抖动（模拟传感器差异，不做"变色"以防伪造温度）；
      * Depth 光度：有效区乘性噪声（模拟测距抖动，无效区保持 0）；
      * Depth 几何：随机平移（模拟对齐残差，标签不动——标签锚定 RGB）。
    """

    # ---- 几何（三图同步）----
    flip_p: float = 0.5              # 水平翻转概率
    vflip_p: float = 0.0             # 垂直翻转概率（默认关：上下语义失真）
    # ---- RGB 光度 ----
    hsv_rgb: bool = True             # 是否启用 RGB HSV
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    # ---- IR 光度 ----
    ir_gain: float = 0.15            # 增益抖动幅度（乘性）
    ir_bias: float = 5.0             # 偏置抖动幅度（加性，0~255 尺度）
    # ---- Depth 光度 ----
    depth_noise: float = 0.02        # 有效值乘性噪声比例
    # ---- Depth 几何（对齐残差模拟，训练期）----
    depth_jitter_x: tuple = (-25, 5)
    depth_jitter_y: tuple = (-5, 5)
    depth_jitter_prob: float = 1.0
    # ---- RGB 随机失效（仅训练；防网络只靠 RGB 偷懒，呼应赛题鲁棒性要求）----
    rgb_drop_prob: float = 0.2        # 每样本概率让 RGB 整图失效（不宜过高，主流是 RGB）
    rgb_drop_mode: str = "zero"       # zero=全黑 / gray=丢色保结构 / noise=噪声+压暗
                                       # （标签不动：标签锚定 RGB 坐标系，与内容无关）


@dataclass
class PreprocessParams:
    """
    数据预处理开关 —— 应对"赛题数据未归一化"（赛题原文：所有图像均未经过归一化处理）。

    默认处理策略（与现行为一致）：
      * rgb_mode='div255'  : [0,255] → [0,1]
      * ir_mode='div255'   : [0,255] → [0,1]
      * depth_mode='mm_unit': 毫米 / depth_scale_mm → [0,1]（赛题有效 [0,19999]，取 20000）
    可选实验模式：
      * 'none'             : 保留原始值域（如 RGB 8bit、IR 8bit、Depth 毫米原值）——用于
                             "归一化 vs 不归一化"的对照实验（注意：配合网络时需同步调 BN/激活）
      * depth 'raw'        : 深度原值（毫米）直接进网络
      * depth 'div255'     : 深度按 8bit 尺度缩放（仅当 depth 已是 8bit 化的数据时用）
    """

    rgb_mode: str = "div255"          # div255 | none
    ir_mode: str = "div255"           # div255 | none
    depth_mode: str = "mm_unit"       # mm_unit | raw | div255
    depth_scale_mm: float = 20000.0   # mm_unit 的归一化分母（赛题有效距离约 20m）
    depth_invalid_zero: bool = True   # 无效区(0/太小)保持 0（True）；False=填有效均值


@dataclass
class AlignConfig:
    """Depth 对齐（可插拔）：mode=none 关闭；shift 固定平移 + 按分辨率自动缩放。
    shift_x/y 在 ref_size 基准分辨率下实测（赛题示例 1920×1080 测出 -22px）；
    scale_with_res=True 时按目标宽等比换算（如 VDT 640 → -22×640/1920≈-7px）。
    """
    mode: str = "shift"            # none | shift
    shift_x: int = -22
    shift_y: int = 0
    ref_size: int = 1920           # 偏移实测时的基准图像宽度
    scale_with_res: bool = True


@dataclass
class HyperParams:
    """默认训练超参（针对 2000 组三模态小样本 + mAP@50-95 指标）。"""
    model: str = ""                       # 内部权重文件/结构名，由各实例填充
    epochs: int = 200                     # 2000 组小样本，给足轮数 + 早停兜底
    imgsz: int = 1024                     # 分辨率高更利于小目标(uav/ball)，服务器 32G 可撑
    batch: int = 8                        # 三模态 s 档 32G 下折中取值，可按实测显存调整
    device: str = "0"                     # cuda:0；无 GPU 可传 "cpu"
    workers: int = 4
    optimizer: str = "auto"
    lr0: float = 0.01
    amp: bool = True                      # 混合精度省显存
    seed: int = 42
    patience: int = 30                    # 早停
    pretrained_weights: str = "yolo11s.pt"  # COCO 预训练(允许)；换规格改此键

    # ---- Depth 对齐（可插拔配置；兼容旧 depth_shift_x/y 读取）----
    align: AlignConfig = field(default_factory=AlignConfig)

    # ---- Step3/4 多模态增强开关与辅助损失权重 ----
    mega: bool = True               # Step4 MEGA 边缘引导注意力(P4/P5)
    aux_heads: bool = True          # Step3 每模态辅助头(中心分类)
    aux_lambda: float = 0.1         # 辅助损失权重(训练期间线性退火→0)

    # ---- 统一数据增强（三模态全覆盖；各版本可覆写）----
    aug: AugmentParams = field(default_factory=AugmentParams)

    # ---- 数据预处理开关（赛题数据未归一化；三种模态值域策略可配置）----
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)

    # 兼容旧引用：h.depth_shift_x / h.depth_shift_y（读取 align）
    @property
    def depth_shift_x(self) -> int:
        return int(self.align.shift_x)

    @property
    def depth_shift_y(self) -> int:
        return int(self.align.shift_y)


# ============================================================
# 四、版本配置结构体
# ============================================================

@dataclass
class ModelConfig:
    """一个模型版本的完整描述。所有版本共用同一结构体（便于消融对照）。"""
    key: str                       # 全局唯一健，如 "baseline1_3ch"
    name: str                      # 人类可读名（对应文件夹）
    description: str               # 一句话说明
    in_channels: int               # 网络首层输入通道数：3 或 5
    modality: Modality             # 输入模态集合
    fusion: FusionScheme           # 融合阶段
    enabled: bool                  # 本次是否落地为可运行实例
    class_num: int = CLASS_NUM     # 检测头类别数(默认转成赛题 12 类)
    hyper: HyperParams = field(default_factory=HyperParams)
    notes: str = ""                # 备注：实现约束 / 待办
    data_prep: str = ""            # 数据预处理插桩说明，便于在技术报告复用


# ============================================================
# 五、版本注册表（唯一登记处）
# ============================================================

# ---- 基线模型1：3 通道原版 YOLO（单模态 RGB only，对称完整实例）----
BASELINE1_3CH = ModelConfig(
    key="baseline1_3ch",
    name="基线模型1",
    description=("3 通道原版 YOLO（仅可见光 RGB）。作为单模态对照基线，"
                 "用于衡量多模态融合带来的增益。"),
    in_channels=3,
    modality=Modality.RGB_ONLY,
    fusion=FusionScheme.NONE_EARLY,
    enabled=True,
    hyper=HyperParams(pretrained_weights="yolo11s.pt"),
    data_prep="仅读取 RGB 三通道并归一化；Depth/IR 不参与该版本。",
)

# ---- 基线模型2：6 通道早期融合（RGB3 + IR 1 + Depth 2 [距离+掩码]）----
BASELINE2_5CH = ModelConfig(
    key="baseline2_5ch",
    name="基线模型2",
    description=("6 通道早期融合：RGB(3)+IR 单通道灰度(1)+Depth 双通道(2)=[归一化距离,有效掩码]。"
                 "把三模态在输入端拼接成 6 通道，仅改造首层卷积，最小侵入的融合基线。"),
    in_channels=6,
    modality=Modality.RGB_IR_DEPTH,
    fusion=FusionScheme.EARLY,
    enabled=True,
    hyper=HyperParams(pretrained_weights="yolo11s.pt"),
    notes=("首层 Conv3→6 权重初始化：前 3 通道继承预训练 RGB 权重，"
           "新增通道复制其三通道均值后再按比例缩放（见实例 model_builder）。"
           "Step1: Depth 双通道=[归一化距离, 有效掩码]；in_channels=5 可作无掩码消融。"),
    data_prep="RGB/IR 读图后统一转 RGB 序并归一化；IR 取单通道灰度；Depth 毫米值缩放至[0,1]+有效掩码。",
)

# ---- 实验模型1：RGB 主流 + 轻辅助流 + 分级融合（已落地完整实例）----
EXPERIMENT1 = ModelConfig(
    key="experiment1",
    name="实验模型1",
    description=("RGB 主流(yolo11s) + IR/Depth 轻量辅助流 + P3/P4/P5 分级融合 + ModalDropout "
                 "+ MEGA 边缘引导注意力 + 每模态辅助头（Step0-4 方案完整实现）。"),
    in_channels=6,                          # 三路输入总通道：RGB3 + IR1 + Depth2（非单张拼接）
    modality=Modality.RGB_IR_DEPTH,
    fusion=FusionScheme.MIDFUSION,          # 主干 P3/P4/P5 分级中间融合（见 model_builder）
    enabled=True,                           # 已落地可运行实例（config/selfcheck/train/predict）
    hyper=HyperParams(pretrained_weights="yolo11s.pt"),
    notes=("输入为三路 tensor：rgb(B,3,H,W)/ir(B,1,H,W)/depth(B,2,H,W)——"
           "in_channels=6 仅表示总输入通道数，非单张拼接。"
           "训练/推理见 实验模型1/main.py；checkpoint 见 common.train_loop.load_custom_checkpoint。"),
)

_MODELS: List[ModelConfig] = [BASELINE1_3CH, BASELINE2_5CH, EXPERIMENT1]
MODELS: Dict[str, ModelConfig] = {cfg.key: cfg for cfg in _MODELS}


# ============================================================
# 六、查询辅助
# ============================================================

def get_keys() -> List[str]:
    """返回所有已注册版本 key（按登记顺序）。"""
    return list(MODELS.keys())


def get(key: str) -> ModelConfig:
    """按 key 取配置；找不到抛 KeyError（含更友好的提示）。"""
    if key not in MODELS:
        raise KeyError(
            f"未注册的模型版本 key: {key!r}。可用: {get_keys()}。"
        )
    return MODELS[key]


def summarize() -> str:
    """打印版本清单摘要（用于控制台/README 对照）。"""
    lines = ["===== Model versions registry ====="]
    lines.append(f"{'key':<14}{'名称':<10}{'ch':>4}  {'融合':<10}  描述")
    for cfg in _MODELS:
        fuse = cfg.fusion.value
        on = "启用" if cfg.enabled else "占位"
        lines.append(
            f"{cfg.key:<14}{cfg.name:<10}{cfg.in_channels:>4}  "
            f"{fuse:<10}  [{on}] {cfg.description[:40]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())
