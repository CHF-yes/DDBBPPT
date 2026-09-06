# -*- coding: utf-8 -*-
"""
common —— 三版本模型共享的代码层（零实例代码、纯复用逻辑）。

职责：
  - scan_data : 自动探测数据布局，按命名规则把三模态与标签配对，生成 ultralytics data.yaml
  - dataset   : 以「通道数参数化」方式加载 YOLO 格式数据集，
                3 通道只读 RGB；5 通道 = RGB + IR单通道 + Depth单通道。
  - trainer   : 封装基于 ultralytics 的训练入口，统一从 models_config 读取超参。
  - inference : 封装推理/预测入口，从 models_config 读取配置。

所有模块约定：**以 code 根目录为 sys.path 前缀**，内部用顶层 import
`import models_config as MC`（而非相对导入），规避中文包名的相对导入坑。
请通过 ensure_code_root() 或直接以 code 为工作目录运行实例入口。
"""

from __future__ import annotations

import sys
from pathlib import Path

# code 根目录 = 本文件(common/__init__.py)的上一级
CODE_ROOT: Path = Path(__file__).resolve().parent.parent


def ensure_code_root() -> Path:
    """把 code 根目录注入 sys.path，使 `import models_config`/`import common` 生效。"""
    root = str(CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return CODE_ROOT


# 包被 import 时立即确保 code 根在路径上（幂等）
ensure_code_root()

# 暴露公共 API
from .scan_data import scan_samples, build_data_yaml, print_dataset_tree  # noqa: E402
from .dataset import load_dataset_cfg, build_consistent_aug_5ch  # noqa: E402,F401
from .split_data import (  # noqa: E402,F401
    split_samples,
    report,
    write_split_txts,
    write_split_data_yaml,
    class_distribution,
)
from .multimodal_augment import (  # noqa: E402,F401
    flip_lr_consistent,
    letterbox_consistent,
    hsv_only_rgb,
    transform_boxes_letterbox,
    consistent_augment_full,
)

__all__ = [
    "CODE_ROOT",
    "ensure_code_root",
    "scan_samples",
    "build_data_yaml",
    "print_dataset_tree",
    "load_dataset_cfg",
    "build_consistent_aug_5ch",
    "split_samples",
    "report",
    "write_split_txts",
    "write_split_data_yaml",
    "class_distribution",
    "flip_lr_consistent",
    "letterbox_consistent",
    "hsv_only_rgb",
    "transform_boxes_letterbox",
    "consistent_augment_full",
]
