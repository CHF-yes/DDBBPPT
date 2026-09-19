"""正式 RGB 训练的共享包。

多模态数据、训练与推理逻辑集中在 ``mm_yolo``，此处只保留 RGB trainer。
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent


def ensure_code_root() -> Path:
    root = str(CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return CODE_ROOT


ensure_code_root()

__all__ = ["CODE_ROOT", "ensure_code_root"]
