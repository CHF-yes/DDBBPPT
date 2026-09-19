# -*- coding: utf-8 -*-
"""IO 工具：Windows 下 OpenCV 读不了非 ASCII（中文）路径 —— 统一走 imdecode。

赛题数据根目录含中文（`初赛数据集-面向城市场景的多模态目标检测`），
`cv2.imread` 在 Windows 上对该路径一律返回 None（实测 2000/2000 失败），
必须用 `np.fromfile` + `cv2.imdecode` 绕过。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path, flags: int = cv2.IMREAD_UNCHANGED):
    """读图（支持中文/非 ASCII 路径）。失败返回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite_unicode(path, img, ext: str = ".png") -> bool:
    """写图（支持中文路径）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(p))
    return True
