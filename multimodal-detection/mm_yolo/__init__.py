# -*- coding: utf-8 -*-
"""mm_yolo：三模态检测框架（方案 v4 的实现）。

模块：
  config.py   配置层（换数据集只改这里）
  align.py    Stage A 对齐与表示（逐图估计 / 亚像素补偿 / 相对深度 / 可靠性掩码）
  fusion.py   Stage C 融合块（L0–L3 四档 + L4 register 总线）
  model.py    Stage B/D/E 组装 + 自检 CLI
"""
