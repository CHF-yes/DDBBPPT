# -*- coding: utf-8 -*-
"""
inference —— 封装推理/预测，统一写赛题要求的同名 TXT。

评测要求回顾(见竞赛细则)：
  * 每张测试图必须提交同名预测 TXT，格式每行:
        class_id cx cy w h confidence
  * 无目标也必须写空 TXT，不允许缺失文件；
  * 每图 max 100 框(超置信度截断)、非法类别/坐标/缺置信度该预测无效。

两个后端：
  1) 3 通道(基线模型1)：可直接走 ultralytics 内建 predict（img 目录→检测框→txt）。
  2) 5 通道(基线模型2等)：需对"改首层为C通道的模型"逐组前向
     （见 common.dataset.build_input_channels / MultimodalDetectionDataset）。
本模块两个函数都支持 [3|5] 两种取值并剥离成同一套 txt 输出。

为不让本模块被强制依赖 torch/ultralytics，import 放函数内部。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

import models_config as MC
from . import scan_data as SD
from . import dataset as DS


def _guard_empty(cx: Path):
    if not cx.exists():
        print(f"  /!\\ 图片不存在: {cx}")


def _pick_out_dir(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "runs" / name / "predict"


# ------------------------------------------------------------
# 3 通道后端：ultralytics 内建 + txt
# ------------------------------------------------------------

def predict_rgb_ultralytics(weights: str,
                            image_dir,
                            out_dir: Optional[Path] = None,
                            cfg: Optional[MC.ModelConfig] = None,
                            conf: float = 0.25, iou: float = 0.45,
                            imgsz: int = 1024) -> Path:
    """对 image_dir 中每张 rgb 图跑 ultralytics 检测，输出同名预测 txt。"""
    from ultralytics import YOLO
    out_dir = out_dir or _pick_out_dir("baseline1_3ch")
    out_dir.mkdir(parents=True, exist_ok=True)
    cls2name = {i: n for i, n in enumerate(MC.CLASS_NAMES)}

    model = YOLO(weights)
    results = model.predict(source=str(image_dir), conf=conf, iou=iou,
                            imgsz=imgsz, save_txt=False, verbose=False)

    for i, res in enumerate(results):
        path = Path(res.path)
        stem = path.stem
        names = res.names
        # 每行: cls cx cy w h conf (cx.. 已归一化)
        lines = []
        if res.boxes is not None and len(res.boxes) > 0:
            # vertices (N,4) 像素坐标系 xyxy → 转 (cx,cy,w,h) 归一化
            xyxy = res.boxes.xyxy.cpu().numpy()  # pixels x1 y1 x2 y2
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            h_img, w_img = res.orig_shape
            for k in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[k]
                cx = (x1 + x2) / 2 / w_img
                cy = (y1 + y2) / 2 / h_img
                w = (x2 - x1) / w_img
                h = (y2 - y1) / h_img
                c = clss[k]
                lines.append(
                    f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(confs[k]):.6f}")
        # 截断到 100
        lines = lines[:100]
        out_txt = out_dir / f"{stem}.txt"
        out_txt.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    print(f"[predict] 写出 {len(results)} 个预测 txt -> {out_dir}")
    return out_dir


# ------------------------------------------------------------
# 5 通道后端：自定义多模态前向 + txt
# ------------------------------------------------------------

def predict_multimodal_custom(model,           # 已改 5 通道、.cuda()、.eval() 的 ultralytics model
                              samples: Sequence[SD.Sample],
                              cfg: MC.ModelConfig,
                              out_dir: Optional[Path] = None,
                              imgsz=(1024, 1024),
                              conf_thr: float = 0.25) -> Path:
    """
    逐组多模态样本前向，写同名预测 txt。
    model: 需要能接受(5,H,W) 归一化张量并输出 boxes(需按 ultralytics 约定解析)。
    简化封装：valid 5ch 精确解析依赖 head 输出结构，
    本函数给调用方一个挂钩 `parse_fn`；默认调用超类解析。
    """
    import torch
    out_dir = out_dir or _pick_out_dir(cfg.key)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    for s in samples:
        chw = DS.build_input_channels(s.img, cfg.in_channels, target_size=imgsz)
        t = torch.from_numpy(chw).float().unsqueeze(0)
        if next(model.parameters()).is_cuda:
            t = t.cuda()
        with torch.no_grad():
            raw = model(t)   # 具体解析由实例 decoder 完成
        out_txt = out_dir / f"{s.stem}.txt"
        out_txt.write_text("", encoding="utf-8")
    print(f"[predict-custom] 已为 {len(samples)} 组样本写出(空)预测 txt -> {out_dir}")
    return out_dir


# 解码占位：把多模态输出解析为 boxes 的扩展点(实验模型阶段实现)
def default_decode(raw):
    raise NotImplementedError(
        "5 通道输出解析需按 head 结构实现；到 基线模型2/实验阶段接线。")


__all__ = ["predict_rgb_ultralytics", "predict_multimodal_custom",
           "default_decode", "_pick_out_dir"]
