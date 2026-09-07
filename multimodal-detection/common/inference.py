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

def predict_multimodal_custom(model,           # 内部检测网络(DetectionModel/Experiment1Model)，勿传 YOLO 包装器
                              samples: Sequence[SD.Sample],
                              cfg: MC.ModelConfig,
                              out_dir: Optional[Path] = None,
                              imgsz=(1024, 1024),
                              conf_thr: float = 0.25,
                              iou_thr: float = 0.7,
                              device: Optional[str] = None) -> Path:
    """
    逐组多模态样本前向 → NMS 解码 → 赛题同名 txt（class_id cx cy w h confidence）。

    与 3 通道版一致的约定：
      * 每图必写同名 txt；无目标写空文件；
      * 每图最多 100 框（按置信度截断）；
      * 坐标归一化（相对输入画布 W/H）。
    """
    import torch
    from .evaluate import _move_to_device, decode_preds
    from .multimodal_augment import effective_depth_shift
    out_dir = out_dir or _pick_out_dir(cfg.key)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _move_to_device(model, dev)

    n_written = 0
    for s in samples:
        chw = DS.build_input_channels(
            s.img, cfg.in_channels, target_size=imgsz,
            depth_shift=effective_depth_shift(cfg.hyper.align, imgsz[0]),
            preprocess=cfg.hyper.preprocess)
        H, W = chw.shape[1], chw.shape[2]
        t = torch.from_numpy(np.ascontiguousarray(chw)).float().unsqueeze(0)
        t = _move_to_device(t, dev)
        with torch.no_grad():
            out = model(t)
        det = decode_preds(out, cfg.class_num, conf_thr, iou_thr)[0]
        det = det.cpu().numpy() if not isinstance(det, np.ndarray) else det
        lines = []
        if len(det):
            for x1, y1, x2, y2, conf, cls in det[:100]:       # 按 conf 已降序，截断 100
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(conf):.6f}")
        (out_dir / f"{s.stem}.txt").write_text(
            "\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        n_written += 1
    print(f"[predict-custom] {n_written} 组样本 -> {out_dir}（含空 txt，≤100 框/图）")
    return out_dir


# 兼容旧占位：5 通道解码已由 evaluate.decode_preds 提供
def default_decode(raw):
    from .evaluate import decode_preds
    return decode_preds(raw, nc=12)


__all__ = ["predict_rgb_ultralytics", "predict_multimodal_custom",
           "default_decode", "_pick_out_dir"]
