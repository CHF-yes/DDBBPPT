# -*- coding: utf-8 -*-
"""
基线模型1 —— 3 通道原版 YOLO 主入口（单模态可见光对照基线）。

用法（EFYOLO conda 环境）:
    python 基线模型1/main.py config
    python 基线模型1/main.py train --data-yaml data.yaml
    python 基线模型1/main.py predict --weights runs/baseline1_3ch/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _DIR.parent
for p in (str(_CODE_ROOT), str(_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_config as MC                      # noqa: E402
from common import trainer as TR                # noqa: E402
from common import inference as INF             # noqa: E402


def cmd_config():
    print(MC.summarize())
    print(TR.train_config_prints(MC.BASELINE1_3CH))


def cmd_train(args):
    from ultralytics import YOLO
    cfg = MC.BASELINE1_3CH
    kw = TR.build_train_kwargs(cfg)
    kw["data"] = args.data_yaml or str(MC.DATA_ROOT / "data.yaml")
    kw["model"] = cfg.hyper.pretrained_weights
    # ultralytics .train(classes=nc) 或由 data.yaml nc 决定类数
    kw["model"] = kw["model"]
    model = YOLO(kw.pop("model"))
    # 注意：YOLO.train 的 device / data 等来自 kw；此处拆分以远离副作用
    train_kwargs = {k: kw[k] for k in
                    ("data", "epochs", "imgsz", "batch", "device", "workers",
                     "optimizer", "lr0", "amp", "seed", "patience",
                     "project", "name", "exist_ok")}
    print("[baseline1 train] 参数如下（正式跑前请核对 DATA_ROOT/data.yaml 是否正确）：")
    print(train_kwargs)
    # 实际训练一行：model.train(**train_kwargs)  —— 等数据+服务器就绪手动放开
    # model.train(**train_kwargs)


def cmd_predict(args):
    from ultralytics import YOLO
    cfg = MC.BASELINE1_3CH
    images = args.images or str(MC.DATA_ROOT)
    INF.predict_rgb_ultralytics(args.weights, images, cfg=cfg, imgsz=cfg.hyper.imgsz)


def main():
    ap = argparse.ArgumentParser(prog="基线模型1")
    ap.add_argument("task", choices=["config", "train", "predict"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--data-yaml", default=None)
    args = ap.parse_args()
    {"config": cmd_config, "train": cmd_train, "predict": cmd_predict}[args.task](args)


if __name__ == "__main__":
    main()
