# 实验模型1 —— RGB 主流 + 轻辅助流 + 分级融合

> **源码**：基于 `code/vendor/ultralytics`（8.4.138，与 EFYOLO 运行时一致），本目录所有
> `from ultralytics...` 均优先命中 vendor。入口已自动注入 `sys.path`。

## 架构

```
RGB(3) ──▶ 主流 backbone(yolo11s, COCO 预训练) ──▶ P3(1/8) P4(1/16) P5(1/32)
IR(1) ──┐
Depth(1)┴─▶ 每模态轻量 AuxStream(3 级下采样) ──▶ 与 P3/P4/P5 同分辨率
                     │ 模态 dropout(训练随机置零 IR 或 Depth 辅助特征)
                     ▼
           FusionBlock(concat+1×1) ×3  ──▶ SimplePAN(FPN+PAN) ──▶ Detect(nc=12)
```

**体现"互补"的机制**：主流语义（RGB）+ 辅助流（IR 温差 / Depth 几何）在 P3/P4/P5
显式融合，且深度融合可用更重的算子替换 `FusionBlock`（如跨模态注意力，见下）。**

## 文件

| 文件 | 说明 |
|---|---|
| `model_builder.py` | 模型骨架：`Experiment1Model`（`__main__` 可自检 dummy 前向） |
| `dataset_adapter.py` | `build_model_inputs`：三模态独立张量 + 一致性增强 + depth 对齐/抖动 |
| `main.py` | 入口：`config` / `selfcheck` / `train`(占位) / `predict`(占位) |

## 运行

```powershell
& "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" 实验模型1/main.py config
& "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" 实验模型1/model_builder.py   # dummy 前向自检
```

## 待接线（下一步工作）

1. **训练循环**（自定义）：`build_model_inputs` 产数据 → 模型前向 → 用
   `ultralytics.utils.loss.v8DetectionLoss(model)`（或 yolo26 对应损失）算 loss →
   EMA/调度/早停。可参考 `common/trainer.py` 的 kw 组装。
2. **深度融合升级**：把 `FusionBlock`（NiN 级）替换为可学习加权或跨模态注意力
   （参考 ddbbppt 的 `TransformerFusionBlock`/`CrossTransformerFusion`，注意其面向两路，
   需扩展为三路或 IR+D 先融合）。
3. **对齐参数**：`models_config.EXPERIMENT1.hyper` 的 `depth_shift_x/y` 与
   `depth_jitter_*` 已生效于 `build_model_inputs`（与基线模型2 共用同一套配置字段）。
4. **模型规格**：换 yolo11m 等需同步调整 `fusion_ch`（对应 backbone P3/P4/P5 输出通道）。
