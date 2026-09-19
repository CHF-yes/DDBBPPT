# 本地 code 对原生 YOLO11 (ultralytics 8.4.138) 的改动说明

**一句话结论：原生 ultralytics 8.4.138 源码一行未改（vendor 与官方包逐文件一致，`diff -rq` 为 0）；全部"多模态改造"都以"旁挂新文件 + 运行时组合"方式实现，不侵入官方源码。**

---

## 1. 改了什么

### 1.1 直接改动官方源码：**无**

`code/vendor/ultralytics/` 是从 EFYOLO 环境 site-packages 复制的 **8.4.138 原始源码**，
校验命令（结果为 0 个差异文件）：

```bash
diff -rq <site-packages>/ultralytics code/vendor/ultralytics   # 0 差异
```

因此 **YOLO11 的 backbone/neck/head、Loss、标签分配、训练引擎全部保持原版**。

### 1.2 运行时"组合式"改造（零源码侵入，改的是实例而非官方代码）

| 位置 | 做法 | 说明 |
|---|---|---|
| 基线模型2 `model_builder.py` | `rebuild_first_conv`：把 `model.model[0].conv` 的 `nn.Conv2d(3,...)` **替换为 6 通道**，前 3 通道继承 COCO 权重、新增 3 通道（IR/距离/掩码）取 RGB 均值 ×5% 起步 | **早期融合**：输入 RGB3+IR1+Depth2(距离+有效掩码) 拼 6 通道；`in_channels=5` 可作无掩码消融 |
| 实验模型1 `model_builder.py` | 从 vendor `import Conv, C2f, Concat, Detect`，自组 `AuxStream + ModalDropout + FusionBlock + EdgeAttnGate(MEGA) + AuxHead + SimplePAN`；主 backbone **直接复用官方 yolo11.pt**（不改 yaml），按模块索引 4/6/10 取 P3/P4/P5（**P5=C2PSA，非 SPPF**） | **中间融合**：主路全量 + IR/Depth 轻量辅助流，P3/P4/P5 分级融合；Step2 逐模态整路 dropout；Step4 MEGA（固定 Sobel 边缘 + 位置级门 × 模态级权重 + 残差保底，P4/P5）；Step3 每模态辅助头（P4 中心分类 + λ 退火辅助损失） |
| common `dataset.py` / `multimodal_augment.py` | 三模态读取 → 6 通道张量拼装（RGB 序 + 有效掩码）；几何同步增强（flip/letterbox 三图共享、HSV 仅 RGB、Depth 最近邻）；`AlignConfig` 按原图宽换算深度平移 | 不触碰官方 DataLoader，独立数据管道；训练/验证/推理共用 `build_consistent_aug_5ch`（letterbox 一致） |
| `model_utils.py` | `ensure_detect_classes`：重建官方 Detect 头的 `cv3`（nc=80→赛题 12），并**同步外层** `DetectionModel.nc` 与 `yaml["nc"]` | 运行时改参数，不改源码 |
| `train_loop.py` | 自定义训练循环（DataLoader 多进程读图 + ultralytics `v8DetectionLoss` + EMA + 余弦调度 + 早停 + 赛题口径 mAP 评估 + 辅助损失）；`save_ckpt`/`load_custom_checkpoint` 统一 checkpoint 协议 | 复用官方损失/EMA 实现，仅组合不修改 |

### 1.3 全部新增文件（均为独立模块，官方包外）

```
models_config.py                     # 版本注册表（三版本配置 + AlignConfig/PreprocessParams/AugmentParams）
common/                              # 共享层 12 个
  __init__.py dataset.py evaluate.py inference.py mask_to_boxes.py
  model_utils.py multimodal_augment.py scan_data.py split_data.py
  train_loop.py trainer.py vis_boxes.py
experiment1.py                       # 早期冒烟实验
基线模型1/  __init__.py model_builder.py dataset_adapter.py main.py
基线模型2/  __init__.py model_builder.py dataset_adapter.py main.py
实验模型1/  __init__.py model_builder.py dataset_adapter.py main.py
```

---

## 2. 什么没改（保持原生）

- **训练/推理核心组件**：TaskAlignedAssigner 标签分配、`v8DetectionLoss`（BCE + DFL + CIoU）、EMA、学习率调度、`close_mosaic`、自动混合精度、DDP 多卡、NMS 后处理——**全部原版未动**（自定义循环是"调用"而非"改写"）；
- **模型结构文件**：`yolo11.yaml` 及其 backbone（C3k2/SPPF）、neck（PAN-FPN/C2PSA）、Detect 头——**未修改、未新增层**（实验模型1 只是"借用"其主干模块并外挂辅助流/融合/注意力/辅助头）；
- **数据增强管线**：官方 Mosaic/HSV/Flip/RandomPerspective/MixUp 等超参与实现未动；
- **官方内置 DataLoader**：未改造（多模态数据由 `common.dataset` 管道读取，训练主循环用自定义 `train_loop.py`，与官方 loader 并行不冲突）。

---

## 3. 为什么这样设计

1. **可升级/可对照**：vendor 保持逐字节原版，升级、对照上游、排查 bug 成本最低；
2. **交付清晰**：半决赛交代码时，"官方源码(原样) + 我们自己新增的公共层"边界一目了然，技术报告可直接引用本说明；
3. **改动面可控**：所有多模态逻辑集中在 `common/` 与各版本目录，互不干扰、可独立回退（如实验模型1 不要了，删除目录即可，基线1/2 不受影响）。

---

## 4. 验证方式（随时可复跑）

```bash
# vendor 与官方包一致性校验（应输出 0 个差异文件）
diff -rq /path/to/site-packages/ultralytics code/vendor/ultralytics | grep -v __pycache__ | wc -l

# import 冒烟（确认新增层可用）
python -c "import sys; sys.path.insert(0,'code'); import models_config, common; print('OK')"

# 实验模型1 结构自检（构建 + dummy 前向，含 MEGA/AuxHead）
python 实验模型1/main.py selfcheck

# 基线2 构建自检（首层 6 通道 / 外层 nc=12）
python -c "import sys; sys.path.insert(0,'code/基线模型2'); import model_builder as MB; w=MB.build_baseline2(); print(w.model.model[0].conv.in_channels, w.model.nc)"
```
