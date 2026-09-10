# 实验模型1 —— 官方 YOLO11s 全量主干 + 第 4/6/10 层零初始化残差注入

> **源码**：基于 `code/vendor/ultralytics`（8.4.138，与 EFYOLO 运行时一致），本目录所有
> `from ultralytics...` 均优先命中 vendor。入口已自动注入 `sys.path`。

## 架构（修复后）

```
                       官方 YOLO11s Sequential（COCO 权重全部保留）
rgb(3) ──▶ model[0:10] backbone ──▶ P3(层4)/P4(层6)/P5(层10)
                                        │  ▲ 零初始化残差注入（hook）
                                        ▼  │
                                    model[11:22] 预训练 neck（FPN+PAN，Concat 依赖 4/6/10）
                                        │
                                    model[23] Detect（回归分支 cv2+dfl 原样；仅 cv3 末层适配 nc）

ir(1) ─┐
depth(2)┴─▶ AuxStream×2 ─ ModalDropout ─▶ [IR,D] concat ─▶ inject3/4/5.proj(1×1) ─▶ γ 门控残差
                                                                  └ MEGA(仅 P4/P5, α 初始 0)
      └─▶ AuxHead×2（P4 中心分类，Step3 辅助损失）
```

**注入公式**（`FusionInjector`，注册在第 4/6/10 层的 forward hook 上）：

```
F' = F + tanh(gamma) · proj([IR, Depth])      # gamma 初始 0
F' = F' · (1 + alpha · Σ_m W_m·A_m)           # MEGA，alpha 初始 0（仅 P4/P5）
```

`gamma=0`、`alpha=0` 时两条式子严格恒等，因此 **RGB 三通道输入的完整前向
（P3/P4/P5 → 预训练 neck → Detect）与官方 YOLO11s 逐位相同**——自检脚本会断言这一点
（`python 实验模型1/model_builder.py`）。

## 与旧版的差别（本次修复）

| 项 | 旧版（有 bug） | 新版 |
|---|---|---|
| 主干 | 只取 `model[0:10]`，neck/Detect 丢弃 | 完整保留 `model[0:23]`（含预训练 neck + Detect） |
| Neck | 自建 `SimplePAN`（随机初始化） | 官方预训练 neck，COCO 权重继承 |
| Detect | 全新 `Detect(nc)`（回归分支随机） | 原 Detect，**仅** `cv3` 末层 1×1 换成 nc 类；cv2/dfl 原样 |
| 融合 | `FusionBlock` 替换 P3/P4/P5 | 第 4/6/10 层输出后零初始化残差注入（Sequential 不动） |
| 初始化 | 主干有 COCO 权重，其余随机 | 499 个 COCO 张量中 493 个逐位继承，仅 6 个类别输出层张量适配 |

## 文件

| 文件 | 说明 |
|---|---|
| `model_builder.py` | `AuxStream / ModalDropout / EdgeAttnGate / AuxHead / FusionInjector / Experiment1Model`（`__main__` 含完整自检） |
| `dataset_adapter.py` | `build_model_inputs`：三模态独立张量(Depth=[距离,掩码]) + 一致性增强 + AlignConfig 对齐 |
| `main.py` | 入口：`config` / `selfcheck` / `train` / `predict` |

## 运行

```powershell
$py = "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe"
& $py 实验模型1/main.py config
& $py 实验模型1/model_builder.py                  # 自检：等价性断言 + 梯度 + EMA + 优化器分组
& $py 实验模型1/main.py selfcheck
# 本地 VDT 数据（无 depth 偏移 → 加 --no-depth-align）
& $py 实验模型1/main.py train   --data-root "<VDT Train>" --class-num 45 --imgsz 640 --no-depth-align
& $py 实验模型1/main.py predict --weights runs/experiment1/weights/best.pt --data-root "<VDT Test>" `
      --class-num 45 --imgsz 640 --no-depth-align --out pred_exp1
```

## 关键实现

1. **COCO 初始化**：`YOLO(yolo11s.pt).model` 整网载入（含 EMA 权重），`self.backbone` 即该
   DetectionModel（命名沿用 `backbone.` 前缀，使 `common/train_loop._build_optimizer` 的
   `backbone_lr_mult=0.1` 分组继续生效）；`model` 属性返回 Sequential 本体，
   供 `v8DetectionLoss` 取 `model.model[-1]`。
2. **必须走 DetectionModel.forward**：Sequential 内 12/15/18/21 层的 Concat 依赖
   `from: [-1, 4/6/10/13]`，不能当普通 Sequential 直接调用。
3. **零初始化残差注入**：`FusionInjector.hook` 挂在预训练层上，`set_context()` 在每次前向
   注入辅助特征；γ=0 → 恒等。注意 γ=0 时 `proj` 的梯度为 0（`d/dW=γ·…`），第一次更新后
   γ≠0 即恢复，这是零初始化残差的正常行为。
4. **Detect 类别输出层适配**：新建 `Conv2d(c3, nc, 1)`，先置官方类别先验偏置
   `log(5/nc/(640/stride)²)`，再按**类别名**从 COCO 同行迁移（赛题 12 类命中
   person/boat/bicycle/car）；不调用 `detect.bias_init()`（它会覆写回归分支 bias）。
5. **Step0 对齐**：`models_config.AlignConfig`（`h.align`，mode=none|shift + 按原图宽等比换算）。
   本地 VDT 数据无偏移 → 用 `--no-depth-align` 同时关闭固定平移与随机平移增强。
6. **Step2 ModalDropout**：IR/Depth 各自整路独立置零（每样本每模态一次采样）。
7. **Step3 辅助头**：`AuxHead`（P4 中心分类）+ `_aux_center_loss`（λ 线性退火）；梯度只经 AuxStream。
8. **Step4 MEGA**：固定 Sobel 边缘（零参数）→ `A_m=σ(conv([F,E_rgb,E_ir,E_dep]))` × 模态权重
   → 残差保底门控；只作用于 P4/P5；开关 `h.mega`。
9. **训练循环**：`common/train_loop.train_custom`（复用 ultralytics `v8DetectionLoss`/EMA/调度/早停
   + 赛题口径 mAP 评估）。

## 注意事项

- **消融开关**：`h.mega=False`、`h.aux_heads=False`、`cls_remap=False`（不迁移同名类行）。
- **模型规格**：换 yolo11m/l 时，P3/P4/P5 通道由探测自动取得，`aux_ch` 与注入器保持通道自洽即可。
- **预训练权重**：`yolo11s.pt` 必须置于 `code/` 根（离线环境缺失即报错，不联网下载）。
- **deepcopy/EMA**：若在任意前向之后 `ModelEMA(model)` 或 `deepcopy(model)`，需先清空
  `model._aux_logits`（非叶子张量不支持 deepcopy）；`train_loop` 在训练前创建 EMA，不受影响。
