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

## 评测口径（必须统一）

**赛题规则**：按"提交的全部框按置信度排序"算 AP、**每图 ≤100 框**、**无置信度阈值**。
因此本框架的评测/选优口径固定为：

```
conf_thres = 0.001      # 不是 0.25！
iou_nms    = 0.7
max_det    = 100        # 赛题每图上限
```

> 历史教训：早期用 `conf=0.25` 评测，会把置信度尚未校准好的模型低估 **约 30%**
> （实测同一 checkpoint：conf=0.001 得 0.1537，conf=0.25 只有 0.0523）。
> `实验模型1/eval_test.py` 的默认值已改为赛题口径；训练期日志里打印的 mAP 仍是
> conf=0.25（仅用于选 best，不用于对外报数）。

```powershell
# 同口径对比（exp1 与基线1）
$py = "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe"
& $py 实验模型1/eval_test.py --root "<VDT Test>" --imgsz 640 --no-depth-align --nc 45 `
      --exp1 code/runs/<run>/weights/best.pt --base1 code/runs/baseline1_3ch/train/weights/best.pt
```

## checkpoint 结构自描述（重要）

训练产出的 checkpoint 现在带 `structure` 字段，记录决定 **state_dict 键集合** 的全部开关：

```
mega / per_modality_gate / aux_depth_head / dist_head_src / aux_ch / gamma_init / dropout_p / nc
```

- 加载请用 `model_builder.load_experiment1_checkpoint(path, **overrides)`：
  按 `structure` **自动重建**结构并**严格加载**；`missing/unexpected` 非空**直接报错**；
- 旧格式（本次改动前训练的）checkpoint 无 `structure` → 需用
  `eval_test.py --struct "mega=false"` 之类显式对齐，否则会报错；
- **历史教训**：S3 是 `--no-mega` 训练的（0 个 MEGA 张量），按默认 `mega=True` 重建会
  得到 `missing=12`，MEGA 停在 α=0 恒等态——成绩被误读成"边缘融合有效"。

## 当前结果（VDT 本地数据，非赛题数据；Test 1000 组，赛题口径）

| 模型 | mAP50-95 | mAP50 | P | R |
|---|---|---|---|---|
| 实验模型1 **s4_edge**（best, ep103，真边缘头） | **0.5561** | 0.6844 | 0.769 | 0.669 |
| 实验模型1 s3_fixed（last, ep120） | 0.5543 | 0.6842 | 0.727 | 0.679 |
| 实验模型1 s4_edge（last, ep120） | 0.5517 | 0.6759 | 0.738 | 0.701 |
| 实验模型1 s3_fixed（best, ep104） | 0.5515 | 0.6785 | 0.735 | 0.669 |
| 基线模型1（仅 RGB，ultralytics mosaic） | 0.5385 | 0.6929 | 0.732 | 0.711 |
| 实验模型1 v3（旧版，无 mosaic） | 0.4480 | 0.6057 | 0.694 | 0.572 |
| 实验模型1 s2（mosaic 错位 bug） | 0.3844 | 0.5451 | 0.561 | 0.553 |

**结论（谨慎）**：多模态比纯 RGB 基线高 **+0.013~0.018**，且 **mAP50/召回反而更低**
（R 0.669 vs 0.711）→ 优势集中在高 IoU 区间，**尚不能证明融合方案有稳定优势**。
同权重消融：关融合 −0.054、关 IR −0.050、关 Depth **−0.013（近乎中性）**。
真边缘头（s4_edge）相对 s3 的 +0.0046（best vs best）**在噪声范围内**，见下文 Step4b 结论。

## Step4b：深度对齐诊断 + 真边缘头（隔离实验）

### 1. 深度↔RGB 对齐：赛题数据确有固定偏移，本地数据没有

测法：`check_align.py` 在**边缘域**做互相关（深度梯度图 ↔ 可见光梯度图），粗到细搜索
（`--work-w --range --min-peak`），只在峰值可信时计票。

| 数据 | 结果 |
|---|---|
| 赛题样例（`visible/infrared/depth` 各 18 组，1920×1080） | 中位 **shift_x = −21.0 px@1920**（IQR −27.2..−16.5；17 组中 8 组峰值可信），对齐后边缘相关 0.159 → **0.199** |
| 本地 VDT-2048 | ≈ **+1.5 px@1920（≈0）** |

→ 配置结论：赛题数据必须用 `align.mode="shift", shift_x=-22, shift_y=0, ref_size=1920`
（`models_config.AlignConfig`，按原图宽等比换算）；本地 VDT 全程 `--no-depth-align`。

**但对齐本身能带来的收益上限很小。** 在 S3 上用 300 组 Test 子集做"注入深度偏移"敏感性扫描：

| 注入 shift_x | +0 px | +10 px | +20 px | +40 px |
|---|---|---|---|---|
| mAP50-95 | 0.6272 | 0.6396 | 0.6379 | 0.6122 |

20 px 的深度错位只掉 <0.013、甚至在噪声内升高 → **模型对深度几乎不敏感**。
因此对齐是"输入管线正确性"修复（赛题数据必须做），**不是涨点手段**。

### 2. 真边缘头（MEGA → EdgeHead）：val 中期领先，**收敛后归零、Test 不迁移**（判定：无收益）

S4 (`experiment1_s4_edge`) 与 S3 的数据/划分/增强/种子/120 轮 cosine 调度完全相同
（两次日志均记录"从 train 抽出 209 组作验证集 ratio=0.2"，划分一致；S4 张量 918 个比
S3 的 910 个多出的 8 个正是 `EdgeHead` 的 conv+BN+1×1）,
唯一差异是 `--edge-head`（`EdgeHead` 用 **GT 框边界**做 BCE 监督，并把预测边缘图喂进
融合门控，取代 MEGA 的固定 Sobel），所以同轮次配对差分可直接解读。

#### 判定口径（预注册，先于看到结果写入）

- 统计量：`Δ_e = mAP50-95(S4, ep e) − mAP50-95(S3, ep e)`（训练日志 val，conf=0.25），
  对共同轮次取均值 ± 标准误，并给 95% 置信区间；
- 判据（用户给定）：**增益 ≥ +0.01 才算有收益**；
- 功效：单轮配对差标准差 ≈ 0.037 → n=60 轮时 95% 半宽 ≈ ±0.009；
  n=9 时半宽 ≈ ±0.024（只能排除 ±0.024 以上的效应）。

#### 结果：val 中期领先 → **收敛后归零，Test 不迁移**（最终判定：无收益）

**A. 同轮次配对 val 差分（120 轮全跑完，同数据/划分/增强/种子/调度）**

| 轮次区间 | mAP50-95 Δ（配对） | 95% CI | mAP50 Δ | 检测损失 Δ |
|---|---|---|---|---|
| ep1–9 | +0.0007 ± 0.0123 | −0.023..+0.025 | +0.019 | −0.007 |
| ep1–37 | +0.0243 ± 0.0070 | +0.011..+0.038 | +0.041 | +0.020 |
| ep1–60 | +0.0302 ± 0.0051 | +0.020..+0.040 | +0.046 | +0.019 |
| **ep1–120（全部）** | **+0.0253 ± 0.0030** | **+0.019..+0.031** | +0.034 | +0.015 |
| **最近 40 轮** | +0.0177 ± 0.0027 | +0.012..+0.023 | +0.015 | +0.013 |
| **最近 20 轮** | +0.0054 ± 0.0021 | +0.001..+0.010 | +0.000 | +0.015 |
| **最近 10 轮** | **+0.0009 ± 0.0025** | **−0.004..+0.006** | −0.008 | +0.015 |

→ 全轮次配对看是"有收益"（+0.025，CI 下界 +0.019 > +0.01），但**优势全部来自
ep18–100 的中期**；mosaic 关闭（ep103）后两者**收敛到同一水平**（最近 10 轮 ≈ 0）。
`edge_loss` 从 0.445 降到 0.17–0.19，边缘头确实是学出来的，只是**最终没留下差距**。

**B. 最终判据：Test 1000 组、赛题口径（conf=0.001 / max_det=100）**

| 模型 | mAP50-95 | mAP50 | P | R |
|---|---|---|---|---|
| S4_edge best (ep103) | 0.5561 | 0.6844 | 0.769 | 0.669 |
| S4_edge last (ep120) | 0.5517 | 0.6759 | 0.738 | 0.701 |
| S3 best (ep104) | 0.5515 | 0.6785 | 0.735 | 0.669 |
| S3 last (ep120) | 0.5543 | 0.6842 | 0.727 | 0.679 |
| 纯 RGB 基线 best | 0.5385 | 0.6929 | 0.732 | 0.711 |

- **best vs best：0.5561 − 0.5515 = +0.0046**（< +0.01 判据）；
  **last vs last：0.5517 − 0.5543 = −0.0026**。两者符号相反、量级 ±0.005 → **噪声范围内**。
- 结论：**真边缘头在赛题判据上没有收益**（按用户规则：无收益则停止，不再推进该方向）。

#### 机制：即使有中期优势，也**不**来自门控的边缘通道

`check_edge_effect.py`（把边缘头输出强制为 −20 → sigmoid≈0 → 门控边缘通道 ≡ 0，
其余完全不变，CPU 前向对比）：

| checkpoint | P3 相对差 | P4 | P5 | Detect 输出相对差 |
|---|---|---|---|---|
| S4 ep17 | 0.115% | 0.195% | 0.173% | **0.028%** |
| S4 ep22 | 0.140% | 0.227% | 0.183% | **0.029%** |

即：**推理时把边缘信息整条拿掉，网络输出只变 0.03%**——门控边缘通道本身几乎无用。
因此 val 中期的领先（若有训练层面的作用）只能来自训练期：边缘 BCE 的梯度回传到共享的
IR 辅助流 (`f_ir[1]`) 起正则作用，而**不是**来自"门控用不用边缘图"。

#### 教训（写下来避免重复踩）

1. **val 中期领先 ≠ 收益**：n=120 配对 +0.025 的 CI 下界（+0.019）明明越过了 +0.01，
   但它在收敛后归零、在 Test 上也不复现（+0.0046）。**判据必须用最终模型在 Test 上的
   赛题口径成绩**；配对 val 曲线只能当"是否值得继续"的早期信号。
2. **训练轨迹随机性不可忽视**：`EdgeHead` 初始化会消耗 RNG，使类别输出层等后续随机初始化
   与 S3 不同源；单次 run 对单次 run 的 ±0.005 差异无法归因。要下"某改动有效"的结论，
   至少需要第二种子复现（两次 run × 同轮次配对）。
3. **过早下结论两个方向都错**：ep9 时判"无效"（当时 Δ=+0.0007）→ 中期看像"有效"
   （+0.030）→ 最终判"无收益"。预注册口径 + 跑完再判是唯一稳妥做法。

→ 仍然成立的推论：融合强度（`tanh(γ)` ≈ 0.04–0.08；γ=0.99 时 ‖ΔF‖/‖F‖ 也只有
0.09–0.13）是这套架构的**上限瓶颈**，下一步该动的是注入强度（proj 输出归一化/
LayerScale、γ 去饱和/下限、模态竞争）+ depth 表示改造，见
[下一步实验设计.md](下一步实验设计.md)。

#### 结论边界（必须写清）

- 上面 A 表是**单次配对的 val 结果**（S3 vs S4，209 组 val，conf=0.25 仅用于选 best）；
  B 表是**最终判据**（Test 1000 组，赛题口径）。
- `EdgeHead` 的初始化会消耗 RNG，使后续随机初始化的类别输出层与 S3 不完全同源，
  因此**不能排除"训练轨迹随机性"**；要确证任何 ≥ +0.01 的收益都需要第二种子复现。
- 旧格式 checkpoint（Step4b 之前，如 S3）加载时必须同时给 `--struct "mega=false,gate_edge=false"`
  （门控输入通道少 1 个"预测边缘"），否则会 size mismatch 报错。

## 实现要点

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

- **消融开关**：`h.mega=False`、`h.aux_heads=False`、`cls_remap=False`（不迁移同名类行）；
  命令行：`--no-mega` / `--no-per-modality-gate` / `--no-dist-head` / `--mosaic-p` /
  `--fusion-warmup` / `--patience`（这些开关会改变 state_dict 键集合，**评估时必须与训练一致**，
  新 checkpoint 已自带 `structure` 自动对齐）。
- **模型规格**：换 yolo11m/l 时，P3/P4/P5 通道由探测自动取得，`aux_ch` 与注入器保持通道自洽即可。
- **预训练权重**：`yolo11s.pt` 必须置于 `code/` 根（离线环境缺失即报错，不联网下载）。
- **deepcopy/EMA**：若在任意前向之后 `ModelEMA(model)` 或 `deepcopy(model)`，需先清空
  `model._aux_logits`（非叶子张量不支持 deepcopy）；`train_loop` 在训练前创建 EMA，不受影响。
- **三模态必须同分辨率**：mosaic/letterbox 用同一组仿射矩阵处理 rgb/ir/depth，
  任一模态分辨率不同就会整体错位（`_read_source` 内有硬断言）。曾被"辅助源半分辨率解码"
  优化破坏过（实测框内深度中位数偏差 70%），已回退并加了回归检查。
