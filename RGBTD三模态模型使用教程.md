# RGBTD 三模态目标检测模型使用教程

> 本项目在原有 YOLOv11-RGBT（可见光 + 红外）基础上，重构新增了 **RGBTD 三模态（可见光 + 红外 + 深度）** 支持，并重点实现了 **"复用现有模型参数简化训练"** 能力（通过预训练权重迁移，用 COCO 预训练权重初始化三路分支，大幅提升训练效率与最终精度）。
>
> 所有改动为**增量式**，不影响原有 RGBT / RGBRGB6C 等功能的正常使用，运行环境与原来完全一致。

---

## 0. 当前状态：AIC2026 赛题适配（已完成训练）

本教程对应的项目现已**完成赛题适配，并已用官方 12 类数据训练出可用模型**。

### 0.1 已完成事项

| 项目 | 状态 |
|---|---|
| 数据集配置 | ✅ 新增 `ultralytics/cfg/datasets/aic2026-rgbtd.yaml`（12 类，指向数据盘） |
| 训练脚本 | ✅ `train_RGBTD.py` 已改为赛题版（12 类 + 全量训练集 + 三模态增强） |
| 验证脚本 | ✅ `val_RGBTD.py` 已升级为赛题版（参数化路径、深度对齐参数与训练一致） |
| 模型训练 | ✅ 300 epoch 跑完，最优 `best.pt` 已备份到本地 |
| 核心代码 | ✅ 本地与服务器逐文件 md5 一致（base/loaders/build/augment/utils/conv/tasks/predictor 等） |

### 0.2 训练成果（已下载到本地）

```
训练成果备份/AIC2026_12类_yolo11x-rgbtd/
├── best.pt     209.7 MB  ⭐ 最优权重（md5 a9c7980b674f71390f21024e01d54f31）
├── last.pt     209.7 MB
├── results.csv / args.yaml / results.png / confusion_matrix*.png
└── train_aic2026.log
```

| 指标 | 值（val_split，同源子集） |
|---|---|
| mAP50-95（最优） | 0.9065（epoch 290） |
| mAP50（最优） | 0.9867（epoch 284） |
| 训练耗时 | 19,932 秒 ≈ 5.54 小时（RTX 4090） |

> ⚠️ **口径提醒**：`val_split` 是从训练集切出的 200 张**同源**样本，指标天然偏高。
> 官方独立测试集（1000 张、无标签）上的真实分数预计在 **0.78~0.86**，须以平台评测为准。

### 0.3 赛题数据格式（官方）

```
初赛数据集-面向城市场景的多模态目标检测/
├── 训练集/AIC2026_Train_2000/
│   ├── visible/    2000 张  8bit RGB（.png/.jpg 混存，640×360 与 1920×1080 混存）
│   ├── infrared/   2000 张  8bit RGB（红外灰度堆叠 3 通道），与 visible 同名同扩展名
│   ├── depth/      2000 张  16bit 单通道灰度 PNG，与 visible 同名同扩展名
│   └── labels/     2000 个 YOLO txt（12 类，越界框已 clamp）
└── 测试集/AIC2026_PHASE_1_1000/   visible/infrared/depth 各 1000 张，**无标签**
```

**12 类名（class_id 0–11）**：
```
person, boat, animal, seat, sign, bicycle, car, ball, light, garbage_can, uav, tricycle
```

### 0.4 服务器训练时的关键配置（`train_RGBTD.py`）

```python
model = YOLO("yolo11x-RGBTD-pretrained.pt")   # 复用迁移后的三模态预训练权重
model.train(
    data="ultralytics/cfg/datasets/aic2026-rgbtd.yaml",
    imgsz=640, epochs=300, batch=8,           # 9 通道显存约为 RGB 的 3 倍
    close_mosaic=10, workers=8, device="0", optimizer="SGD", seed=0,
    use_simotm="RGBTD",                       # 三模态
    channels=9,                               # BGR(3)+IR(3)+Depth(3)
    pairs_rgb_ir=["visible", "infrared", "depth"],
    depth_shift_x=0, depth_shift_y=0,         # 官方数据已对齐，平移量为 0
    rgb_drop_prob=0.2, rgb_drop_mode="zero",  # RGB 随机失效，防单一模态依赖
    ir_gain=0.15, ir_bias=5.0,                # 红外增益/偏置抖动
    depth_noise=0.02,                         # 深度乘性噪声
    depth_jitter_x=[-25, 5], depth_jitter_y=[-5, 5], depth_jitter_prob=1.0,
    cache=False,                              # 三模态增强须在加载期随机执行
)
```

> **两处易错点**：① `depth_shift_x` 必须与数据实际对齐情况匹配（本赛题官方数据为 0）；
> ② 训练 / 验证 / 推理三处的三模态参数必须**完全一致**，否则会出现通道误判或指标失真。

---

## 1. 模型使用环境

与原有 RGBT 项目完全相同，无需额外改动：

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows / Linux 均可 |
| Python | ≥ 3.8（推荐 3.9~3.11） |
| PyTorch | 按 CUDA 版本安装（推荐 2.x + CUDA 11.8/12.1） |
| CUDA + GPU 驱动 | `nvidia-smi` 可识别 GPU；显存建议 ≥ 16GB（9 通道输入，x 模型显存占用约为 RGB 的 3 倍） |
| 依赖 | 项目根目录 `requirements.txt`（torch、torchvision、opencv-python、numpy、pandas、matplotlib、seaborn、einops、timm、efficientnet-pytorch 等） |

**安装注意**：本项目是 fork 的 ultralytics，**不要**再 `pip install ultralytics`（会覆盖成官方包），直接用项目根目录下的 `ultralytics/` 源码包。

```bash
cd DDBBPPT
pip install -r requirements.txt   # torch 建议按 CUDA 版本单独装（requirements 中已注释）
```

---

## 2. 新增的模型参数

本次重构新增/修改的参数与模块如下：

### 2.1 训练参数（`model.train(...)` 传入）

| 参数 | 取值 | 说明 |
|---|---|---|
| `use_simotm` | `"RGBTD"` | 新增的输入模态类型：可见光 + 红外 + 深度三模态 |
| `channels` | `9` | 输入通道数 = BGR(3) + IR(3) + Depth(3) |
| `pairs_rgb_ir` | `["visible", "infrared", "depth"]` | 三个**同级目录名**（原为二目录，现扩展支持三目录），加载时按字符串替换自动定位三模态文件 |
| `depth_shift_x` | `-22` | 深度图相对 RGB 的系统性水平偏移修正（<0 左移，实测约 −22px@1920×1080），训练/验证都做 |
| `depth_shift_y` | `0` | 深度图垂直偏移修正 |
| `rgb_drop_prob` | `0.2` | RGB 整图随机失效概率（仅训练），防网络只依赖 RGB、提升鲁棒性 |
| `rgb_drop_mode` | `"zero"` | RGB 失效模式：`zero` 全黑 / `gray` 灰度保结构 / `noise` 压暗+噪声 |
| `ir_gain` | `0.15` | 红外灰度增益抖动幅度（乘性，仅训练） |
| `ir_bias` | `5.0` | 红外灰度偏置抖动幅度（加性，0~255，仅训练） |
| `depth_noise` | `0.02` | 深度有效区乘性噪声比例（模拟测距抖动，仅训练） |
| `depth_jitter_x` | `[-25, 5]` | 深度水平随机平移区间（模拟对齐残差，仅训练） |
| `depth_jitter_y` | `[-5, 5]` | 深度垂直随机平移区间（仅训练） |
| `depth_jitter_prob` | `1.0` | 深度随机平移触发概率（仅训练） |

### 2.2 模型结构（新增 YAML）

`ultralytics/cfg/models/11-RGBT/yolo11-RGBTD-midfusion.yaml`

- 输入 `ch: 9`；
- **三条分支**（可见光 / 红外 / 深度），每条均为与官方 YOLO11 backbone 完全同构的 3 通道结构（Conv + C3k2 下采样到 P3/P4/P5 三尺度）；
- **融合层 `ModalConcat`**：三路 Concat + 1×1 卷积，把通道压回官方对应尺度（P3/P4/P5），使 SPPF/C2PSA/检测头与官方同构；
- 支持 `n/s/m/l/x` 全部 scale（文件名 `yolo11x-RGBTD-midfusion.yaml` 自动解析为 x）。

### 2.3 新增融合模块

`ModalConcat`（定义于 `ultralytics/nn/modules/conv.py`，已注册到 `tasks.py`）：
> 多路通道拼接 + 1×1 卷积 + SiLU，输出通道由 YAML 指定，用于三模态特征融合并**对齐官方尺度通道以复用预训练权重**。

### 2.4 深度预处理（数据层新增）

在 `ultralytics/data/base.py`（训练/验证）与 `ultralytics/data/loaders.py`（推理）中新增 `RGBTD` 加载分支：

- 深度图 **16bit 毫米 → 8bit [0,255]**，逐帧 min-max 归一化；
- 无效深度值（0 或过小）置 0；
- 深度单通道 replicate 成 3 通道；红外单通道文件也兼容转 3 通道；
- **兼容混入的 8bit 3 通道深度可视化图**（样例中 `depth/00000008.jpg`），自动转单通道灰度后再归一化，避免崩溃。

### 2.5 三模态对齐修正与一致性增强（新增）

为最大化**准确度 / 泛化 / 鲁棒性**，本次在数据层新增了三项关键处理（策略借鉴同赛题另一实现 `feature/multimodal-detection-framework` 的一致性口径）：

1. **深度对齐修正（配准，训练/验证都做）**：三模态虽已空间对齐，但深度相对 RGB 存在**系统性偏移**（实测约右偏 22px@1920×1080）。加载时用最近邻平移 `depth_shift_x=-22` 修正，标签不跟随（标签锚定 RGB 坐标系）。这是数据级硬伤，不修则深度分支学到错位特征。

2. **深度最近邻插值（防伪值）**：深度图 resize 用 `INTER_NEAREST` 而非线性插值，避免在"0=无效"与"有效深度"边界处引入插值伪值。

3. **三模态鲁棒性增强（仅训练期）**：
   - **RGB 随机失效**（`rgb_drop_prob`）：以一定概率让 RGB 整图全黑/灰度/噪声，迫使网络学会在 RGB 不可用时依赖 IR/Depth；
   - **红外增益/偏置抖动**（`ir_gain`/`ir_bias`）：模拟传感器响应差异；
   - **深度值噪声 + 随机平移**（`depth_noise`/`depth_jitter_*`）：模拟测距抖动与未对齐残差。
   - 几何操作（flip / mosaic / letterbox）由 ultralytics 内建 pipeline 在 9 通道拼接后统一执行，**三模态天然几何同步**。

4. **9 通道 HSV 增强 `RandomHSV9C`**（`ultralytics/data/augment.py`）：前 3 通道 BGR 做 HSV 抖动，后 6 通道（IR+Depth）只做亮度抖动，既不报错也不伪造温度/深度语义。

---

## 3. 训练集的格式与内容

### 3.1 目录结构（四目录同级、文件名一一对应）

**以下为官方样例的实际结构**（本重构已实测适配）：

```
<数据集根目录>/
├── visible/            # 可见光 RGB，8bit 3 通道，.png / .jpg 混存
│   ├── 00000008.jpg
│   ├── 000016.png
│   └── ...
├── infrared/           # 红外，8bit 3 通道（单通道灰度堆叠 3 份），与 visible 同名
│   ├── 00000008.jpg
│   ├── 000016.png
│   └── ...
├── depth/              # 深度，16bit 单通道灰度 PNG（样例混少量 8bit 3 通道 jpg 可视化图）
│   ├── 00000008.jpg
│   ├── 000016.png
│   └── ...
└── labels/             # YOLO 标注，与 visible 同名 .txt
    ├── 00000008.txt
    ├── 000016.txt
    └── ...
```

> 样例为**扁平结构**（无 `train/`/`val/` 子目录）。最终训练集若带 `train/`/`test/` 子目录，只需把 YAML 的 `train` 指向 `visible/train` 等，加载逻辑自动兼容。

**三种图像格式（样例实测）**：

| 模态 | 位深 / 通道 | 常见分辨率 | 格式 |
|---|---|---|---|
| 可见光 visible | 8bit · RGB 3 通道 | 1920×1080（jpg 为 640×360） | .png / .jpg |
| 红外 infrared | 8bit · RGB 3 通道（单通道灰度堆叠 3 份） | 同上 | .png / .jpg |
| 深度 depth | 16bit · 单通道灰度（正式）；样例混少量 8bit 3 通道可视化图 | 同上 | .png（.jpg 可视化图） |
| 标注 labels | — | — | .txt（YOLO） |

**关键约定**：

1. `train`/`val` 在数据集 YAML 中**只指向 `visible` 目录**，红外与深度靠 `pairs_rgb_ir` 的字符串替换自动定位；
2. 三模态文件**文件名必须一一对应**（已对齐），加载时 `file_path.replace("visible", "infrared")` / `replace("visible", "depth")` 得到另两模态路径；
3. **同一个 base name 的三种模态扩展名需一致**（如都 .png 或都 .jpg），否则字符串替换定位不到；不一致时请先统一扩展名；
4. **可见光**：常规 RGB 图像（jpg/png，代码以 BGR 读入）；
5. **红外**：3 通道（单通道灰度堆叠 3 份，赛题形式）或单通道灰度，代码均兼容（单通道自动 replicate 成 3 通道）；
6. **深度**：16bit 单通道（png/tif），单位毫米；0 或过小视为无效深度；混入的 8bit 3 通道可视化图自动转灰度处理；
7. **标注**：YOLO 格式 txt，每行 `class_id cx cy w h`（归一化到 [0,1]）；既支持独立 `labels/` 目录（与三模态目录同级），也支持与可见光图同目录，两种约定均已适配。

### 3.2 数据集 YAML 格式（示例见 `ultralytics/cfg/datasets/coco8-rgbtd.yaml`）

```yaml
path: /absolute/path/to/dataset    # 数据集根目录（绝对路径）
train: visible                     # 指向可见光目录（扁平结构）；带子目录则写 visible/train
val: visible                       # TODO: 从 train 切分出的验证集，避免与 train 同集
nc: 12                             # 类别数
# 赛题 12 类（按 class_id 0-11 顺序）
names: ["person", "boat", "animal", "seat", "sign", "bicycle",
        "car", "ball", "light", "garbage_can", "uav", "tricycle"]
```

> **赛题仅提供 train / test，无独立 val**。本地调参请从 train 切分出一部分作为 val（如 `visible/val`），否则验证指标虚高、无法反映真实泛化。

### 3.3 数据工程脚本（新增：自动配对 + 分层切分）

新增两个自包含脚本，放在项目根目录，用于赛题数据（2000 组、默认不分 train/val）的落地准备：

**① 布局探测 + 生成 data.yaml**（`scan_data.py`）

```bash
python scan_data.py --root /path/to/dataset --out data.yaml
```

- 自动探测 `visible/ infrared/ depth/ labels/` 四个同级目录（支持别名 rgb/ir/d 等）；
- 按 base name 自动配对三模态 + 标签，打印缺失统计（缺红外/深度、缺标签的样本数）；
- 兼容 `train/val/test` 分组布局（`--split-dirs train,val` 指定）。

**② 类别近似分层切分**（`split_data.py`）

```bash
python split_data.py --root /path/to/dataset --ratios 0.8,0.1,0.1 --seed 42 --out splits
```

- 把样本按 `8:1:1`（可调）切分为 train/val/test，**按类别近似分层**（稀有类优先落桶，避免某类全落进 test）；
- `--seed` 可复现（同数据 + 同种子 = 同划分）；`--no-stratify` 退化为纯随机；
- 只写 `train.txt/val.txt/test.txt` 图片清单 + `data.yaml`，**不移动/复制原文件**，原始数据保持只读；
- 生成的 `data.yaml` 的 `train/val` 指向同名 txt 清单（每行一个可见光图绝对路径），加载器会自动按目录名替换定位红外/深度。

> 建议流程：`scan_data.py` 确认布局可识别 → `split_data.py` 切分 → 把生成的 `data.yaml` 路径填进 `train_RGBTD.py` 的 `data=`。

---

## 4. 现成模型参数的使用方法（核心功能）

### 4.1 为什么需要"迁移"而不是直接 `model.load()`

本项目 `load()` 走 `intersect_dicts`（按 **key 名 + shape 完全一致** 才加载），而三模态分支的层命名与官方 YOLO11 不同（多了 Silence/SilenceChannel、三条分支各自占一段索引），所以直接 `model.load("yolo11x.pt")` **几乎迁移不了任何 backbone 权重**。因此提供专用迁移脚本。

### 4.2 迁移脚本使用

```bash
# 下载官方 COCO 预训练权重 yolo11x.pt（若本地没有）
python transfer_pretrained.py --src yolo11x.pt --out yolo11x-RGBTD-pretrained.pt
```

脚本做的事（对应模型结构设计）：

| 模型部分 | 预训练来源 | 处理 |
|---|---|---|
| 可见光分支（9 层 Conv/C3k2） | 官方 COCO `yolo11x.pt` | 逐层复制 |
| 红外分支 | 官方 COCO `yolo11x.pt` | 逐层复制 |
| 深度分支 | **复用红外分支迁移后的权重**（同为单通道灰度性质，最接近） | 复制初始化 |
| SPPF / C2PSA / 检测头 | 官方 COCO `yolo11x.pt` | 从 SPPF 起逐层对齐（通道已由 ModalConcat 对齐） |
| Detect 分类头 | — | nc 不同，shape 不一致自动跳过（回归部分保留） |
| 融合层 ModalConcat | — | 随机初始化（新增层） |

### 4.3 三个预训练来源（按优先级）

1. **官方 RGBT 两模态预训练权重**（README 网盘链接，若有）：可见光 + 红外两分支直接加载（完全同构），深度分支再从红外分支复制；
2. **官方 COCO `yolo11x.pt`**（最通用）：按 4.2 迁移到三路分支；
3. **迁移产物 `yolo11x-RGBTD-pretrained.pt`**：迁移脚本输出，训练时直接加载。

---

## 5. 模型使用教程（完整流程）

```bash
# 步骤 0：准备数据 —— 赛题 12 类配置已写好（ultralytics/cfg/datasets/aic2026-rgbtd.yaml）
#         只需把该 YAML 里的 path 改成你数据集的绝对路径

# 步骤 1：迁移预训练权重（复用现有模型参数，简化训练）
#         赛题部署时已生成 yolo11x-RGBTD-pretrained.pt，可直接跳到步骤 2
python transfer_pretrained.py --src yolo11x.pt --out yolo11x-RGBTD-pretrained.pt

# 步骤 2：训练（脚本已适配赛题：12 类 + 全量训练集 + 三模态增强，直接跑）
python train_RGBTD.py

# 步骤 3：验证 / 测试（参数化，默认即赛题配置）
python val_RGBTD.py                    # 常规验证
python val_RGBTD.py --tta              # 额外跑一次 TTA（多尺度 + 翻转）
python val_RGBTD.py --weights "训练成果备份/AIC2026_12类_yolo11x-rgbtd/best.pt"
```

**train_RGBTD.py 关键配置（AIC2026 赛题版，已落地）**：

```python
model = YOLO("yolo11x-RGBTD-pretrained.pt")   # 从迁移后的预训练权重开始
model.train(
    data="ultralytics/cfg/datasets/aic2026-rgbtd.yaml",  # 赛题 12 类数据集
    imgsz=640, epochs=300, batch=8,   # 9 通道显存约 RGB 的 3 倍；OOM 就降到 6
    close_mosaic=10, workers=8, device="0", optimizer="SGD", seed=0,
    use_simotm="RGBTD",
    channels=9,
    pairs_rgb_ir=["visible", "infrared", "depth"],  # 按实际目录名改
    depth_shift_x=0, depth_shift_y=0,  # 官方数据已对齐；换数据集需重新校准
    rgb_drop_prob=0.2, ir_gain=0.15, ir_bias=5.0, depth_noise=0.02,
    cache=False,                       # 三模态增强须在加载期随机执行
)
```

**val_RGBTD.py 支持的命令行参数**（升级后）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--weights` | `runs/RGBTD/aic2026-yolo11x-rgbtd/weights/best.pt` | 权重路径 |
| `--data` | `ultralytics/cfg/datasets/aic2026-rgbtd.yaml` | 数据集 YAML |
| `--imgsz` / `--batch` / `--device` | 640 / 8 / `0` | 与训练保持一致 |
| `--tta` | 关闭 | 开启测试时增强（先确认赛题是否允许） |

也可用环境变量 `RGBTD_WEIGHTS` / `RGBTD_DATA` 覆盖默认路径。

**若目录名不是 `visible/infrared/depth`**：改 `pairs_rgb_ir` 即可，例如 `["rgb", "ir", "depth"]` 或 `["images", "images_ir", "images_depth"]`。

---

## 6. 模型测试方法

### 6.1 验证集评估（推荐，比赛用 mAP）

```bash
python val_RGBTD.py                    # 常规验证
python val_RGBTD.py --tta              # 附带 TTA
```

脚本内包含两种模式：

1. **常规验证**：`model.val(data=..., use_simotm="RGBTD", channels=9, depth_shift_x=0, ...)` → 输出 mAP50 / mAP50-95；
2. **TTA 验证**：`model.val(..., augment=True)` → 测试时增强（多尺度 + 翻转），零训练成本提升 0.5~1.5 点。

> ⚠️ **必须一致**：验证时的 `use_simotm / channels / pairs_rgb_ir / depth_shift_x` 必须与训练完全相同，
> 否则会出现"9 通道被当成 1 通道"的报错，或指标失真（这是实测踩过的坑）。
>
> ⚠️ **赛题测试集（`AIC2026_PHASE_1_1000`）没有标签**，无法在本地计算 mAP。
> 它只能用于"推理 + 按平台格式产出预测结果文件"提交评测；要拿真实分数须以平台反馈为准。

### 6.2 单图 / 目录推理

```python
from ultralytics import YOLO

# 服务器训练产出，或本地备份：训练成果备份/AIC2026_12类_yolo11x-rgbtd/best.pt
model = YOLO("runs/RGBTD/aic2026-yolo11x-rgbtd/weights/best.pt")

# 预测时同样需指定三模态参数（与训练一致）
results = model.predict(
    source="path/to/visible/image_or_dir",
    imgsz=640,
    use_simotm="RGBTD",
    channels=9,
    pairs_rgb_ir=["visible", "infrared", "depth"],
    depth_shift_x=0,
    depth_shift_y=0,
    save=True,
)
```

> 推理走的是 `ultralytics/data/loaders.py` 的 `LoadImagesAndVideos`，
> 其中的深度对齐参数由 `predictor.py → build.py` 从配置透传，与训练共享同一套预处理。

---

## 7. 本次重构改动清单（速查）

| 文件 | 改动 |
|---|---|
| `ultralytics/data/base.py` | 新增 `RGBTD` 加载分支、三目录支持、深度归一化、9 通道合并、深度对齐修正、三模态鲁棒性增强 |
| `ultralytics/data/augment.py` | 新增 `RandomHSV9C`（9 通道 HSV 增强）+ `v8_transforms` 挂接 |
| `ultralytics/data/loaders.py` | 推理加载新增 `RGBTD` 分支、三目录支持 |
| `ultralytics/cfg/default.yaml` | 新增 `depth_shift_*` / `rgb_drop_*` / `ir_*` / `depth_*` 对齐与增强参数 |
| `ultralytics/nn/modules/conv.py` | 新增 `ModalConcat` 三路融合模块 |
| `ultralytics/nn/modules/__init__.py` | 导出 `ModalConcat` |
| `ultralytics/nn/tasks.py` | 导入并注册 `ModalConcat`（parse_model） |
| `ultralytics/models/yolo/detect/train.py` / `val.py` | 可视化判断纳入 `RGBTD` |
| `ultralytics/cfg/models/11-RGBT/yolo11-RGBTD-midfusion.yaml` | 新增三模态模型结构 |
| `ultralytics/cfg/datasets/coco8-rgbtd.yaml` | 新增数据集格式示例 |
| `transfer_pretrained.py` | 新增预训练权重迁移脚本 |
| `train_RGBTD.py` / `val_RGBTD.py` | 新增训练 / 测试脚本（含对齐与增强参数） |
| `scan_data.py` / `split_data.py` | 新增数据布局探测 + 类别近似分层切分脚本 |
| 本教程 | 新增使用说明 |

---

## 8. 常见问题

**Q1：`pairs_rgb_ir` 目录名对不上怎么办？**
在训练/验证脚本里改 `pairs_rgb_ir=["你的可见光目录", "你的红外目录", "你的深度目录"]`。

**Q2：深度是 16bit，会不会"淹掉"RGB/红外？**
不会，加载时已做逐帧 min-max 归一化到 [0,255]，并把无效值（0/过小）置 0。

**Q3：红外是 3 通道还是 1 通道？**
赛题红外为"单通道灰度堆叠 3 份"（3 通道）。代码兼容单通道文件（自动 replicate 成 3 通道）。

**Q4：显存不够怎么办？**
调小 `batch` 和 `imgsz`，或改用 `yolo11s/l/m` 版本（`yolo11s-RGBTD-midfusion.yaml`）。

**Q5：为什么不用 5 通道（BGR3+IR1+Depth1）而用 9 通道？**
为了三条分支都与官方 3 通道 backbone 完全同构，实现预训练权重 100% 复用（无需卷积核压缩等有损操作）。竞赛场景不计较通道冗余带来的显存开销。

**Q6：样例里 .jpg 和 .png 混存，会影响加载吗？**
不影响，只要**同一个文件（base name）的三种模态扩展名一致**即可（加载靠字符串替换，扩展名原样保留）。若出现 visible 是 .png、depth 是 .jpg 这类不一致，需先统一扩展名。

**Q7：样例深度里混了一张 8bit 3 通道的 jpg 可视化图，会不会报错？**
不会。深度预处理已兼容 3 通道输入（自动转单通道灰度后再归一化）。正式训练集的深度请统一为 16bit 单通道 PNG，效果最佳。

**Q8：`depth_shift_x=-22` 这个值怎么来的？要不要改？**
这是同赛题实现实测得到的深度相对 RGB 的系统性偏移量（1920×1080 分辨率下约右偏 22px）。如果你的数据对齐做得很好、或分辨率不同，可先用可视化工具叠加 RGB 与深度边缘确认，再调整该值；设为 0 即关闭修正。

**Q9：赛题数据没有 train/val 划分，怎么准备训练？**
用新增的 `split_data.py`：`python split_data.py --root 数据根 --ratios 0.8,0.1,0.1 --seed 42 --out splits`，会按类别近似分层切分并生成 `train.txt/val.txt/test.txt` + `data.yaml`，把该 yaml 填进 `train_RGBTD.py` 的 `data=` 即可。

**Q10：为什么加了 RGB dropout、深度噪声这些增强？会不会反而降精度？**
这是针对竞赛"泛化 + 鲁棒性"要求的定向增强：RGB dropout 防网络只依赖可见光（单模态过拟合）、红外/深度抖动模拟传感器差异、深度噪声/平移模拟测距抖动与对齐残差。在 2000 组小样本上能有效压低过拟合、提升跨场景鲁棒性。若数据质量极高、场景单一，可把 `rgb_drop_prob` 调低（如 0.1）或 `depth_jitter_prob` 设为 0。

**Q11：RGBTD 训练能不能开 `cache=True`？**
不建议。三模态鲁棒性增强（RGB dropout / 红外抖动 / 深度噪声平移）在**图像加载阶段**执行，`cache=True` 会把某一次增强结果缓存固定、丧失随机性；且 9 通道 cache 的内存/磁盘占用约为 RGB 的 3 倍，本身不划算。`train_RGBTD.py` 默认 `cache=False`，请保持。
