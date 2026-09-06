# RGBTD 三模态目标检测模型使用教程

> 本项目在原有 YOLOv11-RGBT（可见光 + 红外）基础上，重构新增了 **RGBTD 三模态（可见光 + 红外 + 深度）** 支持，并重点实现了 **"复用现有模型参数简化训练"** 能力（通过预训练权重迁移，用 COCO 预训练权重初始化三路分支，大幅提升训练效率与最终精度）。
>
> 所有改动为**增量式**，不影响原有 RGBT / RGBRGB6C 等功能的正常使用，运行环境与原来完全一致。

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
# 步骤 0：准备数据（见第 3 节），改好数据集 YAML 的 path/nc/names

# 步骤 1：迁移预训练权重（复用现有模型参数，简化训练）
python transfer_pretrained.py --src yolo11x.pt --out yolo11x-RGBTD-pretrained.pt

# 步骤 2：训练（编辑 train_RGBTD.py 里的 data 路径与 pairs_rgb_ir 目录名）
python train_RGBTD.py

# 步骤 3：验证 / 测试（编辑 val_RGBTD.py 里的 data 路径与权重路径）
python val_RGBTD.py
```

**train_RGBTD.py 关键配置**：

```python
model = YOLO("yolo11x-RGBTD-pretrained.pt")   # 从迁移后的预训练权重开始
model.train(
    data=R"ultralytics/cfg/datasets/coco8-rgbtd.yaml",  # 你的数据集
    imgsz=640,          # 显存允许可调 1280
    epochs=300,
    batch=8,            # 9 通道显存占用大，按 GPU 调整
    use_simotm="RGBTD",
    channels=9,
    pairs_rgb_ir=["visible", "infrared", "depth"],  # 按实际目录名改
)
```

**若目录名不是 `visible/infrared/depth`**：改 `pairs_rgb_ir` 即可，例如 `["rgb", "ir", "depth"]` 或 `["images", "images_ir", "images_depth"]`。

---

## 6. 模型测试方法

### 6.1 验证集评估（推荐，比赛用 mAP）

```bash
python val_RGBTD.py
```

脚本内包含两种模式：

1. **常规验证**：`model.val(data=..., use_simotm="RGBTD", channels=9, ...)` → 输出 mAP50 / mAP50-95；
2. **TTA 验证**：`model.val(..., augment=True)` → 测试时增强（多尺度 + 翻转），零训练成本提升 0.5~1.5 点。

### 6.2 单图 / 目录推理

```python
from ultralytics import YOLO
model = YOLO("runs/RGBTD/RGBTD-yolo11x-midfusion/weights/best.pt")
# 预测时同样需指定三模态参数
results = model.predict(
    source="path/to/visible/image_or_dir",
    imgsz=640,
    use_simotm="RGBTD",
    channels=9,
    pairs_rgb_ir=["visible", "infrared", "depth"],
    save=True,
)
```

---

## 7. 本次重构改动清单（速查）

| 文件 | 改动 |
|---|---|
| `ultralytics/data/base.py` | 新增 `RGBTD` 加载分支、三目录支持、深度归一化、9 通道合并 |
| `ultralytics/data/loaders.py` | 推理加载新增 `RGBTD` 分支、三目录支持 |
| `ultralytics/nn/modules/conv.py` | 新增 `ModalConcat` 三路融合模块 |
| `ultralytics/nn/modules/__init__.py` | 导出 `ModalConcat` |
| `ultralytics/nn/tasks.py` | 导入并注册 `ModalConcat`（parse_model） |
| `ultralytics/models/yolo/detect/train.py` / `val.py` | 可视化判断纳入 `RGBTD` |
| `ultralytics/cfg/models/11-RGBT/yolo11-RGBTD-midfusion.yaml` | 新增三模态模型结构 |
| `ultralytics/cfg/datasets/coco8-rgbtd.yaml` | 新增数据集格式示例 |
| `transfer_pretrained.py` | 新增预训练权重迁移脚本 |
| `train_RGBTD.py` / `val_RGBTD.py` | 新增训练 / 测试脚本 |
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
