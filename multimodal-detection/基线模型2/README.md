# 基线模型2 —— 三模态 5 通道前期融合 YOLO

## 简洁定位

在**一个 YOLO**里，把 RGB、IR、Depth 三张空间对齐图在**输入端**拼接成 `5` 通道
（RGB 3 + IR 单通道 1 + Depth 归一化单通道 1），只改网络的**首层卷积**即可开训。
这是三模态融合里改动最小、基线最稳的一种。

| 项 | 值 |
|---|---|
| cfg key | `baseline2_5ch`（models_config.py） |
| 输入通道 | 5 = RGB(3) + IR(1) + Depth(1) |
| 首层改造 | 3→5，新增两通道用预训练 RGB 均值的 5% 起步（`strategy='mean_rgb'`） |
| 检测头 | COCO 80 类 → 赛题 12 类（经 data.yaml 的 nc，或 train 传 `nc=12`） |
| 类别表 | 见 `models_config.CLASS_NAMES`（person … tricycle，共 12） |

## 文件

- `model_builder.py`  构建：预训练 + 首层 3→5 权重继承 + 类头设置
- `dataset_adapter.py` 数据适配：按根目录格式扫描/配对三模态与标签，提供 `preview_sample`
- `main.py`           入口：`config / train / predict` 三子命令
- `__init__.py`       （占位，保证可作 Python 包）

## 数据格式适配说明（重要）

赛题格式：
- 一个样本 = 空间对齐的 RGB + IR（实为三通道一致的灰阶图，信息在单通道）+ Depth
  （16bit 毫米单通道、无效区多为 0）三种图 + 一个**同名** `.txt` 标签
- 标签每行：`class_id cx cy w h`（已归一化 0-1）

本文件夹的处理 `common.dataset`：
- IR 取 `[:,:,0]` 得到单通道（三通道冗余）
- Depth 按 `[0, 20000mm]` clip 缩放到 `[0,1]`，无效像素归 0
- 按 cfg.in_channels==5 拼成 (5,H,W)

## 使用

```powershell
# 0) 环境
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe -c "import ultralytics"

# 1) 先把赛题数据根路径配置好（改 models_config.py 的 DATA_ROOT，或用环境变量覆盖）
$env:MULTIMODAL_DATA_ROOT = "你自己的数据根目录"

# 2) 探测数据布局（打印树 + 配对 —— 正式数据首跑）
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe common/scan_data.py --root 你的数据根 --out data.yaml

# 3) 打印本版本配置
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型2/main.py config

# 4) 预览一个样本的 5 通道合成
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型2/dataset_adapter.py
```

> 注：`train` 子命令目前的实现聚焦"打印应交给训练器/自定义循环的超参"，因为
> ultralytics 内建 DataLoader 只能读"单图"、不能把三张对齐图合成一张 5 通道输入；
> 实际多模态训练需要自定义 Dataset（`common.dataset.MultimodalDetectionDataset`，
> 在 实验模型1/进阶 阶段完善）。此阶段交付的是**可跑通的构建/配置/数据组装骨架**。

## 关于权重策略

新增通道（IR、Depth）用预训练 RGB 三通道的**逐通道均值 × 5%** 作为初始，既不抛弃
预训练浅层结构，又给新模态一个"小信号"让它自由适应；也可改用 `strategy='zero'`
（彻底从零学两条新投影）。实验时可做两版对照选优。
