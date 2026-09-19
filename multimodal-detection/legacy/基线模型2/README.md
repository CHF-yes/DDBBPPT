# 基线模型2 —— 三模态 6 通道前期融合 YOLO

## 简洁定位

在**一个 YOLO**里，把 RGB、IR、Depth 三张空间对齐图在**输入端**拼接成 `6` 通道
（RGB 3 + IR 单通道 1 + Depth 双通道 2 = [归一化距离, 有效掩码]），只改网络的**首层卷积**即可开训。
这是三模态融合里改动最小、基线最稳的一种。

| 项 | 值 |
|---|---|
| cfg key | `baseline2_5ch`（models_config.py；key 保留旧名以兼容，实际为 6 通道） |
| 输入通道 | 6 = RGB(3) + IR(1) + Depth(2)（`in_channels=5` 可作无掩码消融） |
| 首层改造 | 3→6，新增通道用预训练 RGB 均值的 5% 起步（`strategy='mean_rgb'`） |
| 检测头 | COCO 80 类 → 赛题 12 类（`ensure_detect_classes` 重建 cv3 + 同步外层 nc/yaml） |
| 类别表 | 见 `models_config.CLASS_NAMES`（person … tricycle，共 12） |

## 文件

- `model_builder.py`  构建：预训练 + 首层 3→6 权重继承 + 类头设置（外层 `nc`/`yaml.nc` 同步）
- `dataset_adapter.py` 数据适配：`build_5ch_from_sample_paths`（默认 6 通道，走 AlignConfig 对齐）
- `main.py`           入口：`config / train / predict` 三子命令（train 用自定义循环 `train_loop.train_custom`）
- `__init__.py`       （占位，保证可作 Python 包）

## 数据格式适配说明（重要）

赛题格式：
- 一个样本 = 空间对齐的 RGB + IR（实为三通道一致的灰阶图，信息在单通道）+ Depth
  （16bit 毫米单通道、无效区多为 0）三种图 + 一个**同名** `.txt` 标签
- 标签每行：`class_id cx cy w h`（已归一化 0-1）

本文件夹的处理 `common.dataset`：
- IR 取 `[:,:,0]` 得到单通道（三通道冗余）
- Depth 按 `[0, 20000mm]` clip 缩放到 `[0,1]`，无效像素归 0；另生成**有效掩码**通道
  （0=无效/1=有效；掩码与距离通道均用 NEAREST 插值，无效区距离恒为 0）
- 按 cfg.in_channels==6 拼成 (6,H,W)（RGB 序 `[R,G,B,IR,D,mask]`）；`in_channels=5` 无掩码
- Depth 固定对齐由 `models_config.AlignConfig` 控制（`h.align`），平移量按**原始图宽**等比换算

## 使用

```powershell
# 0) 环境
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe -c "import ultralytics"

# 1) 先把赛题数据根路径配置好（改 models_config.py 的 DATA_ROOT，或用环境变量覆盖）
$env:MULTIMODAL_DATA_ROOT = "你自己的数据根目录"

# 2) 探测数据布局（打印树 + 配对 —— 正式数据首跑；V/T/D 目录布局自动回退）
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe common/scan_data.py --root 你的数据根 --out data.yaml

# 3) 打印本版本配置
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型2/main.py config

# 4) 预览一个样本的 6 通道合成
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型2/dataset_adapter.py
```

> 训练：`train` 子命令走 `common/train_loop.train_custom`（自定义循环：DataLoader 多进程读图 +
> v8DetectionLoss + EMA + 余弦调度 + 早停 + 赛题口径 mAP 评估），`--imgsz` 可快捷覆盖分辨率；
> 数据扫描为 0 样本时立即报错。验证/推理与训练共用同一 letterbox 预处理（几何一致）。
> 预测：`predict --weights <训练产物 best.pt> --imgsz <与训练一致> --out <目录>`，
> checkpoint 由 `common/train_loop.load_custom_checkpoint` 统一加载（兼容自定义与原生格式）。

## 关于权重策略

新增通道（IR、距离、掩码）用预训练 RGB 三通道的**逐通道均值 × 5%** 作为初始，既不抛弃
预训练浅层结构，又给新模态一个"小信号"让它自由适应；也可改用 `strategy='zero'`
（彻底从零学新投影）。实验时可做两版对照选优。
