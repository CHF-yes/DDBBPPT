# 基线模型1 —— 3 通道原版 YOLO（单模态可见光对照）

## 定位

只用 RGB 可见光的**原版 YOLO**：不改网络、不改首层通道。它是多模态实验的**对照基线**，
用来回答"我融了三模态到底比只用 RGB 好多少"。

| 项 | 值 |
|---|---|
| cfg key | `baseline1_3ch` |
| 输入 | RGB(3) 仅可见光 |
| 首层改造 | 无（原版 3 通道） |
| 检测头 | COCO 80 → 赛题 12 类 |
| 结构与 baseline2 | **对称**（便于两路消融对照） |

## 文件
- `model_builder.py`  加载预训练 3ch 权重 + 类头设定到 12 类
- `dataset_adapter.py`仅 RGB：扫描样本 + 说明 data.yaml 图像目录(即 RGB 图目录)
- `main.py`           `config / train / predict`
- `__init__.py`       （占位）

## 使用

```powershell
# 数据根配置
$env:MULTIMODAL_DATA_ROOT = "数据根"

# 探测数据布局并生成 ultralytics 12 类 data.yaml（内容含 RGB 图目录）
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe common/scan_data.py --root ... --out data.yaml

# 打印配置
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型1/main.py config

# 训练
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型1/main.py train --data-yaml data.yaml

# 对测试图目录预测 → 生成同名 txt
D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe 基线模型1/main.py predict --weights best.pt --images 测试图目录
```

> 说明：基线模型1 用 RGB 单图目录即可走 ultralytics 原版训练，代码中的 `train` 命令
> 已把 ultralytics 参数打印出来，正式跑只需放开 `model.train(**...)` 那一行
> （需 data.yaml 指向真实的训练/验证 RGB 图像目录）。
