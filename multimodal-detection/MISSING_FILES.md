# 提交缺失文件清单 & 获取方式

> 提交：`43e4b3e`（分支 `feature/multimodal-detection-framework`）
> 本清单列出该提交**有意未包含**的文件，以及它们应如何获取。
> 排除原则：大文件不入库（GitHub 不适合托管模型权重/二进制产物）、本地生成物不入库、赛题保密数据严禁公开。

## 1. 缺失文件总表

| 路径 | 类型 | 大小 | 缺失原因 | 是否必需 |
|---|---|---|---|---|
| `yolo11s.pt` | 模型权重 | 19.3 MB | 二进制大文件，不入库 | 必需（推理/训练起点，但可随时下载） |
| `bus.jpg` | 示例图片 | 137 KB | 测试用示例，不入库 | 非必需（仅 experiment1.py 冒烟用） |
| `_align_check_out/*.jpg`（42 张） | 对齐诊断图 | ~10 MB | 本地诊断生成物，不入库 | 非必需（仅人工目测用） |

> 以上三类之外，其他文件（`models_config.py`、`common/`、`基线模型1/2`、`实验模型1`、`vendor/ultralytics` 源码）均已完整提交。

## 2. 获取方式

### 2.1 `yolo11s.pt`（ultralytics 官方 COCO 预训练权重）

- **方式 A（自动下载）**：运行任何需要权重的代码时，ultralytics 会自动下载到当前目录：
  ```powershell
  & "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
  ```
- **方式 B（GitHub 官方 release，直连）**：
  ```
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
  ```
- **方式 C（GitHub 加速镜像，本项目实际使用过，速度更快）**：
  ```
  https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
  ```
  下载命令：
  ```powershell
  curl.exe -sL -o yolo11s.pt "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"
  ```

### 2.2 `bus.jpg`（ultralytics 官方示例图）

- **官方地址**：`https://ultralytics.com/images/bus.jpg`
- **GitHub raw（加速镜像）**：
  ```
  https://gh-proxy.com/https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
  ```
- 用途：仅 `experiment1.py` 的 CPU/GPU 推理冒烟测试；手头没有时可直接省略该测试。

### 2.3 `_align_check_out/`（40+ 张对齐诊断图）

- **性质**：2026-09 对齐诊断期间由临时脚本生成的**本地产物**（RGB↔Depth 边缘叠加、平移对比图），**无网络来源、不可重新下载**。
- **是否影响运行**：完全不影响——框架代码不依赖这些图。
- **重新生成**：若需复现，需运行对齐诊断逻辑（边缘图平移 IoU 搜索 + 相位相关），该临时脚本未随提交入库；需要时可联系维护者获取脚本。

## 3. 特别声明：赛题数据集（不在此提交、也不会公开）

- 仓库运行所需的**三模态训练/示例数据**（`visible/ infrared/ depth/ labels/`，共 18 组示例 + 2000 组正式数据）**不属于本提交**，且按赛题数据保密要求**禁止上传/公开传播**。
- **获取途径**（赛题官方）：
  - 示例数据：https://pan.baidu.com/s/1FBMH8t-boH2j4YiRjrk9NQ（提取码 `sxwu`）
  - 正式数据：报名后由赛事方开放下载。
- 克隆本仓库后需自行准备数据目录，并通过 `models_config.py` 的 `DATA_ROOT`（或环境变量 `MULTIMODAL_DATA_ROOT`）指向数据根。

## 4. 其他不入库项（运行后生成）

| 路径 | 说明 |
|---|---|
| `runs/<key>/train/weights/best.pt` `last.pt` | 训练产物，训练后自动生成 |
| `__pycache__/`、`*.pyc` | Python 缓存 |
| `.reasonix/` | 编辑器/工具状态目录 |
