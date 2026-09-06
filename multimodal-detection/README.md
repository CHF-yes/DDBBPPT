# 面向城市场景的视觉多模态目标检测 —— 训练框架 (code/)

本目录是一个"多版本分离"的 YOLO 训练/推理骨架，用于赛题：
**三模态（RGB + Infrared + Depth）12 类目标检测，指标 mAP@50-95**。

## 目录结构

```
code/
├── models_config.py          ★ 唯一版本注册表（改配置就改这一处）
├── common/                   共享代码层（不掺实例逻辑）
│   ├── __init__.py           sys.path 引导 + 公共 API
│   ├── scan_data.py          数据布局探测 + 生成 ultralytics data.yaml
│   ├── split_data.py         train/val/test 划分（设种子 + 类别分层）+ 清单/yaml
│   ├── dataset.py            三/五通道读取与拼装；标签/预测 txt 读写
│   ├── multimodal_augment.py 三模态一致性增强（几何同步 + 仅RGB变色 + Depth最近邻）
│   ├── model_utils.py        YOLO 首层 3→5 改造 / 类头设定 / 权重继承
│   ├── trainer.py            训练超参封装（从 models_config 读）
│   └── inference.py          推理 + 输出赛题同名预测 txt
├── 基线模型1/                3 通道原版 YOLO（单模态 RGB 对照）
│   ├── model_builder.py  dataset_adapter.py  main.py  README.md  __init__.py
├── 基线模型2/                5 通道前期融合（RGB3+IR1+Depth1）
│   ├── 同基线模型1 的对称四件套
├── 实验模型1/                [占位] 进阶融合预留，不建实例（仅 README + __init__.py）
└── vendor/ultralytics/       本地钉死的 ultralytics 8.4.138 源码（与 EFYOLO 运行时同版本）
                             实验模型1 源码级改造时使用；基线1/2 继续用 pip 版，互不影响
```

依赖运行环境建议：EFYOLO conda（已装 ultralytics + CUDA）。

## 三版本差异（消融对照表）

| 版本 cfg key | 目录 | 输入 | 融合 | 首层 | enabled | 说明 |
|---|---|---|---|---|---|---|
| `baseline1_3ch` | 基线模型1 | RGB 3ch | 无 | 3(原版) | ✅ | 只读 RGB 的对照基线 |
| `baseline2_5ch` | 基线模型2 | RGB+IR+Depth 5ch | 前期(cat) | 3→5 | ✅ | 三模态拼接的最小融合基线 |
| `experiment1`  | 实验模型1 | 5ch(待定) | 进阶(待) | 待定 | ❌占位 | 中间/注意力/门控/模态dropout |

> 对分数来源的取向（见讨论）：**融合模块设计 > 输入分辨率 > 网络规格(n/s/m/l/x)**；
> 在 2000 组、12 类、mAP@50-95 的高框精度需求下，优先 s/m + 高 imgsz + 扎实验证融合。

## 使用流程

```powershell
$env:MULTIMODAL_DATA_ROOT = "填你自己的赛题数据根"   # 或改 models_config.py 的 DATA_ROOT

# ① 建模型骨架环境（已在 EFYOLO 装好 ultralytics/cuda）
code> & "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" -c "import ultralytics, torch; print(torch.cuda.is_available())"

# ② 探测数据根布局、按命名自动配对三模态+标签、生成 data.yaml
code> & "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" common/scan_data.py --root $env:MULTIMODAL_DATA_ROOT --out data.yaml
#    提示：若没有 train/val 分组，先用 --split-dirs train,val 或先切出验证集

# ③ 查看各版本配置 / 构建
code> & "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" 基线模型1/main.py config
code> & "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" 基线模型2/main.py config

# ④ 训练 / 预测 —— 详见各子目录 README
#    3 通道可走 ultralytics 内建（基线1）；5 通道多输入需后续(实验)阶段自定义循环。
```

## 训练 / 验证 / 测试 划分接口（common/split_data.py）

赛题 2000 组默认不分 train/val/test，落地前需先划分。本模块提供，**可设种子复现、近似按类别分层**：

```powershell
# 把数据根目录扫描到的样本划分为 train/val/test=8:1:1，并生成图片清单 + data.yaml
code> & "D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe" common/split_data.py `
      --root $env:MULTIMODAL_DATA_ROOT --ratios 0.8,0.1,0.1 --seed 42 --out splits

# 常用参数
#   --seed       随机种子（默认 42）—— 同一数据 + 同一种子 = 同一划分（可复现实）
#   --ratios     train,val,test 比例（默认 0.8,0.1,0.1）
#   --no-stratify 关闭按类别分层，退化为纯随机切分
#   --split-dirs 只划分根目录下指定子目录（如不分组用默认把所有样本并入一份划分）
#   --out        清单/yaml 输出目录（默认 <数据根>/splits，产出 train.txt/val.txt/test.txt/data.yaml）
```

是否真正分层取决于你数据里每张图的类别数：
- 多数情形图片内含单类目标多框 → 分层效果好；
- 若单图常含多个类（多标签）→ 用 `--no-stratify` 纯随机即可，代码已对两种都支持。

> 早停说明：`models_config.HyperParams` 里 `epochs` 是"轮数上限"，
> `patience` 才是决定停止的早停窗口
> （验证指标连续 N 轮不升自动停止并保留最优权重 `best.pt`）。提交/评估应优先用 `best.pt` 而非 `last.pt`。

## 三模态一致性增强（common/multimodal_augment.py）

三模态(RGB+IR+Depth)是**空间对齐**的；训练时的数据增强必须保证**几何逐像素一致**，
否则随机各翻各的会立刻破坏对齐、毁掉跨模态互补。

增援策略（核心口径）：
- **几何操作(改像素位置)**：flip / crop / letterbox / scale 对 RGB/IR/Depth **共享同一组参数**——
  要么一起都被增广、要么都不，保证三张图仍一一对应。
- **颜色/光度(不改位置)**：HSV 抖动**只作用于 RGB**；IR/Depth 永不参与，避免伪造温度/距离语义。
- **Depth 特殊性**：插值用 `INTER_NEAREST`(最近邻)，letterbox 无效区填 0，
  防止在“0=无效”与“有效 mm”边界因插值引入伪值。
- **bbox 同步**：翻转 `cx→1-cx`；letterbox 按同一 scale+pad 把归一化框映射到新画布。

可用原语（`import common` 后直接可用）：
- `common.flip_lr_consistent` / `common.letterbox_consistent` / `common.hsv_only_rgb`
- `common.consistent_augment_full`   —— flip→仅RGB-Hsv→同步letterbox 的一次性组合
- `common.build_consistent_aug_5ch`  —— dataset 侧统一入口：读三模态→同步增强→拼 (5,H,W)+标 box

> 现状与边界：ultralytics 内建 DataLoader 面向“单图固定通道”，无法直接对三张对齐图同步增广；
> 因此这套一致性原语先作为 **common 层接口**落地，供实验模型/基线2阶段的自定义训练循环引用
> （届时在 DataLoader 里每次 `build_consistent_aug_5ch` 即可保持几何一致）。

## vendor 源码（实验模型1 源码级改造基础）

`code/vendor/ultralytics/` 是从 EFYOLO 环境**复制并钉死**的 ultralytics 8.4.138 源码
（与运行时版本完全一致，防止 pip 升级悄悄破坏你的改动）。

**何时用 vendor**：只改配置/通道数/数据（基线1、2）不需要它；一旦要动网络结构
（三流 backbone、逐 stage 融合、模态 dropout、改训练循环），建议在 vendor 上改。

**如何启用 vendor（替换 pip 版）**：在训练/改造入口的最前面注入 sys.path 即可，
之后所有 `import ultralytics` 都会命中 vendor 源码：

```python
import sys
sys.path.insert(0, r"C:\...\code\vendor")   # 放在 import ultralytics 之前
import ultralytics
print(ultralytics.__file__)                 # 应显示 ...\code\vendor\ultralytics\__init__.py
```

验证过：注入 vendor 后加载路径指向 `code/vendor`（版本 8.4.138，C2f/SPPF/Detect/DetectionModel
均可导入）；不注入时仍走 EFYOLO site-packages 的 pip 版，两者互不影响。

**改造守则（强烈建议）**
1. 官方文件**只做最小必要微调**，且改动处加 `# MOD: <说明>` 注释，便于对照上游与写技术报告；
2. 新增的融合模块/多流网络/自定义 DataLoader 一律放在**你自己的目录**（如 `实验模型1/`），
   通过 `from ultralytics.nn.modules import C2f, ...` 复用官方组件，避免整包"改花"；
3. vendor 目录属于**交付物**（半决赛要交代码），改动历史可写入各模型 README；
4. 想恢复官方行为：删掉 sys.path 注入即回到 pip 版，无需卸载/重装。

## 进阶提示（实验模型1 起点）
- 参考竞赛细则「解题思路」：数据增强；抽取网络特征；合理超参 + 自划验证集。
- 输出要求：每图同名 txt、`class_id cx cy w h confidence`、conf 缺失即无效、每图≤100 框。
- 禁止：测试集手工标注、投票/平均式简单集成、使用非官方扩展数据（允许 COCO 预训练）。
- 复赛提交：测试 tc、训练模型、环境说明；半决赛：再加技术报告 PDF。
