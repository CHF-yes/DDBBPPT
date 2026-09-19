# 三模态距离信息实验（先生成命令，再逐组执行）

`experiments.py` **只打印命令，不启动训练/评测**。本机与 Linux 服务器均使用同一入口，
`--python` 默认当前 Python；服务器传自己的预训练权重、数据集和划分文件路径即可。
需要无人值守串行执行某一阶段时，使用独立的 `run_experiment_queue.py`；它遇错误即停止，
检查 `last.pt` 严格续训，不允许 `--stage all`，并以 `active.lock` 防止同一阶段双开。

在项目根目录的 EFYOLO 环境中：

```powershell
$py = 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe'
& $py code/mm_yolo/experiments.py --stage diagnose `
  --root '初赛数据集-面向城市场景的多模态目标检测/train_extracted' `
  --labels '初赛数据集-面向城市场景的多模态目标检测/训练集/new_labels_2000' `
  --weights code/yolo11s.pt `
  --split-file code/runs/b2_depth4_s_safe/split.json `
  --baseline-ckpt code/runs/b2_depth4_s_safe/weights/best.pt
```

只改变 `--stage` 为 `modalities`、`standalone`、`depth`、`augment`，即可分别打印后续命令。
不要直接从 B1/B2 权重初始化训练对照：四组训练必须从**同一 COCO 权重**独立开始。
输出实验名独立，`train.py` 遇到已有 `last.pt` 会拒绝覆盖。

决定执行时，仅对选中的阶段运行队列，例如 `modalities`：

```powershell
& $py code/mm_yolo/run_experiment_queue.py --stage modalities `
  --root '初赛数据集-面向城市场景的多模态目标检测/train_extracted' `
  --labels '初赛数据集-面向城市场景的多模态目标检测/训练集/new_labels_2000' `
  --weights code/yolo11s.pt `
  --split-file code/runs/b2_depth4_s_safe/split.json
```

状态：`code/runs/experiment_queue_modalities_s42/status.json`；
每组训练：`code/runs/exp_mod_<模式>_s42/train.log`；完整控制台信息：同目录 `console.log`。
队列中断时先确认没有 Python 训练进程，再查 `active.lock`；不要并发启动同一阶段。

## 顺序和解释

1. `diagnose`：现有 B2 全开/去 IR/去 Depth/去 RGB/仅绝对深度置零；按 PNG/JPG
   和亮度、Depth 有效比例切片。置零仅作**分布外诊断**，不能当成因果结论。
2. `modalities`：RGB、RGB+IR、RGB+Depth、RGB+IR+Depth，各自从同一起点训练；
   相同划分、种子、画布、轮次及增强。既看增益，也看 IR×Depth 交互。
3. `standalone`：单独训练 IR-only 与 Depth-only 信号组。RGB 图片只用于读取尺寸和
   框坐标，送入网络的 RGB 像素严格为零；Depth-only 强制 P3/P4/P5 都接 Depth。
   这是同框架下的单模态能力测试，不等同于独立设计并调优的纯深度检测器。
4. `depth`：五种表示/初始化。`relative_only` 只用逐图相对深度；
   `metric_fallback` 的 PNG 使用毫米/20m、JPG 回退到相对深度；
   `metric_log_fallback` 把米数映射到 `log(1+z)/log(21)`；
   `both_relative` 是原始初始化；`both_balanced` 两路各以 0.5 初始化。
   五组都关闭以相对深度构建的先验 FiLM 和质量图，隔离直接输入通道的作用。
5. `augment`：相对于 `modalities` 阶段的 `all`，每次只改目标裁剪、RGB 低照、
   新增 RGB 轻量颜色扰动、IR 单调增益、Depth 无效块或错位扰动中的**一个**概率/幅度。
   新增增强默认关闭，不改变旧实验。Mosaic/copy-paste 没接入此批实验，
   当前没有伪造跨模态一致的贴图实现；需要先证明距离通道有增益再扩充增强种类。

各训练命令之后的 `formal_eval.json` 使用 `conf=0.001`、`batch=1`、全量 400 张
验证集；训练期间和正式评测统一使用 `val_conf=0.001`，避免稀有类的低置信度召回
被提前截断后选错 `best.pt`。
JPG 验证集仅 29 张，分组差异需谨慎解释；ball/tricycle 框数也非常少。

所有阶段都只使用官方训练集，验证划分固定，不用测试集调参。多种实验模型用于
选择一个最终方案，**不做规则禁止的简单模型集成**。选好配置后再多种子/组感知
交叉验证，最终以全部 2000 张数据训练单一提交模型。

脚本默认仅生成命令，逐组执行前确认服务器环境路径、显存和剩余时间；
`--stage all` 也**只打印**，绝不自动连续启动十余次训练。
