# 三模态空间记忆 v1

本版只训练一个新模型；旧八项对照队列保持暂停。旧 run、权重、1600/400 划分及官方标签均不覆盖。

## 已实现

- 每轮先覆盖全部 1600 张训练图，再追加 160 张稀有类图像，每张最多额外 3 次。增强种子包含 epoch、图像、出现次数，persistent worker 直接接收这三个索引，不依赖父进程修改。
- 末 20% 轮次渐弱几何/裁剪/退化/模态失活；末轮缩放 0.95–1.05、平移 0.02、裁剪/错位/失活为 0。翻转保留。没有声称启用 Mosaic/copy-paste。
- `nearest_valid_v2`：原始深度与有效掩码一致最近邻变换，无效距离置零。旧权重缺版本字段时继续走旧双线性预处理。训练、评测、提交读取同一结构配置。
- 输入 RGB 3ch、IR 1ch、Depth 4ch（相对、绝对毫米/20000、valid、metric）。PNG 保留毫米距离；JPG 不伪造绝对距离。新增独立绝对距离编码，在线性/log 距离、有效比例、米制可用标志上学习，输出以非零 0.1 残差接入 Depth 各尺度；没有逐图归一化该分支。
- 三路纯模态 encoder 输出 P3/P4/P5，再融合并送入原 YOLO11 neck/head；融合后的特征不再冒充纯 RGB。共享卷积、逐模态 BN，Depth 独立浅层 stem。
- 三路空间嵌入、模态身份和有效性；P4/P5 使用当前融合查询与辅助模态 key 的 3×3 匹配，加 null 选项。P3 只做同位置轻量注入，Depth 直接注入 P4/P5。
- 每尺度两轮共享参数更新，始终读取保留的原始证据；每路独立 sigmoid 门控，不以三路 softmax 强迫争抢份额。RGB 锚定、小非零残差初始化。
- 8×64 记忆：RGB/IR/Depth 各 2 个、共享 2 个；从 P3→P4→P5 更新，每次 forward 重新初始化。自身记忆只读纯模态特征，共享记忆交换汇总；池化按有效像素数归一，带坐标与有效比例。
- 最终记忆在 neck P3 进行一次空间注意力回写。纯 RGB 或辅助模态全无效时保持原 YOLO 路径，不引入随机融合扰动。

## 本机正式配方

YOLO11s COCO 初始化，约 9.99M 参数；608×1088，batch=3，accum=5，100 轮，4 个读图 worker；完整 400 图每轮验证，conf=0.001，按 mAP50–95 选 EMA best。

AdamW task LR=3e-4、encoder LR=1.5e-4；5 轮 warmup，余弦尾比 0.02，首轮冻结 encoder。BN 在首轮冻结主干时保留对应统计，其后以 momentum=0.03 适应；这是有依据的新配方选择，不是已证明最优的结论。RGB-only 入口新增 `--rgb-batch`，本轮不启动 RGB 对照。

裁剪初始 60；warmup 和解冻结束后收集 128 个有限、未裁剪范数，取 1.5×P90（20–200 范围）作为固定尖峰保护阈值。阈值/样本随 checkpoint 保存。日志记录范数、P90、clip 次数及 AMP skip；裁剪比例不等于 AdamW 学习率比例。新版 AMP 初始 scale=1024，避免默认 65536 在解冻时大量溢出。

## 运行、查看和续训

在项目根目录，用 EFYOLO Python：

```powershell
& 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe' -u code/mm_yolo/run_spatial_memory.py
Get-Content 'code/runs/spatial_memory_s_v1_s42/train.log' -Encoding UTF8 -Tail 20 -Wait
# 退出/断电后用相同配方恢复，不要与已运行进程同时启动
& 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe' -u code/mm_yolo/run_spatial_memory.py --resume
```

服务器同一入口，显式传 `--root --labels --split-file --weights --out`；新开 run 可改变 `--batch/--accum/--workers/--imgsz`。精确续训保持 batch、accum、画布、结构和训练配方，只允许设备/路径/worker 等非语义选项变化。加载跨平台 checkpoint 评测/提交时可用 `--base-weights /path/yolo11s.pt` 重建。

保存 `weights/best.pt`、`weights/last.pt`、`val_best.json`、`val_latest.json`、`launch.json`；last 包含原模型、EMA、优化器、AMP、RNG 和裁剪校准状态。新旧结构严格分开，不能拿旧权重冒充新结构的精确续训。

## 验证与边界

新增测试覆盖完整采样、重复增强、persistent worker 调度、Depth 边界、null 匹配、RGB 恒等、记忆重置、所有关键分支梯度、EMA、保存加载；另做真实图 608×1088 两轮 smoke 和 checkpoint 恢复检查。smoke 不是精度实验。

数据审计报告：`code/runs/spatial_memory_preflight/data_audit.json`。2000 图无缺失模态、无格式错误；64 条边界超出图像的标注及 1 条重复行仅记录，不擅改官方标签。已有框变换按画布裁剪。这不证明标签语义全部正确，人工视觉审查仍可能发现问题。

首轮采样模拟覆盖 1600/1600，三轮车原始 14 框变为采样曝光 46 框；这不是增加真实标注或真实场景多样性。固定验证集中的长尾类仍小，不能凭一次波动断言收益。

本实现是检测任务监督下的可学习对应，不是相机标定；没有额外对齐真值损失，也不声称向量差异已经证明“独有语义”。高质量/具体分数必须等正式完整验证。当前“全量”指完整 1600 训练划分，400 验证不参与训练；最终使用全部 2000 图重训是另一个无独立保留集分数的提交阶段。

## 启动验收记录（2026-09-19）

34 项 unittest 全部通过；旧 RGB+IR ep28 的 checkpoint 按原配方通过精确恢复校验。
真实图两轮 smoke（解冻主干、AMP 初始 scale=1024）无跳步，峰值分配/保留显存 4.01/4.16 GiB；实际参数梯度检查也覆盖了绝对距离输入通道本身。

正式 run `spatial_memory_s_v1_s42` 已通过第一轮完整验证并保存 best/last：
1600/1600 图、1760 次采样、118 次优化更新、clip=0/118、AMP skip=0；
400 图、12 类验证 mAP50–95=0.2846、mAP50=0.4754。这是第 1 轮，不能当作最终性能。
日志旧格式的“用时/轮”目前指训练段，不包含其后的完整验证；不要据此直接推算总训练时长。
