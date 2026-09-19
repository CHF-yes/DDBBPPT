# 面向城市场景的三模态目标检测

当前代码只保留两条正式链路：一个可审计的 RGB 上限模型，以及一个真正进行
RGB + Infrared + Depth 特征交互的空间记忆模型。早期基线、旧实验队列和一次性诊断脚本
已经移除；历史实现仍可从 Git 提交记录恢复。

## 目录

```text
multimodal-detection/
├── train_rgb.py             # 正式 RGB YOLO11m 训练
├── predict_rgb.py           # RGB 测试集推理、严格校验与提交压缩包
├── models_config.py         # RGB 配方
├── common/trainer.py        # RGB trainer、逐图 NMS、梯度监测
├── configs/split_s42.json   # 固定 1600/400 划分
├── mm_yolo/                 # 正式三模态模型
│   ├── config.py            # 自描述结构配置
│   ├── data.py              # 三模态读取、Depth 表示、同步增强
│   ├── align.py             # 跨模态对齐
│   ├── fusion.py            # 局部匹配与门控融合
│   ├── memory_fusion.py     # 跨尺度空间记忆
│   ├── model.py             # 三路编码器、融合与 YOLO neck/head
│   ├── train.py             # 训练与严格 checkpoint 恢复
│   ├── eval.py              # 完整验证
│   ├── submit.py            # 单模型提交生成及校验
│   └── run_spatial_memory.py# 本机/服务器统一启动器
└── vendor/ultralytics/      # 固定版本运行时
```

## RGB 正式训练

```bash
python train_rgb.py \
  --data-root /data/train_extracted \
  --labels /data/new_labels_2000 \
  --split-file configs/split_s42.json \
  --weights /weights/yolo11m.pt \
  --device 0 --batch 16 --workers 12
```

默认模型为 YOLO11m，输入画布为 **960×960 正方形**。原图按长宽比缩放后
letterbox 补边，并非拉伸成正方形。训练先使用固定 1600/400 划分选择 `best.pt`，
随后用全部 2000 张图低学习率精修 18 轮。中断后在完全相同的命令末尾加
`--resume`，以恢复 optimizer、GradScaler 和 EMA。

生成 RGB 提交包：

```bash
python predict_rgb.py --weights /runs/phase2_full/weights/last.pt \
  --source /data/test_extracted/visible --out /runs/submission_txt \
  --zip /runs/submission.zip --imgsz 544x960 --device 0 --batch 16 --half
```

## 三模态正式训练

```bash
python mm_yolo/run_spatial_memory.py \
  --root /data/train_extracted \
  --labels /data/new_labels_2000 \
  --split-file /data/split_s42.json \
  --weights /weights/yolo11s.pt
```

该模型保持 RGB、IR 和 Depth 的独立证据路径，在 P3/P4/P5 做局部匹配、拒绝不可靠
对应、独立残差门控和两轮互补读取；模态记忆及共享记忆随尺度更新，最后回写 YOLO
neck。Depth 输入保留相对深度、绝对米制距离、有效掩码和米制可用标志。

默认三模态画布是 **608×1088 长方形**，以 RGB 网格为坐标参考；三模态执行完全相同的
几何变换，Depth 距离与有效掩码采用一致的有效性重采样。

## 约束

- 只使用一个模型、一个权重、一次前向和一次标准 NMS；不使用投票、WBF 或 TTA。
- 数据和预训练权重不纳入仓库，所有入口均接受本机或服务器绝对路径。
- checkpoint 保存结构及预处理版本；结构不匹配时禁止静默续训。
- 当前有效的正确性检查位于 `mm_yolo/tests/`。
