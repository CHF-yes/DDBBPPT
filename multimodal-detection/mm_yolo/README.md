# 三模态空间记忆模型

这是当前保留的正式 RGB + Infrared + Depth 路线，不是旧八项消融队列。

## 结构

1. RGB、IR、Depth 分别编码，保留各自证据来源；RGB/IR 可继承 COCO stem，Depth
   使用独立输入适配器和米制距离分支。
2. P3、P4、P5 上保留三路空间特征，以 RGB 网格为参考做邻域匹配，并允许拒绝不可靠
   对应。
3. 每个尺度进行两次按需读取，通过独立残差门控注入 IR 与 Depth，避免一次融合后
   永久丢失模态信息。
4. 三组模态记忆与共享记忆跨尺度更新；空间特征负责定位，记忆负责全局上下文，最后
   在 neck 回写。
5. Depth 为四通道表示：相对深度、绝对距离/20m、有效掩码、米制深度可用标志。

正式默认画布为 `608x1088`（高×宽）长方形。三路执行同一几何变换，Depth 距离和
有效掩码使用一致的有效性重采样。

## 训练

```bash
python mm_yolo/run_spatial_memory.py \
  --root /data/train_extracted \
  --labels /data/new_labels_2000 \
  --split-file /data/split_s42.json \
  --weights /weights/yolo11s.pt
```

本地中断后，用相同参数加 `--resume`。启动器会拒绝在已有 checkpoint 上误开新训练。

若从已完成的 bounded-v2 最佳权重做定位精修：

```bash
python mm_yolo/run_spatial_memory.py --localization-finetune \
  --init-checkpoint /path/to/best.pt [其余路径参数]
```

训练输出包含 `train.log`、`metrics.jsonl`、`weights/best.pt` 和 `weights/last.pt`。

## 推理与提交

```bash
python mm_yolo/submit.py --ckpt /path/to/best.pt \
  --root /data/test_extracted --out /data/submission --zip
```

提交器以 visible 图像清单为准，为每张测试图生成同名 TXT，严格检查类别、归一化坐标、
置信度顺序和每图最多 100 框。推理使用 checkpoint 内保存的结构、模态及画布配置。

小目标切片试验使用同一权重的全图结果加同步的 RGB/IR/Depth 切片结果。每片重新计算
质量图与深度先验，映射回原图后按类别去重，最后统一限制每图 100 框；默认不会启用。
启用后默认只接收 `ball,bicycle,sign` 且映射回全图画布后短边小于 32px 的切片框。
全图结果优先，切片只补充未重叠的目标。`--tile-classes all` 可复现旧版全类别、
按置信度合并的方式；也可用 `--tile-classes ball,bicycle` 单独比较类别组合。
首次对照保持与当前提交相同的置信度、权重和后处理参数：

```bash
python mm_yolo/submit.py --ckpt /path/to/best.pt \
  --root /data/test_extracted --out /data/submission_tiled \
  --conf 0.25 --tiled --tile-fraction 0.6 --tile-overlap 0.2 \
  --tile-merge-iou 0.6 --tile-batch 2 \
  --tile-classes ball,bicycle,sign --tile-max-short-side 32 --zip
```

`eval.py` 也接受相同的切片参数，可与普通推理分别运行并比较逐类 AP。切片增加推理耗时，
其分数收益需要实测；V4.4 的 400 张留出图可用于此权重的 A/B，已加入全量训练的
权重不能再把这 400 张当作独立验证集。

## 维护边界

- 正确性回归位于 `tests/`；它们不是实验队列。
- `SPATIAL_MEMORY_V2_REPAIR.md` 记录 bounded-v2 的设计证据和权重迁移边界。
- 不再维护旧 early-fusion、旧 hook 基线、模态消融队列及一次性诊断入口。
