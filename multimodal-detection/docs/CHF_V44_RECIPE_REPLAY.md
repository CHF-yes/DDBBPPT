# YOLO26m 使用 V4.4 两阶段训练配方

启动日期：2026-10-03。服务器实验目录：`/root/autodl-tmp/runs/CHF_v44_recipe_YOLO26m_20261003`。

本实验从 CHF v1 的 RGB 最佳权重（累计第 34 轮）初始化 YOLO26m-P2，使用原 1600/400 划分和 V4.4 的 Stage A、Stage B 配方重新训练。它复刻的是 **V4.4 两阶段训练参数**，不复刻 V4.4 的 YOLO11s 架构，也不是 CHF v1 Stage B 的精确断点续训。

## 参数来源与差异

- [本实验配置](../configs/mm_yolo26m_p2_v44_recipe_s42.json)由 [V4.4 原配置](../configs/mm_v44_multimodal_ceiling.json)生成；逐项比较只有 `name` 和 `architecture` 不同。画布 `736×1280`、物理 batch 4、梯度累积 4、BF16、Stage A/B 各 36 轮，以及增强、学习率、冻结、损失权重、验证频率等配置值均与 V4.4 配置一致。
- 固定使用 `configs/split_s42.json`（1600 训练、400 验证）和 `/root/autodl-tmp/data/new_labels_2000`。没有将 400 张验证图加入训练。
- 起点是 `/root/autodl-tmp/runs/CHF_v1_RGB_extend18_20261002/rgb_extend18/weights/best.pt`，SHA256 `1369464ec132486317ddfc2aa1bc01ac4e54e260cc344352a4dce98c8ad12da1`。它此前在 `864×1536` 上训练；迁到 `736×1280` 后，优化器、EMA 和随机状态从新实验初始化。
- 本实验的主干、检测头和损失仍为 YOLO26m-P2。V4.4 使用 YOLO11s-P2，两个架构的权重不能互换。因此即使两阶段配置一致，也不能称为 V4.4 模型的逐位复现或只改变一个变量的对照。
- 与历史 V4.4 实际训练是否逐项一致，目前可核对的是仓库中的 V4.4 配置；历史运行命令及代码版本未纳入此次复刻的逐字节审计。

## 启动与验证

服务器代码：`/root/autodl-tmp/DDBBPPT_CHF_test/multimodal-detection`；环境：`/root/autodl-tmp/venvs/ddbbppt26`。先用独立的 24 样本冒烟实验 `CHF_v44_recipe_probe_20261003` 完成 Stage A、Stage B 的加载、训练、验证和权重交接；Stage B 报告的显存峰值为 26.85 GiB，低于 RTX 5090 的 32 GiB。冒烟验证只有 5 张图、5 个有效类，其 mAP 不用于模型比较。

正式实验运行于独立 `screen` 会话 `CHF_v44_recipe_YOLO26m_20261003`：

```bash
cd /root/autodl-tmp/DDBBPPT_CHF_test/multimodal-detection
/root/autodl-tmp/venvs/ddbbppt26/bin/python -u train_multimodal.py \
  --config configs/mm_yolo26m_p2_v44_recipe_s42.json \
  --root /root/autodl-tmp/data/train_extracted \
  --labels /root/autodl-tmp/data/new_labels_2000 \
  --weights /root/autodl-tmp/weights/yolo26m.pt \
  --split-file configs/split_s42.json \
  --init-checkpoint /root/autodl-tmp/runs/CHF_v1_RGB_extend18_20261002/rgb_extend18/weights/best.pt \
  --out /root/autodl-tmp/runs \
  --name CHF_v44_recipe_YOLO26m_20261003
```

`pipeline_status.json` 和 `console.log` 记录实际阶段、命令及训练进度；每阶段的 `weights/best.pt`、`last.pt` 均保存在该实验目录。Stage B 从本实验 Stage A 的 `best.pt` 初始化。比较结果时使用同一 400 张和相同推理设置，并报告整体与逐类 AP；本地 mAP 不能直接换算为官方分数。
