# YOLO26m-P2 三模态 5090 运行记录

日期：2026-10-02。分支：`CHF_test`。实验配置：
[`mm_yolo26m_p2_864_v44_s42.json`](../configs/mm_yolo26m_p2_864_v44_s42.json)。

## 启动前已核对

- 服务器：RTX 5090 32 GB，驱动 595.71.05；Python 3.12、PyTorch 2.7.0+cu128 支持 `sm_120`。环境为 `/root/autodl-tmp/venvs/ddbbppt26`。
- 官方 `yolo26m.pt` 在 `/root/autodl-tmp/weights/yolo26m.pt`，SHA256 为 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。
- 代码在 `/root/autodl-tmp/DDBBPPT_CHF_test`。模型用官方 YOLO26m 初始化四尺度 P2 结构，共迁移 606 个张量、19,048,500 个值；P2 新层随机初始化。YOLO26 端到端双头、原生 E2E 损失与类别重映射保留。
- 合成三模态输入的前向、损失、反向梯度和推理输出已通过。864×1536、BF16、物理 batch 1 的单步峰值约 10.3 GB，前向+反向约 2.14 秒；这是不含真实数据读取、评估和 checkpoint 保存的下限测试。
- 在同一画布以物理 batch 2 执行包含 AdamW 更新与 EMA 的完整单步，PyTorch 峰值已分配约 21.9 GB、保留约 22.9 GB；仍须以真实图像和实际增强测稳定峰值。
- 24 张明确标记的合成图完成了三阶段各 1 轮的完整训练、验证、保存和 `best.pt` 阶段交接；`pipeline_status.json` 为 `completed`。合成集只有一个类别，分数没有参考价值。
- 从 Stage B 的 `best.pt` 重新加载模型后，现有 `submit.py` 对 4 张合成图生成了 4 个 TXT，提交格式校验为 0 个问题。

## 正式实验顺序

1. `rgb_bootstrap`：先在固定 1600 张训练集上训练同结构的 YOLO26m-P2 RGB 任务起点，避免把只迁移了 4 个同名 COCO 类的头直接当成 V4.4 基线。
2. `stage_a`：从 RGB 最佳权重初始化，独立训练 IR/Depth 检测分支。
3. `stage_b`：从 Stage A 最佳权重初始化，训练三模态残差融合。

配置为 864×1536、BF16、物理 batch 2、累积 8；初筛轮数分别为 18、18、24。训练脚本会在每阶段保存 `best.pt` 和 `last.pt`，固定使用 `configs/split_s42.json` 的 1600/400 划分。初筛结果不能与 V4.4 的完整 36+36 轮直接解释为架构上限比较。

## 等待确认的数据

新机器数据盘在准备时没有训练数据。启动前需要确认：

- 数据根目录含 `visible/`、`infrared/`、`depth/`，同名图像为同一样本。
- 标签目录含官方修正的 2000 份 YOLO `.txt` 标注。
- `configs/split_s42.json` 的每个 stem 都能在三模态目录和标签目录找到；训练/验证无交叉。
- 用真实数据先跑一次少量 step，量出 batch 2 的峰值显存、数据吞吐和损失稳定性；若显存不足再改为 batch 1、累积 16。

数据放妥后，先用同一命令加 `--dry-run` 检查三个阶段的实际参数。完整训练命令如下（执行前替换两个数据路径）：

```bash
PY=/root/autodl-tmp/venvs/ddbbppt26/bin/python
CODE=/root/autodl-tmp/DDBBPPT_CHF_test/multimodal-detection
$PY -u "$CODE/train_multimodal.py" \
  --config "$CODE/configs/mm_yolo26m_p2_864_v44_s42.json" \
  --root /root/autodl-tmp/data/train_extracted \
  --labels /root/autodl-tmp/data/new_labels_2000 \
  --weights /root/autodl-tmp/weights/yolo26m.pt \
  --split-file "$CODE/configs/split_s42.json" \
  --out /root/autodl-tmp/runs
```

服务器任务应在 `screen` 中运行，以免 SSH 断开中止训练。训练结果应先和同一 400 张、同一 `conf=0.001` 与 NMS 配置的 V4.4 权重比较，确认整体 12 类 AP 有收益后再考虑 2000 张全量训练与官方提交。
