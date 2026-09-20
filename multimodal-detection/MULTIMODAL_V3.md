# 三模态独立主干 / P2 / 空间互补记忆 v3

正式入口：`train_multimodal.py`；配置：`configs/mm_v3_server.json`。
`model_catalog.py` 同时列出服务器原 RGB 配方与新版三模态配方，不覆盖 RGB 配置。

## 当前结构

- 三个独立 YOLO11m 编码器（含 stem、卷积、BN），均迁移 COCO；IR/Depth 单通道首层使用 RGB 核之和。
- Depth 四通道保持 `[relative, mm/20000, valid, metric_available]`；线性/对数距离独立、非零路径注入 P2–P5。
- 每个尺度保留三路空间共同/专属嵌入。P5→P4→P3 局部双向匹配、null 拒绝、循环可靠性；P2 接收尺度换算后的偏移。
- 模态身份只进入内容门控，不进入共同匹配。未知对应不是物体不存在；低置信专属证据只保留有限语义注入。
- 两轮残差读取，纯证据不被融合状态覆盖；三组模态记忆及共享记忆（4×4×128）随 P2→P5 更新，每图重置。
- 扩展 P2 PAN，neck 输出通道 m 为 128/256/512/512；P3–P5 保留兼容预训练分支，新增 P2 单独初始化。
- 边界旁路进入同一检测器的回归分支；一个模型、一次前向、标准逐图 NMS，无投票/TTA/WBF。

## 数据与训练

736×1280 长方形画布。三模态同步几何、目标裁剪、低概率同步 Mosaic。
Mosaic 的绝对距离不混合，分位相对深度按来源处理，scene_id 禁止跨场景对应。
Depth 最近邻有效性重采样；拖尾使用软可靠性，不因 RGB 没边缘就删除。
完整覆盖采样后追加 10% 稀有类，worker 使用 epoch/draw 确定增强。

服务器正式配方 batch=3、accum=5（有效 batch=15）、BF16、编码器重算（恢复 BN buffers，避免重复统计）。
完整 m 模型 45.67M 参数，RGB/IR/Depth 主干各约 10.35M；batch=3 的5步检查峰值分配/预留约 20.18/20.63GiB。
真实数据检查发现并修复无效匹配 sqrt(0) 反向异常；初始裁剪仅设 1000 的安全上限，
warmup 后以 128 个有限更新的 1.5×P90 校准（上限 1000），不沿用旧阈值 10/60。
固定 1600/400 训练 160 轮；最后 20% 关闭 Mosaic 并渐弱增强，不使用提前终止。
随后从 dev/best.pt 用全部 2000 张精修 18 轮，不再将原验证集当作独立验证集。
新架构不续训旧架构；resume 恢复 raw model、optimizer、EMA、RNG、裁剪校准状态。

## 启动与日志

```bash
python train_multimodal.py --root /root/autodl-tmp/data/train_extracted \
  --labels /root/autodl-tmp/data/new_labels_2000 \
  --weights /root/autodl-tmp/weights/yolo11m.pt --out /root/autodl-tmp/runs
```

相同命令加 `--resume` 恢复。`--smoke` 仅作独立正确性检查，不能据其分数判断效果。
`console.log` 为全流程日志，`dev/train.log` 为开发训练摘要，`pipeline_status.json` 为状态。

## 归档与兼容

服务器部署前先保存可恢复源码压缩包；过时顶层模型目录移至同一归档，不删除训练权重/数据。
`mm_yolo/model.py` 中旧模型及 `run_spatial_memory.py` 仅用于兼容历史权重/复现，不再是新训练入口。
部署清单保护 `train_rgb.py`、`models_config.py`、`common/trainer.py`、`predict_rgb.py` 原件哈希。
本版本不承诺达到某个排行榜分数；空间共同/专属分解也不等于具备唯一语义解释。
