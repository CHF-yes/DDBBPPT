# 空间记忆 v2：有证据的修复与迁移

2026-09-19。旧 `spatial_memory_s_v1_s42` 停在已保存的 ep19，best mAP50–95=0.3649395，last/best 均保留；停止时 ep20 的未完成更新不续接。旧八项队列不恢复。

## 发现与证据

1. **局部查询被全局记忆幅度主导、门控饱和。** 对 ep16 EMA 做 CPU FP32、原画布608×1088、固定随机12张验证图检查：P5 context RMS=4.026，局部query RMS=1.025；P4/P5 IR与Depth门控100%的位置>0.99。保持相同权重/图片，仅将局部context置零，P5 IR门控均值0.99991→0.83410、Depth 0.99995→0.90267，二者>0.99比例均为0。该干预只定位饱和来源，不证明屏蔽记忆提高mAP。
2. **覆盖采样后的稀有类尾段会集中改写BN统计。** 160次追加约54个batch，momentum=.03时占末尾统计权重约81%，与全体训练分布不同。这是可避免的统计偏置，不声称已经测出其分数损失。
3. **此前小规模冒烟样本覆盖不足。** 按文件名取前32图恰好全为JPG，因此没有测试真实毫米分支。新增分层真实数据检查使用20 PNG+10 JPG，验证三模态、米制入口、记忆模块有限非零梯度。JPG米制分支零梯度是正确语义。
4. **DFL固定积分层被全局requires_grad_(True)误解冻。** 当前权重核仍精确为0..15，训练路径不调用其解码，未发生漂移，故不是目前低分原因；已显式保持冻结，避免未来误用。

补查当前验证框过滤：400张原标注3252框，608×1088下评测保留3252框，没有因2px过滤丢框。存在1条官方标签重复行，未自动改标签。标注语义与物理对齐正确性不能由这些检查保证。

## 修改

- 新结构字段 `fusion.memory_control=bounded_v2`；缺字段时仍为`unbounded_v1`，旧权重严格加载/恢复不变。
- 共享记忆进入局部查询前做无仿射LayerNorm，再将投影RMS平滑限制至≤0.25。保留全部跨尺度记忆和neck注意力，不删除用户的idea。
- 门控只使用尺度归一化的query/read描述，真实值特征注入保持原幅度；sigmoid使用FP32。
- 只重新初始化9个门控的末层（18个参数张量），encoder、检测头、距离分支、value投影、记忆权重等严格迁移。明示`--reset-fusion-gates`才允许v1→v2，且检查原划分；不能用严格resume偷换架构。
- `adaptive_no_tail`：完整覆盖段正常更新BN，含稀有追加的尾段（含边界混合batch）momentum=0，但仍使用批统计并参与反向传播。epoch开始恢复momentum=.03。
- 记录整轮mean gate、gate>0.99比例、记忆/查询RMS；新增含验证的墙钟时间，解决旧用时字段漏计验证。

## 新阶段配置

从旧ep19最佳EMA迁移，另开 `spatial_memory_s_v2_repair_s42`，不是COCO从零重训、不是精确续训。
80轮、task LR=1e-4、encoder LR=5e-5、warmup3轮、不冻结主干；裁剪初值48.49，warmup后重新收集128步校准。其余608×1088、batch3×accum5、1600/400划分、完整400图验证、采样和增强预算保持。

运行（EFYOLO Python，项目根目录）：

```powershell
& 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe' -u code/mm_yolo/run_spatial_memory.py --repair-v2 --init-checkpoint code/runs/spatial_memory_s_v1_s42/weights/best.pt
Get-Content 'code/runs/spatial_memory_s_v2_repair_s42/train.log' -Encoding UTF8 -Tail 20 -Wait
# 已停止后恢复新阶段，不再传init-checkpoint
& 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe' -u code/mm_yolo/run_spatial_memory.py --repair-v2 --resume
```

服务器使用同一入口并覆盖数据/权重/划分路径。旧版需要的字段默认值未改。

## 验证边界

37项单元测试通过；真实PNG/JPG混合、两步有效batch15的GPU检查中无AMP跳步，范数29.00/15.46，峰值分配3.95GiB；P3/P4/P5 context RMS约0.19–0.20、门控饱和比例0。旧v1 ep19精确恢复检查通过。

这些结果证明修复路径可以训练，不代表最终分数一定提高。仍未解决/未证明：小目标严格定位、长尾场景多样性不足、局部匹配真实几何精度、三模态各自净收益、记忆向量的语义解释。检测loss下降也不能替代这些验证。

证据：`code/runs/spatial_memory_preflight/health_ep16.json`、`health_ep16_no_context.json`、`repair_smoke.json`。

## 首轮正式训练验收（2026-09-19 03:00）

新阶段ep1已完成，best/last均已保存，进入ep2。固定400图、12类验证：mAP50–95=0.3922484，AP50=0.6571014；源ep19为0.3649395。此处同时改变了控制方式、门控初始化、BN策略并新增训练，不能将2.73个百分点的差值单独归因于其中某项。

完整覆盖1600/1600、总抽样1760；梯度均值/最大14.52/32.51，裁剪0/118，无AMP跳步。整轮所有记录的gate>0.99比例为0，context RMS为0.1915–0.1989。训练5.2分钟，含验证7.48分钟，峰值分配/保留显存3.95/4.10GiB。

另有非阻断效率问题：`evaluate_model`虽然接受workers参数，实际以列表推导串行读取样本，未启用并行DataLoader；本轮不为此中断训练。当前仍是1600训练+400验证，不是最终2000图的提交训练。

## 定位精修阶段

主训练结束后可从其最终 `best.pt` 启动独立的20轮弱增强精修。该阶段不重置融合门控，不恢复旧优化器：task/encoder LR分别为2e-5/1e-5；缩放0.90–1.10、平移0.03，关闭目标裁剪、模拟错位、Depth孔洞和模态失活，并保留轻量成像退化。第0轮先完整评测并保存迁移起点，避免精修退化时丢失原最佳权重。

```powershell
& 'D:\Development_Tools\anaconda3\envs\EFYOLO\python.exe' -u code/mm_yolo/run_spatial_memory.py `
  --localization-finetune `
  --init-checkpoint code/runs/spatial_memory_s_v2_repair_s42/weights/best.pt
```

精修不是结构收益证明；判断标准是固定400图的mAP50–95是否超过起点，并重点观察严格定位指标。精确恢复使用同一命令去掉`--init-checkpoint`并加`--resume`。
