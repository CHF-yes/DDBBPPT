# CHF_v3 数据增强与对齐：论文核对和极简验证方案

2026-10-04。状态：完成研究和实验设计，尚未新增训练或运行MINIMA。当前DEIMv2继续运行。

## 论文及可迁移部分

1. AR-CNN，ICCV2019，Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection：https://arxiv.org/abs/1901.02645 。区域特征对齐、可靠性重加权及RoI扰动，原研究还使用模态独立配对标注。CHF没有此类成对框，不能直接声称原方案可复现；可以提取小幅位移鲁棒训练的思路，位移增强不等于解决错位。
2. MINIMA，CVPR2025，Modality Invariant Image Matching：https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html 。跨模态匹配覆盖RGB/IR/Depth，适合替代不稳定SIFT作配准诊断。匹配成功不等于检测AP提升；不能未经置信度/空间支持筛查就warp全图。先作离线训练图诊断，不自动将外部权重加入比赛推理链。
3. Random Erasing：https://arxiv.org/abs/1708.04896 。训练随机遮挡是成熟正则化思路；本实验借鉴到depth小块缺失，必须同时清零有效掩码。这是自设计传感器退化，不是复现论文全部方法，更不能宣称同等收益。
4. Unbiased Teacher，ICLR2021：https://arxiv.org/abs/2102.09480 。弱/强增强框架来自半监督检测。CHF已有训练标签，且测试不可参与训练；本方案不采用测试伪标签或teacher-student，只参考增强应区分几何/外观的设计原则。

## 仓库证据

旧V4.4配置 `configs/mm_v44_multimodal_ceiling.json`：Stage B rgb_color_p=.10, ir_noise_p=.06, ir_gain_p=.10, depth_hole_p=.06, aux_dropout=.06, misalign_px=1；scale=.93–1.08, translate=.025, target_crop=.04, mosaic=.08。具体实现见 `mm_yolo/data.py`，不能原样复用旧letterbox返回值至RF原图坐标链。

现V3仅fixed864×1536与同步flip.5。此前RGB裁剪无稳定收益，局部门控/有限RGB解冻无额外收益。新增增强先保持原V3结构与冻结策略，避免结构和增强一起变化。

已有24图SIFT审计：IR可信5/24、退化5/24、不足14/24；Depth可信0/24、退化1/24、不足23/24。可信IR近单位变换；证据不支持全数据统一仿射，匹配不足不是错位证明。原报告尚未运行MINIMA。

## 第一轮极简训练：三组各4轮

所有组从原CHF_v3权重初始化，1600/400固定split，seed42，864×1536，FP32/batch1积8，冻结RGB与Transformer，融合lr2e-5、分类回归头lr5e-6；新优化器、相同样本顺序与flip随机流。每轮FP32完整400图、原图坐标合法类别全局top100，无conf.25过滤，保存AP95/AP50/AP75/小目标AP/AR及loss。

- A control：原V3数据处理，仅flip。
- B IR gain：仅训练IR，触发p=.10，原始[0,1]IR做gain U(.9,1.1)、bias U(-.02,.02)、clip[0,1]，然后原V3归一化；RGB/Depth/框不改。
- C depth holes：仅训练depth，触发p=.06，随机一个矩形孔面积占全图.1%–.5%，宽高比.5–2，深度值与mask同时置0；RGB/IR/框不改。

这些强度是本数据的保守设计值，不是论文推荐最优。B和C分别隔离一个因素，不混RGB颜色、mosaic、几何裁剪和模态dropout。验证关闭增强。单因素有稳定迹象再做B+C组合。加入整模态dropout须在投影/融合输出处置零，不能只将归一化输入置0而称模态缺失（卷积bias和归一化仍会产生特征）。

运行前：合成深度掩码一致性检查、三模态几何不变检查、确定性采样顺序、8批真实梯度烟测。独立随机数流防止增强随机抽样改变control的flip与shuffle。

粗略预算：按上一轮真实每轮约3分钟，三组12轮约36分钟，加baseline和检查约45–60分钟；未运行不能精确报时。4轮结果只用于筛选，不能判定充分收敛。

## 如何判断有效

比较相同轮次、末两轮均值及最佳，不能只挑一个最高点。原V3同流程.5103743，之前6轮control峰值.5108826，本次先对A做对照。差值只有.0001–.0005时不当作确定提升；.001以上也只是候选信号，第二seed或训练内独立划分复核后才能称较稳定。配对bootstrap可以作为不确定性描述，不能消除反复调400验证集的选择偏差。

## 对齐极简验证：先诊断，再小范围检测A/B

从train1600固定抽64对RGB/IR，包含近远目标和不同光照；目标类别标签只用于训练图诊断。用跨模态匹配检查内点数、覆盖范围、变换退化、边缘对齐前后和局部目标偏移；同时检查原图尺寸，不能仅在统一resize后推断传感器参数。

只有确认偏移稳定才从训练样本估计一个受约束平移/仿射；RGB及GT不动，只warpIR，并在边界保存有效性。若偏移依赖目标距离，拒绝全局变换，转局部特征对齐。Depth须配对warp值/mask，避免无效值插值扩散。

可信参数冻结后对同一400图以原V3权重比较identity与IR变换，失败图自动identity回退。此无训练A/B只能检验原V3是否敏感，不能证明训练对齐方案无效：推理突变可能造成分布偏移。若检测指标/局部偏差都有正向证据，再作train+infer一致对齐的4轮control实验。

不大规模搜索验证集平移；训练图上估参，验证只比较预先确定方案。不根据测试图人工改框/变换，不用测试标签。
