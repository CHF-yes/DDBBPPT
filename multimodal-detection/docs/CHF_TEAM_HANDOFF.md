# CHF_v2 / CHF_v3 队友接手指南

更新：2026-10-05。分支 `CHF_test`。推荐实验起点是原 **CHF_v3**，不是full2000，也不是局部门控实验last。

## 1. 下载固定权重

[CHF_v2/v3 Release](https://github.com/CHF-yes/DDBBPPT/releases/tag/CHF_v2_v3_20261005)

```bash
gh release download CHF_v2_v3_20261005 --repo CHF-yes/DDBBPPT --pattern 'CHF_v*.pth' --dir weights
sha256sum weights/CHF_v2.pth weights/CHF_v3.pth
```

| 权重 | SHA256 | 留出集成绩 |
|---|---|---|
| CHF_v2.pth | 29c80a33803ddfc0ceeaa1d14a55c56e85bc2b26f95ad4cdf2be8ac6ba10befa | AP95 .5098751359 |
| CHF_v3.pth | 936755d6f850ce7b19b12deb27b3af52df804ef4559f93cdabe1719b8ca7e977 | 修正评分AP95 .5103658436；pycoco .5103743165 |

Release文件是已交付原版本，包含RGB检测器和辅助融合参数。不是从零训练预训练文件。数据和测试标签不公开进入Git或Release，向项目负责人取得赛事数据。

## 2. 环境准备

实际服务器为torch2.7.0+cu128/torchvision0.22.0+cu128、RF-DETR1.12.0.dev0。开发版源码无Git历史，故提供实际src、pyproject、README、LICENSE快照：`repro/chf_v3/rfdetr_runtime.tar.gz`，SHA256 `f128eae96d7b97ef975ee6f9f47616805bad0558bb4c046527d3622c8ed990f7`。保留原许可证；非完整上游仓库。

Linux NVIDIA CUDA示例：

```bash
cd multimodal-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
mkdir -p runtime/rfdetr
tar -xzf repro/chf_v3/rfdetr_runtime.tar.gz -C runtime/rfdetr
python -m pip install -e runtime/rfdetr
python -m pip install -r repro/chf_v3/requirements-observed.txt
python -m pip check
```

`requirements-observed.txt`是服务器观察版本，不是全量lock。新环境还需按pyproject安装依赖；出现API差异不能改成静默跳过加载。需要CUDA，不建议直接改成CPU/MPS后宣称等价性能。

## 3. 数据布局与类别

```text
/data/train_extracted/{visible,infrared,depth}/同名图像
/data/chf_arch_rgb_coco_s42/export_manifest.json
/data/chf_arch_rgb_coco_s42/train/_annotations.coco.json
/data/chf_arch_rgb_coco_s42/valid/_annotations.coco.json
```

COCO图像名称对应原三模态路径，不是另做letterbox图。1600训练/400验证，split见仓库 `configs/split_s42.json`，SHA `4165ce54e9b34be51ec3f87d3065977eee310da3ce700fa4b5b19a85202ba00a`。类别顺序0–11：person, boat, animal, seat, sign, bicycle, car, ball, light, garbage_can, uav, tricycle。

COCO转换工具：`tools/export_rgb_coco_split.py`，先 `--help` 查看参数，按固定split导出。保留原图宽高、原图xywh及0起始类别，不用旧letterbox坐标GT直接喂RF脚本。

## 4. 第一次验证：必须先复现原V3

```bash
python tools/evaluate_chf_checkpoint.py --version v3 \
  --weights weights/CHF_v3.pth --data /data/train_extracted \
  --coco /data/chf_arch_rgb_coco_s42/valid/_annotations.coco.json \
  --report runs/v3_baseline.json
```

应接近pycoco AP95 .5103743165/AP50 .7936739812。V2替换version/weights，AP95 .5098751359/AP50 .7949133276。该便携入口已语法检查，尚未在队友新机器端到端运行；使用已实测的同一模型和evaluate函数。先复现再实验，不能在结果异常时直接用新分数比较。

## 5. 从V3接着做对照实验

便携入口 `tools/run_chf_v3_experiment.py` 支持weights/data/coco/out/epochs，无需修改服务器绝对路径。保留原V3哈希验证；其他权重若要作起点，应明确修改身份校验和实验说明。

```bash
python tools/run_chf_v3_experiment.py --arm control \
  --weights weights/CHF_v3.pth --data /data/train_extracted \
  --coco /data/chf_arch_rgb_coco_s42 --epochs 4 --out runs/control_smoke --smoke
python tools/run_chf_v3_experiment.py --arm control \
  --weights weights/CHF_v3.pth --data /data/train_extracted \
  --coco /data/chf_arch_rgb_coco_s42 --epochs 4 --out runs/control_4ep
```

out必须不存在。smoke用独立目录，不可用smoke更新后的模型作为正式起点。control训练融合+检测头，RGB/Transformer冻结；auxlr2e-5/headlr5e-6，FP32batch1积8，seed42，864×1536。支持local/local_rgb用于复核先前无收益实验；不建议将它们当已成功升级。

原服务器 `run_chf_v3_upgrade.py` 已完成三组各6轮，报告见技术文档。便携副本仅改路径/轮次参数，语法检查通过，GPU训练逻辑来源于已运行脚本；便携副本本身未跑完新的GPU实验。

## 6. 源码与下一步方案

- [完整技术说明](CHF_V2_V3_VS_YOLO_TECHNICAL.md)：张量、门控、损失、冻结、梯度、恢复。
- `tools/run_rf_dynamic_trimodal_chf.py`：原V3融合结构与原6轮训练。
- `tools/run_rf_trimodal_chf.py`：原V2静态融合。
- `tools/run_rf_highres_chf.py` / `run_rf_arch_baseline.py`：RGB12轮微调/640基础24轮。
- [增强和对齐实验方案](CHF_V3_AUG_ALIGNMENT_PLAN.md)：IR增益、深度孔洞各自4轮对照，尚未实现/启动这些新增强。
- `tools/chf_alignment_sift_audit.py` / `chf_alignment_contact.py`：旧配准诊断；只读分析，不是已验证的自动配准部署。

新增增强建议继承Triples，在归一化之前处理IR；Depth孔洞必须同改值/mask。增强只对training=True执行。保持几何/框一致，独立随机数流避免变更flip/shuffle序列。

## 7. 导出与坐标警告

`tools/export_chf_v3_test1000.py`是已经实际用于交付的服务器专用入口，依赖 `audit_rf_small_objects_chf.py`，路径固定且先复现400验证。队友新机器使用前必须修改root/data/weights/out并检查覆盖逻辑；不是上面便携验证入口的一部分。它只接受原V3，权重变更需更新哈希与预期AP，不能绕过检查。

预测TXT为原图归一化class cx cy w h confidence，最多100合法框，0–11类，空图保留空TXT。RF post sizes=[H,W]。以前.44错误来自将原图归一化预测直接当1280×736 letterbox坐标；评分应同步处理有效720高度和上下8padding。不同原图按自身变换计算。

V3必须同时加载net和chf_aux。仅用原生RFDETRMedium推理会丢融合。验证AP不设conf.25；单独阈值包不继承完整预测AP。测试只推理，不选权重/调参/手工改框。

## 8. 当前实测结论和交接边界

局部门控及RGB最后两层解冻未优于control；control最佳pycoco .5108826，较原V3仅+.000508，单seed不能称稳定提升。原图/L/小目标裁剪亦无稳定超过当前M收益。full2000已用旧400训练，不能以该400验证泛化。

本指南不包含私钥、访问令牌、赛事数据；Release下载凭仓库权限。当前远端DEIM实验由负责人监控，队友不要改动原训练进程。每个新实验记录git提交、权重SHA、splitSHA、预处理、seed及原始metrics，输出目录独立，保留原V3。
