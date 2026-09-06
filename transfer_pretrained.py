# -*- coding: utf-8 -*-
"""
三模态(RGBTD)模型预训练权重迁移脚本
=====================================
将官方 YOLO11 的 COCO 预训练权重迁移到 RGBTD 三模态模型，实现"使用现有模型参数简化训练"。

迁移策略（对应 yolo11-RGBTD-midfusion.yaml 的结构设计）：
1. 三条分支(可见光/红外/深度)均为与官方 backbone 同构的 3 通道结构，逐层复制官方
   backbone 的 Conv/C3k2 权重（官方 model.0~8 -> 三条分支各 9 层）；
2. 深度分支直接复用红外分支迁移后的权重（两者同为单通道灰度性质，最接近）；
3. SPPF/C2PSA/检测头由于融合层 ModalConcat 已把通道对齐官方尺度，与官方同构，
   从 SPPF 起逐层对齐迁移；Detect 分类头因 nc 不同 shape 不一致自动跳过（回归部分保留）。

用法：
    python transfer_pretrained.py --src yolo11x.pt \
        --dst ultralytics/cfg/models/11-RGBT/yolo11x-RGBTD-midfusion.yaml \
        --out yolo11x-RGBTD-pretrained.pt

说明：--dst 的 `yolo11x-RGBTD-midfusion.yaml` 是 Ultralytics 的 scale 命名约定（并非真实
存在的文件）。Ultralytics 的 yaml_model_load 会把它归一化为 `yolo11-RGBTD-midfusion.yaml`
并自动应用 scale='x'；若改写成 `yolo11-RGBTD-midfusion.yaml`，scale 会退化为默认 'n'。
"""

import argparse

import torch

from ultralytics import YOLO


def find_branch_layers(model):
    """定位三条分支的 Conv/C3k2 层索引（每条分支从 SilenceChannel 后连续 9 层）。"""
    branches, current = [], []
    for i, m in enumerate(model.model):
        t = getattr(m, "type", "")
        if "SilenceChannel" in t:
            current = []  # 新分支开始
        elif t.endswith("Conv") or "C3k2" in t:
            current.append(i)
            if len(current) == 9:
                branches.append(list(current))
                current = []
    return branches


def find_sppf_index(model):
    """定位 SPPF 层索引，作为尾部(head)对齐起点。"""
    for i, m in enumerate(model.model):
        if "SPPF" in getattr(m, "type", ""):
            return i
    raise RuntimeError("未找到 SPPF 层")


def transfer_layer(src_state, dst_state, si, di, transferred=None):
    """把 src 第 si 层的参数按 key 复制到 dst 第 di 层（shape 不一致自动跳过）。"""
    ps, pd = f"model.{si}.", f"model.{di}."
    n = 0
    for k in dst_state:
        if k.startswith(pd):
            sk = ps + k[len(pd):]
            if sk in src_state and src_state[sk].shape == dst_state[k].shape:
                dst_state[k] = src_state[sk]
                if transferred is not None:
                    transferred.add(k)
                n += 1
    return n


def transfer_layers(src_state, dst_state, src_indices, dst_indices, transferred=None):
    """批量逐层迁移。"""
    total = 0
    for si, di in zip(src_indices, dst_indices):
        total += transfer_layer(src_state, dst_state, si, di, transferred)
    return total


def main():
    parser = argparse.ArgumentParser(description="迁移 COCO 预训练权重到 RGBTD 三模态模型")
    parser.add_argument("--src", default="yolo11x.pt", help="官方 COCO 预训练权重路径")
    parser.add_argument("--dst", default="ultralytics/cfg/models/11-RGBT/yolo11x-RGBTD-midfusion.yaml",
                        help="三模态模型 YAML 路径(scale 后缀命名，会自动解析到 yolo11-RGBTD-midfusion.yaml + scale x)")
    parser.add_argument("--out", default="yolo11x-RGBTD-pretrained.pt", help="输出迁移后权重路径")
    args = parser.parse_args()

    # 1) 加载官方预训练权重（优先 ema，效果更好）
    ckpt = torch.load(args.src, map_location="cpu")
    src_model = (ckpt.get("ema") or ckpt["model"]).float()
    src_state = src_model.state_dict()
    src_n_layers = len(src_model.model)

    # 2) 构建三模态模型
    dst_model = YOLO(args.dst).model
    dst_state = dst_model.state_dict()
    dst_n_layers = len(dst_model.model)

    # 3) 定位三条分支与尾部(head)对齐起点
    branches = find_branch_layers(dst_model)
    assert len(branches) == 3, f"预期 3 条分支，实际 {len(branches)} 条"

    src_sppf = find_sppf_index(src_model)
    dst_sppf = find_sppf_index(dst_model)

    # 4) 迁移 backbone：官方 model.0~8 -> 三条分支
    transferred = set()  # 记录已成功迁移的 dst key，用于报告"保持随机初始化的层"
    src_backbone = list(range(9))  # 官方 backbone 的 9 个 Conv/C3k2 层
    n_vis = transfer_layers(src_state, dst_state, src_backbone, branches[0], transferred)
    n_ir = transfer_layers(src_state, dst_state, src_backbone, branches[1], transferred)
    # 深度分支复用红外分支迁移后的权重（性质最接近）
    n_dp = transfer_layers(dst_state, dst_state, branches[1], branches[2], transferred)

    # 5) 迁移尾部(head)：官方 SPPF 起逐层对齐到末尾（含 SPPF/C2PSA/head/Detect 回归）
    src_tail = list(range(src_sppf, src_n_layers))
    dst_tail = list(range(dst_sppf, dst_n_layers))
    # 防御性校验：若尾部层数不一致，zip 会静默截断导致语义错位，必须显式告警
    if len(src_tail) != len(dst_tail):
        print(f"[警告] 尾部(head)层数不一致：官方 {len(src_tail)} 层 vs RGBTD {len(dst_tail)} 层，"
              f"zip 将静默截断，请检查 RGBTD 的 head 是否与官方 YOLO11 同构！")
    n_tail = transfer_layers(src_state, dst_state, src_tail, dst_tail, transferred)

    # 6) 写回并保存
    dst_model.load_state_dict(dst_state, strict=False)

    # 6.5) 报告保持随机初始化的层（新增层/形状不匹配层，无法从官方权重迁移）
    unmigrated = []
    for i, m in enumerate(dst_model.model):
        t = getattr(m, "type", "")
        keys = [k for k in dst_state if k.startswith(f"model.{i}.")]
        if keys and all(k not in transferred for k in keys):
            unmigrated.append(f"{i}:{t}")
    if unmigrated:
        print(f"[注意] 以下层保持随机初始化(新增层或形状不匹配，无法迁移，需在训练中学习): {unmigrated}")

    ckpt["model"] = dst_model
    ckpt.pop("ema", None)  # 移除旧 ema，训练时会重建
    torch.save(ckpt, args.out)

    print(f"[迁移完成] 可见光分支 {n_vis} 项 | 红外分支 {n_ir} 项 | 深度分支(复用红外) {n_dp} 项 | 尾部(head) {n_tail} 项")
    print(f"[输出] {args.out}")
    print("[提示] 之后用 YOLO('{0}').train(...) 即可从预训练权重开始训练".format(args.out))


if __name__ == "__main__":
    main()
