"""同一 checkpoint 的全图加三模态同步切片推理。"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from data import MMDataset, boxes_norm_to_canvas, canvas_to_orig_norm, collate


def tile_windows(height: int, width: int, fraction: float = 0.6,
                 overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """生成覆盖整图的原图坐标窗口；相邻窗口保留重叠。"""
    if not (0 < fraction < 1) or not (0 <= overlap < 1):
        raise ValueError("tile fraction must be in (0,1), overlap in [0,1)")
    if height < 2 or width < 2:
        raise ValueError("image must be at least 2x2")

    def starts(length: int) -> tuple[list[int], int]:
        size = min(length, max(2, int(round(length * fraction))))
        step = max(1, int(round(size * (1 - overlap))))
        positions = list(range(0, length - size + 1, step))
        if positions[-1] != length - size:
            positions.append(length - size)
        return positions, size

    xs, tw = starts(width)
    ys, th = starts(height)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def merge_detections(dets: Sequence[np.ndarray], iou: float = 0.6,
                     max_det: int = 100) -> np.ndarray:
    """按类别做整图 NMS，再按置信度取最多 max_det 个框。"""
    if not (0 < iou < 1) or max_det < 1:
        raise ValueError("merge IoU must be in (0,1) and max_det >= 1")
    nonempty = [np.asarray(d, np.float32).reshape(-1, 6) for d in dets if len(d)]
    if not nonempty:
        return np.zeros((0, 6), np.float32)
    candidates = np.concatenate(nonempty)
    candidates = candidates[np.isfinite(candidates).all(axis=1)]
    candidates = candidates[(candidates[:, 2] > candidates[:, 0]) &
                            (candidates[:, 3] > candidates[:, 1])]
    order = np.argsort(-candidates[:, 4], kind="stable")
    kept: list[int] = []
    while len(order) and len(kept) < max_det:
        cur = int(order[0])
        kept.append(cur)
        rest = order[1:]
        if len(rest) == 0:
            break
        same = candidates[rest, 5] == candidates[cur, 5]
        a = candidates[cur, :4]
        b = candidates[rest, :4]
        lt = np.maximum(a[:2], b[:, :2])
        rb = np.minimum(a[2:], b[:, 2:])
        wh = np.maximum(rb - lt, 0)
        inter = wh[:, 0] * wh[:, 1]
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        overlap = inter / np.maximum(area_a + area_b - inter, 1e-9)
        order = rest[~(same & (overlap > iou))]
    return candidates[kept]


def tile_to_full_canvas(dets: np.ndarray, tile_M: np.ndarray, full_M: np.ndarray,
                        orig_hw: Sequence[float], window: Sequence[int],
                        canvas) -> np.ndarray:
    """切片画布框 → 原图框 → 全图画布框；丢弃内部切边截断框。"""
    if len(dets) == 0:
        return np.zeros((0, 6), np.float32)
    norm = canvas_to_orig_norm(dets[:, :4], tile_M, orig_hw, canvas)
    H, W = float(orig_hw[0]), float(orig_hw[1])
    x0, y0, x1, y1 = window
    left = (norm[:, 0] - norm[:, 2] / 2) * W
    right = (norm[:, 0] + norm[:, 2] / 2) * W
    top = (norm[:, 1] - norm[:, 3] / 2) * H
    bottom = (norm[:, 1] + norm[:, 3] / 2) * H
    valid = (norm[:, 2] > 0) & (norm[:, 3] > 0)
    # 内部切边上的不完整框交给重叠切片或全图结果处理。
    if x0 > 0:
        valid &= left > x0 + 2
    if x1 < W:
        valid &= right < x1 - 2
    if y0 > 0:
        valid &= top > y0 + 2
    if y1 < H:
        valid &= bottom < y1 - 2
    if not np.any(valid):
        return np.zeros((0, 6), np.float32)
    boxes = boxes_norm_to_canvas(norm[valid], full_M, orig_hw, canvas)
    return np.column_stack((boxes, dets[valid, 4:6])).astype(np.float32)


def decode_full_and_tiles(model, dataset: MMDataset, samples: Sequence[dict],
                          full_batch: dict, full_dets: Sequence[np.ndarray], device,
                          conf: float, iou: float, max_det: int, modalities: str,
                          off, imgsz, fraction: float = 0.6, overlap: float = 0.2,
                          merge_iou: float = 0.6, tile_batch: int = 2) -> list[np.ndarray]:
    """复用 MMDataset 为每片重新生成三模态输入、质量图和深度先验。"""
    if tile_batch < 1 or len(samples) != len(full_dets):
        raise ValueError("invalid tile batch or mismatched full-image predictions")
    from eval import _decode
    tile_samples: list[dict] = []
    owners: list[int] = []
    for owner, (sample, hw) in enumerate(zip(samples, full_batch["orig_hw"])):
        for window in tile_windows(int(hw[0]), int(hw[1]), fraction, overlap):
            tile_samples.append({**sample, "crop_xyxy": window})
            owners.append(owner)
    tiles = MMDataset(dataset.root, tile_samples, imgsz=imgsz, train=False,
                      aug=dataset.aug, prior_stride=dataset.prior_stride,
                      enabled=dataset.enabled)
    parts = [[np.asarray(d, np.float32)] for d in full_dets]
    for start in range(0, len(tiles), tile_batch):
        items = [tiles[j] for j in range(start, min(start + tile_batch, len(tiles)))]
        if any(item is None for item in items):
            raise RuntimeError("tile image read failed; refusing an incomplete submission")
        batch = collate(items)
        decoded = _decode(model, batch, device, conf, iou, max_det, modalities, off, imgsz)
        for j, det in enumerate(decoded):
            index = start + j
            owner = owners[index]
            projected = tile_to_full_canvas(
                det, batch["M"][j].numpy(), full_batch["M"][owner].numpy(),
                full_batch["orig_hw"][owner].numpy(),
                tile_samples[index]["crop_xyxy"], imgsz)
            parts[owner].append(projected)
    return [merge_detections(p, merge_iou, max_det) for p in parts]
