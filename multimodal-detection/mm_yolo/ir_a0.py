"""Offline IR A0 analysis for V5.1.1.

This module never rewrites source imagery.  It derives conservative masks,
processing-chain descriptors and bounded RGB-reference -> IR-source sampling
affines.  A weak estimate is explicitly unsupervised rather than being turned
into a false zero-motion label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import json
import math

import cv2
import numpy as np

try:
    from .io_utils import imread_unicode
except ImportError:  # tools may import this file as a top-level module
    from io_utils import imread_unicode

A0_VERSION = 1
A0_QUALITY_NAMES = (
    "intensity", "local_contrast", "edge", "invalid_border", "saturation",
    "blur", "double_edge", "rgb_leakage", "thermal_confidence",
    "registration_confidence",
)


@dataclass(frozen=True)
class SearchCfg:
    work_width: int = 480
    angle_limit: float = 3.0
    angle_step: float = 0.25
    scale_limit: float = 0.03
    scale_step: float = 0.01
    max_shift_frac: float = 0.02
    residual_angle: float = 0.75
    residual_scale: float = 0.005
    residual_shift_px: float = 12.0
    min_confidence: float = 0.45
    strong_confidence: float = 0.75


def read_modalities(rgb_path: Path, ir_path: Path):
    rgb_bgr = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
    ir = imread_unicode(ir_path, cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or ir is None:
        raise OSError(f"unreadable RGB/IR pair: {rgb_path} | {ir_path}")
    if ir.ndim == 2:
        ir3 = np.repeat(ir[..., None], 3, 2)
    else:
        ir3 = ir[..., :3]
    if ir3.dtype == np.uint16:
        ir3 = ir3.astype(np.float32) / 256.0
    else:
        ir3 = ir3.astype(np.float32)
    ir3 = np.clip(ir3, 0, 255)
    thermal = np.median(ir3, axis=2).astype(np.float32)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if thermal.shape != rgb.shape[:2]:
        thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        ir3 = cv2.resize(ir3, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    return rgb, thermal, ir3


def _local_stats(x: np.ndarray, k: int = 15):
    x = x.astype(np.float32)
    mean = cv2.blur(x, (k, k))
    var = cv2.blur(x * x, (k, k)) - mean * mean
    return mean, np.sqrt(np.maximum(var, 0))


def border_masks(thermal: np.ndarray):
    """Return visible and conservative geometry masks.

    Darkness alone is not invalid.  A candidate must be low-texture and
    connected to the image boundary, preventing ordinary dark objects from
    being removed.
    """
    mean, std = _local_stats(thermal, 15)
    low = min(12.0, float(np.percentile(thermal, 2)) + 3.0)
    high = max(243.0, float(np.percentile(thermal, 98)) - 3.0)
    candidate = (((mean <= low) | (mean >= high)) & (std < 5.0)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    invalid = np.zeros_like(candidate)
    h, w = candidate.shape
    min_area = max(32, round(h * w * 0.0005))
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        touches = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if touches and area >= min_area:
            invalid[labels == i] = 1
    invalid = cv2.morphologyEx(invalid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    visible = 1 - cv2.dilate(invalid, np.ones((3, 3), np.uint8), iterations=1)
    geometry = cv2.erode(visible, np.ones((7, 7), np.uint8), iterations=1)
    return visible.astype(np.float32), geometry.astype(np.float32), invalid.astype(np.float32)


def _normalized_edge(gray: np.ndarray, valid: Optional[np.ndarray] = None):
    x = gray.astype(np.float32)
    x = cv2.GaussianBlur(x, (5, 5), 0)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    if valid is not None:
        edge *= valid.astype(np.float32)
    scale = float(np.percentile(edge[edge > 0], 95)) if np.any(edge > 0) else 1.0
    return np.clip(edge / max(scale, 1e-6), 0, 1)


def leakage_maps(rgb: np.ndarray, ir3: np.ndarray, valid: np.ndarray):
    rgbf = rgb.astype(np.float32)
    irf = ir3.astype(np.float32)
    # Two independent chroma coordinates are sufficient and avoid depending on
    # OpenCV/RGB channel order after input conversion.
    rc = np.stack((rgbf[..., 0] - rgbf[..., 1], rgbf[..., 2] - rgbf[..., 1]), -1) / 255.0
    ic = np.stack((irf[..., 0] - irf[..., 1], irf[..., 2] - irf[..., 1]), -1) / 255.0
    chroma_energy = np.sqrt(np.square(rc).sum(2))
    mask = (valid > .5) & (chroma_energy > np.percentile(chroma_energy[valid > .5], 55)
                                 if np.any(valid > .5) else False)
    if mask.sum() < 64:
        return np.zeros(valid.shape, np.float32), 0.0, 0.0, 1.0
    x, y = rc[mask].reshape(-1, 2), ic[mask].reshape(-1, 2)
    # IRLS-Huber scalar coefficient.  It estimates processing leakage only;
    # it is never subtracted from the thermal intensity.
    alpha = float(np.clip((x * y).sum() / (np.square(x).sum() + 1e-8), -0.5, 0.5))
    for _ in range(5):
        residual = y - alpha * x
        norm = np.sqrt(np.square(residual).sum(1))
        delta = max(float(np.median(norm) * 1.5), .005)
        weight = np.minimum(1.0, delta / np.maximum(norm, 1e-6))
        alpha = float(np.clip((weight[:, None] * x * y).sum() /
                              ((weight[:, None] * x * x).sum() + 1e-8), -0.5, 0.5))
    residual = ic - alpha * rc
    err = np.sqrt(np.square(residual).sum(2))
    fit = np.exp(-err / .035) * valid
    base = float(np.sqrt(np.square(ic[mask]).sum(1)).mean() + 1e-6)
    fit_err = float(np.sqrt(np.square(residual[mask]).sum(1)).mean())
    confidence = float(np.clip(abs(alpha) / .15, 0, 1) * np.clip(1 - fit_err / base, 0, 1))
    return fit.astype(np.float32), alpha, confidence, fit_err


def quality_maps(rgb: np.ndarray, thermal: np.ndarray, ir3: np.ndarray,
                 out_hw: tuple[int, int] = (96, 160)):
    visible, geometry, invalid = border_masks(thermal)
    mean, std = _local_stats(thermal)
    edge = _normalized_edge(thermal, geometry)
    lap = np.abs(cv2.Laplacian(thermal, cv2.CV_32F, ksize=3))
    lap_scale = float(np.percentile(lap[geometry > .5], 95)) if np.any(geometry > .5) else 1.0
    sharp = np.clip(lap / max(lap_scale, 1e-6), 0, 1)
    blur = 1 - sharp
    dense_edge = cv2.blur((edge > .25).astype(np.float32), (7, 7))
    double_edge = np.clip(edge * dense_edge * 3, 0, 1)
    saturation = ((mean < 2) | (mean > 253)).astype(np.float32)
    leak, alpha, leak_conf, fit_err = leakage_maps(rgb, ir3, geometry)
    thermal_conf = geometry * np.clip(std / 24.0, 0, 1) * (1 - .5 * double_edge)
    channels = [mean / 255.0, np.clip(std / 64.0, 0, 1), edge, invalid,
                saturation, blur, double_edge, leak, thermal_conf,
                np.zeros_like(thermal_conf)]
    oh, ow = out_hw
    q = np.stack([cv2.resize(x.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
                  for x in channels], 0)
    return q.astype(np.float32), visible, geometry, {
        "alpha_global": alpha, "leakage_confidence": leak_conf,
        "chroma_fit_error": fit_err, "valid_ratio": float(visible.mean()),
        "ghost_score": float((double_edge * geometry).sum() / max(geometry.sum(), 1)),
        "thermal_confidence": float(thermal_conf.sum() / max(geometry.sum(), 1)),
    }


def _resize_work(rgb: np.ndarray, thermal: np.ndarray, geometry: np.ndarray, width: int):
    scale = min(1.0, width / max(rgb.shape[1], 1))
    wh = (max(64, round(rgb.shape[1] * scale)), max(36, round(rgb.shape[0] * scale)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (cv2.resize(gray, wh, interpolation=cv2.INTER_AREA),
            cv2.resize(thermal, wh, interpolation=cv2.INTER_AREA),
            cv2.resize(geometry, wh, interpolation=cv2.INTER_NEAREST), scale)


def affine_matrix(angle_deg: float, tx: float, ty: float, scale: float,
                  shape: tuple[int, int]):
    h, w = shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle_deg, scale).astype(np.float32)
    m[:, 2] += (tx, ty)
    return m


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    keep = mask > .5
    if keep.sum() < 128:
        return -1.0
    x, y = a[keep].astype(np.float32), b[keep].astype(np.float32)
    x -= x.mean(); y -= y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8)
    return float((x * y).sum() / denom)


def _candidate_score(rgb_edge, ir_edge, ir_mask, matrix):
    h, w = rgb_edge.shape
    warped_edge = cv2.warpAffine(ir_edge, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpAffine(ir_mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _corr(rgb_edge, warped_edge, warped_mask), warped_mask.mean()


def estimate_affine(rgb: np.ndarray, thermal: np.ndarray, geometry: np.ndarray,
                    cfg: SearchCfg, prior: Optional[Sequence[float]] = None):
    """Estimate an IR-source -> RGB-destination OpenCV affine candidate.

    The cache writer inverts it before storage because the model consumes a
    reference-output -> IR-source sampling transform.
    """
    rg, th, vm, work_scale = _resize_work(rgb, thermal, geometry, cfg.work_width)
    re, ie = _normalized_edge(rg), _normalized_edge(th, vm)
    base = np.asarray(prior if prior is not None else (0., 0., 0., 1.), np.float32)
    angle_lim = cfg.residual_angle if prior is not None else cfg.angle_limit
    scale_lim = cfg.residual_scale if prior is not None else cfg.scale_limit
    angle_step = min(cfg.angle_step, max(angle_lim / 3, .1))
    scale_step = min(cfg.scale_step, max(scale_lim, .0025))
    angles = base[0] + np.arange(-angle_lim, angle_lim + 1e-6, angle_step)
    scales = base[3] + np.arange(-scale_lim, scale_lim + 1e-7, scale_step)
    candidates = []
    h, w = re.shape
    max_shift = (cfg.residual_shift_px * work_scale if prior is not None
                 else cfg.max_shift_frac * max(h, w))
    for angle in angles:
        for scale in scales:
            m0 = affine_matrix(float(angle), float(base[1] * work_scale),
                               float(base[2] * work_scale), float(scale), (h, w))
            rotated = cv2.warpAffine(ie, m0, (w, h), flags=cv2.INTER_LINEAR)
            mask0 = cv2.warpAffine(vm, m0, (w, h), flags=cv2.INTER_NEAREST)
            try:
                shift, response = cv2.phaseCorrelate(
                    (rotated * mask0).astype(np.float32), (re * mask0).astype(np.float32))
            except cv2.error:
                shift, response = (0., 0.), 0.
            dx = float(np.clip(shift[0], -max_shift, max_shift))
            dy = float(np.clip(shift[1], -max_shift, max_shift))
            m = m0.copy(); m[:, 2] += (dx, dy)
            score, support = _candidate_score(re, ie, vm, m)
            score = score * math.sqrt(max(support, 1e-4))
            candidates.append((score, float(response), angle,
                               (m[0, 2] - affine_matrix(angle, 0, 0, scale, (h, w))[0, 2]) / work_scale,
                               (m[1, 2] - affine_matrix(angle, 0, 0, scale, (h, w))[1, 2]) / work_scale,
                               scale, m))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    second = candidates[min(len(candidates) - 1, max(1, len(candidates) // 20))]
    identity = affine_matrix(float(base[0]), float(base[1] * work_scale),
                             float(base[2] * work_scale), float(base[3]), (h, w))
    identity_score, _ = _candidate_score(re, ie, vm, identity)
    improvement = best[0] - identity_score
    uniqueness = best[0] - second[0]
    confidence = float(np.clip((best[0] + .05) / .35, 0, 1) *
                       np.clip((improvement + .01) / .08, 0, 1) *
                       np.clip((uniqueness + .002) / .025, 0, 1))
    return {
        "params": np.asarray([best[2], best[3], best[4], best[5]], np.float32),
        "score": float(best[0]), "identity_score": float(identity_score),
        "improvement": float(improvement), "uniqueness": float(uniqueness),
        "phase_response": float(best[1]), "confidence": confidence,
    }


def robust_sequence_prior(rows: Sequence[dict]):
    good = [r for r in rows if r["confidence"] >= .45]
    if not good:
        return np.asarray([0., 0., 0., 1.], np.float32), 0.0
    x = np.stack([r["params"] for r in good])
    weight = np.asarray([r["confidence"] for r in good], np.float32)
    centre = np.median(x, axis=0)
    mad = np.median(np.abs(x - centre), axis=0) + np.asarray([.1, 1., 1., .002])
    keep = (np.abs(x - centre) / mad < 3.5).all(1)
    if not keep.any():
        keep[:] = True
    x, weight = x[keep], weight[keep]
    order = np.argsort(x, axis=0)
    result = []
    for col in range(4):
        values, ww = x[order[:, col], col], weight[order[:, col]]
        result.append(float(values[np.searchsorted(np.cumsum(ww), ww.sum() / 2)]))
    return np.asarray(result, np.float32), float(np.clip(weight.mean() * min(1, len(x) / 3), 0, 1))


def source_to_sampling(params: Sequence[float], shape: tuple[int, int]):
    """Convert source->destination parameters to destination->source sampling."""
    m = affine_matrix(float(params[0]), float(params[1]), float(params[2]),
                      float(params[3]), shape)
    return cv2.invertAffineTransform(m).astype(np.float32)


def sampling_params(matrix: np.ndarray, shape: tuple[int, int]):
    """Decompose destination->source sampling matrix into model convention."""
    h, w = shape
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = math.sqrt(max(a * a + b * b, 1e-12))
    angle = math.degrees(math.atan2(b, a))
    centre = np.asarray([(w - 1) / 2, (h - 1) / 2], np.float32)
    mapped = matrix[:, :2] @ centre + matrix[:, 2]
    shift = mapped - centre
    return np.asarray([angle, shift[0] / max(w, 1), shift[1] / max(h, 1), scale - 1], np.float32)


def save_sample(path: Path, *, stem: str, params: Sequence[float], confidence: float,
                quality: np.ndarray, visible: np.ndarray, geometry: np.ndarray,
                meta: dict, shape: tuple[int, int], sequence: str,
                sequence_prior: Sequence[float], sequence_confidence: float,
                min_confidence: float = .45):
    sampling = source_to_sampling(params, shape)
    model_params = sampling_params(sampling, shape)
    q = quality.copy()
    q[9].fill(float(confidence))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, version=np.int32(A0_VERSION), stem=np.asarray(stem),
        source_to_rgb_params=np.asarray(params, np.float32),
        sampling_matrix=sampling, affine_physical=model_params,
        affine_confidence=np.float32(confidence),
        affine_supervised=np.uint8(confidence >= min_confidence),
        sequence=np.asarray(sequence), sequence_prior=np.asarray(sequence_prior, np.float32),
        sequence_confidence=np.float32(sequence_confidence),
        quality_maps=q.astype(np.float16),
        visible_mask=cv2.resize(visible, (q.shape[2], q.shape[1]), interpolation=cv2.INTER_AREA).astype(np.float16),
        geometry_mask=cv2.resize(geometry, (q.shape[2], q.shape[1]), interpolation=cv2.INTER_AREA).astype(np.float16),
        orig_hw=np.asarray(shape, np.int32),
        **{k: np.float32(v) for k, v in meta.items()},
    )


def load_sample(cache_root: Path, stem: str):
    path = Path(cache_root) / "samples" / f"{stem}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        if int(z["version"]) != A0_VERSION:
            raise ValueError(f"A0 cache version mismatch: {path}")
        return {k: z[k] for k in z.files}


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
