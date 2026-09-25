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

# Version 4 moves RGB-ghost suppression ahead of final border detection and
# affine estimation.  The suppression creates an analysis proxy only: source
# IR pixels are never rewritten or cached as replacement imagery.
A0_VERSION = 4
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


def _provisional_dark_border(thermal: np.ndarray):
    """Return a loose border seed without declaring any pixel invalid.

    RGB leakage may draw texture over the true black frame, so this mask is
    deliberately high-recall and is used only to calibrate the leakage model.
    Final border detection is performed after deghosting.
    """
    x = thermal.astype(np.float32)
    h, w = x.shape
    sigma = max(2.0, min(h, w) / 180.0)
    smooth = cv2.GaussianBlur(x, (0, 0), sigma)
    _, std = _local_stats(x, 15)
    p10, p35 = np.percentile(smooth, (10, 35))
    cutoff = float(np.clip(p10 + 20.0, 20.0, 80.0))
    band_size = max(8, round(min(h, w) * .18))
    band = np.zeros((h, w), np.uint8)
    band[:band_size] = 1; band[-band_size:] = 1
    band[:, :band_size] = 1; band[:, -band_size:] = 1
    candidate = band.astype(bool) & (
        (smooth <= cutoff) | ((smooth <= p35) & (std < 24.0)))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    seed = np.zeros_like(candidate)
    min_area = max(64, round(h * w * .00075))
    for i in range(1, n):
        xx, yy, ww, hh, area = stats[i]
        touches = xx == 0 or yy == 0 or xx + ww >= w or yy + hh >= h
        if touches and area >= min_area:
            seed[labels == i] = 1
    # A weak fallback remains a calibration seed, never a crop/invalid mask.
    if seed.mean() < .004:
        seed = (band.astype(bool) & (smooth <= p35) & (std < 32.0)).astype(np.uint8)
    return cv2.dilate(seed, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)


def _shift_image(x: np.ndarray, dx: int, dy: int):
    h, w = x.shape[:2]
    return cv2.warpAffine(x, np.asarray([[1, 0, dx], [0, 1, dy]], np.float32),
                          (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _fit_leakage_regression(rgb_high: np.ndarray, ir_high: np.ndarray,
                            mask: np.ndarray):
    """Fit a robust 3x3 RGB-high-frequency to IR-residual mapping."""
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    keep = mask > .5
    train = keep & (((xx + 2 * yy) % 4) != 0)
    valid = keep & ~train
    if train.sum() < 192 or valid.sum() < 64:
        return np.zeros((3, 3), np.float32), 0.0
    xt, yt = rgb_high[train], ir_high[train]
    xv, yv = rgb_high[valid], ir_high[valid]
    # Bound runtime on full-resolution frames without random sampling.
    if len(xt) > 60000:
        stride = int(math.ceil(len(xt) / 60000))
        xt, yt = xt[::stride], yt[::stride]
    ridge = np.eye(3, dtype=np.float32) * max(float((xt * xt).sum()) * 1e-6, 1e-4)
    weight = np.ones(len(xt), np.float32)
    beta = np.zeros((3, 3), np.float32)
    for _ in range(4):
        xw = xt * weight[:, None]
        beta = np.linalg.solve(xw.T @ xt + ridge, xw.T @ yt).astype(np.float32)
        residual = yt - xt @ beta
        norm = np.sqrt(np.square(residual).sum(1))
        delta = max(float(np.median(norm) * 1.5), .5)
        weight = np.minimum(1.0, delta / np.maximum(norm, 1e-6)).astype(np.float32)
    prediction = xv @ beta
    baseline = float(np.square(yv).mean() + 1e-6)
    score = float(np.clip(1.0 - np.square(yv - prediction).mean() / baseline, 0, 1))
    return beta, score


def deghost_for_a0(rgb: np.ndarray, thermal: np.ndarray, ir3: np.ndarray):
    """Build a conservative deghosted proxy used only by offline A0 analysis.

    A real RGB ghost is expected to create a sharp zero-displacement fit.  The
    fit must outperform deliberately shifted RGB controls before any component
    is subtracted.  Low-frequency thermal content is preserved.
    """
    rgbf = rgb.astype(np.float32)
    irf = ir3.astype(np.float32)
    seed = _provisional_dark_border(thermal)
    sigma = max(1.5, min(thermal.shape) / 320.0)
    rgb_high = rgbf - cv2.GaussianBlur(rgbf, (0, 0), sigma)
    ir_high = irf - cv2.GaussianBlur(irf, (0, 0), sigma)
    # A thermal boundary is normally common to all IR channels, whereas RGB
    # leakage often carries chroma.  Score chroma separately so a strong frame
    # edge cannot hide a weak ghost.  Also score an eroded dark interior for
    # grayscale leakage, where the true thermal signal should be nearly flat.
    ir_chroma = ir_high - ir_high.mean(2, keepdims=True)
    fit_seed = cv2.erode((seed > .5).astype(np.uint8),
                         np.ones((9, 9), np.uint8), iterations=1).astype(np.float32)
    if fit_seed.sum() < 256:
        fit_seed = seed
    beta_chroma, chroma_fit = _fit_leakage_regression(
        rgb_high, ir_chroma, seed)
    beta_full, full_fit = _fit_leakage_regression(
        rgb_high, ir_high, fit_seed)
    if full_fit >= chroma_fit:
        beta, identity_fit = beta_full, full_fit
        fit_target, control_seed = ir_high, fit_seed
    else:
        beta, identity_fit = beta_chroma, chroma_fit
        fit_target, control_seed = ir_chroma, seed
    offset = max(4, round(min(thermal.shape) * .012))
    control_fits = []
    for dx, dy in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
        _, score = _fit_leakage_regression(
            _shift_image(rgb_high, dx, dy), fit_target, control_seed)
        control_fits.append(score)
    control_fit = max(control_fits, default=0.0)
    specificity = max(0.0, identity_fit - control_fit)
    coverage = float(np.clip(seed.mean() / .025, 0, 1))
    presence = float(
        np.clip((identity_fit - .04) / .30, 0, 1) *
        np.clip((specificity - .01) / .14, 0, 1) * coverage)
    prediction = np.einsum("...c,cd->...d", rgb_high, beta).astype(np.float32)
    seed_values = np.abs(ir_high[seed > .5])
    cap = max(4.0, float(np.percentile(seed_values, 99)) * 1.5) if seed_values.size else 4.0
    prediction = np.clip(prediction, -cap, cap)
    residual = fit_target - prediction
    pred_mag = np.sqrt(np.square(prediction).mean(2))
    residual_mag = np.sqrt(np.square(residual).mean(2))
    local_fit = pred_mag / (pred_mag + residual_mag + 1e-3)
    positive = pred_mag[pred_mag > 0]
    magnitude_scale = float(np.percentile(positive, 90)) if positive.size else 1.0
    strength = float(np.clip((presence - .05) / .45, 0, 1))
    magnitude = np.clip(pred_mag / max(magnitude_scale, 1e-3), 0, 1)
    ghost_mask = np.clip(strength * magnitude * local_fit * 1.5, 0, 1).astype(np.float32)
    clean = np.clip(irf - prediction * (strength * local_fit)[..., None], 0, 255)
    clean_thermal = np.median(clean, axis=2).astype(np.float32)
    meta = {
        "alpha_global": float(np.linalg.norm(beta) / math.sqrt(beta.size)),
        "leakage_confidence": presence,
        "chroma_fit_error": float(1.0 - identity_fit),
        "ghost_score": presence,
        "ghost_identity_fit": float(identity_fit),
        "ghost_chroma_fit": float(chroma_fit),
        "ghost_full_fit": float(full_fit),
        "ghost_control_fit": float(control_fit),
        "ghost_specificity": float(specificity),
        "ghost_seed_ratio": float(seed.mean()),
        "ghost_mask_ratio": float((ghost_mask > .35).mean()),
        "deghost_mean_change": float(np.abs(clean_thermal - thermal).mean()),
    }
    return clean_thermal, clean.astype(np.float32), ghost_mask, seed, meta


def border_masks(thermal: np.ndarray):
    """Return visible and conservative geometry masks.

    Darkness alone is not invalid.  A candidate must be low-texture and
    connected to the image boundary, preventing ordinary dark objects from
    being removed.
    """
    mean, std = _local_stats(thermal, 15)
    low = min(12.0, float(np.percentile(thermal, 2)) + 3.0)
    high = max(243.0, float(np.percentile(thermal, 98)) - 3.0)
    # A local mean smears a narrow registration border with adjacent scene
    # content and detects only its outer half.  Include raw extreme pixels when
    # their neighbourhood is still low texture, then retain *only* components
    # connected to an image boundary below.  Thus an interior black object is
    # never removed merely because it is dark.
    smooth_extreme = ((mean <= low) | (mean >= high)) & (std < 5.0)
    raw_extreme = (thermal <= low) | (thermal >= high)
    candidate = (smooth_extreme | raw_extreme).astype(np.uint8)
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
                 out_hw: tuple[int, int] = (96, 160),
                 return_preprocessed: bool = False):
    clean_thermal, clean_ir3, ghost, _, ghost_meta = deghost_for_a0(rgb, thermal, ir3)
    visible, geometry, invalid = border_masks(clean_thermal)
    # Strong remaining RGB-explainable pixels must not vote on the transform.
    ghost_exclude = cv2.dilate((ghost > .45).astype(np.uint8),
                               np.ones((5, 5), np.uint8), iterations=1)
    reduced_geometry = geometry * (1 - ghost_exclude.astype(np.float32))
    if reduced_geometry.mean() >= max(.10, geometry.mean() * .55):
        geometry = reduced_geometry
    mean, std = _local_stats(clean_thermal)
    edge = _normalized_edge(clean_thermal, geometry)
    lap = np.abs(cv2.Laplacian(clean_thermal, cv2.CV_32F, ksize=3))
    lap_scale = float(np.percentile(lap[geometry > .5], 95)) if np.any(geometry > .5) else 1.0
    sharp = np.clip(lap / max(lap_scale, 1e-6), 0, 1)
    blur = 1 - sharp
    dense_edge = cv2.blur((edge > .25).astype(np.float32), (7, 7))
    double_edge = np.clip(edge * dense_edge * 3, 0, 1)
    saturation = ((mean < 2) | (mean > 253)).astype(np.float32)
    leak = ghost
    thermal_conf = (geometry * np.clip(std / 24.0, 0, 1) *
                    (1 - .5 * double_edge) * (1 - .75 * ghost))
    channels = [mean / 255.0, np.clip(std / 64.0, 0, 1), edge, invalid,
                saturation, blur, double_edge, leak, thermal_conf,
                np.zeros_like(thermal_conf)]
    oh, ow = out_hw
    q = np.stack([cv2.resize(x.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
                  for x in channels], 0)
    meta = {
        **ghost_meta, "valid_ratio": float(visible.mean()),
        "thermal_confidence": float(thermal_conf.sum() / max(geometry.sum(), 1)),
    }
    result = (q.astype(np.float32), visible, geometry, meta)
    if return_preprocessed:
        return (*result, clean_thermal)
    return result


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


def _common_support_metrics(rgb_edge, ir_edge, ir_mask, candidate, baseline):
    """Compare a candidate with its baseline on exactly the same pixels."""
    h, w = rgb_edge.shape
    cand_edge = cv2.warpAffine(ir_edge, candidate, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cand_mask = cv2.warpAffine(ir_mask, candidate, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    base_edge = cv2.warpAffine(ir_edge, baseline, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    base_mask = cv2.warpAffine(ir_mask, baseline, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    common = ((cand_mask > .5) & (base_mask > .5)).astype(np.float32)
    cand_score = _corr(rgb_edge, cand_edge, common)
    base_score = _corr(rgb_edge, base_edge, common)
    gains, locations = [], []
    for row in range(3):
        y0, y1 = round(row * h / 3), round((row + 1) * h / 3)
        for col in range(3):
            x0, x1 = round(col * w / 3), round((col + 1) * w / 3)
            tile = np.zeros_like(common)
            tile[y0:y1, x0:x1] = common[y0:y1, x0:x1]
            if tile.sum() < max(96, (y1 - y0) * (x1 - x0) * .08):
                continue
            c = _corr(rgb_edge, cand_edge, tile)
            b = _corr(rgb_edge, base_edge, tile)
            if c > -1 and b > -1:
                gains.append(c - b); locations.append((row, col))
    positive = [loc for gain, loc in zip(gains, locations) if gain > .003]
    # Sparse, strong RGB edges are a conservative foreground proxy: dense
    # foliage/road texture is downweighted while isolated people, vehicles and
    # signs retain a veto over a background-dominated global improvement.
    edge_seed = (rgb_edge > .20) & (common > .5)
    foreground_candidate = foreground_baseline = foreground_gain = 0.0
    foreground_support = 0.0
    if edge_seed.sum() >= 128:
        density = cv2.blur(edge_seed.astype(np.float32), (21, 21))
        density_limit = float(np.percentile(density[edge_seed], 60))
        sparse = edge_seed & (density <= density_limit) & (rgb_edge > .28)
        foreground = cv2.dilate(sparse.astype(np.uint8),
                                np.ones((13, 13), np.uint8), iterations=1)
        foreground = foreground.astype(np.float32) * common
        if foreground.sum() >= 128:
            foreground_candidate = _corr(rgb_edge, cand_edge, foreground)
            foreground_baseline = _corr(rgb_edge, base_edge, foreground)
            foreground_gain = foreground_candidate - foreground_baseline
            foreground_support = float(foreground.mean())
    return {
        "candidate_score": float(cand_score),
        "baseline_score": float(base_score),
        "gain": float(cand_score - base_score),
        "support": float(common.mean()),
        "tile_count": len(gains),
        "tile_positive": len(positive),
        "tile_median": float(np.median(gains)) if gains else -1.0,
        "tile_worst": float(min(gains)) if gains else -1.0,
        "positive_rows": len({x[0] for x in positive}),
        "positive_cols": len({x[1] for x in positive}),
        "foreground_candidate": float(foreground_candidate),
        "foreground_baseline": float(foreground_baseline),
        "foreground_gain": float(foreground_gain),
        "foreground_support": float(foreground_support),
    }


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
    # Phase correlation can lock onto a dominant background edge.  Give the
    # translation-only mode an explicit bounded grid so a small correction can
    # compete with a saturated phase-correlation shift.
    translation_base = affine_matrix(
        float(base[0]), float(base[1] * work_scale),
        float(base[2] * work_scale), float(base[3]), (h, w))
    small_shift = min(max_shift, cfg.residual_shift_px * work_scale)
    grid_step = max(.75, small_shift / 4.0)
    grid = np.arange(-small_shift, small_shift + .5 * grid_step, grid_step)
    for dx in grid:
        for dy in grid:
            m = translation_base.copy(); m[:, 2] += (float(dx), float(dy))
            score, support = _candidate_score(re, ie, vm, m)
            score = score * math.sqrt(max(support, 1e-4))
            candidates.append((score, 0.0, float(base[0]),
                               float(base[1] + dx / work_scale),
                               float(base[2] + dy / work_scale),
                               float(base[3]), m))
    candidates.sort(key=lambda x: x[0], reverse=True)
    identity = affine_matrix(float(base[0]), float(base[1] * work_scale),
                             float(base[2] * work_scale), float(base[3]), (h, w))
    angle_eps = max(angle_step * .55, .06)
    scale_eps = max(scale_step * .55, .0015)
    grouped = {"translation": [], "wide_translation": [], "rigid": [], "full": []}
    for item in candidates:
        da, ds = abs(item[2] - base[0]), abs(item[5] - base[3])
        shift_delta = max(abs(item[3] - base[1]), abs(item[4] - base[2]))
        if da <= angle_eps and ds <= scale_eps:
            mode = ("translation" if shift_delta <= cfg.residual_shift_px * 1.05
                    else "wide_translation")
        else:
            mode = "rigid" if ds <= scale_eps else "full"
        if len(grouped[mode]) < 12:
            grouped[mode].append(item)
    policies = {
        "translation": {"min_gain": .025, "min_tiles": 2, "worst": -.070,
                        "penalty": .004, "spread": 1},
        "wide_translation": {"min_gain": .060, "min_tiles": 4, "worst": -.020,
                             "penalty": .025, "spread": 2},
        "rigid": {"min_gain": .020, "min_tiles": 3, "worst": -.035,
                  "penalty": .015, "spread": 2},
        "full": {"min_gain": .040, "min_tiles": 4, "worst": -.025,
                 "penalty": .020, "spread": 2},
    }
    accepted, diagnostics = [], {}
    for mode, items in grouped.items():
        best_mode = None
        for item in items:
            metrics = _common_support_metrics(re, ie, vm, item[6], identity)
            policy = policies[mode]
            extra = 0.0
            if mode != "translation" and abs(item[2] - base[0]) > 1.5:
                extra += .008
            if mode == "full" and abs(item[5] - base[3]) > .015:
                extra += .008
            if mode == "full" and abs(item[5] - base[3]) >= .90 * scale_lim:
                extra += .025
            if metrics["baseline_score"] >= .35:
                if mode in ("translation", "wide_translation"):
                    extra += .040
                elif mode == "rigid":
                    extra += .040
                elif mode == "full":
                    extra += .045
            shift_work = np.hypot((item[3] - base[1]) * work_scale,
                                  (item[4] - base[2]) * work_scale)
            if shift_work > .85 * max_shift:
                extra += .010
            worst_limit = policy["worst"]
            if (mode == "full" and metrics["foreground_gain"] > .08 and
                    metrics["tile_positive"] >= 8):
                worst_limit = -.18
            passes = (
                metrics["candidate_score"] >= .10 and
                metrics["gain"] >= policy["min_gain"] + extra and
                metrics["tile_positive"] >= policy["min_tiles"] and
                metrics["tile_median"] >= 0 and
                metrics["tile_worst"] >= worst_limit and
                max(metrics["positive_rows"], metrics["positive_cols"]) >=
                policy["spread"])
            objective = metrics["candidate_score"] - policy["penalty"]
            record = (objective, item, metrics, passes, mode)
            if best_mode is None or objective > best_mode[0]:
                best_mode = record
            if passes:
                accepted.append(record)
        if best_mode is not None:
            item, metrics = best_mode[1], best_mode[2]
            diagnostics[mode] = {
                "params": [float(item[2]), float(item[3]), float(item[4]), float(item[5])],
                **metrics, "accepted": bool(best_mode[3]),
            }
    if accepted:
        accepted.sort(key=lambda x: x[0], reverse=True)
        _, best, best_metrics, _, selected_mode = accepted[0]
    else:
        identity_score, identity_support = _candidate_score(re, ie, vm, identity)
        best = (identity_score * math.sqrt(max(identity_support, 1e-4)), 0.0,
                float(base[0]), float(base[1]), float(base[2]), float(base[3]), identity)
        best_metrics = {
            "candidate_score": float(identity_score), "baseline_score": float(identity_score),
            "gain": 0.0, "support": float(identity_support), "tile_count": 0,
            "tile_positive": 0, "tile_median": 0.0, "tile_worst": 0.0,
            "positive_rows": 0, "positive_cols": 0,
            "foreground_candidate": float(identity_score),
            "foreground_baseline": float(identity_score),
            "foreground_gain": 0.0, "foreground_support": 0.0,
        }
        selected_mode = "identity"
    alternatives = sorted((x[1][0] for x in accepted
                           if x[1] is not best), reverse=True)
    identity_score = best_metrics["baseline_score"]
    improvement = best_metrics["gain"]
    uniqueness = max(0.0, best_metrics["candidate_score"] -
                     (alternatives[0] if alternatives else identity_score))
    absolute = float(np.clip((best_metrics["candidate_score"] - .08) / .32, 0, 1))
    identity_delta = np.asarray(
        [best[2] - base[0], best[3] - base[1], best[4] - base[2],
         (best[5] - base[3]) * 100], np.float32)
    near_prior = float(np.exp(-(
        abs(identity_delta[0]) / .35 +
        np.hypot(identity_delta[1], identity_delta[2]) / 4.0 +
        abs(identity_delta[3]) / .8)))
    # A well-aligned pair must be allowed to produce a reliable identity label:
    # requiring improvement over identity makes exact registration impossible
    # to supervise.  Non-identity corrections still need a unique improvement;
    # low absolute cross-modal correlation remains explicitly unsupervised.
    correction_evidence = float(np.clip((improvement + .002) / .06, 0, 1) *
                                np.clip((uniqueness + .001) / .02, 0, 1))
    identity_evidence = near_prior * float(np.clip((identity_score - .10) / .25, 0, 1))
    confidence = absolute * max(identity_evidence, correction_evidence)
    return {
        "params": np.asarray([best[2], best[3], best[4], best[5]], np.float32),
        "score": float(best[0]), "identity_score": float(identity_score),
        "improvement": float(improvement), "uniqueness": float(uniqueness),
        "phase_response": float(best[1]), "confidence": confidence,
        "selected_mode": selected_mode, "mode_diagnostics": diagnostics,
        "common_support": float(best_metrics["support"]),
        "tile_positive": int(best_metrics["tile_positive"]),
        "tile_median_gain": float(best_metrics["tile_median"]),
        "tile_worst_gain": float(best_metrics["tile_worst"]),
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


def select_affine_candidate(coarse: dict, refined: dict,
                            sequence_prior: Sequence[float],
                            sequence_confidence: float,
                            has_sequence_prior: bool):
    """Select a conservative A0 label without inventing cross-image motion.

    A sequence prior is meaningful only when the filename exposes a real
    source/sequence and at least two samples support it.  Unidentified
    ``PLAIN`` images and singleton sources keep their own coarse-to-fine
    estimate; they must never inherit a transform aggregated from unrelated
    images.
    """
    if refined["confidence"] >= max(.35, .8 * coarse["confidence"]):
        return dict(refined), "refined"
    if has_sequence_prior and float(sequence_confidence) > 0:
        chosen = dict(coarse)
        chosen["params"] = np.asarray(sequence_prior, np.float32)
        chosen["confidence"] = min(float(sequence_confidence),
                                   float(coarse["confidence"]))
        return chosen, "sequence_prior"
    return dict(coarse), "coarse_individual"


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


def affine_model_contract(params: Sequence[float], shape: tuple[int, int],
                          canvas: tuple[int, int] = (736, 1280),
                          angle_limit: float = 3.0,
                          shift_limit: float = 16.0,
                          scale_limit: float = .04,
                          tolerance: float = 1.05):
    """Return whether an A0 affine is representable by the V5.1.1 head.

    The cached candidate is expressed in original-image pixels while the
    aligner predicts normalized motion on the letterboxed training canvas.
    Conjugating through the nominal letterbox makes offline supervision and
    the DataLoader's fail-closed range check use the same geometry.
    """
    h, w = (int(shape[0]), int(shape[1]))
    hc, wc = (int(canvas[0]), int(canvas[1]))
    factor = min(wc / max(w, 1), hc / max(h, 1))
    ox, oy = (wc - w * factor) / 2.0, (hc - h * factor) / 2.0
    letterbox = np.asarray([[factor, 0, ox], [0, factor, oy], [0, 0, 1]],
                           np.float32)
    sampling = np.vstack((source_to_sampling(params, shape), [0, 0, 1])).astype(np.float32)
    on_canvas = letterbox @ sampling @ np.linalg.inv(letterbox)
    physical = sampling_params(on_canvas[:2], canvas)
    normalized = np.asarray((
        physical[0] / max(angle_limit, 1e-6),
        physical[1] * wc / max(shift_limit, 1e-6),
        physical[2] * hc / max(shift_limit, 1e-6),
        physical[3] / max(scale_limit, 1e-6),
    ), np.float32)
    return bool(np.max(np.abs(normalized)) <= tolerance), normalized


def save_sample(path: Path, *, stem: str, params: Sequence[float], confidence: float,
                quality: np.ndarray, visible: np.ndarray, geometry: np.ndarray,
                meta: dict, shape: tuple[int, int], sequence: str,
                sequence_prior: Sequence[float], sequence_confidence: float,
                min_confidence: float = .45,
                affine_supervised: Optional[bool] = None,
                affine_contract: Optional[Sequence[float]] = None):
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
        affine_supervised=np.uint8(
            confidence >= min_confidence if affine_supervised is None else affine_supervised),
        affine_contract=np.asarray(
            np.zeros(4, np.float32) if affine_contract is None else affine_contract,
            np.float32),
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
