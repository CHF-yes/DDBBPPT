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

# Version 10 strengthens high-confidence RGB-ghost removal and allows a
# directionally supported rigid correction to beat periodic-grid translations.
# fit one rotated sensor rectangle against the image canvas, and estimate A0
# only from the remaining thermal geometry.  The cache contract contains only
# the four spatial maps used by the geometry path.
A0_VERSION = 10
A0_QUALITY_NAMES = (
    "ghost_probability", "black_invalid_mask", "geometry_mask",
    "thermal_confidence",
)


@dataclass(frozen=True)
class SearchCfg:
    work_width: int = 480
    # Offline A0 may correct rotations up to the agreed +/-25 degree range.
    # The learned residual head remains independently bounded to +/-15 degrees.
    angle_limit: float = 25.0
    angle_step: float = 5.0
    refine_angle_step: float = 1.0
    fine_angle_step: float = 0.25
    scale_min: float = 0.80
    scale_max: float = 1.25
    scale_step: float = 0.10
    refine_scale_step: float = 0.025
    scale_limit: float = 0.25
    max_shift_frac: float = 0.20
    max_shift_x_frac: float = 0.20
    max_shift_y_frac: float = 0.20
    residual_angle: float = 15.0
    residual_scale: float = 0.25
    residual_shift_px: float = 0.0
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


def _ghost_fit_support(rgb_high: np.ndarray):
    """Select distributed RGB detail without making any black-frame guess."""
    energy = np.sqrt(np.square(rgb_high).mean(2))
    positive = energy[energy > 0]
    if positive.size == 0:
        return np.ones(energy.shape, np.float32)
    cutoff = float(np.percentile(positive, 55))
    support = (energy >= cutoff).astype(np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8))
    # Keep support spatially distributed.  This is a ghost-fitting mask, not a
    # border mask, and therefore deliberately spans the complete image.
    if support.mean() < .05:
        support = (energy >= np.percentile(positive, 35)).astype(np.uint8)
    return support.astype(np.float32)


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
    sigma = max(1.5, min(thermal.shape) / 320.0)
    rgb_high = rgbf - cv2.GaussianBlur(rgbf, (0, 0), sigma)
    ir_high = irf - cv2.GaussianBlur(irf, (0, 0), sigma)
    support = _ghost_fit_support(rgb_high)
    # A thermal boundary is normally common to all IR channels, whereas RGB
    # leakage often carries chroma.  Score chroma separately.  No black-frame
    # estimate is allowed here: ghost removal must precede the one and only
    # polygon-frame detection pass.
    ir_chroma = ir_high - ir_high.mean(2, keepdims=True)
    beta_chroma, chroma_fit = _fit_leakage_regression(
        rgb_high, ir_chroma, support)
    beta_full, full_fit = _fit_leakage_regression(
        rgb_high, ir_high, support)
    if full_fit >= chroma_fit:
        beta, identity_fit = beta_full, full_fit
        fit_target = ir_high
    else:
        beta, identity_fit = beta_chroma, chroma_fit
        fit_target = ir_chroma
    offset = max(4, round(min(thermal.shape) * .012))
    control_fits = []
    for dx, dy in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
        _, score = _fit_leakage_regression(
            _shift_image(rgb_high, dx, dy), fit_target, support)
        control_fits.append(score)
    control_fit = max(control_fits, default=0.0)
    specificity = max(0.0, identity_fit - control_fit)
    coverage = float(np.clip(support.mean() / .12, 0, 1))
    presence = float(
        np.clip((identity_fit - .04) / .30, 0, 1) *
        np.clip((specificity - .01) / .14, 0, 1) * coverage)
    # Chroma is the safest signal for declaring a ghost, but subtracting only
    # chroma leaves the same RGB projection visible in the median/intensity
    # image.  Once the identity fit is unequivocal, remove the fitted full IR
    # component while retaining chroma specificity as the activation gate.
    removal_beta, removal_target = beta, fit_target
    if presence >= .75 and specificity >= .20 and full_fit >= .45:
        removal_beta, removal_target = beta_full, ir_high
    prediction = np.einsum(
        "...c,cd->...d", rgb_high, removal_beta).astype(np.float32)
    support_values = np.abs(ir_high[support > .5])
    cap = (max(4.0, float(np.percentile(support_values, 99)) * 1.5)
           if support_values.size else 4.0)
    prediction = np.clip(prediction, -cap, cap)
    residual = removal_target - prediction
    pred_mag = np.sqrt(np.square(prediction).mean(2))
    residual_mag = np.sqrt(np.square(residual).mean(2))
    local_fit = pred_mag / (pred_mag + residual_mag + 1e-3)
    positive = pred_mag[pred_mag > 0]
    magnitude_scale = float(np.percentile(positive, 90)) if positive.size else 1.0
    strength = float(np.clip((presence - .05) / .45, 0, 1))
    magnitude = np.clip(pred_mag / max(magnitude_scale, 1e-3), 0, 1)
    # Keep the original detection footprint for diagnostics only.  Geometry
    # must later be gated by what remains after cleaning, not by pixels that
    # have already been successfully repaired.
    ghost_detected = np.clip(
        strength * magnitude * local_fit * 1.5, 0, 1).astype(np.float32)
    if presence >= .75 and specificity >= .20:
        high_weight = strength * np.clip(.85 + .25 * local_fit, 0, 1)
    else:
        high_weight = strength * local_fit
    clean = irf - prediction * high_weight[..., None]

    # A strong zero-displacement chromatic ghost can also contain a broader
    # halo that is not represented by the first high-frequency regression.
    # Fit one guarded medium-frequency layer only after the sharp ghost has
    # already passed the shifted-control specificity test.
    mid_fit = mid_control_fit = mid_specificity = 0.0
    if presence >= .75 and specificity >= .20:
        mid_sigma = max(sigma * 5.0, 4.0)
        rgb_mid = (cv2.GaussianBlur(rgbf, (0, 0), sigma) -
                   cv2.GaussianBlur(rgbf, (0, 0), mid_sigma))
        ir_mid = (cv2.GaussianBlur(irf, (0, 0), sigma) -
                  cv2.GaussianBlur(irf, (0, 0), mid_sigma))
        mid_support = _ghost_fit_support(rgb_mid)
        mid_chroma = ir_mid - ir_mid.mean(2, keepdims=True)
        beta_mid_full, mid_full_fit = _fit_leakage_regression(
            rgb_mid, ir_mid, mid_support)
        beta_mid_chroma, mid_chroma_fit = _fit_leakage_regression(
            rgb_mid, mid_chroma, mid_support)
        if mid_full_fit >= .45:
            beta_mid, mid_fit, mid_target = beta_mid_full, mid_full_fit, ir_mid
        elif mid_full_fit >= mid_chroma_fit:
            beta_mid, mid_fit, mid_target = beta_mid_full, mid_full_fit, ir_mid
        else:
            beta_mid, mid_fit, mid_target = beta_mid_chroma, mid_chroma_fit, mid_chroma
        mid_controls = []
        for dx, dy in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
            _, score = _fit_leakage_regression(
                _shift_image(rgb_mid, dx, dy), mid_target, mid_support)
            mid_controls.append(score)
        mid_control_fit = max(mid_controls, default=0.0)
        mid_specificity = max(0.0, mid_fit - mid_control_fit)
        mid_presence = float(
            presence * np.clip((mid_fit - .03) / .24, 0, 1) *
            np.clip((mid_specificity - .01) / .12, 0, 1))
        if mid_presence > 0:
            mid_prediction = np.einsum(
                "...c,cd->...d", rgb_mid, beta_mid).astype(np.float32)
            mid_values = np.abs(mid_target[mid_support > .5])
            mid_cap = (max(3.0, float(np.percentile(mid_values, 99)) * 1.25)
                       if mid_values.size else 3.0)
            mid_prediction = np.clip(mid_prediction, -mid_cap, mid_cap)
            mid_residual = mid_target - mid_prediction
            mid_pred_mag = np.sqrt(np.square(mid_prediction).mean(2))
            mid_residual_mag = np.sqrt(np.square(mid_residual).mean(2))
            mid_local_fit = mid_pred_mag / (
                mid_pred_mag + mid_residual_mag + 1e-3)
            mid_positive = mid_pred_mag[mid_pred_mag > 0]
            mid_scale = (float(np.percentile(mid_positive, 90))
                         if mid_positive.size else 1.0)
            mid_weight = mid_presence * np.clip(.55 + .40 * mid_local_fit, 0, .95)
            clean -= mid_prediction * mid_weight[..., None]
            mid_mask = np.clip(
                mid_presence * np.clip(mid_pred_mag / max(mid_scale, 1e-3), 0, 1) *
                np.clip(.5 + mid_local_fit, 0, 1), 0, 1).astype(np.float32)
            ghost_detected = np.maximum(ghost_detected, mid_mask)

    clean = np.clip(clean, 0, 255)
    clean_high = clean - cv2.GaussianBlur(clean, (0, 0), sigma)
    clean_chroma = clean_high - clean_high.mean(2, keepdims=True)
    residual_beta_full, residual_full_fit = _fit_leakage_regression(
        rgb_high, clean_high, support)
    residual_beta_chroma, residual_chroma_fit = _fit_leakage_regression(
        rgb_high, clean_chroma, support)
    if residual_full_fit >= residual_chroma_fit:
        residual_beta = residual_beta_full
        residual_fit = residual_full_fit
        residual_target = clean_high
    else:
        residual_beta = residual_beta_chroma
        residual_fit = residual_chroma_fit
        residual_target = clean_chroma

    residual_controls = []
    for dx, dy in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
        _, score = _fit_leakage_regression(
            _shift_image(rgb_high, dx, dy), residual_target, support)
        residual_controls.append(score)
    residual_control_fit = max(residual_controls, default=0.0)
    residual_specificity = max(0.0, residual_fit - residual_control_fit)
    residual_presence = float(
        np.clip((residual_fit - .04) / .30, 0, 1) *
        np.clip((residual_specificity - .01) / .14, 0, 1) * coverage)
    # Below this level the remaining zero-displacement RGB explanation is too
    # weak to justify excluding already-cleaned pixels from affine scoring.
    if residual_fit < .08:
        residual_presence = 0.0
    residual_prediction = np.einsum(
        "...c,cd->...d", rgb_high, residual_beta).astype(np.float32)
    residual_pred_mag = np.sqrt(np.square(residual_prediction).mean(2))
    residual_error = residual_target - residual_prediction
    residual_error_mag = np.sqrt(np.square(residual_error).mean(2))
    residual_local_fit = residual_pred_mag / (
        residual_pred_mag + residual_error_mag + 1e-3)
    residual_positive = residual_pred_mag[residual_pred_mag > 0]
    residual_scale = (float(np.percentile(residual_positive, 90))
                      if residual_positive.size else 1.0)
    residual_strength = float(np.clip((residual_presence - .05) / .45, 0, 1))
    ghost_residual = np.clip(
        residual_strength *
        np.clip(residual_pred_mag / max(residual_scale, 1e-3), 0, 1) *
        residual_local_fit * 1.5, 0, 1).astype(np.float32)
    clean_thermal = np.median(clean, axis=2).astype(np.float32)
    detected_ratio = float((ghost_detected > .35).mean())
    residual_ratio = float((ghost_residual > .35).mean())
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
        "ghost_mid_fit": float(mid_fit),
        "ghost_mid_control_fit": float(mid_control_fit),
        "ghost_mid_specificity": float(mid_specificity),
        "ghost_residual_fit": float(residual_fit),
        "ghost_residual_control_fit": float(residual_control_fit),
        "ghost_residual_specificity": float(residual_specificity),
        "ghost_residual_score": float(residual_presence),
        "ghost_seed_ratio": float(support.mean()),
        "ghost_detected_mask_ratio": detected_ratio,
        "ghost_residual_mask_ratio": residual_ratio,
        # Compatibility key now describes the mask actually used downstream.
        "ghost_mask_ratio": residual_ratio,
        "deghost_mean_change": float(np.abs(clean_thermal - thermal).mean()),
    }
    return clean_thermal, clean.astype(np.float32), ghost_residual, support, meta


def border_masks(thermal: np.ndarray):
    """Fit the useful field as canvas intersected with one rotated rectangle.

    The deghosted image has a very dark boundary-connected exterior.  We fit a
    rotated sensor rectangle to its complement; OpenCV rasterization naturally
    intersects that rectangle with the image canvas.  No area, side-count,
    side-length or exterior-fraction acceptance rules are used.
    """
    x = thermal.astype(np.float32)
    h, w = x.shape
    all_valid = np.ones((h, w), np.float32)
    all_invalid = np.zeros((h, w), np.float32)
    smooth = cv2.GaussianBlur(x, (0, 0), max(1.2, min(h, w) / 420.0))
    band_width = max(3, round(min(h, w) * .025))
    band = np.zeros((h, w), np.uint8)
    band[:band_width] = 1; band[-band_width:] = 1
    band[:, :band_width] = 1; band[:, -band_width:] = 1
    central = smooth[h // 4: max(h // 4 + 1, 3 * h // 4),
                     w // 4: max(w // 4 + 1, 3 * w // 4)]
    exterior_level = float(np.median(smooth[band > 0]))
    interior_level = float(np.median(central)) if central.size else float(np.median(smooth))
    # A black frame is much darker than the thermal field.  This photometric
    # condition replaces the old collection of geometric size heuristics.
    if interior_level < 8.0 or exterior_level > max(8.0, .38 * interior_level):
        return all_valid, all_valid.copy(), all_invalid

    threshold = float(exterior_level + .35 * (interior_level - exterior_level))
    dark = (smooth <= threshold).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            np.ones((max(3, round(min(h, w) * .012)),) * 2,
                                    np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    exterior = np.zeros_like(dark)
    for i in range(1, n):
        xx, yy, ww, hh, _ = stats[i]
        if xx == 0 or yy == 0 or xx + ww >= w or yy + hh >= h:
            exterior[labels == i] = 1
    exterior = cv2.morphologyEx(exterior, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))

    content = (1 - exterior).astype(np.uint8)
    contours, _ = cv2.findContours(content, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return all_valid, all_valid.copy(), all_invalid
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return all_valid, all_valid.copy(), all_invalid

    rectangle = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
    visible = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(visible, rectangle, 1)
    outside = visible == 0
    inside = visible > 0
    if not inside.any():
        return all_valid, all_valid.copy(), all_invalid
    if outside.any() and float(np.median(smooth[outside])) > .45 * float(np.median(smooth[inside])):
        return all_valid, all_valid.copy(), all_invalid

    invalid = 1 - visible
    guard = max(5, round(min(h, w) * .012))
    geometry = cv2.erode(visible, np.ones((guard, guard), np.uint8), iterations=1)
    return visible.astype(np.float32), geometry.astype(np.float32), invalid.astype(np.float32)


def _normalized_edge(gray: np.ndarray, valid: Optional[np.ndarray] = None):
    x = gray.astype(np.float32)
    safe = None
    if valid is not None:
        valid8 = (valid > .5).astype(np.uint8)
        if valid8.mean() > .05 and valid8.mean() < .999:
            outside = ((1 - valid8) * 255).astype(np.uint8)
            x = cv2.inpaint(np.clip(x, 0, 255).astype(np.uint8), outside,
                            3, cv2.INPAINT_TELEA).astype(np.float32)
        safe = cv2.erode(valid8, np.ones((3, 3), np.uint8), iterations=1)
    x = cv2.GaussianBlur(x, (5, 5), 0)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    if safe is not None:
        edge *= safe.astype(np.float32)
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
    clean_thermal, clean_ir3, ghost_residual, _, ghost_meta = deghost_for_a0(
        rgb, thermal, ir3)
    visible, geometry, invalid = border_masks(clean_thermal)
    # Only remaining RGB-explainable pixels are excluded.  The original
    # detection footprint is diagnostic and must not suppress repaired edges.
    ghost_exclude = cv2.dilate((ghost_residual > .45).astype(np.uint8),
                               np.ones((5, 5), np.uint8), iterations=1)
    geometry = geometry * (1 - ghost_exclude.astype(np.float32))
    _, std = _local_stats(clean_thermal)
    thermal_conf = (geometry * np.clip(std / 24.0, 0, 1) *
                    (1 - .75 * ghost_residual)).astype(np.float32)
    channels = [ghost_residual, invalid, geometry, thermal_conf]
    oh, ow = out_hw
    q = np.stack([cv2.resize(x.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
                  for x in channels], 0)
    meta = {
        **ghost_meta,
        "black_border_area_px": float(invalid.sum()),
        "source_height": float(thermal.shape[0]),
        "source_width": float(thermal.shape[1]),
        "a0_preprocess_version": 11.0,
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


def _homogeneous_affine(matrix: np.ndarray):
    return np.vstack((np.asarray(matrix, np.float32),
                      np.asarray((0., 0., 1.), np.float32)))


def _source_params_from_matrix(matrix: np.ndarray,
                               shape: tuple[int, int]):
    matrix = np.asarray(matrix, np.float32)
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = math.sqrt(max(a * a + b * b, 1e-12))
    angle = math.degrees(math.atan2(b, a))
    h, w = shape
    centre = np.asarray([(w - 1) / 2, (h - 1) / 2], np.float32)
    mapped = matrix[:, :2] @ centre + matrix[:, 2]
    shift = mapped - centre
    return np.asarray((angle, shift[0], shift[1], scale), np.float32)


def compose_source_affine(base_params: Sequence[float],
                          delta_params: Sequence[float],
                          shape: tuple[int, int]):
    """Compose source-to-RGB similarities while preserving one final resample.

    Source pixels first receive the cached A0/base transform and then the
    residual transform, hence A_total = A_delta @ A_base.  In inverse-sampling
    coordinates this is exactly S_total = S_base @ S_delta.
    """
    base = affine_matrix(float(base_params[0]), float(base_params[1]),
                         float(base_params[2]), float(base_params[3]), shape)
    delta = affine_matrix(float(delta_params[0]), float(delta_params[1]),
                          float(delta_params[2]), float(delta_params[3]), shape)
    total = (_homogeneous_affine(delta) @ _homogeneous_affine(base))[:2]
    return _source_params_from_matrix(total, shape), total.astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    keep = mask > .5
    if keep.sum() < 128:
        return -1.0
    x, y = a[keep].astype(np.float32), b[keep].astype(np.float32)
    x -= x.mean(); y -= y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8)
    return float((x * y).sum() / denom)


def _profile_corr(a: np.ndarray, b: np.ndarray, support: np.ndarray):
    keep = support > 1e-4
    if keep.sum() < 8:
        return -1.0
    x, y = a[keep].astype(np.float32), b[keep].astype(np.float32)
    x -= x.mean(); y -= y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8)
    return float((x * y).sum() / denom)


def _directional_projection_score(reference: np.ndarray, moving: np.ndarray,
                                  mask: np.ndarray):
    """Compare edge-energy projections at 0, 90, 45 and 135 degrees."""
    h, w = reference.shape
    yy, xx = np.indices((h, w))
    weight = np.clip(mask.astype(np.float32), 0, 1)
    directions = ((xx, w), (yy, h), (xx + yy, w + h - 1),
                  (xx - yy + h - 1, w + h - 1))
    scores = []
    for index, bins in directions:
        flat = index.reshape(-1)
        support = np.bincount(flat, weights=weight.reshape(-1), minlength=bins)
        ref = np.bincount(flat, weights=(reference * weight).reshape(-1), minlength=bins)
        mov = np.bincount(flat, weights=(moving * weight).reshape(-1), minlength=bins)
        ref = ref / np.maximum(support, 1e-4)
        mov = mov / np.maximum(support, 1e-4)
        score = _profile_corr(ref, mov, support)
        if score > -1:
            scores.append(score)
    return float(np.mean(scores)) if scores else -1.0


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
    projection_candidate = _directional_projection_score(
        rgb_edge, cand_edge, common)
    projection_baseline = _directional_projection_score(
        rgb_edge, base_edge, common)
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
        "projection_candidate": float(projection_candidate),
        "projection_baseline": float(projection_baseline),
        "projection_gain": float(projection_candidate - projection_baseline),
    }


def _estimate_affine_v4(rgb: np.ndarray, thermal: np.ndarray, geometry: np.ndarray,
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


def estimate_affine(rgb: np.ndarray, thermal: np.ndarray, geometry: np.ndarray,
                    cfg: SearchCfg, prior: Optional[Sequence[float]] = None):
    """Search raw and old-A0 baselines over the enlarged A0 ranges."""
    rg, th, vm, work_scale = _resize_work(rgb, thermal, geometry, cfg.work_width)
    re, ie = _normalized_edge(rg), _normalized_edge(th, vm)
    h, w = re.shape
    raw = np.asarray((0., 0., 0., 1.), np.float32)
    old = np.asarray(prior if prior is not None else raw, np.float32).copy()
    old[0] = np.clip(old[0], -cfg.angle_limit, cfg.angle_limit)
    old[3] = np.clip(old[3], cfg.scale_min, cfg.scale_max)

    def make_matrix(params):
        return affine_matrix(float(params[0]), float(params[1] * work_scale),
                             float(params[2] * work_scale), float(params[3]),
                             (h, w))

    def compose(base, delta):
        base_work = np.asarray((base[0], base[1] * work_scale,
                                base[2] * work_scale, base[3]), np.float32)
        delta_work = np.asarray((delta[0], delta[1] * work_scale,
                                 delta[2] * work_scale, delta[3]), np.float32)
        params, total = compose_source_affine(base_work, delta_work, (h, w))
        params[1:3] /= work_scale
        return params, total

    def evaluate(params, origin, mode, response=0.0, matrix=None, delta=None):
        params = np.asarray(params, np.float32)
        matrix = make_matrix(params) if matrix is None else np.asarray(matrix, np.float32)
        score, support = _candidate_score(re, ie, vm, matrix)
        return {
            "params": params, "origin": np.asarray(origin, np.float32),
            "mode": mode, "matrix": matrix, "score": float(score),
            "support": float(support), "rank": float(
                score * math.sqrt(max(support, 1e-4))),
            "response": float(response),
            "delta": np.asarray((0., 0., 0., 1.) if delta is None else delta,
                                np.float32),
        }

    max_shift_x = cfg.max_shift_x_frac * w
    max_shift_y = cfg.max_shift_y_frac * h

    def phase_item(angle, scale, tx, ty, origin, mode):
        _, m0 = compose(origin, (angle, tx, ty, scale))
        rotated = cv2.warpAffine(ie, m0, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask0 = cv2.warpAffine(vm, m0, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        try:
            shift, response = cv2.phaseCorrelate(
                (rotated * mask0).astype(np.float32),
                (re * mask0).astype(np.float32))
        except cv2.error:
            shift, response = (0., 0.), 0.
        dx = float(np.clip(shift[0], -max_shift_x, max_shift_x)) / work_scale
        dy = float(np.clip(shift[1], -max_shift_y, max_shift_y)) / work_scale
        delta = np.asarray((angle, tx + dx, ty + dy, scale), np.float32)
        params, total = compose(origin, delta)
        return evaluate(params, origin, mode, response, total, delta)

    raw_item = evaluate(raw, raw, "raw")
    old_item = evaluate(old, old, "old_a0")
    distinct_old = bool(prior is not None and np.max(np.abs(old - raw)) > 1e-4)
    simple = (old_item if distinct_old and
              old_item["score"] > raw_item["score"] + .010 else raw_item)
    simple_score = float(simple["score"])
    candidates = []

    # Translation-only candidates cover the complete +/-20 percent range.
    fractions = (-.20, -.10, 0., .10, .20)
    bases = [(raw, "raw_translation")]
    if distinct_old:
        bases.append((old, "old_translation"))
    for base, mode in bases:
        phase = phase_item(0., 1., 0., 0., base, mode)
        candidates.append(phase)
        for fx in fractions:
            for fy in fractions:
                delta = np.asarray((0., fx * w / work_scale,
                                    fy * h / work_scale, 1.), np.float32)
                params, total = compose(base, delta)
                candidates.append(evaluate(params, base, mode, 0., total, delta))
        # Search small translations explicitly instead of relying on one
        # phase-correlation peak.  This keeps examples needing only a modest
        # tx/ty correction from being displaced by a background-dominated
        # phase estimate or promoted to a rotation/scale candidate.
        for fx in (-.05, -.025, -.0125, 0., .0125, .025, .05):
            for fy in (-.05, -.025, -.0125, 0., .0125, .025, .05):
                delta = np.asarray((0., fx * w / work_scale,
                                    fy * h / work_scale, 1.), np.float32)
                params, total = compose(base, delta)
                candidates.append(evaluate(params, base, mode, 0., total, delta))
        centre = phase["delta"]
        for fx in (-.025, 0., .025):
            for fy in (-.025, 0., .025):
                delta = np.asarray((0., centre[1] + fx * w / work_scale,
                                    centre[2] + fy * h / work_scale, 1.), np.float32)
                params, total = compose(base, delta)
                candidates.append(evaluate(params, base, mode, 0., total, delta))

    coarse_scales = (cfg.scale_min, .90, 1.0, 1.10, 1.20, cfg.scale_max)
    coarse = []
    # Residual search around old A0 uses true matrix composition and is bounded
    # to +/-15 degrees, +/-20 percent translation and scale 0.8--1.25.
    if distinct_old:
        residual_step = max(2.5, cfg.residual_angle / 5.0)
        for angle in np.arange(-cfg.residual_angle,
                               cfg.residual_angle + 1e-6, residual_step):
            for scale in coarse_scales:
                mode = "old_rigid" if abs(scale - 1.0) < .015 else "old_full"
                item = phase_item(float(angle), float(scale), 0., 0., old, mode)
                candidates.append(item); coarse.append(item)

    # Raw rigid rescue gets its own coarse pool so scale candidates cannot use
    # every refinement slot and hide a valid rotation-only solution.
    angles = list(np.arange(-cfg.angle_limit, cfg.angle_limit + 1e-6,
                            cfg.angle_step)) + [0.0, float(old[0])]
    raw_rigid_coarse = []
    for angle in sorted(set(round(float(x), 4) for x in angles)):
        if abs(angle) >= cfg.fine_angle_step:
            item = phase_item(angle, 1.0, 0.0, 0.0, raw, "raw_rescue")
            candidates.append(item); coarse.append(item)
            raw_rigid_coarse.append(item)
        for scale in coarse_scales:
            if abs(scale - 1.0) < .015:
                continue
            item = phase_item(angle, float(scale), 0.0, 0.0, raw, "raw_full")
            candidates.append(item); coarse.append(item)

    refined = []
    for seed in sorted(coarse, key=lambda x: x["rank"], reverse=True)[:5]:
        prefix = "old" if seed["mode"].startswith("old") else "raw"
        centre_angle = float(seed["delta"][0])
        centre_scale = float(seed["delta"][3])
        angle_limit = cfg.residual_angle if prefix == "old" else cfg.angle_limit
        for angle in np.arange(max(-angle_limit, centre_angle - cfg.angle_step),
                               min(angle_limit, centre_angle + cfg.angle_step) + 1e-6,
                               cfg.refine_angle_step):
            for scale in np.arange(max(cfg.scale_min, centre_scale - .08),
                                   min(cfg.scale_max, centre_scale + .08) + 1e-7,
                                   cfg.refine_scale_step):
                if prefix == "raw":
                    mode = "raw_rescue" if abs(scale - 1.0) < .015 else "raw_full"
                else:
                    mode = "old_rigid" if abs(scale - 1.0) < .015 else "old_full"
                base = old if prefix == "old" else raw
                item = phase_item(float(angle), float(scale), 0., 0., base, mode)
                candidates.append(item); refined.append(item)

    for seed in sorted(refined, key=lambda x: x["rank"], reverse=True)[:3]:
        prefix = "old" if seed["mode"].startswith("old") else "raw"
        centre_angle = float(seed["delta"][0])
        centre_scale = float(seed["delta"][3])
        angle_limit = cfg.residual_angle if prefix == "old" else cfg.angle_limit
        for angle in np.arange(max(-angle_limit, centre_angle - cfg.refine_angle_step),
                               min(angle_limit, centre_angle + cfg.refine_angle_step) + 1e-7,
                               cfg.fine_angle_step):
            for scale in np.arange(max(cfg.scale_min, centre_scale - .03),
                                   min(cfg.scale_max, centre_scale + .03) + 1e-7,
                                   .015):
                if prefix == "raw":
                    mode = "raw_rescue" if abs(scale - 1.0) < .015 else "raw_full"
                else:
                    mode = "old_rigid" if abs(scale - 1.0) < .015 else "old_full"
                base = old if prefix == "old" else raw
                candidates.append(phase_item(float(angle), float(scale),
                                             0., 0., base, mode))

    # Independently refine the best raw rigid seeds at 0.5 then 0.25 degrees.
    # This specifically preserves the stable 15--17 degree basin that can be
    # missed when full-similarity candidates dominate the shared ranking.
    rigid_refined = []
    for seed in sorted(raw_rigid_coarse,
                       key=lambda x: x["rank"], reverse=True)[:3]:
        centre_angle = float(seed["delta"][0])
        for angle in np.arange(max(-cfg.angle_limit, centre_angle - cfg.angle_step),
                               min(cfg.angle_limit, centre_angle + cfg.angle_step) + 1e-7,
                               .5):
            item = phase_item(float(angle), 1.0, 0., 0., raw, "raw_rescue")
            candidates.append(item); rigid_refined.append(item)

    rigid_fine = []
    for seed in sorted(rigid_refined,
                       key=lambda x: x["rank"], reverse=True)[:3]:
        centre_angle = float(seed["delta"][0])
        for angle in np.arange(max(-cfg.angle_limit, centre_angle - .5),
                               min(cfg.angle_limit, centre_angle + .5) + 1e-7,
                               .25):
            item = phase_item(float(angle), 1.0, 0., 0., raw, "raw_rescue")
            candidates.append(item); rigid_fine.append(item)

    # With angle fixed, compare zero translation and the phase estimate using
    # a small local grid.  Periodic fences can otherwise produce a plausible
    # angle paired with the wrong phase-correlation peak.
    for seed in sorted(rigid_fine,
                       key=lambda x: x["rank"], reverse=True)[:3]:
        angle = float(seed["delta"][0])
        phase_tx, phase_ty = (float(seed["delta"][1]),
                              float(seed["delta"][2]))
        # Use a few work-image pixels rather than a fraction of the full
        # canvas.  On 1920-wide inputs the old 1.25% step was 24 source pixels
        # and skipped the stable translation basin near the phase estimate.
        local_step = 2.5 / work_scale
        for centre_tx, centre_ty in ((0., 0.), (phase_tx, phase_ty)):
            for dx in (-local_step, 0., local_step):
                for dy in (-local_step, 0., local_step):
                    delta = np.asarray((angle,
                                        centre_tx + dx,
                                        centre_ty + dy,
                                        1.), np.float32)
                    params, total = compose(raw, delta)
                    candidates.append(evaluate(
                        params, raw, "raw_rescue", 0., total, delta))

    policies = {
        "raw_translation": (.020, 2, -.070, 1, .004),
        "old_translation": (.016, 2, -.060, 1, .004),
        "old_rigid": (.030, 3, -.035, 2, .015),
        "raw_rescue": (.040, 4, -.025, 2, .018),
        "old_full": (.045, 4, -.025, 2, .024),
        "raw_full": (.055, 5, -.020, 2, .028),
    }
    grouped = {name: [] for name in policies}
    for item in sorted(candidates, key=lambda x: x["rank"], reverse=True):
        mode_cap = 32 if item["mode"] == "raw_rescue" else 16
        if len(grouped[item["mode"]]) < mode_cap:
            grouped[item["mode"]].append(item)

    accepted, diagnostics = [], {}
    for mode, items in grouped.items():
        min_gain, min_tiles, worst, spread, penalty = policies[mode]
        best_mode = None
        for item in items:
            metrics = _common_support_metrics(
                re, ie, vm, item["matrix"], make_matrix(item["origin"]))
            da, dtx, dty, ds = (float(x) for x in item["delta"])
            large = (abs(da) > 15.0 or abs(dtx) > .10 * w / work_scale or
                     abs(dty) > .10 * h / work_scale or ds < .90 or ds > 1.12)
            tiles = max(min_tiles, 6 if large else 0)
            required_spread = max(spread, 3 if abs(da) > 20.0 else spread)
            boundary = (abs(dtx) >= .19 * w / work_scale or
                        abs(dty) >= .19 * h / work_scale or
                        ds <= cfg.scale_min + .01 or ds >= cfg.scale_max - .01)
            boundary_extra = .030 if boundary else 0.0
            high_baseline_extra = .040 if metrics["baseline_score"] >= .35 else 0.0
            foreground_ok = not (
                metrics["foreground_support"] >= .01 and
                metrics["foreground_gain"] < -.005)
            strong_rotation_consensus = (
                abs(da) >= 10.0 and
                metrics["candidate_score"] >= .25 and
                metrics["gain"] >= .10 and
                metrics["tile_positive"] >= 6 and
                metrics["tile_median"] >= .04 and
                metrics["positive_rows"] >= 3 and
                metrics["positive_cols"] >= 3 and
                metrics["foreground_gain"] >= .02 and
                metrics["projection_candidate"] >= .20)
            distributed_consensus = (
                mode not in ("raw_translation", "old_translation") and
                metrics["candidate_score"] >= .25 and
                metrics["gain"] >= .10 and
                metrics["tile_count"] >= 8 and
                metrics["tile_positive"] >= max(6, metrics["tile_count"] - 2) and
                metrics["tile_median"] >= .04 and
                metrics["positive_rows"] >= 3 and
                metrics["positive_cols"] >= 3 and
                metrics["foreground_gain"] >= .02 and
                metrics["projection_gain"] >= .02)
            directional_rigid_consensus = (
                mode in ("raw_rescue", "old_rigid") and
                abs(da) >= 8.0 and
                metrics["candidate_score"] >= .20 and
                metrics["gain"] >= .08 and
                metrics["tile_positive"] >= 5 and
                metrics["tile_median"] >= .01 and
                max(metrics["positive_rows"], metrics["positive_cols"]) >= 3 and
                min(metrics["positive_rows"], metrics["positive_cols"]) >= 2 and
                metrics["foreground_gain"] >= .05 and
                metrics["projection_candidate"] >= .35 and
                metrics["projection_gain"] >= .12 and
                metrics["tile_worst"] >= -.08)
            projection_ok = (mode in ("raw_translation", "old_translation") or
                             (metrics["projection_candidate"] >= .05 and
                              metrics["projection_gain"] >= .002) or
                             strong_rotation_consensus or
                             directional_rigid_consensus)
            worst_limit = worst
            if (strong_rotation_consensus and
                    mode in ("raw_rescue", "old_rigid")):
                worst_limit = min(worst_limit, -.080)
            if directional_rigid_consensus:
                worst_limit = min(worst_limit, -.080)
            tile_consensus_ok = (metrics["tile_worst"] >= worst_limit or
                                 distributed_consensus)
            # After aggressive deghosting only a small part of the frame may
            # remain eligible.  A relative gain over a negative baseline is
            # then misleading, so low-support candidates need a real absolute
            # cross-modal match before any motion is accepted.
            absolute_floor = .16 if metrics["support"] < .25 else .10
            passes = (
                metrics["candidate_score"] >= max(absolute_floor,
                                                   simple_score + .004) and
                metrics["gain"] >= min_gain + high_baseline_extra + boundary_extra and
                metrics["tile_positive"] >= tiles and
                metrics["tile_median"] >= 0 and
                tile_consensus_ok and
                metrics["positive_rows"] >= required_spread and
                metrics["positive_cols"] >= required_spread and
                foreground_ok and projection_ok)
            objective = metrics["candidate_score"] - penalty
            record = (objective, item, metrics, passes, mode)
            if best_mode is None or objective > best_mode[0]:
                best_mode = record
            if passes and objective > simple_score + .005:
                accepted.append(record)
        if best_mode is not None:
            item, metrics = best_mode[1], best_mode[2]
            diagnostics[mode] = {
                "params": [float(x) for x in item["params"]],
                "residual": [float(x) for x in item["delta"]],
                **metrics, "accepted": bool(best_mode[3]),
            }

    if accepted:
        accepted.sort(key=lambda x: x[0], reverse=True)
        top = accepted[0]
        directional_override = None
        if top[4] in ("raw_translation", "old_translation"):
            top_delta = top[1]["delta"]
            top_large_shift = (
                abs(float(top_delta[1])) >= .095 * w / work_scale or
                abs(float(top_delta[2])) >= .095 * h / work_scale)
            directional = [x for x in accepted
                           if x[4] in ("raw_rescue", "old_rigid") and
                           abs(float(x[1]["delta"][0])) >= 8.0 and
                           x[2]["foreground_gain"] >= .05 and
                           x[2]["projection_gain"] >= .12 and
                           x[2]["tile_positive"] >= 5 and
                           x[2]["tile_worst"] >= -.08]
            if top_large_shift and directional:
                directional.sort(key=lambda x: x[0], reverse=True)
                if (float(top[2]["candidate_score"]) -
                        float(directional[0][2]["candidate_score"]) <= .050):
                    directional_override = directional[0]
        if directional_override is not None:
            _, best, best_metrics, _, selected_mode = directional_override
        else:
            top_score = float(accepted[0][2]["candidate_score"])
            complexity = {
                "raw_translation": 0, "old_translation": 0,
                "raw_rescue": 1, "old_rigid": 1,
                "raw_full": 2, "old_full": 2,
            }
            near_best = [x for x in accepted
                         if top_score - float(x[2]["candidate_score"]) <= .030]
            near_best.sort(key=lambda x: (complexity[x[4]], -x[0]))
            _, best, best_metrics, _, selected_mode = near_best[0]
    else:
        best, selected_mode = simple, simple["mode"]
        best_metrics = {
            "candidate_score": simple["score"], "baseline_score": simple["score"],
            "gain": 0.0, "support": simple["support"], "tile_count": 0,
            "tile_positive": 0, "tile_median": 0.0, "tile_worst": 0.0,
            "positive_rows": 0, "positive_cols": 0,
            "foreground_candidate": simple["score"],
            "foreground_baseline": simple["score"], "foreground_gain": 0.0,
            "foreground_support": 0.0,
            "projection_candidate": simple["score"],
            "projection_baseline": simple["score"], "projection_gain": 0.0,
        }

    alternatives = ([x[2]["candidate_score"] for x in accepted if x[1] is not best]
                    + [simple_score])
    improvement = float(best_metrics["gain"])
    uniqueness = max(0.0, float(best_metrics["candidate_score"]) -
                     max(alternatives, default=simple_score))
    absolute = float(np.clip((best_metrics["candidate_score"] - .08) / .32, 0, 1))
    if selected_mode in ("raw", "old_a0"):
        evidence = float(np.clip((best_metrics["candidate_score"] - .10) / .25, 0, 1))
    else:
        evidence = float(np.clip((improvement + .002) / .06, 0, 1) *
                         np.clip((uniqueness + .001) / .02, 0, 1))
    return {
        "params": np.asarray(best["params"], np.float32),
        "residual_params": np.asarray(best["delta"], np.float32),
        "score": float(best["rank"]),
        "identity_score": float(raw_item["score"]),
        "old_a0_score": float(old_item["score"]),
        "baseline_mode": simple["mode"],
        "improvement": improvement, "uniqueness": float(uniqueness),
        "phase_response": float(best["response"]),
        "confidence": absolute * evidence,
        "selected_mode": selected_mode, "mode_diagnostics": diagnostics,
        "common_support": float(best_metrics["support"]),
        "tile_positive": int(best_metrics["tile_positive"]),
        "tile_median_gain": float(best_metrics["tile_median"]),
        "tile_worst_gain": float(best_metrics["tile_worst"]),
        "directional_projection_gain": float(best_metrics["projection_gain"]),
        "search_bounds": {
            "raw_angle": float(cfg.angle_limit),
            "residual_angle": float(cfg.residual_angle),
            "shift_x_fraction": float(cfg.max_shift_x_frac),
            "shift_y_fraction": float(cfg.max_shift_y_frac),
            "scale_min": float(cfg.scale_min),
            "scale_max": float(cfg.scale_max),
        },
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
                          angle_limit: float = 25.0,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    ghost_confidence = float(meta.get("ghost_score", 0.0))
    coarse_available = float(confidence >= min_confidence)
    align_exclude = np.maximum(q[1], 1 - q[2]).astype(np.float16)
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
        ghost_probability=q[0].astype(np.float16),
        ghost_transform_confidence=np.float32(ghost_confidence),
        coarse_candidate_available=np.float32(coarse_available),
        thermal_confidence_map=q[3].astype(np.float16),
        hard_mask=q[1].astype(np.float16),
        align_exclude_mask=align_exclude,
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
