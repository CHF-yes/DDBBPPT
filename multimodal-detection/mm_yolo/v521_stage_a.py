"""V5.2.1 explicit A0 input and large residual alignment.

The detector receives masked raw IR and the A0 coarse transform explicitly.
RGB is optional and is reduced to a fixed, detached Sobel edge map used only
by the geometry heads.  No RGB semantic feature is passed to the IR detector.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from v52_stage_a import resample


# Frozen from the complete 1,998-sample A0 cache on 2026-09-25.  These are
# dataset-level constants, never per-image percentiles: q50 and q99 of the
# cached ghost_probability distribution.  This turns the very small A0 proxy
# values into a useful 0-1 confidence scale without inventing new evidence.
A0_GHOST_Q50 = 0.000925
A0_GHOST_Q99 = 0.015935
A0_GHOST_GAMMA = 0.75


def calibrate_a0_ghost(raw_ghost):
    unit = ((raw_ghost.float() - A0_GHOST_Q50) /
            max(A0_GHOST_Q99 - A0_GHOST_Q50, 1e-8)).clamp(0, 1)
    return unit.pow(A0_GHOST_GAMMA)


def rgb_ghost_support(rgb_edge, kernel=7):
    """Broad low-level RGB support; cannot create ghost evidence alone."""
    support = F.max_pool2d(rgb_edge.float().clamp(0, 1), kernel, stride=1,
                           padding=kernel // 2)
    return support.sqrt().detach()


def _sobel_edge(image):
    if image is None:
        return None
    gray = image.mean(1, keepdim=True).float()
    kx = gray.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)
    scale = edge.flatten(2).quantile(.95, dim=2, keepdim=True).clamp_min(1e-4)
    return (edge / scale[..., None]).clamp(0, 1).detach()


def residual_sampling(normalized, canvas, max_angle=15., max_shift=.20,
                      max_log_scale=math.log(1.25)):
    """Return output-reference -> coarse-source sampling and physical values."""
    h, w = int(canvas[0]), int(canvas[1])
    n = normalized.float().clamp(-1, 1)
    angle = n[:, 0] * math.radians(float(max_angle))
    tx = n[:, 1] * (float(w) * float(max_shift))
    ty = n[:, 2] * (float(h) * float(max_shift))
    log_scale = n[:, 3] * float(max_log_scale)
    scale = log_scale.exp()
    c, s = angle.cos() * scale, angle.sin() * scale
    cx, cy = (w - 1.) * .5, (h - 1.) * .5
    out = n.new_zeros((len(n), 3, 3))
    out[:, 0, 0], out[:, 0, 1] = c, s
    out[:, 1, 0], out[:, 1, 1] = -s, c
    out[:, 0, 2] = cx + tx - c * cx - s * cy
    out[:, 1, 2] = cy + ty + s * cx - c * cy
    out[:, 2, 2] = 1
    physical = torch.stack((angle * 180. / math.pi, tx, ty, scale), 1)
    return out, physical


def _weighted_edge_mismatch(reference, moving, weight):
    weight = weight.float().clamp(0, 1)
    numerator = ((reference.float() - moving.float()).abs() * weight).flatten(1).sum(1)
    support = weight.flatten(1).sum(1)
    return numerator / support.clamp_min(1e-4), support


def geometry_acceptance(rgb_edge, raw_edge, coarse_edge, candidate_edge, weight):
    """Detached multi-region validation for a learned global candidate.

    A candidate must improve both raw and A0 on the complete valid field,
    improve a fixed checkerboard holdout, and improve at least four spatially
    separated regions without a large local regression.  This is deliberately
    stricter than the network's confidence score: confidence is necessary but
    cannot manufacture evidence that the proposed warp is better.
    """
    with torch.no_grad():
        rgb_edge = rgb_edge.float()
        raw_edge = raw_edge.float()
        coarse_edge = coarse_edge.float()
        candidate_edge = candidate_edge.float()
        weight = weight.float().clamp(0, 1)
        raw_score, _ = _weighted_edge_mismatch(rgb_edge, raw_edge, weight)
        coarse_score, _ = _weighted_edge_mismatch(rgb_edge, coarse_edge, weight)
        candidate_score, _ = _weighted_edge_mismatch(rgb_edge, candidate_edge, weight)

        h, w = weight.shape[-2:]
        regional_base, regional_candidate, regional_valid = [], [], []
        holdout = torch.zeros_like(weight)
        for row in range(3):
            y0, y1 = row * h // 3, (row + 1) * h // 3
            for col in range(4):
                x0, x1 = col * w // 4, (col + 1) * w // 4
                tile = torch.zeros_like(weight)
                tile[:, :, y0:y1, x0:x1] = 1
                tile_weight = weight * tile
                base, support = _weighted_edge_mismatch(rgb_edge, coarse_edge, tile_weight)
                cand, _ = _weighted_edge_mismatch(rgb_edge, candidate_edge, tile_weight)
                regional_base.append(base)
                regional_candidate.append(cand)
                regional_valid.append(support >= 6.)
                if (row + col) % 2:
                    holdout[:, :, y0:y1, x0:x1] = 1
        regional_base = torch.stack(regional_base, 1)
        regional_candidate = torch.stack(regional_candidate, 1)
        regional_valid = torch.stack(regional_valid, 1)
        improved_regions = ((regional_candidate <= regional_base * .98) &
                            regional_valid).sum(1)
        worsened_regions = ((regional_candidate > regional_base * 1.08 + .01) &
                            regional_valid).sum(1)
        valid_regions = regional_valid.sum(1)
        holdout_weight = weight * holdout
        holdout_base, holdout_support = _weighted_edge_mismatch(
            rgb_edge, coarse_edge, holdout_weight)
        holdout_candidate, _ = _weighted_edge_mismatch(
            rgb_edge, candidate_edge, holdout_weight)

        geometry_ok = (
            (candidate_score <= raw_score * .98) &
            (candidate_score <= coarse_score * .97) &
            (holdout_candidate <= holdout_base * .98) &
            (holdout_support >= 12.) &
            (valid_regions >= 6) &
            (improved_regions >= 4) &
            (worsened_regions == 0))
        return {
            'geometry_ok': geometry_ok,
            'raw_score': raw_score,
            'coarse_score': coarse_score,
            'candidate_score': candidate_score,
            'holdout_base_score': holdout_base,
            'holdout_candidate_score': holdout_candidate,
            'valid_regions': valid_regions,
            'improved_regions': improved_regions,
            'worsened_regions': worsened_regions,
        }


class ExplicitCoarseIRInput(nn.Module):
    """A0 coarse warp + large global residual + bounded local flow.

    A0 candidates remain inputs, not presumed truth.  Confidence gates make
    the initial function exactly the coarse transform; rejected refinements
    therefore fall back to A0 instead of becoming an apparent alignment.
    """
    def __init__(self, p4_channels, max_angle=15., max_shift=.20,
                 max_scale=1.25, local_shift=.06):
        super().__init__()
        self.max_angle = float(max_angle)
        self.max_shift = float(max_shift)
        self.max_log_scale = math.log(float(max_scale))
        self.local_shift = float(local_shift)
        # Global motion must retain spatial layout.  The previous head applied
        # two stride-1 convolutions and then averaged every location, which
        # discarded the very information needed to distinguish left/right
        # translation and clockwise/counter-clockwise rotation.  Estimate the
        # global transform from explicit paired geometry maps on a fixed grid;
        # reserve detector features for the local residual branch.
        # Coarse-space geometry input uses only explicit V7 maps plus derived
        # RGB/IR geometry.  Removed blur, double-edge and constant availability
        # maps are not duplicated in the trainable adapter.
        geometry_channels = 22
        self.global_features = nn.Sequential(
            nn.Conv2d(geometry_channels, 96, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv2d(96, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 5)))
        self.global_head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 3 * 5, 128), nn.SiLU(),
            nn.Linear(128, 5))
        self.local_head = nn.Sequential(
            nn.Conv2d(int(p4_channels) + geometry_channels, 48, 3, padding=1, bias=False),
            nn.GroupNorm(8, 48), nn.SiLU(), nn.Conv2d(48, 3, 1))
        # A bounded refinement of the deterministic pseudo target
        # calibrated_A0 * RGB_structure_support.  Multiplication by the target
        # below guarantees that this head cannot manufacture a ghost region
        # where A0 supplied no evidence or RGB supplied no structural support.
        self.ghost_refiner = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8), nn.SiLU(), nn.Conv2d(8, 1, 1))
        # Start as an exact identity residual.  The output layer learns on the
        # first optimizer step; spatial feature layers receive gradients from
        # subsequent steps without presenting an untrained warp to detection.
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)
        nn.init.zeros_(self.ghost_refiner[-1].weight)
        nn.init.zeros_(self.ghost_refiner[-1].bias)
        with torch.no_grad():
            self.global_head[-1].bias[4] = -2.2
            self.local_head[-1].bias[2] = -2.2
        self.raw_mix = nn.Parameter(torch.tensor(-3.0))
        self.last_penalty = None
        self.last_global_normalized = None
        self.last_global_logit = None
        self.last_global_physical = None
        self.last_local_delta = None
        self.last_acceptance = None
        self.last_acceptance_target = None
        self.last_acceptance_metrics = None
        self.last_coarse_ghost_raw = None
        self.last_coarse_ghost_calibrated = None
        self.last_rgb_ghost_support = None
        self.last_coarse_ghost = None
        self.last_coarse_thermal_weight = None
        self.last_ghost_calibration_loss = None
        self.last_a0_condition = None
        self.last_stats = {}

    @staticmethod
    def _map(quality, name, fallback, shape):
        value = quality.get(name) if quality else None
        value = fallback if value is None else value
        return F.interpolate(value.float(), shape, mode='bilinear', align_corners=False)

    def _sampling_condition(self, sampling, canvas, shape):
        h, w = int(canvas[0]), int(canvas[1])
        a, b = sampling[:, 0, 0], sampling[:, 0, 1]
        d = sampling[:, 1, 1]
        ox, oy = sampling[:, 0, 2], sampling[:, 1, 2]
        scale = torch.sqrt(a.square() + b.square()).clamp_min(1e-6)
        angle = torch.atan2(b, a)
        cx, cy = (w - 1.) * .5, (h - 1.) * .5
        tx = ox - cx + a * cx + b * cy
        ty = oy - cy - b * cx + d * cy
        values = torch.stack((
            angle / math.radians(max(self.max_angle, 1e-6)),
            tx / max(float(w) * self.max_shift, 1e-6),
            ty / max(float(h) * self.max_shift, 1e-6),
            scale.log() / max(self.max_log_scale, 1e-6),
        ), 1).clamp(-2, 2)
        return values[:, :, None, None].expand(-1, -1, shape[0], shape[1])

    def _refine_ghost(self, calibrated, support):
        pseudo_target = (calibrated * support).clamp(0, 1)
        features = torch.cat((calibrated, support, pseudo_target), 1)
        correction = .25 * self.ghost_refiner(features).tanh()
        refined = (pseudo_target * (1 + correction)).clamp(0, 1)
        return refined, pseudo_target

    def forward(self, raw, quality, canvas, rgb=None, ir=None):
        if quality is None or 'v52_ir_sampling' not in quality:
            raise ValueError('V5.2.1 requires explicit A0 sampling at train and eval time')
        if 'availability' not in quality:
            raise ValueError('V5.2.1 requires modality availability')
        sampling = quality['v52_ir_sampling'].float()
        valid = quality['availability'][:, 1:2].float()
        p4 = raw['p4']
        shape = p4.shape[-2:]
        valid_p4 = F.interpolate(valid, shape, mode='nearest')
        coarse = resample(p4 * valid_p4.to(p4.dtype), sampling, canvas)

        zero = torch.zeros_like(valid_p4)
        one = torch.ones_like(valid_p4)
        raw_invalid = self._map(
            quality, 'v521_hard_mask', zero, shape).clamp(0, 1)
        raw_ghost = self._map(
            quality, 'v521_ghost_probability', zero, shape).clamp(0, 1)
        raw_ghost_conf = self._map(
            quality, 'v521_ghost_confidence', zero, shape).clamp(0, 1)
        raw_thermal = self._map(
            quality, 'v521_thermal_confidence', one, shape).clamp(0, 1)
        raw_geometry = self._map(
            quality, 'v521_geometry_mask', 1 - raw_invalid, shape).clamp(0, 1)
        raw_exclude = self._map(
            quality, 'v521_align_exclude', raw_invalid, shape).clamp(0, 1)

        raw_ir_edge = _sobel_edge(ir)
        raw_ir_edge = (torch.zeros_like(raw_ghost) if raw_ir_edge is None else
                       F.interpolate(raw_ir_edge, shape, mode='bilinear', align_corners=False))
        raw_ir_edge = raw_ir_edge * valid_p4
        raw_ir_gray = (torch.zeros_like(raw_ghost) if ir is None else
                       F.interpolate(ir.mean(1, keepdim=True).float(), shape,
                                     mode='bilinear', align_corners=False)) * valid_p4
        ir_edge = resample(raw_ir_edge, sampling, canvas)
        coarse_gray = resample(raw_ir_gray, sampling, canvas)
        rgb_edge = _sobel_edge(rgb)
        rgb_edge = (torch.zeros_like(ir_edge) if rgb_edge is None else
                    F.interpolate(rgb_edge, shape, mode='bilinear', align_corners=False))

        # The A0 mask, quality maps, availability, and IR features must share
        # the same coarse coordinate system before the geometry head sees them.
        coarse_valid = resample(valid_p4, sampling, canvas).clamp(0, 1)
        coarse_invalid = resample(raw_invalid * valid_p4, sampling, canvas).clamp(0, 1)
        coarse_ghost_raw = resample(raw_ghost * valid_p4, sampling, canvas).clamp(0, 1)
        coarse_ghost_calibrated = calibrate_a0_ghost(coarse_ghost_raw)
        coarse_ghost_conf = resample(raw_ghost_conf * valid_p4, sampling, canvas).clamp(0, 1)
        coarse_thermal = resample(raw_thermal * valid_p4, sampling, canvas).clamp(0, 1)
        coarse_geometry = resample(
            raw_geometry * valid_p4, sampling, canvas).clamp(0, 1)
        coarse_exclude = resample(raw_exclude * valid_p4, sampling, canvas).clamp(0, 1)
        ghost_rgb_support = rgb_ghost_support(rgb_edge)
        coarse_ghost, coarse_ghost_target = self._refine_ghost(
            coarse_ghost_calibrated, ghost_rgb_support)
        self.last_ghost_calibration_loss = F.smooth_l1_loss(
            coarse_ghost.float(), coarse_ghost_target.float(), beta=.05)
        coarse_weight = ((1 - coarse_ghost) * coarse_thermal *
                         (1 - coarse_invalid) * (1 - coarse_exclude) *
                         coarse_geometry * coarse_valid).clamp(0, 1)
        masked_ir_edge = ir_edge * coarse_weight
        edge_signed = rgb_edge - masked_ir_edge
        edge_difference = edge_signed.abs()
        edge_overlap = rgb_edge * masked_ir_edge
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, shape[0], device=p4.device, dtype=torch.float32),
            torch.linspace(-1, 1, shape[1], device=p4.device, dtype=torch.float32),
            indexing='ij')
        coord_x = xx[None, None].expand(len(p4), 1, -1, -1)
        coord_y = yy[None, None].expand(len(p4), 1, -1, -1)
        a0_condition = self._sampling_condition(sampling, canvas, shape)
        maps = torch.cat((coarse_invalid, coarse_ghost_raw, coarse_ghost_calibrated,
                          ghost_rgb_support, coarse_ghost,
                          coarse_ghost_conf, coarse_thermal, coarse_exclude,
                          coarse_geometry, coarse_valid, coarse_gray, rgb_edge,
                          masked_ir_edge, edge_signed, edge_difference, edge_overlap,
                          coord_x, coord_y, a0_condition), 1).to(coarse.dtype)

        global_raw = self.global_head(self.global_features(maps))
        normalized = global_raw[:, :4].tanh()
        global_conf = global_raw[:, 4:5].sigmoid()
        residual, physical = residual_sampling(
            normalized, canvas, self.max_angle, self.max_shift, self.max_log_scale)
        s3 = sampling.new_zeros((len(sampling), 3, 3))
        s3[:, :2] = sampling
        s3[:, 2, 2] = 1
        total_sampling = torch.bmm(s3, residual)[:, :2]
        candidate_ir_edge = resample(raw_ir_edge, total_sampling, canvas)
        raw_ghost_calibrated = calibrate_a0_ghost(raw_ghost)
        raw_ghost_supported, _ = self._refine_ghost(
            raw_ghost_calibrated, ghost_rgb_support)
        raw_weight = ((1 - raw_ghost_supported) * raw_thermal * raw_geometry *
                      (1 - raw_invalid) * (1 - raw_exclude) * valid_p4).clamp(0, 1)
        candidate_base_weight = resample(
            raw_thermal * raw_geometry * (1 - raw_invalid) *
            (1 - raw_exclude) * valid_p4,
            total_sampling, canvas).clamp(0, 1)
        candidate_ghost_raw = resample(
            raw_ghost * valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_ghost_calibrated = calibrate_a0_ghost(candidate_ghost_raw)
        candidate_ghost, _ = self._refine_ghost(
            candidate_ghost_calibrated, ghost_rgb_support)
        candidate_weight = (candidate_base_weight * (1 - candidate_ghost)).clamp(0, 1)
        common_weight = torch.minimum(raw_weight, torch.minimum(coarse_weight, candidate_weight))
        acceptance = geometry_acceptance(
            rgb_edge, raw_ir_edge, ir_edge, candidate_ir_edge, common_weight)
        geometry_ok = acceptance['geometry_ok']
        hard_accept = geometry_ok & (global_conf[:, 0] >= .5)

        global_p4 = resample(p4 * valid_p4.to(p4.dtype), total_sampling, canvas)
        candidate_invalid = resample(raw_invalid * valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_ghost_conf = resample(
            raw_ghost_conf * valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_thermal = resample(raw_thermal * valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_geometry = resample(
            raw_geometry * valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_exclude = resample(raw_exclude * valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_valid = resample(valid_p4, total_sampling, canvas).clamp(0, 1)
        candidate_gray = resample(raw_ir_gray, total_sampling, canvas)
        candidate_masked_edge = candidate_ir_edge * candidate_weight
        candidate_signed = rgb_edge - candidate_masked_edge
        local_maps = torch.cat((
            candidate_invalid, candidate_ghost_raw, candidate_ghost_calibrated,
            ghost_rgb_support, candidate_ghost,
            candidate_ghost_conf, candidate_thermal, candidate_exclude,
            candidate_geometry, candidate_valid, candidate_gray, rgb_edge,
            candidate_masked_edge, candidate_signed, candidate_signed.abs(),
            rgb_edge * candidate_masked_edge, coord_x, coord_y, a0_condition),
            1).to(global_p4.dtype)
        local_input = torch.cat((global_p4 * candidate_weight.to(global_p4.dtype),
                                 local_maps), 1)
        local_raw = self.local_head(local_input)
        unit = local_raw[:, :2].tanh()
        local_conf = local_raw[:, 2:3].sigmoid()
        bounds = unit.new_tensor([canvas[1] * self.local_shift,
                                  canvas[0] * self.local_shift])[None, :, None, None]
        delta = unit * bounds
        smooth = ((unit[:, :, 1:] - unit[:, :, :-1]).square().mean() +
                  (unit[:, :, :, 1:] - unit[:, :, :, :-1]).square().mean())
        self.last_penalty = .05 * normalized.square().mean() + unit.square().mean() + .2 * smooth

        # Never mix differently positioned feature maps.  During training the
        # confidence scales the transform parameters and the source is warped
        # exactly once.  Evaluation uses a hard, evidence-validated decision:
        # either the full candidate is used or the A0 transform is retained.
        global_gate = (global_conf if self.training else
                       hard_accept.to(global_conf.dtype)[:, None])
        local_gate = (local_conf if self.training else
                      (local_conf >= .5).to(local_conf.dtype))
        used_normalized = normalized * global_gate
        used_residual, _ = residual_sampling(
            used_normalized, canvas, self.max_angle, self.max_shift, self.max_log_scale)
        used_sampling = torch.bmm(s3, used_residual)[:, :2]
        used_delta = delta * local_gate * global_gate[:, :, None, None]

        result = {}
        for scale_name, feature in raw.items():
            v = F.interpolate(valid, feature.shape[-2:], mode='nearest').to(feature.dtype)
            safe_raw = feature * v
            result[scale_name] = resample(
                safe_raw, used_sampling, canvas, used_delta.to(feature.dtype))

        self.last_global_normalized = normalized
        self.last_global_logit = global_raw[:, 4]
        self.last_global_physical = physical
        self.last_local_delta = delta
        self.last_acceptance = hard_accept
        self.last_acceptance_target = geometry_ok.to(global_conf.dtype)
        self.last_acceptance_metrics = acceptance
        self.last_coarse_ghost_raw = coarse_ghost_raw.detach()
        self.last_coarse_ghost_calibrated = coarse_ghost_calibrated.detach()
        self.last_rgb_ghost_support = ghost_rgb_support.detach()
        self.last_coarse_ghost = coarse_ghost.detach()
        self.last_coarse_thermal_weight = coarse_weight.detach()
        self.last_a0_condition = a0_condition.detach()
        self.last_stats = {
            'angle_abs_mean_deg': float(physical[:, 0].detach().abs().mean()),
            'translation_rms_px': float(physical[:, 1:3].detach().square().mean().sqrt()),
            'scale_mean': float(physical[:, 3].detach().mean()),
            'global_confidence': float(global_conf.detach().mean()),
            'local_flow_rms_px': float(delta.detach().square().mean().sqrt()),
            'local_confidence': float(local_conf.detach().mean()),
            'pixel_blend_enabled': False,
            'accepted_fraction': float(hard_accept.float().mean()),
            'geometry_improves_fraction': float(geometry_ok.float().mean()),
            'improved_regions_mean': float(acceptance['improved_regions'].float().mean()),
            'worsened_regions_mean': float(acceptance['worsened_regions'].float().mean()),
            'ghost_raw_mean': float(coarse_ghost_raw.detach().mean()),
            'ghost_calibrated_mean': float(coarse_ghost_calibrated.detach().mean()),
            'ghost_rgb_support_mean': float(ghost_rgb_support.detach().mean()),
            'ghost_mean': float(coarse_ghost.detach().mean()),
            'ghost_calibration_loss': float(self.last_ghost_calibration_loss.detach()),
            'ghost_confidence': float(coarse_ghost_conf.detach().mean()),
            'geometry_support': float(coarse_geometry.detach().mean()),
            'thermal_weight_mean': float(coarse_weight.detach().mean()),
        }
        return result

    def supervision_loss(self, normalized_target, supervised, target_confidence=None):
        if self.last_global_normalized is None:
            raise RuntimeError('supervision_loss requires a preceding forward pass')
        target = normalized_target.float().clamp(-1, 1)
        supervised = supervised.float().reshape(-1).clamp(0, 1)
        # A0 candidate confidence is an input-quality descriptor, not a label
        # that the learned residual should be accepted.  Only samples with an
        # exact synthetic or approved pseudo-transform may have a positive gate
        # target; candidate-only samples must learn the low-confidence fallback.
        label_quality = (torch.ones_like(supervised) if target_confidence is None else
                         target_confidence.float().reshape(-1).clamp(0, 1))
        observed_improvement = (torch.zeros_like(supervised)
                                if self.last_acceptance_target is None else
                                self.last_acceptance_target.float().reshape(-1))
        confidence_target = torch.where(
            supervised > 0, supervised * label_quality, observed_improvement)
        error = F.smooth_l1_loss(self.last_global_normalized.float(), target,
                                 reduction='none', beta=.10).mean(1)
        weight = supervised * confidence_target.clamp_min(.05)
        geometry = (error * weight).sum() / weight.sum().clamp_min(1)
        confidence = F.binary_cross_entropy_with_logits(
            self.last_global_logit.float(), confidence_target)
        ghost_calibration = (self.last_ghost_calibration_loss if
                             self.last_ghost_calibration_loss is not None else
                             geometry.new_zeros(()))
        return geometry + .25 * confidence + .10 * ghost_calibration
