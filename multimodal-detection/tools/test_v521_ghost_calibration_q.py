"""Contract checks for dataset-global A0 ghost calibration and RGB support."""
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from v52_stage_a import resample
from v521_stage_a import (
    ExplicitCoarseIRInput,
    _sobel_edge,
    calibrate_a0_ghost,
    residual_sampling,
    rgb_ghost_support,
)
from train_v521_geometry_warmup import aligned_ir_reference


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    canvas = (96, 160)
    shape = (24, 40)
    module = ExplicitCoarseIRInput(8).to(device).eval()
    assert module.global_features[0].in_channels == 24
    assert module.local_head[0].in_channels == 8 + 24

    p4 = torch.rand(1, 8, *shape, device=device, requires_grad=True)
    ir = torch.rand(1, 1, *canvas, device=device)
    rgb = torch.zeros(1, 3, *canvas, device=device, requires_grad=True)
    with torch.no_grad():
        rgb[:, :, 20:76, 42:116] = 1.
    sampling = torch.tensor([[[1., 0., 12.], [0., 1., -6.]]], device=device)
    ghost = torch.zeros(1, 1, *shape, device=device)
    ghost[:, :, 7:18, 9:31] = .012
    ones = torch.ones_like(ghost)
    zeros = torch.zeros_like(ghost)
    quality = {
        'v52_ir_sampling': sampling,
        'availability': torch.ones(1, 3, *canvas, device=device),
        'ir': torch.cat((zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                         ghost, ones, ones), 1),
        'v521_ghost_probability': ghost,
        'v521_ghost_confidence': zeros,
        'v521_thermal_confidence': ones,
        'v521_hard_mask': zeros,
        'v521_soft_mask': zeros,
        'v521_align_exclude': zeros,
        'v521_coarse_available': ones,
    }
    output = module({'p4': p4}, quality, canvas, rgb=rgb, ir=ir)

    expected_raw = resample(ghost, sampling, canvas).clamp(0, 1)
    expected_calibrated = calibrate_a0_ghost(expected_raw)
    expected_rgb_edge = F.interpolate(_sobel_edge(rgb), shape, mode='bilinear',
                                      align_corners=False)
    expected_support = rgb_ghost_support(expected_rgb_edge)
    expected_supported = expected_calibrated * expected_support
    raw_error = float((module.last_coarse_ghost_raw - expected_raw).abs().max())
    calibrated_error = float((module.last_coarse_ghost_calibrated -
                              expected_calibrated).abs().max())
    support_error = float((module.last_rgb_ghost_support - expected_support).abs().max())
    supported_error = float((module.last_coarse_ghost - expected_supported).abs().max())
    assert raw_error < 1e-6, raw_error
    assert calibrated_error < 1e-6, calibrated_error
    assert support_error < 1e-6, support_error
    assert supported_error < 1e-6, supported_error
    assert float(module.last_coarse_ghost.max()) <= float(expected_calibrated.max()) + 1e-6
    assert float(module.last_coarse_ghost[expected_calibrated == 0].abs().max()) == 0.

    coarse_valid = resample(ones, sampling, canvas).clamp(0, 1)
    expected_weight = ((1 - expected_supported) * coarse_valid.square()).clamp(0, 1)
    weight_error = float((module.last_coarse_thermal_weight - expected_weight).abs().max())
    assert weight_error < 1e-6, weight_error

    coarse = resample(p4.detach(), sampling, canvas)
    fallback_error = float((output['p4'].detach() - coarse).abs().max())
    assert fallback_error < 1e-6, fallback_error
    output['p4'].sum().backward()
    assert rgb.grad is None or float(rgb.grad.abs().max()) == 0.0

    target = torch.tensor([[.2, -.15, .1, .05]], device=device)
    dummy = SimpleNamespace(v52_ir_input=module)
    reference = aligned_ir_reference(dummy, ir, target, quality)
    residual, _ = residual_sampling(target, canvas, module.max_angle,
                                    module.max_shift, module.max_log_scale)
    s3 = torch.zeros(1, 3, 3, device=device)
    s3[:, :2] = sampling
    s3[:, 2, 2] = 1
    expected_reference = resample(ir, torch.bmm(s3, residual)[:, :2], canvas)
    reference_error = float((reference - expected_reference).abs().max())
    assert reference_error < 1e-6, reference_error
    print({
        'passed': True,
        'geometry_channels': 24,
        'raw_error': raw_error,
        'calibrated_error': calibrated_error,
        'rgb_support_error': support_error,
        'supported_error': supported_error,
        'coarse_weight_error': weight_error,
        'fallback_error': fallback_error,
        'reference_composition_error': reference_error,
        'rgb_semantic_gradient': 0.0,
    })


if __name__ == '__main__':
    main()
