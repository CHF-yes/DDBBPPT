"""Contract checks for deterministic A0 plus quality-only IR input."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'mm_yolo'):
    sys.path.insert(0, str(path))

import torch
import cv2
import numpy as np

from v521_stage_a import ExplicitCoarseIRInput
from data import _fill_ir_exterior


def main():
    torch.manual_seed(7)
    canvas = (128, 224)
    raw = {
        'p2': torch.rand(2, 24, 32, 56, requires_grad=True),
        'p3': torch.rand(2, 32, 16, 28, requires_grad=True),
        'p4': torch.rand(2, 48, 8, 14, requires_grad=True),
        'p5': torch.rand(2, 64, 4, 7, requires_grad=True),
    }
    sampling = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]).repeat(2, 1, 1)
    availability = torch.ones(2, 3, *canvas)
    availability[:, 1, :, :24] = 0
    qshape = (32, 56)
    quality = {
        'v52_ir_sampling': sampling,
        'availability': availability,
        'v521_ghost_probability': torch.full((2, 1, *qshape), .01),
        'v521_ghost_confidence': torch.full((2, 1, 1, 1), .8),
        'v521_coarse_available': torch.tensor([1., 0.]).reshape(2, 1, 1, 1),
        'v521_content_weight': torch.tensor([1., .25]).reshape(2, 1, 1, 1),
        'v521_alignment_weight': torch.tensor([1., .10]).reshape(2, 1, 1, 1),
        'v521_fallback_raw': torch.tensor([0., 1.]).reshape(2, 1, 1, 1),
        'v521_thermal_confidence': torch.ones(2, 1, *qshape),
        'v521_hard_mask': torch.zeros(2, 1, *qshape),
        'v521_align_exclude': torch.zeros(2, 1, *qshape),
        'v521_geometry_mask': torch.ones(2, 1, *qshape),
    }
    module = ExplicitCoarseIRInput(48)
    out1 = module.quality_only(raw, quality, canvas)
    # Learned global/local geometry heads must be irrelevant in this mode.
    with torch.no_grad():
        for head in (module.global_features, module.global_head, module.local_head):
            for parameter in head.parameters():
                parameter.normal_(0, 5)
    out2 = module.quality_only(raw, quality, canvas)
    for key in raw:
        assert out1[key].shape == raw[key].shape
        assert torch.isfinite(out1[key]).all()
        assert torch.allclose(out1[key], out2[key], atol=0, rtol=0)
        assert out1[key][1].abs().mean() < out1[key][0].abs().mean() * .45
    # The second sample is explicitly fallback_raw: its output remains close
    # to the unwarped thermal feature instead of trusting the A0 sample.
    assert out1['p4'][1].abs().mean() <= raw['p4'][1].abs().mean() * 1.05
    assert module.last_stats['mode'] == 'a0_quality_only'
    assert module.last_stats['learned_geometry'] is False
    loss = sum(value.float().mean() for value in out1.values())
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in raw.values())

    # The detector may consume an accepted correction, but its loss must not
    # update geometry-head parameters.  Explicit geometry supervision remains
    # able to train the global head.
    module = ExplicitCoarseIRInput(48, max_angle=5., max_shift=.04,
                                   max_scale=1.06, local_shift=.015).train()
    raw2 = {
        'p2': torch.rand(2, 24, 32, 56, requires_grad=True),
        'p3': torch.rand(2, 32, 16, 28, requires_grad=True),
        'p4': torch.rand(2, 48, 8, 14, requires_grad=True),
        'p5': torch.rand(2, 64, 4, 7, requires_grad=True),
    }
    guarded_quality = dict(quality)
    guarded_quality['v521_content_weight'] = torch.ones(2, 1, 1, 1)
    guarded_quality['v521_alignment_weight'] = torch.ones(2, 1, 1, 1)
    guarded_quality['v521_fallback_raw'] = torch.zeros(2, 1, 1, 1)
    rgb = torch.rand(2, 3, *canvas)
    ir = torch.rand(2, 1, *canvas)
    guarded = module(raw2, guarded_quality, canvas, rgb=rgb, ir=ir)
    sum(value.float().mean() for value in guarded.values()).backward()
    geometry_parameters = list(module.global_features.parameters()) + list(module.global_head.parameters())
    assert not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in geometry_parameters)
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in raw2.values())
    assert module.last_stats['soft_gate_mean'] >= 0
    assert 'hard_accept_fraction' in module.last_stats

    module.zero_grad(set_to_none=True)
    raw3 = {key: value.detach().clone().requires_grad_(True)
            for key, value in raw2.items()}
    module(raw3, guarded_quality, canvas, rgb=rgb, ir=ir)
    module.supervision_loss(torch.zeros(2, 4), torch.ones(2),
                            torch.ones(2)).backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and
               parameter.grad.abs().sum() > 0 for parameter in geometry_parameters)

    # A filled exterior must not leave the original hard black seam.
    image = np.full((96, 160), 120, np.float32)
    invalid = np.zeros_like(image, np.uint8)
    invalid[:, :24] = 1
    image[:, :24] = 0
    filled = _fill_ir_exterior(image, invalid)
    assert float(filled[:, :20].mean()) > 80
    seam = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    assert float(np.abs(seam[:, 22:27]).mean()) < 35

    print({'passed': True, 'stats': module.last_stats,
           'filled_exterior_mean': float(filled[:, :20].mean())})


if __name__ == '__main__':
    main()
