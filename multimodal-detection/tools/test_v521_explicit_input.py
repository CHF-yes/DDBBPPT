"""CPU contract checks for the V5.2.1 explicit A0 geometry input."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'mm_yolo'):
    sys.path.insert(0, str(path))

import torch

from v521_stage_a import ExplicitCoarseIRInput, residual_sampling


def main():
    torch.manual_seed(42)
    canvas = (128, 224)
    raw = {
        'p2': torch.rand(2, 24, 32, 56),
        'p3': torch.rand(2, 32, 16, 28),
        'p4': torch.rand(2, 48, 8, 14),
        'p5': torch.rand(2, 64, 4, 7),
    }
    sampling = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]).repeat(2, 1, 1)
    availability = torch.ones(2, 3, *canvas)
    availability[:, 1, :, :20] = 0
    quality_ir = torch.zeros(2, 10, 32, 56)
    quality_ir[:, 8] = 1
    quality_ir[:, 9] = .8
    quality = {'v52_ir_sampling': sampling, 'availability': availability,
               'ir': quality_ir,
               'v521_ghost_probability': torch.full((2, 1, 32, 56), .15),
               'v521_ghost_confidence': torch.full((2, 1, 32, 56), .4),
               'v521_coarse_available': torch.ones(2, 1, 32, 56)}
    rgb = torch.rand(2, 3, *canvas)
    ir = torch.rand(2, 1, *canvas)
    module = ExplicitCoarseIRInput(48)
    out = module(raw, quality, canvas, rgb=rgb, ir=ir)
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        k: tuple(v.shape) for k, v in raw.items()}
    assert torch.isfinite(sum(v.float().mean() for v in out.values()))
    assert module.last_global_physical[:, 0].abs().max() == 0
    assert torch.allclose(module.last_global_physical[:, 3], torch.ones(2))
    assert module.last_stats['global_confidence'] < .11
    assert out['p2'][:, :, :, :2].abs().max() < 1e-5
    loss = module.supervision_loss(torch.zeros(2, 4), torch.ones(2), torch.ones(2))
    assert torch.isfinite(loss) and loss > 0
    matrix, physical = residual_sampling(
        torch.tensor([[1., 1., -1., 1.], [-1., -1., 1., -1.]]), canvas)
    assert matrix.shape == (2, 3, 3)
    assert torch.allclose(physical[:, 0], torch.tensor([15., -15.]), atol=1e-4)
    assert torch.allclose(physical[:, 1].abs(), torch.tensor([44.8, 44.8]), atol=1e-4)
    assert torch.allclose(physical[:, 2].abs(), torch.tensor([25.6, 25.6]), atol=1e-4)
    assert torch.allclose(physical[:, 3], torch.tensor([1.25, .8]), atol=1e-4)
    print({'passed': True, 'stats': module.last_stats,
           'large_range_physical': physical.tolist()})


if __name__ == '__main__':
    main()
