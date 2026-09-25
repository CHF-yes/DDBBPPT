import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mm_yolo'))

from v52_stage_a import resample
from v521_stage_a import ExplicitCoarseIRInput, geometry_acceptance, residual_sampling


class V521GateFixTest(unittest.TestCase):
    def _edge_pattern(self):
        value = torch.zeros(1, 1, 24, 32)
        for row in range(3):
            for col in range(4):
                y = row * 8 + 2 + col % 3
                x = col * 8 + 2 + row % 3
                value[:, :, y:y + 3, x] = 1
                value[:, :, y, x:x + 4] = 1
        return value

    def test_multi_region_candidate_accepts_only_distributed_improvement(self):
        rgb = self._edge_pattern()
        coarse = torch.roll(rgb, shifts=2, dims=-1)
        weight = torch.ones_like(rgb)
        good = geometry_acceptance(rgb, coarse, coarse, rgb, weight)
        self.assertTrue(bool(good['geometry_ok'][0]))
        self.assertGreaterEqual(int(good['improved_regions'][0]), 4)

        one_tile = coarse.clone()
        one_tile[:, :, :8, :8] = rgb[:, :, :8, :8]
        weak = geometry_acceptance(rgb, coarse, coarse, one_tile, weight)
        self.assertFalse(bool(weak['geometry_ok'][0]))
        self.assertLess(int(weak['improved_regions'][0]), 4)

    def test_worse_candidate_is_rejected(self):
        rgb = self._edge_pattern()
        coarse = torch.roll(rgb, shifts=1, dims=-1)
        worse = torch.roll(rgb, shifts=5, dims=-1)
        result = geometry_acceptance(rgb, coarse, coarse, worse, torch.ones_like(rgb))
        self.assertFalse(bool(result['geometry_ok'][0]))
        self.assertGreater(float(result['candidate_score'][0]),
                           float(result['coarse_score'][0]))

    def test_training_gate_warps_once_in_parameter_space(self):
        torch.manual_seed(3)
        aligner = ExplicitCoarseIRInput(4)
        aligner.train()
        with torch.no_grad():
            for parameter in aligner.global_features.parameters():
                parameter.zero_()
            for parameter in aligner.global_head.parameters():
                parameter.zero_()
            aligner.global_head[-1].bias[:] = torch.tensor([.6, -.4, .3, .2, 0.])
            for parameter in aligner.local_head.parameters():
                parameter.zero_()
            aligner.local_head[-1].bias[2] = -20.

        feature = torch.zeros(1, 4, 16, 16)
        feature[:, :, 5, 7] = 1
        raw = {'p4': feature}
        identity = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        quality = {
            'v52_ir_sampling': identity,
            'availability': torch.ones(1, 3, 64, 64),
            'ir': torch.zeros(1, 10, 64, 64),
        }
        quality['ir'][:, 8:10] = 1
        rgb = torch.zeros(1, 3, 64, 64)
        ir = torch.zeros(1, 3, 64, 64)
        output = aligner(raw, quality, (64, 64), rgb=rgb, ir=ir)['p4']

        confidence = torch.tensor([[.5]])
        normalized = torch.tensor([[.6, -.4, .3, .2]]).tanh() * confidence
        residual, _ = residual_sampling(
            normalized, (64, 64), aligner.max_angle, aligner.max_shift,
            aligner.max_log_scale)
        expected = resample(feature, residual[:, :2], (64, 64))
        self.assertTrue(torch.allclose(output, expected, atol=1e-5, rtol=1e-5))
        self.assertFalse(aligner.last_stats['pixel_blend_enabled'])


if __name__ == '__main__':
    unittest.main()
