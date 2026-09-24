import sys
import unittest
from pathlib import Path

import torch

MM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MM))

from independent_fusion import V511TrustedIRFusion


class V511FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        torch.manual_seed(17)
        raw = [torch.randn(2, 32, 8, 12) for _ in range(3)]
        common = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        aligned = [common[0], torch.randn(2, 16, 8, 12), common[2]]
        private = [torch.randn(2, 16, 8, 12) for _ in range(3)]
        valid = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        match = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        reliable = [torch.ones(2, 1, 8, 12) for _ in range(3)]
        memory = torch.randn(2, 4, 3, 24)
        quality = [torch.ones(2, 3, 8, 12), torch.zeros(2, 10, 8, 12),
                   torch.ones(2, 3, 8, 12)]
        quality[1][:, 8:10] = 1
        return raw, common, aligned, private, valid, match, reliable, memory, quality

    def test_zero_initialized_plugin_is_anchor_identity(self):
        module = V511TrustedIRFusion(32, dim=16, memory_dim=24).train()
        values = self.inputs()
        anchor = torch.randn(2, 32, 8, 12)
        fused, shared, depth, state = module(*values[:-1], quality=values[-1], anchor=anchor)
        self.assertTrue(torch.equal(fused, anchor))
        self.assertEqual(float(shared.abs().max()), 0.)
        self.assertEqual(float(depth.abs().max()), 0.)
        self.assertEqual(tuple(state.shape), (2, 6, 16))
        (fused.mean() + shared.mean() + depth.mean()).backward()
        for block in (module.shared, module.private, module.depth_support):
            self.assertIsNotNone(block[-1].weight.grad)

    def test_private_ir_cannot_change_localization_support(self):
        module = V511TrustedIRFusion(32, dim=16, memory_dim=24).eval()
        with torch.no_grad():
            module.private[-1].weight.normal_(0, .02)
        values = list(self.inputs())
        anchor = torch.randn(2, 32, 8, 12)
        first = module(*values[:-1], quality=values[-1], anchor=anchor)
        values[3] = list(values[3])
        values[3][1] = values[3][1] + 20
        second = module(*values[:-1], quality=values[-1], anchor=anchor)
        self.assertFalse(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_cross_scale_state_advances(self):
        module = V511TrustedIRFusion(32, dim=16, memory_dim=24).eval()
        values = self.inputs()
        anchor = torch.randn(2, 32, 8, 12)
        first = module(*values[:-1], quality=values[-1], anchor=anchor)
        second = module(*values[:-1], quality=values[-1],
                        cross_scale_state=first[3], anchor=anchor)
        self.assertFalse(torch.equal(first[3], second[3]))


if __name__ == "__main__":
    unittest.main()
