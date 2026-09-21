import unittest
import torch
from src.train.RFT.runtime.patch_embed_linear import eligible, install


class PatchEmbedTest(unittest.TestCase):
    def test_forward_backward_and_state_dict_are_preserved(self):
        original = torch.nn.Conv3d.forward
        try:
            for bias in (True, False):
                torch.manual_seed(7)
                conv = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4), bias=bias).double()
                x = torch.randn(5, 3, 2, 4, 4, dtype=torch.float64, requires_grad=True)
                params = [x, conv.weight] + ([conv.bias] if bias else [])
                baseline = original(conv, x)
                grads = torch.autograd.grad(baseline.square().sum(), params)
                state = {k: v.clone() for k, v in conv.state_dict().items()}
                install()
                seen = []
                hook = conv.register_forward_pre_hook(lambda *args: seen.append(True))
                candidate = conv(x)
                candidate_grads = torch.autograd.grad(candidate.square().sum(), params)
                hook.remove()
                self.assertEqual(seen, [True])
                torch.testing.assert_close(candidate, baseline, rtol=1e-12, atol=1e-12)
                for a, b in zip(grads, candidate_grads):
                    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
                for k, v in conv.state_dict().items():
                    torch.testing.assert_close(v, state[k], rtol=0, atol=0)
                torch.nn.Conv3d.forward = original
        finally:
            torch.nn.Conv3d.forward = original

    def test_non_patch_convolution_uses_original_path(self):
        conv = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4)).double()
        x = torch.randn(2, 3, 4, 8, 8, dtype=torch.float64)
        self.assertFalse(eligible(conv, x))
        baseline = conv(x)
        original = torch.nn.Conv3d.forward
        try:
            install()
            torch.testing.assert_close(conv(x), baseline, rtol=0, atol=0)
        finally:
            torch.nn.Conv3d.forward = original


if __name__ == "__main__":
    unittest.main()
