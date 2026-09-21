"""Opt-in algebraically equivalent fast path for single-patch Conv3d.

Preserves module calls/FSDP hooks and the original parameter names/shapes.
Only a convolution producing a single output voxel per sample is eligible.
"""
import torch.nn.functional as F
from torch import nn


def eligible(module, value):
    return (value.ndim == 5 and tuple(value.shape[-3:]) == tuple(module.kernel_size)
            and tuple(module.stride) == tuple(module.kernel_size)
            and module.groups == 1 and tuple(module.dilation) == (1, 1, 1)
            and tuple(module.padding) == (0, 0, 0) and module.padding_mode == "zeros")


def install():
    if getattr(nn.Conv3d.forward, "_tooleqa_patch_linear", False):
        return
    original = nn.Conv3d.forward

    def forward(self, value):
        if eligible(self, value):
            output = F.linear(value.flatten(1), self.weight.flatten(1), self.bias)
            return output[:, :, None, None, None]
        return original(self, value)

    forward._tooleqa_patch_linear = True
    nn.Conv3d.forward = forward
