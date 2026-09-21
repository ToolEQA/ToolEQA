"""Compare Qwen's single-patch Conv3d with exactly equivalent flattened GEMM."""
import json
import time

import torch
from torch.nn import functional as F


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


torch.manual_seed(42)
for dtype, batch in ((torch.float64, 8), (torch.bfloat16, 128), (torch.bfloat16, 2048)):
    conv = torch.nn.Conv3d(3, 1152, (2, 16, 16), stride=(2, 16, 16), bias=True).to(device="cuda", dtype=dtype)
    x = torch.randn(batch, 3, 2, 16, 16, device="cuda", dtype=dtype, requires_grad=True)
    reference, conv_seconds = timed(lambda: conv(x))
    optimized, linear_seconds = timed(lambda: F.linear(x.flatten(1), conv.weight.flatten(1), conv.bias)[:, :, None, None, None])
    tol = 1e-9 if dtype == torch.float64 else .016
    torch.testing.assert_close(reference, optimized, rtol=tol, atol=tol)
    cotangent = torch.randn_like(reference)
    grad_original, grad_conv_seconds = timed(lambda: torch.autograd.grad(reference, (x, conv.weight, conv.bias), cotangent))
    grad_fast, grad_linear_seconds = timed(lambda: torch.autograd.grad(optimized, (x, conv.weight, conv.bias), cotangent))
    errors = []
    for a, b in zip(grad_original, grad_fast):
        # BF16 reductions differ in rounding; compare normalized RMS as well as FP64 exact tolerance.
        relative = ((a.double() - b.double()).square().mean().sqrt() / a.double().square().mean().sqrt().clamp_min(1e-12)).item()
        if dtype == torch.float64:
            assert relative < 1e-9, relative
        errors.append(relative)
    oracle_errors = []
    if dtype == torch.bfloat16:
        xd = x.detach().double().requires_grad_()
        wd = conv.weight.detach().double().requires_grad_()
        bd = conv.bias.detach().double().requires_grad_()
        oracle = F.linear(xd.flatten(1), wd.flatten(1), bd)[:, :, None, None, None]
        oracle_grads = torch.autograd.grad(oracle, (xd, wd, bd), cotangent.double())
        for slow, fast, gold in zip(grad_original, grad_fast, oracle_grads):
            scale = gold.square().mean().sqrt().clamp_min(1e-12)
            slow_error = ((slow.double()-gold).square().mean().sqrt()/scale).item()
            fast_error = ((fast.double()-gold).square().mean().sqrt()/scale).item()
            assert fast_error < .005, fast_error
            assert fast_error <= max(.003, slow_error * 1.1), (slow_error, fast_error)
            oracle_errors.append({"conv": slow_error, "linear": fast_error})
    print(json.dumps({"dtype": str(dtype), "patches": batch, "conv_forward_s": conv_seconds,
                      "linear_forward_s": linear_seconds, "conv_backward_s": grad_conv_seconds,
                      "linear_backward_s": grad_linear_seconds, "max_forward_difference": (reference-optimized).abs().max().item(),
                      "gradient_relative_rms_errors": errors, "gradient_errors_vs_fp64": oracle_errors}), flush=True)
