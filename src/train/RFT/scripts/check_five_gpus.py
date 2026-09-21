"""Check actual CUDA allocation on the five surviving physical GPUs."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((
    "GPU-7808853a-db8b-a474-6fd7-5f44d0fcf812",
    "GPU-fedce279-6a20-845c-efa1-0c2822e9857c",
    "GPU-fbe6bf75-8a8f-e62b-56f5-9797153df810",
    "GPU-2f5ec3f0-1cb7-6c4f-34d2-93784c0fee86",
    "GPU-dd87e394-2826-1441-19c5-e01befc9c8e3",
))

import torch

count = torch.cuda.device_count()
if count != 5:
    raise SystemExit(f"Expected five working CUDA GPUs, found {count}; evaluation remains stopped")
for index in range(5):
    value = torch.ones(16, device=f"cuda:{index}")
    value = value + value
    torch.cuda.synchronize(index)
    assert value.sum().item() == 32
    print(f"CUDA {index}: allocation and compute passed", flush=True)
