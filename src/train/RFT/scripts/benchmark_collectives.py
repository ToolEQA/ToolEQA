"""Bounded correctness/throughput probe on the explicitly visible CUDA devices."""
import argparse
import datetime
import json
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, rendezvous, backend):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=45))
    for size in (1, 16, 64):
        x = torch.ones(size * 1024 * 1024 // 4, device=f"cuda:{rank}")
        dist.all_reduce(x)
        assert torch.all(x == world).item()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            x.fill_(1)
            dist.all_reduce(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 5
        assert torch.all(x == world).item()
        if rank == 0:
            print(json.dumps({"backend": backend, "ranks": world, "MiB": size,
                              "seconds": elapsed, "effective_MiB_s": size / elapsed}), flush=True)
    if backend == "nccl":
        x = torch.full((262144,), rank + 1.0, device=f"cuda:{rank}")
        gathered = torch.empty(world * x.numel(), device=x.device)
        dist.all_gather_into_tensor(gathered, x)
        for other in range(world):
            assert torch.all(gathered.view(world, -1)[other] == other + 1).item()
        full = torch.full_like(gathered, rank + 1.0)
        dist.reduce_scatter_tensor(x, full)
        assert torch.all(x == world * (world + 1) / 2).item()
        x.fill_(7 if rank == 0 else 0)
        dist.broadcast(x, src=0)
        assert torch.all(x == 7).item()
        if rank == 0:
            print("Full-tensor all_reduce/all_gather/reduce_scatter/broadcast correctness: PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tooleqa-collectives-") as directory:
        mp.spawn(worker, args=(3, "file://" + directory + "/store", args.backend), nprocs=3, join=True)
