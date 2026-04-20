import os
import torch
import torch.distributed as dist

def main():
    # 初始化进程组
    dist.init_process_group(backend="nccl")

    # 每个进程都会有 LOCAL_RANK 环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 获取全局 rank
    rank = dist.get_rank()

    # 测试 all_reduce
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t)
    print(f"[Rank {rank}] local_rank={local_rank}, tensor={t.item()}")

if __name__ == "__main__":
    main()