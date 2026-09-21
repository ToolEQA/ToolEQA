"""Bounded in-process Python stack sampling via an existing Ray actor.

No ptrace/security setting changes; records code locations, never frame locals.
The callback is queued between actor methods and does not alter model state.
"""
import argparse
import json
from pathlib import Path


def start_sampling(self, output, seconds):
    import json
    import sys
    import threading
    import time
    import torch

    group = torch.distributed.distributed_c10d._get_default_group()
    metadata = {
        "backend": torch.distributed.get_backend(),
        "cuda_backend": type(group._get_backend(torch.device("cuda", torch.cuda.current_device()))).__name__,
        "threads": torch.get_num_threads(),
        "device": torch.cuda.current_device(),
        "fast_patch_embed": bool(getattr(torch.nn.Conv3d.forward, "_tooleqa_patch_linear", False)),
    }

    def sample():
        deadline = time.monotonic() + seconds
        with open(output, "x") as handle:
            handle.write(json.dumps({"metadata": metadata}) + "\n")
            while time.monotonic() < deadline:
                stacks = []
                for ident, frame in sys._current_frames().items():
                    if ident == threading.get_ident():
                        continue
                    stack = []
                    while frame and len(stack) < 50:
                        stack.append([frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno])
                        frame = frame.f_back
                    stacks.append(stack)
                handle.write(json.dumps({"time": time.time(), "stacks": stacks}) + "\n")
                handle.flush()
                time.sleep(2)
    thread = threading.Thread(target=sample, daemon=True, name="tooleqa-speed-stack-sampler")
    thread.start()
    return metadata


if __name__ == "__main__":
    import ray
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    ray.init(address="auto", log_to_driver=False)
    actor = ray.get_actor(args.actor, namespace=args.namespace)
    print(json.dumps(ray.get(actor.__ray_call__.remote(fn=start_sampling, output=args.output, seconds=args.seconds), timeout=1200)), flush=True)
    ray.shutdown()
