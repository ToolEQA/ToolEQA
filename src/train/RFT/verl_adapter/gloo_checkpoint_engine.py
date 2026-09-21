"""Gloo checkpoint transport for disaggregated verl rollouts.

This compatibility backend is used when NCCL cannot initialize, for example
when NVML reports a broken physical GPU. Weights travel through CPU buffers,
so this is deliberately a correctness fallback rather than a fast path.
"""

from __future__ import annotations

import pickle
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncGenerator, Generator

import ray
import torch
import torch.distributed as dist

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.net_utils import get_free_port


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int


@CheckpointEngineRegistry.register("gloo")
class GlooCheckpointEngine(CheckpointEngine):
    """Broadcast full model weights between disaggregated workers over CPU."""

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "tooleqa_gloo_checkpoint",
        rebuild_group: bool = True,
        is_master: bool = False,
        **_: object,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.is_master = is_master
        self.rank = None
        self.world_size = None
        if is_master:
            self.ip = ray.util.get_node_ip_address().strip("[]")
            self.listen_port, _ = get_free_port(self.ip)

    def prepare(self):
        self.send_buf = torch.empty(self.bucket_size, dtype=torch.uint8, device="cpu")
        self.recv_buf = torch.empty(self.bucket_size, dtype=torch.uint8, device="cpu")
        return MasterMetadata(self.ip, self.listen_port) if self.is_master else None

    def finalize(self):
        self.process_group = None
        self.store = None
        self.rank = None
        self.world_size = None
        self.send_buf = None
        self.recv_buf = None

    @classmethod
    def build_topology(cls, trainer_world_size, rollout_world_size, metadata):
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "master_metadata": [metadata[0]] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank, world_size, master_metadata):
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return
        self.rank = rank
        self.world_size = world_size
        timeout = timedelta(minutes=30)
        self.store = dist.TCPStore(
            master_metadata.zmq_ip,
            master_metadata.zmq_port,
            world_size,
            rank == 0,
            timeout,
        )
        # Construct the process group directly. Unlike dist.new_group(), this is
        # independent of verl's pre-existing per-worker default process group.
        self.process_group = dist.ProcessGroupGloo(self.store, rank, world_size, timeout)

    def _broadcast(self, bucket):
        options = dist.BroadcastOptions()
        options.rootRank = 0
        options.rootTensor = 0
        self.process_group.broadcast([bucket], options).wait()

    def _send_bucket(self, index, bucket, metadata):
        self.store.set(f"metadata_{index}", pickle.dumps(metadata))
        self._broadcast(bucket)

    def _receive_bucket(self, index, bucket):
        metadata = pickle.loads(self.store.get(f"metadata_{index}"))
        self._broadcast(bucket)
        return metadata

    @staticmethod
    def _edge_probe(byte_tensor: torch.Tensor) -> tuple[int, int, int]:
        flat = byte_tensor.reshape(-1)
        width = min(256, flat.numel())
        middle = max(0, (flat.numel() - width) // 2)
        return (
            int(flat[:width].to(torch.int64).sum().item()),
            int(flat[middle : middle + width].to(torch.int64).sum().item()),
            int(flat[-width:].to(torch.int64).sum().item()),
        )

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        assert self.rank is not None and self.rank <= 0
        if self.rank < 0:
            for _name, _weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        bucket_index = 0
        debug_count = 0
        for name, weight in weights:
            weight_bytes = weight.nbytes
            if offset + weight_bytes > self.bucket_size:
                self._send_bucket(
                    bucket_index, send_buf, {"bucket_meta": bucket_meta, "is_last": False}
                )
                bucket_index += 1
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0
            if weight_bytes > self.bucket_size:
                raise ValueError(
                    f"Weight {name} ({weight_bytes} bytes) exceeds checkpoint bucket "
                    f"({self.bucket_size} bytes)"
                )
            cpu_bytes = weight.detach().contiguous().view(torch.uint8).view(-1).cpu()
            probe = self._edge_probe(cpu_bytes)
            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
                "probe": probe,
            }
            if os.environ.get("TOOLEQA_DEBUG_WEIGHT_SYNC") == "1" and debug_count < 8:
                print(f"[Gloo send] {name} shape={tuple(weight.shape)} dtype={weight.dtype} probe={probe}", flush=True)
                debug_count += 1
            send_buf[offset : offset + weight_bytes].copy_(cpu_bytes)
            offset += weight_bytes

        self._send_bucket(bucket_index, send_buf, {"bucket_meta": bucket_meta, "is_last": True})

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        assert self.rank is not None and self.rank > 0
        send_buf, recv_buf = self.send_buf, self.recv_buf
        bucket_index = 0
        metadata = self._receive_bucket(bucket_index, recv_buf)
        send_buf, recv_buf = recv_buf, send_buf

        while not metadata["is_last"]:
            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor_bytes = send_buf[meta["offset"] : meta["offset"] + size]
                probe = self._edge_probe(tensor_bytes)
                if probe != tuple(meta["probe"]):
                    raise RuntimeError(f"Gloo weight corruption for {name}: expected {meta['probe']}, got {probe}")
                tensor = tensor_bytes.view(dtype).view(shape)
                yield name, tensor
            bucket_index += 1
            metadata = self._receive_bucket(bucket_index, recv_buf)
            send_buf, recv_buf = recv_buf, send_buf

        for name, meta in metadata["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor_bytes = send_buf[meta["offset"] : meta["offset"] + size]
            probe = self._edge_probe(tensor_bytes)
            if probe != tuple(meta["probe"]):
                raise RuntimeError(f"Gloo weight corruption for {name}: expected {meta['probe']}, got {probe}")
            tensor = tensor_bytes.view(dtype).view(shape)
            yield name, tensor
