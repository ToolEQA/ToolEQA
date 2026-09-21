"""Process-local compatibility hooks loaded automatically by Python.

This host exposes stale/broken NVML indices in addition to the usable CUDA
devices. vLLM logs every NVML device at import time and aborts on those broken
indices even when CUDA_VISIBLE_DEVICES excludes them. Limit only vLLM's bundled
NVML device count to the number of devices intentionally exposed to the RFT
process. These hooks are opt-in and do not modify the driver or system NVML.
TOOLEQA_FAST_PATCH_EMBED separately enables a process-local Conv3d fast path.
"""

from __future__ import annotations

import os


if os.environ.get("TOOLEQA_FAST_PATCH_EMBED") == "1":
    # Fail closed if this explicitly requested optimization cannot be installed.
    from patch_embed_linear import install
    install()


if os.environ.get("TOOLEQA_LIMIT_NVML_TO_VISIBLE") == "1":
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible:
        try:
            from vllm.third_party import pynvml

            _original_device_count = pynvml.nvmlDeviceGetCount
            _original_get_handle = pynvml.nvmlDeviceGetHandleByIndex

            def _visible_device_count() -> int:
                return min(int(_original_device_count()), len(visible))

            pynvml.nvmlDeviceGetCount = _visible_device_count

            raw_index_map = os.environ.get("TOOLEQA_NVML_INDEX_MAP", "")
            index_map = {}
            for pair in raw_index_map.split(","):
                if not pair.strip():
                    continue
                cuda_index, nvml_index = pair.split(":", 1)
                index_map[int(cuda_index)] = int(nvml_index)

            if index_map:
                def _mapped_device_handle(index: int):
                    return _original_get_handle(index_map.get(int(index), int(index)))

                pynvml.nvmlDeviceGetHandleByIndex = _mapped_device_handle
        except Exception:
            pass


distributed_backend = os.environ.get("TOOLEQA_DISTRIBUTED_BACKEND", "").strip().lower()
if distributed_backend:
    try:
        import verl.utils.device as verl_device

        verl_device.get_nccl_backend = lambda: distributed_backend
    except Exception:
        pass
