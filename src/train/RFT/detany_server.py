"""Launch one DetAny3D shared-memory worker on a reserved physical GPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def cleanup_channel(channel: int) -> None:
    import posix_ipc

    for name in (f"image_data_{channel}", f"result_data_{channel}"):
        try:
            posix_ipc.unlink_shared_memory(name)
        except posix_ipc.ExistentialError:
            pass
        for suffix in ("ready", "done"):
            try:
                posix_ipc.unlink_semaphore(f"/{name}_{suffix}")
            except posix_ipc.ExistentialError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--keep-stale-ipc", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    detany_root = root / "third_party/DetAny3D"
    if not detany_root.is_dir():
        raise FileNotFoundError(f"DetAny3D submodule is missing: {detany_root}")
    os.chdir(detany_root)
    sys.path.insert(0, str(detany_root))
    if not args.keep_stale_ipc:
        cleanup_channel(args.channel)

    from app_mp import worker_process

    worker_process(args.channel)


if __name__ == "__main__":
    main()
