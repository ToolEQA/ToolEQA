"""Persist run liveness and terminal failures; no external notification channel.

The child has its own process group. Status includes timestamps and exit code;
failure writes FAILED.json and does not trigger an automatic restart.
"""
import argparse
import datetime
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import time


def write_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--progress-root', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing command')
    if args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    args.directory.mkdir(parents=True, exist_ok=True)
    # Exclusive log prevents accidental duplicate launch into the same run.
    with (args.directory / 'supervisor.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state = {'command': command, 'pid': child.pid, 'started_at': datetime.datetime.now().astimezone().isoformat()}

        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            code = child.poll()
            state.update(updated_at=datetime.datetime.now().astimezone().isoformat(),
                         status='running' if code is None else ('completed' if code == 0 else 'failed'),
                         exit_code=code)
            log_paths = [args.directory / 'supervisor.log']
            if args.progress_root:
                log_paths += [args.progress_root / 'stage1.log', args.progress_root / 'stage2.log']
            for path in log_paths:
                if not path.exists():
                    continue
                with path.open('rb') as handle:
                    handle.seek(max(0, path.stat().st_size - 262144))
                    tail = handle.read().decode(errors='replace')
                steps = re.findall(r'training/global_step:(\d+)', tail)
                if steps:
                    state['progress'] = {'log': str(path), 'completed_step': int(steps[-1])}
                if 'CUDA out of memory' in tail:
                    state['failure_hint'] = f'CUDA out of memory: {path}'
            write_status(args.directory / 'status.json', state)
            if code is not None:
                if code != 0:
                    write_status(args.directory / 'FAILED.json', state)
                    print(f'RUN FAILED: {args.directory}, exit={code}', flush=True)
                return code
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
