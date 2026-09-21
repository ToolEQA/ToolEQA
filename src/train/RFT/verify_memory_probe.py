"""Read-only gate for the isolated long-trajectory memory probe."""
import argparse
import json
from pathlib import Path
import re


def inspect(directory, run, worker_logs=None):
    state = json.loads((directory / 'status.json').read_text())
    log = (directory / 'supervisor.log').read_text(errors='replace')
    # Ray driver output may deduplicate similar per-rank lines. Prefer archived
    # raw worker logs when supplied; never count both copies of the same event.
    metric_log = log if worker_logs is None else '\n'.join(
        p.read_text(errors='replace') for p in worker_logs.glob('worker-*.out'))
    metrics = [json.loads(m.group(1)) for m in re.finditer(r'TOOLEQA_MEMORY (\{[^\n]+?\})', metric_log)]
    backward = [m for m in metrics if not m['forward_only']]
    steps = [int(s) for s in re.findall(r'training/global_step:(\d+)', log)]
    missing = []
    for step in (1, 3):
        actor = run / 'checkpoints' / f'global_step_{step}' / 'actor'
        for rank in range(3):
            for prefix in ('model', 'optim', 'extra_state'):
                path = actor / f'{prefix}_world_size_3_rank_{rank}.pt'
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path))
        if not (actor / 'fsdp_config.json').is_file():
            missing.append(str(actor / 'fsdp_config.json'))
    by_rank = {}
    for rank in range(3):
        entries = [m for m in backward if m['rank'] == rank]
        by_rank[rank] = {
            'observed_backward_calls': len(entries),
            'max_tokens': max((m['tokens'] for m in entries), default=0),
            'peak_allocated_gib': max((m['peak_allocated_gib'] for m in entries), default=0),
            'peak_reserved_gib': max((m['peak_reserved_gib'] for m in entries), default=0),
        }
    passed = (state['status'] == 'completed' and steps == [1, 2, 3] and not missing
              and backward and all(m['success'] and m['micro_batch_size'] == 1 for m in metrics)
              and all(v['max_tokens'] >= 10000 and v['observed_backward_calls'] >= 6 for v in by_rank.values()))
    return {'passed': bool(passed), 'status': state['status'], 'steps': steps,
            'backward_by_rank': by_rank, 'missing_checkpoint_files': missing,
            'scope': 'Observed real long trajectories only; not proof of all possible inputs or checkpoint reload.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--worker-logs', type=Path)
    args = parser.parse_args()
    result = inspect(args.directory, args.run, args.worker_logs)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
