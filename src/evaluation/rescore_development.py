"""Re-score the exact saved development answers without rerunning navigation.

Resolve padded rollout duplicates using the unique initial-image path recorded
in each retained validation row, never by selecting an arbitrary trajectory.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import urllib.request

from src.evaluation.open_protocol import semantic_judgment
from src.evaluation.openeqa_protocol import JUDGE_PROTOCOL_ID
from src.evaluation.paper_metrics import compute_paper_metrics
from src.evaluation.resume_official import hardware_errors


def resolve_rows(stage, step, manifest):
    table_path = stage / 'validation' / f'{step}.jsonl'
    table = [json.loads(s) for s in table_path.read_text().splitlines() if s.strip()]
    expected = {r['extra_info']['sample_id'] for r in
                (json.loads(s) for s in manifest.read_text().splitlines() if s.strip())}
    trajectories = [(p, json.loads(p.read_text())) for p in
                    (stage / 'trajectories' / f'step_{step}').glob('*.json')]
    trajectories = [(p, r) for p, r in trajectories if r.get('validate')]
    resolved, seen = [], set()
    for row in table:
        matches = [(p, r) for p, r in trajectories
                   if r.get('initial_image') and r['initial_image'] in row['input']]
        if len(matches) != 1:
            raise ValueError(f'Expected one retained trace, found {len(matches)}')
        path, trace = matches[0]
        sample = trace['sample']
        sid = sample['sample_id']
        if sid in seen or sid not in expected or sample['answer'] != row['gts']:
            raise ValueError(f'Identity/reference mismatch: {path}')
        if hardware_errors(trace['tool_trace']):
            raise ValueError(f'Hardware failure in selected trace: {path}')
        old = trace['reward']
        if not math.isclose(old['answer_quality'], row['llm_match'], abs_tol=1e-10):
            raise ValueError(f'Original score mismatch: {path}')
        metrics = compute_paper_metrics(sample, trace['tool_trace'], correct=old['correct'],
                                        answer_quality=old['answer_quality'])
        for key in ('trajectory_length', 'recall_at_5', 'recall_at_10', 'recall_at_15',
                    'epath_at_5', 'epath_at_10', 'epath_at_15'):
            if not math.isclose(metrics[key], row[key], abs_tol=1e-9):
                raise ValueError(f'Original metric mismatch: {path} {key}')
        seen.add(sid)
        resolved.append((row, path, trace))
    if seen != expected or len(table) != len(expected):
        raise ValueError('Development split is incomplete or duplicated')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--stage', default='stage2-joint', choices=['stage1-evidence', 'stage2-joint'])
    parser.add_argument('--step', type=int, default=450)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    stage = args.run_root / args.stage
    checkpoint = stage / 'checkpoints' / f'global_step_{args.step}'
    if not (checkpoint / 'actor/fsdp_config.json').is_file():
        raise FileNotFoundError(checkpoint)
    resolved = resolve_rows(stage, args.step, args.run_root / 'data-v2/dev.jsonl')
    endpoint = os.environ.get('TOOLEQA_FROZEN_JUDGE_SERVICE', 'http://127.0.0.1:18942')
    with urllib.request.urlopen(endpoint, timeout=30) as response:
        identity = json.load(response)
    if identity['protocol_id'] != JUDGE_PROTOCOL_ID:
        raise ValueError('Judge service protocol mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {'checkpoint': str(checkpoint), 'judge': identity, 'count': len(resolved),
                'method': 'same retained development trajectories; answer re-scoring only',
                'source_table_sha256': hashlib.sha256((stage / 'validation' / f'{args.step}.jsonl').read_bytes()).hexdigest(),
                'warning': 'Original 225-task dev retained; known 3 Seen-overlap questions not removed'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    results = []
    with (args.output_dir / 'scores.jsonl').open('x') as handle:
        for row, path, trace in resolved:
            sample = trace['sample']
            judgment = semantic_judgment(sample['question'], sample['answer'], trace['final_answer'], sample.get('extra_answers'))
            metrics = compute_paper_metrics(sample, trace['tool_trace'], correct=judgment['score'] == 5,
                                            answer_quality=judgment['answer_quality'])
            result = {'sample_id': sample['sample_id'], 'trajectory_path': str(path),
                      'trajectory_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'question': sample['question'], 'reference': sample['answer'],
                      'final_answer': trace['final_answer'], 'semantic_judgment': judgment,
                      'semantic_score': judgment['score'], 'llm_match': judgment['answer_quality'],
                      'acc': float(judgment['score'] == 5), 'old_llm_match': row['llm_match'], **metrics,
                      **{k: row[k] for k in ['evidence_coverage', 'evidence_complete', 'tool_call_accuracy', 'forced_final']}}
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(resolved):
                print(f'Progress {len(results)}/{len(resolved)}', flush=True)
    keys = ['llm_match', 'acc', 'recall_at_5', 'recall_at_10', 'recall_at_15',
            'epath_at_5', 'epath_at_10', 'epath_at_15', 'trajectory_length', 'trajectory_steps',
            'evidence_coverage', 'evidence_complete', 'tool_call_accuracy', 'forced_final', 'old_llm_match']
    summary = {**metadata, 'metrics': {k: fmean(r[k] for r in results) for k in keys},
               'score_distribution': dict(sorted(Counter(r['semantic_score'] for r in results).items()))}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
