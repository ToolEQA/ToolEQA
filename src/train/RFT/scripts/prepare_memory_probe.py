"""Create an isolated stress curriculum; never modify the formal manifest."""
import json
from pathlib import Path

source = Path('/mynvme1/ToolEQA_ICLR2027/rft8b-open-20260916/data-v2/train.jsonl')
target = Path('/mynvme1/ToolEQA_ICLR2027/memory-tests-20260916')
rows = source.read_text().splitlines()
# Trainer consumes the unshuffled curriculum; optimizer update6 uses row6.
selected = [rows[5], rows[5], rows[5]]
target.mkdir(parents=True, exist_ok=True)
with (target / 'stress-train.jsonl').open('x') as handle:
    handle.write('\n'.join(selected) + '\n')
print(json.dumps({'source_row_1based': 6, 'sample': json.loads(rows[5])['extra_info']['sample_id'],
                  'stress_only': True, 'select_longest_turn': True}))
