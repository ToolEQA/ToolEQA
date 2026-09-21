import json
from pathlib import Path
import tempfile
import unittest

from src.train.RFT.verify_memory_probe import inspect


class MemoryProbeGateTest(unittest.TestCase):
    def test_requires_long_backward_all_ranks_and_complete_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            monitor, run = root / 'monitor', root / 'run'
            monitor.mkdir()
            (monitor / 'status.json').write_text(json.dumps({'status': 'completed'}))
            rows = ['training/global_step:1', 'training/global_step:2', 'training/global_step:3']
            for rank in range(3):
                for _ in range(6):
                    rows.append('TOOLEQA_MEMORY ' + json.dumps({
                        'rank': rank, 'forward_only': False, 'success': True,
                        'micro_batch_size': 1, 'tokens': 12000,
                        'peak_allocated_gib': 33, 'peak_reserved_gib': 35,
                    }))
            log = monitor / 'supervisor.log'
            log.write_text('\n'.join(rows))
            self.assertFalse(inspect(monitor, run)['passed'])
            for step in (1, 3):
                actor = run / 'checkpoints' / f'global_step_{step}' / 'actor'
                actor.mkdir(parents=True)
                (actor / 'fsdp_config.json').write_text('{}')
                for rank in range(3):
                    for prefix in ('model', 'optim', 'extra_state'):
                        (actor / f'{prefix}_world_size_3_rank_{rank}.pt').write_text('fixture')
            self.assertTrue(inspect(monitor, run)['passed'])
            log.write_text('\n'.join(rows).replace('12000', '2000'))
            self.assertFalse(inspect(monitor, run)['passed'])


if __name__ == '__main__':
    unittest.main()
