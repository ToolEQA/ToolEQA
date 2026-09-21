import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SupervisionTest(unittest.TestCase):
    def test_terminal_status_and_failure_hint(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'run'
                result = subprocess.run([
                    sys.executable, '-m', 'src.train.RFT.supervise_run',
                    '--directory', str(path), '--poll-seconds', '0.02', '--',
                    sys.executable, '-c',
                    f'print("training/global_step:5"); print("CUDA out of memory" if {code} else "ok"); raise SystemExit({code})',
                ], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, code)
                state = json.loads((path / 'status.json').read_text())
                self.assertEqual(state['status'], 'failed' if code else 'completed')
                self.assertEqual(state['progress']['completed_step'], 5)
                self.assertEqual((path / 'FAILED.json').exists(), bool(code))
                if code:
                    self.assertIn('CUDA out of memory', state['failure_hint'])


if __name__ == '__main__':
    unittest.main()
