"""Compatibility and non-mutating evaluation-entry regression tests."""
import contextlib
import fcntl
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation import run_open_eval


class MigrationTest(unittest.TestCase):
    def test_file_based_reward_loader_keeps_callable(self):
        path = run_open_eval.REPO_ROOT / 'src/train/RFT/official_eval.py'
        spec = importlib.util.spec_from_file_location('_migration_legacy_reward', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from src.evaluation.official_eval import compute_score
        self.assertIs(module.compute_score, compute_score)

    def test_old_modules_are_exact_aliases(self):
        for name in ('paper_metrics', 'open_protocol', 'frozen_service', 'official_eval',
                     'summarize_rollouts', 'resume_official', 'select_open_checkpoint'):
            self.assertIs(importlib.import_module('src.train.RFT.' + name),
                          importlib.import_module('src.evaluation.' + name))

    def test_preview_never_launches_or_creates_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'not-created'
            output = io.StringIO()
            with patch('sys.argv', ['run_open_eval', '--run-root', str(root)]), \
                    patch.object(run_open_eval, 'execute') as execute, contextlib.redirect_stdout(output):
                run_open_eval.main()
            execute.assert_not_called()
            self.assertFalse(root.exists())
            self.assertFalse(json.loads(output.getvalue())['execute'])

    def test_active_training_prevents_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'pipeline.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('src.evaluation.select_open_checkpoint.select') as select, \
                        patch('subprocess.run') as run:
                    with self.assertRaisesRegex(RuntimeError, 'still active'):
                        run_open_eval.execute(root)
                    select.assert_not_called()
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
