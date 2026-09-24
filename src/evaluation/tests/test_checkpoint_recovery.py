import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.evaluate_checkpoint import completed_records


class CheckpointRecoveryTest(unittest.TestCase):
    def test_only_completed_unique_nonhardware_results_are_retained(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "seen/trajectories/step_0"
            traces.mkdir(parents=True)
            samples = [{"sample_id": str(i), "question": "q", "answer": "a", "scene": "s"}
                       for i in range(3)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps({"extra_info": s}) + "\n" for s in samples))
            reward = {"semantic_score": 2, "evidence_coverage": 0.2,
                      "evidence_complete": 0.0, "trajectory_length": 1.0}
            for i, sample in enumerate(samples):
                record = {"sample": sample, "validate": True, "initial_image": f"img-{i}",
                          "tool_trace": [], "reward": reward if i != 1 else {}}
                if i == 2:
                    record["tool_trace"] = [{"error": "CUDA out of memory"}]
                (traces / f"{i}.json").write_text(json.dumps(record))
            result = completed_records(root, "seen", manifest)
            self.assertEqual(set(result), {"0"})
            (traces / "duplicate.json").write_text((traces / "0.json").read_text())
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                completed_records(root, "seen", manifest)


if __name__ == "__main__":
    unittest.main()
