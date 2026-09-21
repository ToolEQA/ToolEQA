import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.train.RFT.resume_official import prepare, consolidate, hardware_errors
from src.train.RFT.official_eval import compute_score


class ResumeOfficialTest(unittest.TestCase):
    def test_hardware_failure_is_not_a_model_score(self):
        trace = [{"error": "RuntimeError: CUDA error: unknown error"}]
        self.assertTrue(hardware_errors(trace))
        self.assertFalse(hardware_errors([{"error": "invalid action"}]))
        with self.assertRaisesRegex(RuntimeError, "Hardware failure"):
            compute_score("", "", "A", {"tooleqa_trace": trace})

    def test_resume_keeps_good_ids_and_archives_hardware_failures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data-mounted").mkdir()
            for split in ("seen", "unseen"):
                rows = [{"extra_info": {"sample_id": f"{split}-{i}"}} for i in range(3)]
                (root / "data-mounted" / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
            folder = root / "seen/trajectories/step_0"
            folder.mkdir(parents=True)
            for i in range(2):
                record = {"sample": {"sample_id": f"seen-{i}"}, "trajectory_id": str(i),
                          "reward": {"acc": 0}, "tool_trace": []}
                if i == 1:
                    record["tool_trace"] = [{"error": "CUDA error: unknown error"}]
                (folder / f"{i}.json").write_text(json.dumps(record))
            prepare(root)
            remaining = [json.loads(line)["extra_info"]["sample_id"]
                         for line in (root / "data-resume/seen.jsonl").read_text().splitlines()]
            self.assertEqual(remaining, ["seen-1", "seen-2"])
            self.assertTrue((folder / "0.json").exists())
            self.assertFalse((folder / "1.json").exists())
            self.assertEqual(len(list((root / "hardware_interrupted").rglob("*.json"))), 1)
            with self.assertRaisesRegex(ValueError, "completeness"):
                consolidate(root, "seen")
            for i in (1, 2):
                (folder / f"{i}.json").write_text(json.dumps({
                    "sample": {"sample_id": f"seen-{i}"}, "trajectory_id": str(i),
                    "reward": {"acc": 1}, "tool_trace": []}))
            consolidate(root, "seen")
            summary = json.loads((root / "seen-summary.json").read_text())[0]
            self.assertEqual(summary["records"], 3)
            self.assertAlmostEqual(summary["acc"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
