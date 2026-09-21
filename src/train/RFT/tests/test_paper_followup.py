import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.train.RFT.paper_followup import readiness


class PaperFollowupTest(unittest.TestCase):
    def complete_fixture(self, root):
        (root / "data-mounted").mkdir()
        for split, count in (("seen", 845), ("unseen", 1069)):
            (root / f"{split}-summary.json").write_text(json.dumps([{"records": count}]))
            folder = root / split / "trajectories" / "step_0"
            folder.mkdir(parents=True)
            with (root / "data-mounted" / f"{split}.jsonl").open("w") as manifest:
                for index in range(count):
                    sample_id = f"{split}-{index}"
                    manifest.write(json.dumps({"extra_info": {"sample_id": sample_id}}) + "\n")
                    (folder / f"{index}.json").write_text(json.dumps({
                        "sample": {"sample_id": sample_id}, "reward": {"acc": 0}}))

    def test_waits_for_summary(self):
        with TemporaryDirectory() as folder:
            self.assertFalse(readiness(Path(folder))[0])

    def test_full_identity_gate_and_duplicate_rejection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.complete_fixture(root)
            self.assertTrue(readiness(root)[0])
            (root / "seen/trajectories/step_0/0.json").write_text(json.dumps({
                "sample": {"sample_id": "seen-1"}, "reward": {"acc": 0}}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                readiness(root)

    def test_unscored_episode_does_not_trigger(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.complete_fixture(root)
            (root / "seen/trajectories/step_0/0.json").write_text(json.dumps({
                "sample": {"sample_id": "seen-0"}}))
            self.assertFalse(readiness(root)[0])


if __name__ == "__main__":
    unittest.main()
