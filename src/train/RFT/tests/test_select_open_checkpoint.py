import json
import tempfile
import unittest
from pathlib import Path

from src.train.RFT.open_protocol import PROTOCOL_ID
from src.train.RFT.select_open_checkpoint import select


class SelectionTest(unittest.TestCase):
    def test_only_complete_dev_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data-v2").mkdir()
            infos = [{"sample_id": str(i), "answer_setting": "open",
                      "planner_source": "question-only-frozen-v1", "frozen_protocol_id": PROTOCOL_ID}
                     for i in range(225)]
            (root / "data-v2/dev.jsonl").write_text("\n".join(json.dumps({"extra_info": i}) for i in infos))
            stage = root / "stage2-joint"
            for step, score in ((150, .4), (300, .6), (450, .6)):
                ckpt = stage / "checkpoints" / f"global_step_{step}" / "actor"
                ckpt.mkdir(parents=True)
                (ckpt / "fsdp_config.json").write_text("{}")
                (stage / "validation").mkdir(exist_ok=True)
                (stage / "validation" / f"{step}.jsonl").write_text("\n".join(
                    json.dumps({"llm_match": score}) for _ in infos))
                traces = stage / "trajectories" / f"step_{step}"
                traces.mkdir(parents=True)
                for info in infos:
                    (traces / f"{info['sample_id']}.json").write_text(json.dumps({
                        "validate": True, "sample": info, "tool_trace": [],
                        "reward": {"answer_quality": score, "semantic_judgment": {"score": round(score*5)}}}))
            result = select(root)
            self.assertEqual(result["step"], 300)
            (stage / "trajectories/step_450/0.json").unlink()
            with self.assertRaises(ValueError):
                select(root)


if __name__ == "__main__":
    unittest.main()
