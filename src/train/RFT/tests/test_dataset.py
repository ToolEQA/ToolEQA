from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.train.RFT.dataset import build_record, convert_dataset, iter_json_array


class DatasetTest(unittest.TestCase):
    def test_streaming_and_privileged_field_separation(self) -> None:
        samples = [
            {
                "sample_id": f"s{index}",
                "question": "Where is the chair?",
                "answer": "A",
                "proposals": ["kitchen", "bedroom", "bathroom", "living room"],
                "question_type": "location-location",
                "related_objects": [{"name": "chair", "id": index, "pos": [index, 0, 0]}],
            }
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(samples), encoding="utf-8")
            self.assertEqual(list(iter_json_array(source, chunk_size=17)), samples)
            counts = convert_dataset(source, root / "train.jsonl", root / "val.jsonl", val_ratio=0.2)
            self.assertEqual(counts["source"], 20)
            self.assertGreater(counts["train"], 0)
            self.assertGreater(counts["validation"], 0)

            filtered = convert_dataset(
                source,
                root / "filtered-train.jsonl",
                root / "filtered-val.jsonl",
                val_ratio=0.2,
                allowed_scenes={"missing-scene"},
            )
            self.assertEqual(filtered["source"], 0)

        record = build_record(samples[0], 0)
        prompt_text = record["prompt"][0]["content"]
        self.assertNotIn("related_objects", prompt_text)
        self.assertIn("related_objects", record["extra_info"])
        self.assertEqual(record["extra_info"]["evidence_targets"], samples[0]["related_objects"])
        self.assertTrue(record["extra_info"]["reward_eligible"])
        self.assertEqual(record["extra_info"]["reward_audit"], [])
        self.assertEqual(record["agent_name"], "tooleqa_evidence_agent")

    def test_conversion_quarantines_bad_annotations_and_semantic_duplicates(self) -> None:
        valid = {
            "sample_id": "original",
            "scene": "scene-a",
            "question": "Which is larger, the chair or the table?",
            "answer": "A",
            "proposals": ["chair", "table", "same size", "unknown"],
            "question_type": "attribute-size",
            "related_objects": [
                {"name": "chair", "id": 1, "pos": [0, 0, 0]},
                {"name": "table", "id": 2, "pos": [2, 0, 0]},
            ],
        }
        duplicate = dict(valid, sample_id="duplicate")
        bad = {
            **valid,
            "sample_id": "bad",
            "question": "Which is larger, the chair or the lamp on the table?",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps([valid, duplicate, bad]), encoding="utf-8")
            counts = convert_dataset(
                source,
                root / "train.jsonl",
                root / "val.jsonl",
                val_ratio=0.5,
                quarantine_output=root / "quarantine.jsonl",
            )
            self.assertEqual(counts["source"], 3)
            self.assertEqual(counts["train"] + counts["validation"], 1)
            self.assertEqual(counts["rejected"], 2)
            self.assertEqual(counts["duplicates"], 1)
            quarantined = [
                json.loads(line)
                for line in (root / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            audits = [row["extra_info"]["reward_audit"] for row in quarantined]
            self.assertTrue(any(any("semantic-duplicate" in issue for issue in audit) for audit in audits))
            self.assertTrue(any(any("nonprimary-target" in issue for issue in audit) for audit in audits))

    def test_exact_stratified_validation_is_balanced_and_disjoint(self) -> None:
        samples = []
        for question_type in ("status-status", "relationship-relationship"):
            for index in range(4):
                samples.append(
                    {
                        "sample_id": f"{question_type}-{index}",
                        "scene": "scene-a",
                        "question": f"Is the chair visible in task {question_type} {index}?",
                        "answer": "A",
                        "proposals": ["yes", "no", "unknown", "not applicable"],
                        "question_type": question_type,
                        "related_objects": [
                            {"name": "chair", "id": index, "pos": [index, 0, 0]}
                        ],
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(samples), encoding="utf-8")
            counts = convert_dataset(
                source,
                root / "train.jsonl",
                root / "val.jsonl",
                val_per_question_type=2,
            )
            self.assertEqual(counts["validation_by_question_type"], {
                "relationship-relationship": 2,
                "status-status": 2,
            })
            train_ids = {
                json.loads(line)["extra_info"]["sample_id"]
                for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
            }
            val_ids = {
                json.loads(line)["extra_info"]["sample_id"]
                for line in (root / "val.jsonl").read_text(encoding="utf-8").splitlines()
            }
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(len(train_ids | val_ids), len(samples))


if __name__ == "__main__":
    unittest.main()
