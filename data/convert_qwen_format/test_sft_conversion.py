"""Regression checks for SFT supervision boundaries and temporal context."""

import ast
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

from data.convert_qwen_format.convert_data_to_qwen import (
    REPO_ROOT,
    _has_final_answer_call,
    convert_sample_to_qwen_style,
    iter_samples,
)
from data.convert_qwen_format.convert_qwen import convert_sample_to_qwen_style as legacy_convert


def sample():
    return {
        "sample_id": "example",
        "question": "Which object is taller?",
        "plan": "PRIVILEGED_PLAN",
        "answer": "GOLD_LABEL",
        "trajectory": [
            {"step": "0", "is_key": "false", "image_path": "/rgb/start.jpg", "react": [
                {"thought": "Explore.", "code": "print(GoNextPointTool('move_forward'))",
                 "observation": "NEXT_VIEW"},
                {"thought": "Turn.", "code": "print(GoNextPointTool('turn_left'))",
                 "observation": "TURNED_VIEW"},
            ]},
            {"step": "1", "is_key": True, "image_path": "relative/view.jpg", "react": [
                {"thought": "Measure the chair.",
                 "code": "position, size = ObjectLocation3D(object='chair', image_path='relative/view.jpg')",
                 "observation": "position is [1, 2, 3], size is [0.5, 0.6, 0.7]"},
                {"thought": "Compute its height.", "code": "print(size[2])", "observation": "FUTURE_RESULT"},
                {"thought": "GOLD_FINAL_CONCLUSION", "code": "final_answer('GOLD_FINAL')",
                 "observation": "GOLD_FINAL"},
                {"thought": "AFTER_TERMINAL", "code": "print('AFTER_TERMINAL')", "observation": ""},
            ]},
        ],
    }


class ConverterTests(unittest.TestCase):
    def test_all_nonterminal_turns_once_without_plan_or_answer(self):
        source = sample()
        original = copy.deepcopy(source)
        rows = convert_sample_to_qwen_style(source)
        self.assertEqual(source, original)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["id"] for row in rows}), 4)
        for row in rows:
            self.assertEqual([m["from"] for m in row["conversations"]], ["system", "human", "gpt"])
            self.assertEqual(row["conversations"][1]["value"].count("<image>"), len(row["image"]))
        serialized = json.dumps(rows)
        for forbidden in ["PRIVILEGED_PLAN", "GOLD_LABEL", "GOLD_FINAL", "AFTER_TERMINAL"]:
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(rows[0]["image"], ["/rgb/start.jpg"])
        self.assertEqual(rows[2]["image"], ["relative/view.jpg"])
        self.assertEqual(legacy_convert(source), rows)

    def test_history_and_memory_use_only_preceding_observations(self):
        rows = convert_sample_to_qwen_style(sample())
        first = rows[0]["conversations"][1]["value"]
        self.assertNotIn("NEXT_VIEW", first)
        second = rows[1]["conversations"][1]["value"]
        self.assertIn("NEXT_VIEW", second)
        self.assertNotIn("TURNED_VIEW", second)
        self.assertIn("Explored: 1 viewpoints", second)
        before_measurement = rows[2]["conversations"][1]["value"]
        self.assertNotIn("position=[1.0, 2.0, 3.0]", before_measurement)
        after_measurement = rows[3]["conversations"][1]["value"]
        self.assertIn("[Interaction History]", after_measurement)
        self.assertIn("Detected objects:", after_measurement)
        self.assertIn("chair", after_measurement.split("[Spatial Memory]")[-1])
        self.assertNotIn("FUTURE_RESULT", after_measurement)

    def test_terminal_detection(self):
        for code in ["final_answer ('answer')", "tool.final_answer(answer)", "FinalAnswerTool(answer)",
                     "```py\nfinal_answer('answer')\n```"]:
            self.assertTrue(_has_final_answer_call(code))
        self.assertFalse(_has_final_answer_call("print('final_answer(answer)')"))
        self.assertFalse(_has_final_answer_call("# final_answer(answer)\nprint(1)"))

    def test_existing_training_masks_supervise_only_current_response(self):
        # Execute the actual preprocessing functions without loading models,
        # image libraries, or the training stack. Only tensor storage is stubbed.
        class Tokenizer:
            def apply_chat_template(self, messages):
                return [1, 2, 3] + [ord(c) + 10 for c in messages[0]["content"]] + [4]

        row = convert_sample_to_qwen_style(sample())[-1]
        for filename in ["data_qwen.py", "data_qwen_packed.py"]:
            with self.subTest(loader=filename):
                path = REPO_ROOT / "src/train/SFT/qwen-vl-finetune/qwenvl/data" / filename
                tree = ast.parse(path.read_text())
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == "preprocess_qwen_2_visual")
                namespace = {
                    "copy": copy, "IGNORE_INDEX": -100,
                    "torch": SimpleNamespace(tensor=lambda values, **kw: values, long=None),
                }
                exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
                with redirect_stdout(io.StringIO()):
                    result = namespace["preprocess_qwen_2_visual"](
                        [row["conversations"]], Tokenizer(), grid_thw_image=[1])
                supervised = [token for token in result["labels"][0] if token != -100]
                expected = Tokenizer().apply_chat_template([
                    {"content": row["conversations"][-1]["value"]}])[3:]
                self.assertEqual(supervised, expected)

    def test_json_and_jsonl_input(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ["json", "jsonl"]:
                path = Path(directory) / f"input.{suffix}"
                rows = [sample(), sample()]
                path.write_text(json.dumps(rows) if suffix == "json" else
                                "\n".join(json.dumps(row) for row in rows))
                self.assertEqual(list(iter_samples(path)), rows)


if __name__ == "__main__":
    unittest.main()
