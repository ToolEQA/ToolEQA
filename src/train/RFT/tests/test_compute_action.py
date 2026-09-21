from __future__ import annotations

import unittest

from src.utils.compute_action import (
    ComputeActionGuard,
    DuplicateComputeActionError,
    calls_registered_action,
    compute_action_signature,
)


class ComputeActionTest(unittest.TestCase):
    def test_signature_ignores_formatting_but_includes_external_inputs(self) -> None:
        compact = "total=sum(values)\nprint(total)"
        formatted = "total = sum(values)\nprint(total)\n"
        first = compute_action_signature(compact, {"values": [1, 2, 3]})
        second = compute_action_signature(formatted, {"values": [1, 2, 3]})
        changed = compute_action_signature(formatted, {"values": [1, 2, 4]})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_assigned_output_is_not_mistaken_for_an_external_input(self) -> None:
        code = "total = sum(values)\nprint(total)"
        before = compute_action_signature(code, {"values": [1, 2, 3]})
        after = compute_action_signature(code, {"values": [1, 2, 3], "total": 6})
        self.assertEqual(before, after)

    def test_registered_action_detection_does_not_treat_print_as_a_tool(self) -> None:
        actions = {"ObjectLocation3D", "final_answer"}
        self.assertFalse(calls_registered_action("print(sum(values))", actions))
        self.assertTrue(
            calls_registered_action(
                'positions, sizes = ObjectLocation3D(object="chair", image_path=image_path)',
                actions,
            )
        )
        self.assertTrue(
            calls_registered_action(
                'locate = ObjectLocation3D\nlocate(object="chair", image_path=image_path)',
                actions,
            )
        )

    def test_guard_rejects_only_after_success_is_recorded(self) -> None:
        guard = ComputeActionGuard()
        signature = guard.check("print(sum(values))", {"values": [1, 2]})
        guard.record(signature)
        with self.assertRaises(DuplicateComputeActionError):
            guard.check("print(sum(values))", {"values": [1, 2]})
        guard.check("print(sum(values))", {"values": [1, 3]})


if __name__ == "__main__":
    unittest.main()
