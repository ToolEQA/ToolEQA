import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from src.evaluation.open_protocol import PROTOCOL_ID, request, recover_planner_output


class PlannerRecoveryTest(unittest.TestCase):
    def payload(self, raw):
        return {"protocol_id": PROTOCOL_ID, "error_type": "ValueError", "retryable": False,
                "error": "Invalid planner output: " + raw}

    def test_preamble_only_is_removed(self):
        result = recover_planner_output(self.payload("Explanation.\n\nPlan:\n1. Find chair.\n2. Inspect it."))
        self.assertEqual(result["plan"], "Plan:\n1. Find chair.\n2. Inspect it.")
        self.assertEqual(result["planner_repair"], "strip-preamble")

    def test_no_valid_plan_uses_generic_guidance(self):
        result = recover_planner_output(self.payload("No plan"))
        self.assertEqual(result["planner_repair"], "generic-question-only-fallback")

    def test_wrong_protocol_is_not_recovered(self):
        payload = self.payload("Plan:\n1. x\n2. y")
        payload["protocol_id"] = "wrong"
        self.assertIsNone(recover_planner_output(payload))

    def test_http_recovery_is_planner_only(self):
        for operation in ("plan", "judge"):
            error = urllib.error.HTTPError("http://local", 500, "bad", {}, io.BytesIO(
                json.dumps(self.payload("Intro\nPlan:\n1. x\n2. y")).encode()))
            with patch("urllib.request.urlopen", side_effect=error):
                if operation == "plan":
                    self.assertEqual(request(operation, question="q")["planner_repair"], "strip-preamble")
                else:
                    with self.assertRaises(RuntimeError):
                        request(operation, question="q")


if __name__ == "__main__":
    unittest.main()
