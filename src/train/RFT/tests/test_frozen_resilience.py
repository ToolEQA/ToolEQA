import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from src.train.RFT.frozen_service import parse_judgment
from src.train.RFT.open_protocol import PROTOCOL_ID, request


class FrozenResilienceTest(unittest.TestCase):
    def test_quote_repair_preserves_score_and_reason(self):
        for score in range(6):
            result, repaired = parse_judgment(
                '{"score": %d, "reason": "It is "against the wall" but incomplete."}' % score)
            self.assertTrue(repaired)
            self.assertEqual(result, {"score": score, "reason": 'It is "against the wall" but incomplete.'})

    def test_valid_unchanged(self):
        value = {"score": 2, "reason": 'It is "against the wall".'}
        self.assertEqual(parse_judgment(json.dumps(value)), (value, False))

    def test_bad_scores_and_ambiguous_outputs_fail_closed(self):
        for raw in ('{"score": 6}', '{"score": 2.0}', '{"score": true}', '[]',
                    '{"score": 2, "reason": "x", "score": 5 broken"}',
                    '{"score": 2, "reason": "truncated'):
            with self.assertRaises(ValueError):
                parse_judgment(raw)

    def test_transient_http_retry(self):
        error = urllib.error.HTTPError('http://local', 503, 'busy', {}, io.BytesIO(b'{"error":"busy"}'))
        response = io.BytesIO(json.dumps({"score": 2, "protocol_id": PROTOCOL_ID}).encode())
        with patch('urllib.request.urlopen', side_effect=[error, response]) as call, patch('time.sleep'):
            self.assertEqual(request('judge', question='q')["score"], 2)
            self.assertEqual(call.call_count, 2)

    def test_permanent_error_detail_preserved(self):
        error = urllib.error.HTTPError('http://local', 500, 'bad', {},
                                     io.BytesIO(b'{"error":"bad format", "retryable":false}'))
        with patch('urllib.request.urlopen', side_effect=error) as call:
            with self.assertRaisesRegex(RuntimeError, 'bad format'):
                request('judge', question='q')
            self.assertEqual(call.call_count, 1)

    def test_retries_bounded(self):
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('offline')) as call, patch('time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
                request('judge', question='q')
            self.assertEqual(call.call_count, 3)
