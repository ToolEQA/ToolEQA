import unittest
from unittest.mock import patch
from src.evaluation.openeqa_protocol import (
    PROMPT, EXTRA_PROMPT, parse_score, normalize_score, preprocess_prediction, build_messages,
    JUDGE_PROTOCOL_ID,
)
from src.evaluation.open_protocol import semantic_judgment, PROTOCOL_ID


class OpenEQAProtocolTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual([normalize_score(s) for s in range(6)], [0, 0, .25, .5, .75, 1])
        self.assertEqual(normalize_score(8), 1)

    def test_upstream_parsing(self):
        for text, expected in [('3', 3), ('Your mark: 4', 4), ('explanation\nYour mark: 2\nrest', 2), ('6', 6)]:
            self.assertEqual(parse_score(text), expected)
        for text in ['{"score": 5}', '5 because correct', 'unknown']:
            with self.assertRaises(ValueError):
                parse_score(text)

    def test_preprocessing(self):
        for before, after in [(None, None), ('', ''), ('red', 'red'), ('Red. extra', 'Red.'),
                              ('One. Two. rest', 'One. Two.'), ('A.', 'A.')]:
            self.assertEqual(preprocess_prediction(before), after)

    def test_prompts_single_user_message(self):
        self.assertEqual(build_messages('q', 'r', 'c'),
                         [{'role': 'user', 'content': PROMPT.format(question='q', answer='r', prediction='c')}])
        self.assertEqual(build_messages('q', 'r', 'c', []),
                         [{'role': 'user', 'content': EXTRA_PROMPT.format(question='q', answer='r', prediction='c', extra_answers=[])}])
        self.assertNotEqual(PROTOCOL_ID, JUDGE_PROTOCOL_ID)

    def test_empty_and_letter_predictions_are_judged(self):
        for candidate in ('', 'A'):
            with patch('src.evaluation.open_protocol.request', return_value={'score': 3}) as request:
                self.assertEqual(semantic_judgment('q', 'r', candidate)['answer_quality'], .5)
                request.assert_called_once()

    def test_reward_uses_new_scale_only_when_explicit(self):
        from src.train.RFT.reward import compute_reward
        from src.train.RFT.tests.test_reward import SAMPLE, final
        sample = {**SAMPLE, 'answer_setting': 'open', 'answer': 'yes'}
        for score in range(1, 6):
            result = compute_reward(sample, [final('yes')], semantic_score=score, score_protocol='openeqa')
            self.assertEqual(result['answer_quality'], (score - 1) / 4)
        self.assertEqual(compute_reward(sample, [final('yes')], semantic_score=3)['answer_quality'], .6)
