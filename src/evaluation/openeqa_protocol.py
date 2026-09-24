"""OpenEQA scoring protocol, with the judge backend replaced by local Qwen3-VL.

Prompts and parsing adapted from facebookresearch/open-eqa (MIT).
Copyright (c) Meta Platforms, Inc. and affiliates.
See openeqa_LICENSE.txt for the upstream license.
"""
import hashlib
import json

PROMPT = """You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3
Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}"""

EXTRA_PROMPT = """You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer or any of the extra answers.
1 means that the response is completely different from the answer and all of the extra answers.
Example 1:
Question: Is it overcast?
Answer: no
Extra Answers: ['doesn't look like it', 'no',' it's sunny']
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Extra Answers: ['a woman', 'a lady', 'woman']
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
Response: yes
Your mark: 5
Your Turn:
Question: {question}
Answer: {answer}
Extra Answers: {extra_answers}
Response: {prediction}"""

SETTINGS = {"temperature": 0.2, "seed": 1234, "max_new_tokens": 32,
            "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
JUDGE_PROTOCOL_ID = hashlib.sha256(json.dumps(
    {"prompt": PROMPT, "extra_prompt": EXTRA_PROMPT, "settings": SETTINGS,
     "normalization": "(clip(score,1,5)-1)/4", "preprocess": "truncate-after-last-period",
     "backend": "Qwen3-VL-8B-Instruct", "version": 1}, sort_keys=True).encode()).hexdigest()


def preprocess_prediction(prediction):
    # Exact evaluate-predictions.py preprocessing; None is handled separately.
    if prediction:
        end = prediction.rfind('.')
        if end >= 0 and end + 1 < len(prediction):
            prediction = prediction[:end + 1]
    return prediction


def build_messages(question, reference, candidate, extra_answers=None):
    prompt = PROMPT if extra_answers is None else EXTRA_PROMPT
    return [{"role": "user", "content": prompt.format(
        question=question, answer=reference, prediction=preprocess_prediction(candidate),
        extra_answers=extra_answers)}]


def parse_score(output, tag="Your mark:"):
    # Match upstream exactly: out-of-range integers are clipped at aggregation.
    if output.isdigit():
        return int(output)
    start = output.find(tag)
    if start == -1:
        raise ValueError(f"Invalid output string: {output}")
    end = output.find('\n', start)
    return int(output[start:end if end != -1 else None].replace(tag, '').strip())


def normalize_score(score):
    if type(score) is not int:
        raise ValueError("Expected integer OpenEQA score")
    return (min(5, max(1, score)) - 1) / 4.0
