"""Loopback-only frozen Qwen-VL judge/planner with auditable model-specific cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.evaluation.open_protocol import PLAN_SYSTEM_PROMPT, PROTOCOL_ID
from src.evaluation.openeqa_protocol import JUDGE_PROTOCOL_ID, SETTINGS, build_messages, parse_score, normalize_score


def parse_judgment(raw):
    """Repair only unescaped quotes in a trailing reason; never infer a score."""
    clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    repaired = False
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        match = re.fullmatch(r'\{\s*"score"\s*:\s*([0-5])\s*,\s*"reason"\s*:\s*"(.*)"\s*\}', clean, re.S)
        if not match:
            raise
        reason = match[2]
        # Do not reinterpret extra fields or nested objects as reason text.
        if any(c in reason for c in '{}') or re.search(r'"\s*:', reason):
            raise ValueError("Ambiguous malformed judge reason")
        fixed = re.sub(r'(?<!\\)((?:\\\\)*)"', r'\1\\"', reason)
        reason = json.loads('"' + fixed + '"')
        result = {"score": int(match[1]), "reason": reason}
        repaired = True
    if not isinstance(result, dict) or type(result.get("score")) is not int or not 0 <= result["score"] <= 5:
        raise ValueError(f"Invalid judge output: {raw}")
    return result, repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--port", type=int, default=18941)
    args = parser.parse_args()
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model_class = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
                   "qwen3_vl": Qwen3VLForConditionalGeneration}.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unsupported frozen model type: {config.model_type}")
    model = model_class.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    identity = {"model": str(Path(args.model).resolve()), "protocol_id": PROTOCOL_ID,
                "temperature": 0, "max_new_tokens_judge": 192, "max_new_tokens_plan": 384,
                "config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest()}
    judge_identity = {**identity, "protocol_id": JUDGE_PROTOCOL_ID, "score_protocol": "openeqa",
                      "temperature": 0.2, "seed": 1234, "max_new_tokens_judge": 32,
                      "generation_settings": SETTINGS}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, result):
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200, {"ready": True, **judge_identity, "planner_protocol_id": PROTOCOL_ID})

        def do_POST(self):
            key = None
            raw = None
            response_identity = judge_identity
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid payload size")
                payload = json.loads(self.rfile.read(size))
                supplied_protocol = payload.pop("protocol_id")
                operation = payload.pop("operation")
                response_identity = judge_identity if operation == "judge" else identity
                if supplied_protocol != response_identity['protocol_id']:
                    raise ValueError("Protocol mismatch")
                expected = {"question"} if operation == "plan" else {"question", "reference", "candidate", "extra_answers"}
                if operation not in {"plan", "judge"} or set(payload) != expected:
                    raise ValueError("Unexpected operation or fields")
                if not all(isinstance(v, str) for k, v in payload.items() if k != 'extra_answers'):
                    raise ValueError("Only text fields accepted")
                extra = payload.get('extra_answers')
                if extra is not None and (not isinstance(extra, list) or not all(isinstance(v, str) for v in extra)):
                    raise ValueError("extra_answers must be null or a list of strings")
                if operation == 'judge' and (config.model_type != 'qwen3_vl' or Path(args.model).name != 'Qwen3-VL-8B-Instruct'):
                    raise ValueError("OpenEQA judge requires original Qwen3-VL-8B-Instruct")
                key = hashlib.sha256(json.dumps([response_identity, operation, payload], sort_keys=True).encode()).hexdigest()
                path = cache / f"{key}.json"
                with lock:
                    if path.exists():
                        result = json.loads(path.read_text())["result"]
                    else:
                        messages = [{"role": "system", "content": PLAN_SYSTEM_PROMPT},
                                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                        if operation == 'judge':
                            messages = build_messages(**payload)
                        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                               return_tensors="pt", return_dict=True).to("cuda:0")
                        if inputs["input_ids"].shape[1] > 4096:
                            raise ValueError("Frozen evaluator input exceeds 4096 tokens")
                        with torch.inference_mode():
                            if operation == 'judge':
                                torch.manual_seed(1234)
                                output = model.generate(**inputs, do_sample=True, temperature=0.2,
                                                        top_p=1.0, top_k=0, repetition_penalty=1.0,
                                                        max_new_tokens=32)
                            else:
                                output = model.generate(**inputs, do_sample=False, max_new_tokens=384)
                        raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                        if operation == "judge":
                            score = parse_score(raw)
                            result = {"score": score, "answer_quality": normalize_score(score)}
                        else:
                            if not raw.startswith("Plan:") or "1." not in raw or "2." not in raw:
                                raise ValueError(f"Invalid planner output: {raw}")
                            result = {"plan": raw}
                        result.update(response_identity)
                        record = {"request": payload, "operation": operation, "raw": raw, "result": result}
                        if operation == "judge":
                            record["parser_version"] = "openeqa-integer-v1"
                        temporary = path.with_suffix(".tmp")
                        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                        temporary.replace(path)
                self.respond(200, result)
            except Exception as exc:
                # Fail closed: a crashed/invalid judge must never become a zero reward.
                logging.exception("frozen_request_failed key=%s raw=%r", key, raw)
                self.respond(500, {"error": str(exc), "error_type": type(exc).__name__,
                                   "retryable": not isinstance(exc, (ValueError, KeyError, TypeError)),
                                   "request_key": key, **response_identity})

    print(json.dumps({"ready": True, **judge_identity}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
