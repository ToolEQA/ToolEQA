"""Loopback-only frozen 7B judge/planner. Persistent auditable content-addressed cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.evaluation.open_protocol import JUDGE_PROMPT, PLAN_SYSTEM_PROMPT, PROTOCOL_ID


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
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    identity = {"model": str(Path(args.model).resolve()), "protocol_id": PROTOCOL_ID,
                "temperature": 0, "max_new_tokens_judge": 192, "max_new_tokens_plan": 384,
                "config_sha256": hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest()}
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
            self.respond(200, {"ready": True, **identity})

        def do_POST(self):
            key = None
            raw = None
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid payload size")
                payload = json.loads(self.rfile.read(size))
                if payload.pop("protocol_id") != PROTOCOL_ID:
                    raise ValueError("Protocol mismatch")
                operation = payload.pop("operation")
                expected = {"question"} if operation == "plan" else {"question", "reference", "candidate"}
                if operation not in {"plan", "judge"} or set(payload) != expected:
                    raise ValueError("Unexpected operation or fields")
                if not all(isinstance(v, str) for v in payload.values()):
                    raise ValueError("Only text fields accepted")
                key = hashlib.sha256(json.dumps([identity, operation, payload], sort_keys=True).encode()).hexdigest()
                path = cache / f"{key}.json"
                with lock:
                    if path.exists():
                        result = json.loads(path.read_text())["result"]
                    else:
                        messages = [{"role": "system", "content": PLAN_SYSTEM_PROMPT if operation == "plan" else JUDGE_PROMPT},
                                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
                        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                               return_tensors="pt", return_dict=True).to("cuda:0")
                        if inputs["input_ids"].shape[1] > 4096:
                            raise ValueError("Frozen evaluator input exceeds 4096 tokens")
                        with torch.inference_mode():
                            output = model.generate(**inputs, do_sample=False,
                                                    max_new_tokens=384 if operation == "plan" else 192)
                        raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                        if operation == "judge":
                            result, repaired = parse_judgment(raw)
                            if repaired:
                                logging.warning("judge_reason_quote_repair key=%s raw=%r", key, raw)
                        else:
                            if not raw.startswith("Plan:") or "1." not in raw or "2." not in raw:
                                raise ValueError(f"Invalid planner output: {raw}")
                            result = {"plan": raw}
                        result.update(identity)
                        record = {"request": payload, "operation": operation, "raw": raw, "result": result}
                        if operation == "judge":
                            record["parser_version"] = "reason-quotes-v1"
                            record["format_repaired"] = repaired
                        temporary = path.with_suffix(".tmp")
                        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                        temporary.replace(path)
                self.respond(200, result)
            except Exception as exc:
                # Fail closed: a crashed/invalid judge must never become a zero reward.
                logging.exception("frozen_request_failed key=%s raw=%r", key, raw)
                self.respond(500, {"error": str(exc), "error_type": type(exc).__name__,
                                   "retryable": not isinstance(exc, (ValueError, KeyError, TypeError)),
                                   "request_key": key, **identity})

    print(json.dumps({"ready": True, **identity}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
