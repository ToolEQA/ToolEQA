import ast
import json
import re
from typing import Any, Dict


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
INLINE_JSON_RE = re.compile(r"(\{.*\})", re.S)
TOOL_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", re.S)


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = action.get("action_type") or action.get("tool") or action.get("name")
    if not action_type:
        raise ValueError("Missing action_type in model output.")
    args = action.get("args") or action.get("arguments") or action.get("kwargs") or {}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("Action args must be a dictionary.")
    return {
        "action_type": str(action_type),
        "args": args,
        "raw": action,
    }


def _parse_json_payload(text: str) -> Dict[str, Any]:
    for pattern in (FENCED_JSON_RE, INLINE_JSON_RE):
        match = pattern.search(text)
        if not match:
            continue
        payload = match.group(1)
        try:
            return _normalize_action(json.loads(payload))
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON action payload found.")


def _split_args(args_expr: str) -> Dict[str, Any]:
    args_expr = args_expr.strip()
    if not args_expr:
        return {}
    expr = f"f({args_expr})"
    parsed = ast.parse(expr, mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError("Failed to parse call arguments.")
    kwargs = {}
    for kw in parsed.body.keywords:
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    if parsed.body.args:
        if len(parsed.body.args) == 1:
            kwargs["value"] = ast.literal_eval(parsed.body.args[0])
        else:
            kwargs["value"] = [ast.literal_eval(arg) for arg in parsed.body.args]
    return kwargs


def _parse_tool_style(text: str) -> Dict[str, Any]:
    match = TOOL_CALL_RE.search(text.strip())
    if not match:
        raise ValueError("No tool-style call found.")
    tool_name = match.group(1)
    args_expr = match.group(2)
    args = _split_args(args_expr)

    if tool_name == "final_answer":
        answer = args.get("answer", args.get("final_answer", args.get("value")))
        args = {"answer": answer}
        action_type = "FinalAnswer"
    elif tool_name == "GoNextPointTool":
        direction = args.get("direction", args.get("command", args.get("value")))
        args = {"direction": direction}
        action_type = "Navigate"
    else:
        action_type = tool_name

    return {
        "action_type": action_type,
        "args": args,
        "raw": {"tool_name": tool_name, "args": args},
    }


def parse_action_output(text: str) -> Dict[str, Any]:
    """
    Parse model output into a normalized action dictionary.

    Preferred output format:
    {
      "action_type": "Navigate",
      "args": {"direction": "move_forward"}
    }

    Fallback formats:
    - final_answer("C")
    - GoNextPointTool("turn_right")
    - ObjectLocation2D(object="bed", image_path="...")
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty model output.")

    try:
        return _parse_json_payload(text)
    except ValueError:
        return _parse_tool_style(text)
