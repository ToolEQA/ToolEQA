"""Embodied rollout loop with single-pass Thought-Code actions."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from argparse import Namespace
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import torch

from verl.experimental.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics
from verl.utils.profiler import simple_timer

from src.memory.spatial_memory import SpatialMemory
from src.tools.tool_box import get_tool_box, show_tool_descriptions
from src.train.RFT.evidence import normalize_action, to_jsonable
from src.train.RFT.trajectory_log import trajectory_path, write_trajectory
from src.utils.compute_action import (
    DuplicateComputeActionError,
    compute_action_signature,
    state_changes,
    state_snapshot,
)


_ENV_LOCK: asyncio.Lock | None = None
_AGENT_WORKER_RANK: int | None = None
_AGENT_WORKER_RANK_RESOLVED = False


def _agent_worker_rank() -> int | None:
    """Index of the enclosing AgentLoopWorker Ray actor, if resolvable."""
    global _AGENT_WORKER_RANK, _AGENT_WORKER_RANK_RESOLVED
    if not _AGENT_WORKER_RANK_RESOLVED:
        _AGENT_WORKER_RANK_RESOLVED = True
        try:
            import ray

            name = ray.get_runtime_context().get_actor_name() or ""
            match = re.match(r"agent_loop_worker_(\d+)", name)
            if match:
                _AGENT_WORKER_RANK = int(match.group(1))
        except Exception:
            _AGENT_WORKER_RANK = None
    return _AGENT_WORKER_RANK


class DuplicateToolCallError(RuntimeError):
    """Raised before executing a tool call that cannot add new evidence."""


def _environment_lock() -> asyncio.Lock:
    """Habitat state is mutable; serialize episodes inside each rollout process."""
    global _ENV_LOCK
    if _ENV_LOCK is None:
        _ENV_LOCK = asyncio.Lock()
    return _ENV_LOCK


class _RecordedTool:
    __slots__ = ("_name", "_fn", "_recorder")

    def __init__(self, name: str, fn: Callable[..., Any], recorder: "_ToolExecutionRecorder"):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_recorder", recorder)

    def __getattribute__(self, name: str) -> Any:
        # The controller may call the wrapper but must not walk through it to
        # simulator internals (which contain privileged related-object data).
        raise AttributeError(f"Tool internals are not accessible: {name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name = object.__getattribute__(self, "_name")
        fn = object.__getattribute__(self, "_fn")
        recorder = object.__getattribute__(self, "_recorder")
        started = time.perf_counter()
        before_image = recorder.current_image()
        before_path = recorder.current_path_length()
        before_camera = recorder.current_camera_state()
        previous = recorder.find_duplicate(name, args, kwargs, before_image)
        if previous is not None:
            error = DuplicateToolCallError(
                "Identical tool call rejected. Reuse the recorded result and follow Spatial Memory's "
                "Recommended next evidence actions instead of repeating it."
            )
            recorder.record(
                name,
                args,
                kwargs,
                result=None,
                latency=time.perf_counter() - started,
                ok=False,
                error=f"{type(error).__name__}: {error}",
                image_before=before_image,
                path_before=before_path,
                camera_before=before_camera,
                duplicate_rejected=True,
            )
            raise error
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            recorder.record(
                name,
                args,
                kwargs,
                result=None,
                latency=time.perf_counter() - started,
                ok=False,
                error=f"{type(error).__name__}: {error}",
                image_before=before_image,
                path_before=before_path,
                camera_before=before_camera,
            )
            raise
        recorder.record(
            name,
            args,
            kwargs,
            result=result,
            latency=time.perf_counter() - started,
            ok=True,
            error=None,
            image_before=before_image,
            path_before=before_path,
            camera_before=before_camera,
        )
        return result


class _ToolExecutionRecorder:
    def __init__(self, toolbox: list[Any]):
        self.toolbox = toolbox
        self.trace: list[dict[str, Any]] = []
        self.final_answer: Any = None
        self.memory = SpatialMemory()
        self._signatures: dict[str, dict[str, Any]] = {}
        self._compute_signatures: dict[str, dict[str, Any]] = {}

    def navigation_tool(self) -> Any | None:
        return next((tool for tool in self.toolbox if getattr(tool, "name", "") == "GoNextPointTool"), None)

    def current_image(self) -> str | None:
        tool = self.navigation_tool()
        return getattr(tool, "cur_rgb_path", None) if tool is not None else None

    def current_path_length(self) -> float | None:
        tool = self.navigation_tool()
        backend = getattr(tool, "eqa_modeling", None) if tool is not None else None
        value = getattr(backend, "path_length", None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def current_camera_state(self) -> dict[str, Any] | None:
        """Return the agent pose used by the paper's Recall@D evaluator."""
        tool = self.navigation_tool()
        backend = getattr(tool, "eqa_modeling", None) if tool is not None else None
        agent_state = getattr(backend, "agent_state", None)
        position = getattr(agent_state, "position", None)
        yaw = getattr(backend, "angle", None)
        if position is None or yaw is None:
            return None
        try:
            values = [float(value) for value in position]
            if len(values) != 3:
                return None
            return {"position": values, "yaw": float(yaw)}
        except (TypeError, ValueError):
            return None

    def tool_metadata(self, name: str) -> dict[str, Any]:
        tool = next((item for item in self.toolbox if getattr(item, "name", "") == name), None)
        metadata: dict[str, Any] = {}
        if tool is not None and hasattr(tool, "last_resolved_image_paths"):
            metadata["resolved_image_paths"] = getattr(tool, "last_resolved_image_paths")
        if name == "ObjectLocation3D" and tool is not None:
            metadata["coordinate_frame"] = getattr(tool, "last_coordinate_frame", None)
            metadata["geometry_source"] = getattr(tool, "last_geometry_source", None)
        return metadata

    @staticmethod
    def _value(args: tuple[Any, ...], kwargs: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
        return kwargs.get(key, args[index] if len(args) > index else default)

    def normalize_args(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        if name == "GoNextPointTool":
            return {"direction": self._value(args, kwargs, "direction", 0)}
        if name in {"ObjectLocation2D", "ObjectLocation3D"}:
            return {
                "object": self._value(args, kwargs, "object", 0),
                "image_path": self._value(args, kwargs, "image_path", 1),
            }
        if name == "ObjectCrop":
            return {
                "bounding_box": self._value(args, kwargs, "bounding_box", 0, kwargs.get("bound_boxes")),
                "image_path": self._value(args, kwargs, "image_path", 1),
            }
        if name == "VisualQATool":
            return {
                "question": self._value(args, kwargs, "question", 0),
                "image_path": self._value(args, kwargs, "image_path", 1, ""),
                "image_paths": kwargs.get("image_paths", ""),
            }
        if name in {"FinalAnswerTool", "final_answer"}:
            return {"answer": self._value(args, kwargs, "answer", 0, kwargs.get("final_answer"))}
        return {"args": list(args), **kwargs}

    def signature(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        image_before: str | None,
    ) -> str | None:
        action = normalize_action(name)
        if action == "FinalAnswer":
            return None
        payload = {
            "action": action,
            "args": to_jsonable(self.normalize_args(name, args, kwargs)),
            # Navigation with the same direction from a new viewpoint is a new
            # action; perception calls already include their image path.
            "view": image_before if action == "Navigate" else None,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def find_duplicate(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        image_before: str | None,
    ) -> dict[str, Any] | None:
        signature = self.signature(name, args, kwargs, image_before)
        return self._signatures.get(signature) if signature is not None else None

    def record(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        latency: float,
        ok: bool,
        error: str | None,
        image_before: str | None,
        path_before: float | None,
        camera_before: dict[str, Any] | None,
        duplicate_rejected: bool = False,
    ) -> None:
        normalized_args = self.normalize_args(name, args, kwargs)
        action = normalize_action(name)
        json_result = to_jsonable(result)
        empty_detection = action in {"Location2D", "Location3D"} and json_result in (
            None,
            [],
            [[], []],
        )
        if ok and not empty_detection:
            try:
                self.memory.update(name, to_jsonable(normalized_args), json_result, len(self.trace))
            except Exception:
                # Memory is contextual help, never a reason to lose an otherwise valid rollout.
                pass
        self.memory.record_action(
            name,
            to_jsonable(normalized_args),
            json_result,
            len(self.trace),
            ok=ok,
            error=error,
            duplicate_rejected=duplicate_rejected,
        )
        if action == "FinalAnswer" and ok:
            self.final_answer = json_result
        trace_step = {
            "step": len(self.trace),
            "action_type": action,
            "args": to_jsonable(normalized_args),
            "result": json_result,
            "ok": bool(ok),
            "error": error,
            "duplicate_rejected": bool(duplicate_rejected),
            "latency_s": round(float(latency), 6),
            "image_path_before": image_before,
            "image_path_after": self.current_image(),
            "path_length_before": path_before,
            "path_length_after": self.current_path_length(),
            "camera_state_before": to_jsonable(camera_before),
            "camera_state_after": to_jsonable(self.current_camera_state()),
        }
        trace_step.update(to_jsonable(self.tool_metadata(name)))
        self.trace.append(trace_step)
        signature = self.signature(name, args, kwargs, image_before)
        if signature is not None and not duplicate_rejected:
            self._signatures.setdefault(signature, trace_step)

    def find_compute_duplicate(self, signature: str) -> dict[str, Any] | None:
        return self._compute_signatures.get(signature)

    def record_compute(
        self,
        code: str,
        signature: str,
        result: Any,
        changes: dict[str, Any],
        latency: float,
        *,
        ok: bool,
        error: str | None = None,
        duplicate_rejected: bool = False,
    ) -> None:
        args = {"code": code, "input_signature": signature}
        json_result = to_jsonable(result)
        step = len(self.trace)
        self.memory.record_action(
            "Compute",
            args,
            json_result,
            step,
            ok=ok,
            error=error,
            duplicate_rejected=duplicate_rejected,
        )
        trace_step = {
            "step": step,
            "action_type": "Compute",
            "args": args,
            "result": json_result,
            "state_changes": to_jsonable(changes),
            "ok": bool(ok),
            "error": error,
            "duplicate_rejected": bool(duplicate_rejected),
            "latency_s": round(float(latency), 6),
            "image_path_before": self.current_image(),
            "image_path_after": self.current_image(),
            "path_length_before": self.current_path_length(),
            "path_length_after": self.current_path_length(),
            "camera_state_before": to_jsonable(self.current_camera_state()),
            "camera_state_after": to_jsonable(self.current_camera_state()),
        }
        self.trace.append(trace_step)
        if ok and not duplicate_rejected:
            self._compute_signatures.setdefault(signature, trace_step)

    def record_invalid_code(self, error: Exception, code: str = "") -> None:
        step = len(self.trace)
        args = {"code": code}
        rendered_error = f"{type(error).__name__}: {error}"
        self.memory.record_action(
            "InvalidCode",
            args,
            None,
            step,
            ok=False,
            error=rendered_error,
            duplicate_rejected=False,
        )
        self.trace.append(
            {
                "step": step,
                "action_type": "InvalidCode",
                "args": args,
                "result": None,
                "ok": False,
                "error": rendered_error,
                "duplicate_rejected": False,
                "image_path_before": self.current_image(),
                "image_path_after": self.current_image(),
                "path_length_before": self.current_path_length(),
                "path_length_after": self.current_path_length(),
                "camera_state_before": to_jsonable(self.current_camera_state()),
                "camera_state_after": to_jsonable(self.current_camera_state()),
            }
        )


class ToolEQAEvidenceAgentLoop(AgentLoopBase):
    """Execute controller-generated Python against the original ToolEQA tools.

    This class is deliberately not decorated with VERL's ``register`` helper.
    The external agent-loop YAML already registers its target and parameters.
    Decorating the lazily imported target overwrites that populated registry
    entry with only ``_target_`` after the first trajectory, silently causing
    every later trajectory to fall back to constructor defaults.
    """

    _code_block_pattern = re.compile(r"Code:\s*```(?:py|python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    _fallback_code_pattern = re.compile(r"```(?:py|python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    _no_print_observation = "Code ran successfully but printed no observation."

    def __init__(self, *args: Any, **kwargs: Any):
        final_answer_reserve_tokens = kwargs.pop("final_answer_reserve_tokens", 512)
        forced_final_max_tokens = kwargs.pop("forced_final_max_tokens", 384)
        minimum_normal_turn_tokens = kwargs.pop("minimum_normal_turn_tokens", 128)
        normal_turn_max_tokens = kwargs.pop("normal_turn_max_tokens", 768)
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.model_context_length = int(
            self.rollout_config.get("max_model_len", self.prompt_length + self.response_length)
        )
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.final_answer_reserve_tokens = max(int(final_answer_reserve_tokens), 64)
        self.forced_final_max_tokens = max(int(forced_final_max_tokens), 32)
        self.minimum_normal_turn_tokens = max(int(minimum_normal_turn_tokens), 32)
        self.normal_turn_max_tokens = max(int(normal_turn_max_tokens), 64)
        self._accumulated_grid_thw: list[torch.Tensor] = []
        self._accumulated_pixel_values: list[torch.Tensor] = []
        root = Path(os.environ.get("TOOLEQA_ROOT", Path.cwd())).resolve()
        prompt_path = root / "data/ToolTrajectory/prompts/rft_thought_code_system_prompt.txt"
        self.react_system_prompt_template = prompt_path.read_text(encoding="utf-8")
        self.root = root

    def _reset_vision_accumulators(self) -> None:
        self._accumulated_grid_thw = []
        self._accumulated_pixel_values = []

    def _capture_vision_inputs(self) -> None:
        vision = self._last_vision_inputs
        if vision.get("image_grid_thw") is not None:
            self._accumulated_grid_thw.append(vision["image_grid_thw"])
        if vision.get("pixel_values") is not None:
            self._accumulated_pixel_values.append(vision["pixel_values"])

    def _build_toolbox(self, sample: dict[str, Any]) -> tuple[list[Any], dict[str, Any], _ToolExecutionRecorder]:
        # DetAny3D uses gpu_id as its shared-memory channel, while the
        # navigation/VQA tools use it as a local CUDA index. Keep the two
        # namespaces separate so tool models do not occupy an FSDP train GPU.
        channel_id = int(os.environ.get("TOOLEQA_TOOL_GPU_ID", "0"))
        agent_gpu_id = int(os.environ.get("TOOLEQA_AGENT_GPU_ID", str(channel_id)))
        # With multiple AgentLoopWorkers, spread each worker's Habitat/VQA
        # stack over its own CUDA device so episodes run concurrently.
        gpu_map = [item.strip() for item in os.environ.get("TOOLEQA_AGENT_GPU_MAP", "").split(",") if item.strip()]
        if gpu_map:
            worker_rank = _agent_worker_rank()
            if worker_rank is not None:
                agent_gpu_id = int(gpu_map[worker_rank % len(gpu_map)])
        cfg = self.rollout_config.agent.get("env_config_path", str(self.root / "config/react-eqa.yaml"))
        toolbox = get_tool_box(debug=False, gpu_id=channel_id, args=Namespace(cfg=cfg, open_vocab=True))
        # Concurrent rollouts of one prompt share sample_id, and the tools key
        # their image directories on it. Give each episode a unique id for
        # tool output paths only, so parallel episodes never read an image
        # another episode is still writing. Reward and trajectory metadata
        # keep the original sample.
        tool_sample = dict(sample)
        tool_sample["sample_id"] = f"{sample.get('sample_id', 'sample')}__ep{uuid4().hex[:8]}"
        for tool in toolbox:
            if getattr(tool, "name", "") in {"GoNextPointTool", "VisualQATool"}:
                tool.gpu_id = agent_gpu_id
            if hasattr(tool, "initialize"):
                tool.initialize(tool_sample)
        recorder = _ToolExecutionRecorder(toolbox)
        recorder.memory.configure_task(sample)
        executable = {tool.name: _RecordedTool(tool.name, tool, recorder) for tool in toolbox}
        return toolbox, executable, recorder

    @staticmethod
    def _build_task(raw_prompt: list[dict[str, Any]], sample: dict[str, Any]) -> str:
        question = sample.get("question")
        if not question and raw_prompt:
            question = raw_prompt[-1].get("content", "")
        lines = [str(question or "").strip()]
        if sample.get("planner_source") == "question-only-frozen-v1":
            from src.train.RFT.open_protocol import question_only_plan
            lines.extend(["", "Planner guidance:", question_only_plan(str(question))])
        elif sample.get("plan"):
            lines.extend(["", "Planner guidance:", str(sample["plan"]).strip()])
        proposals = [] if sample.get("answer_setting") == "open" else sample.get("proposals") or []
        if proposals:
            lines.extend(["", "Choices:"])
            lines.extend(f"{chr(65 + index)}. {proposal}" for index, proposal in enumerate(proposals[:4]))
        return "\n".join(lines).strip()

    def _build_system_prompt(self, toolbox: list[Any]) -> str:
        prompt = self.react_system_prompt_template.replace("<<tool_descriptions>>", show_tool_descriptions(toolbox))
        prompt = prompt.replace("<<authorized_imports>>", "math, json, re")
        if "<<" in prompt or ">>" in prompt:
            raise ValueError("Unresolved placeholder in the RFT system prompt")
        return prompt

    def _extract_code(self, generated_text: str) -> str:
        match = self._code_block_pattern.search(generated_text) or self._fallback_code_pattern.search(generated_text)
        return match.group(1).strip() if match else generated_text.strip()

    @staticmethod
    def _extract_thought(generated_text: str) -> str:
        text = str(generated_text or "")
        tagged = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if tagged:
            return tagged.group(1).strip()
        prefix = re.split(r"\bCode\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
        prefix = prefix.replace("</think>", "")
        prefix = re.sub(r"^\s*Thought\s*:\s*", "", prefix, flags=re.IGNORECASE)
        return "\n".join(
            line for line in prefix.splitlines() if not line.strip().startswith("```")
        ).strip()

    @classmethod
    def _response_is_truncated(cls, generated_text: str, token_count: int, token_budget: int) -> bool:
        """Treat a max-length response as truncated only if its Code block is unfinished.

        vLLM currently reports both stop-string and length termination as
        ``completed``. A response may therefore land exactly on the configured
        boundary while still containing a complete action.
        """
        if token_count < token_budget:
            return False
        return cls._code_block_pattern.search(generated_text) is None

    @staticmethod
    def _best_effort_final_answer(generated_text: str, sample: dict[str, Any]) -> str:
        """Recover the controller's intended choice when forced code is malformed."""
        text = str(generated_text or "").strip()
        call = re.search(
            r"final_answer\s*\(\s*(?:answer\s*=\s*)?(['\"])(.*?)\1\s*\)",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if call:
            return call.group(2).strip()

        if sample.get("answer_setting") == "open":
            # Never treat malformed Thought/Code as the answer or invent an option.
            return ""

        option = re.search(r"\b(?:answer|choice|option)\s*(?:is|=|:)?\s*([A-D])\b", text, re.IGNORECASE)
        if option:
            return option.group(1).upper()
        exact_option = re.fullmatch(r"\s*([A-D])(?:[.\)])?\s*", text, re.IGNORECASE)
        if exact_option:
            return exact_option.group(1).upper()

        lowered = text.lower()
        matches = [
            (index, str(proposal))
            for index, proposal in enumerate((sample.get("proposals") or [])[:4])
            if str(proposal).strip() and str(proposal).strip().lower() in lowered
        ]
        if len(matches) == 1:
            return chr(ord("A") + matches[0][0])
        return text[:512] or "A"

    @staticmethod
    def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in {"math", "json", "re"}:
            raise ImportError(f"Import '{name}' is not allowed")
        return __import__(name, globals, locals, fromlist, level)

    def _execute_code(self, code: str, tools: dict[str, Any], state: dict[str, Any]) -> str:
        code = code.strip()
        if not code:
            raise ValueError("Controller emitted no executable code")

        stdout = StringIO()

        def captured_print(*args: Any, **kwargs: Any) -> None:
            print(*args, file=stdout, **kwargs)

        safe_builtins = {
            "__import__": self._safe_import,
            "print": captured_print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
        }
        tool_names = set(tools)
        environment = {
            "__builtins__": safe_builtins,
            **{key: value for key, value in state.items() if key not in tool_names},
            **tools,
        }
        exec(code, environment, environment)
        for key, value in environment.items():
            if not key.startswith("__") and key not in tool_names:
                state[key] = value
        return stdout.getvalue().strip() or self._no_print_observation

    def _execute_turn(
        self,
        code: str,
        tools: dict[str, Any],
        state: dict[str, Any],
        recorder: _ToolExecutionRecorder,
    ) -> str:
        """Execute one action and classify tool-free Python as a Compute action."""
        code = code.strip()
        trace_size = len(recorder.trace)
        before_state = state_snapshot(state)
        signature = compute_action_signature(code, state)
        duplicate = recorder.find_compute_duplicate(signature)
        if duplicate is not None:
            error = DuplicateComputeActionError(
                "Identical Compute action rejected. Reuse its prior Observation or change its inputs."
            )
            recorder.record_compute(
                code,
                signature,
                result=None,
                changes={},
                latency=0.0,
                ok=False,
                error=f"{type(error).__name__}: {error}",
                duplicate_rejected=True,
            )
            return f"Code execution rejected: {error}"

        started = time.perf_counter()
        try:
            observation = self._execute_code(code, tools, state)
        except Exception as error:
            if len(recorder.trace) == trace_size:
                recorder.record_invalid_code(error, code)
            return f"Code execution failed: {type(error).__name__}: {error}"

        if len(recorder.trace) > trace_size:
            return self._recover_unprinted_tool_result(observation, recorder.trace[trace_size:])

        changes = state_changes(before_state, state_snapshot(state))
        if observation == self._no_print_observation and not changes:
            error = ValueError("Controller code produced no tool call, printed output, or state change")
            recorder.record_invalid_code(error, code)
            return f"Code execution rejected: {error}"
        if observation == self._no_print_observation:
            observation = "Compute result (persisted variables): " + json.dumps(
                {name: change["after"] for name, change in changes.items()},
                ensure_ascii=False,
            )
        recorder.record_compute(
            code,
            signature,
            result=observation,
            changes=changes,
            latency=time.perf_counter() - started,
            ok=True,
        )
        return observation

    @classmethod
    def _recover_unprinted_tool_result(cls, observation: str, trace: list[dict[str, Any]]) -> str:
        """Expose a successful tool result when code omitted the requested print call."""
        if observation != cls._no_print_observation:
            return observation
        for step in reversed(trace):
            if step.get("ok"):
                rendered = json.dumps(step.get("result"), ensure_ascii=False)
                return f"Tool result (automatically captured): {rendered}"
        return observation

    @staticmethod
    def _user_content(text: str, image_path: str | None) -> Any:
        if image_path:
            return [{"type": "image", "image": image_path}, {"type": "text", "text": text}]
        return text

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        async with _environment_lock():
            sample = dict(kwargs.get("extra_info", {}) or {})
            manager_trajectory = dict(kwargs.get("trajectory_info", {}) or {})
            trajectory_meta = {
                "global_step": manager_trajectory.get("step", kwargs.get("global_steps")),
                "dataset_index": manager_trajectory.get(
                    "sample_index", sample.get("index", kwargs.get("index"))
                ),
                "rollout_n": manager_trajectory.get("rollout_n"),
                "validate": manager_trajectory.get("validate"),
                "prompt_uid": kwargs.get("uid"),
                "session_id": kwargs.get("session_id"),
            }
            toolbox, executable, recorder = self._build_toolbox(sample)
            try:
                return await self._run_episode(
                    toolbox,
                    executable,
                    recorder,
                    list(kwargs["raw_prompt"]),
                    sample,
                    sampling_params,
                    trajectory_meta,
                )
            finally:
                self._close_simulators(toolbox)

    async def _run_episode(
        self,
        toolbox: list[Any],
        executable: dict[str, Any],
        recorder: _ToolExecutionRecorder,
        raw_prompt: list[dict[str, Any]],
        sample: dict[str, Any],
        sampling_params: dict[str, Any],
        trajectory_meta: dict[str, Any],
    ) -> AgentLoopOutput:
        initial_image = recorder.current_image()
        initial_camera_state = recorder.current_camera_state()
        task = self._build_task(raw_prompt, sample)
        if initial_image:
            task = f"{task}\n\nInitial observation image path: {initial_image}"
        system_prompt = self._build_system_prompt(toolbox)
        if sample.get("answer_setting") == "open":
            from src.train.RFT.open_protocol import open_system_prompt
            system_prompt = open_system_prompt(system_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": self._user_content(task, initial_image),
            },
        ]

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}
        # Keep exactly one uniformly sampled turn from each trajectory for the
        # PPO row.  This preserves the exact prompt used by vLLM without
        # weighting long trajectories more heavily than short trajectories.
        # The full episode trace is still used to compute the shared outcome
        # reward.  Across rollouts/updates, reservoir sampling covers all turns.
        selected_turn: dict[str, Any] | None = None
        turn_context_stats: list[dict[str, Any]] = []
        diagnostic_turns: list[dict[str, Any]] = []
        generated_token_count = 0
        state: dict[str, Any] = {}
        assistant_turns = 0
        user_turns = 0
        forced_final = False
        termination_reason = "unknown"
        last_observation = ""
        truncated_normal_action = False
        per_turn_response_limit = min(self.normal_turn_max_tokens, self.response_length)

        def record_training_turn(
            prompt_ids: list[int],
            generated_ids: list[int],
            log_probs: list[float] | None,
            vision: dict[str, Any],
            *,
            is_forced_final: bool,
            response_mask: list[int] | None = None,
        ) -> None:
            nonlocal selected_turn
            turn_index = len(turn_context_stats)
            # apply_chat_template() has just populated these tensors from the
            # exact prompt. Reuse them during PPO post-processing instead of
            # asking the processor to reconstruct an accumulated multi-image
            # prompt (which can collapse images and misalign image_pad groups).
            turn_multi_modal_data = dict(vision)
            image_grid_thw = self._last_vision_inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.clone()
                turn_multi_modal_data["_image_grid_thw"] = image_grid_thw
                turn_multi_modal_data["_images_seqlens"] = torch.repeat_interleave(
                    image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
                )
            pixel_values = self._last_vision_inputs.get("pixel_values")
            if pixel_values is not None:
                turn_multi_modal_data["_pixel_values"] = pixel_values.clone()
            turn_context_stats.append(
                {
                    "turn_index": turn_index,
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(generated_ids),
                    "trainable_response_tokens": sum(response_mask or [1] * len(generated_ids)),
                    "image_count": len(vision.get("images") or []),
                    "forced_final": is_forced_final,
                    "persistent_assistant_messages": 0,
                }
            )
            candidate = {
                "turn_index": turn_index,
                "prompt_ids": list(prompt_ids),
                "response_ids": list(generated_ids),
                "response_mask": list(response_mask or [1] * len(generated_ids)),
                "response_logprobs": list(log_probs[: len(generated_ids)]) if log_probs else None,
                "multi_modal_data": turn_multi_modal_data,
            }
            # Diagnostic-only: stress the longest observed prompt, never silently
            # enable this selector in a formal experiment.
            stress_longest = os.environ.get("TOOLEQA_STRESS_LONGEST_TURN") == "1"
            replace = (selected_turn is None or (
                len(prompt_ids) + len(generated_ids) > len(selected_turn["prompt_ids"]) + len(selected_turn["response_ids"])
            )) if stress_longest else (selected_turn is None or random.randrange(turn_index + 1) == 0)
            if replace:
                selected_turn = candidate

        while assistant_turns < self.max_assistant_turns:
            vision = await self.process_vision_info(messages)
            prompt_ids = await self.apply_chat_template(
                messages, images=vision.get("images"), videos=vision.get("videos")
            )
            if len(prompt_ids) > self.prompt_length:
                termination_reason = "context_budget"
                break
            if per_turn_response_limit < self.minimum_normal_turn_tokens:
                termination_reason = "response_budget"
                break

            turn_budget = min(
                per_turn_response_limit,
                self.model_context_length - len(prompt_ids),
            )
            if turn_budget < self.minimum_normal_turn_tokens:
                termination_reason = "context_budget"
                break

            turn_params = dict(sampling_params)
            turn_params.pop("max_new_tokens", None)
            turn_params["stop"] = ["<end_action>", "<end action>", "Observation:"]
            turn_params["max_tokens"] = turn_budget
            with simple_timer("generate_sequences", metrics):
                turn_output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_params,
                    image_data=vision.get("images"),
                    video_data=vision.get("videos"),
                )
            preempted = turn_output.num_preempted if turn_output.num_preempted is not None else -1
            metrics["num_preempted"] = max(int(metrics.get("num_preempted", -1)), 0) + max(preempted, 0)

            assistant_turns += 1
            generated_ids = list(turn_output.token_ids)
            response_mask = [1] * len(generated_ids)
            log_probs = (
                list(turn_output.log_probs[: len(generated_ids)])
                if turn_output.log_probs is not None
                else None
            )
            generated_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            generated_token_count += len(generated_ids)
            record_training_turn(
                prompt_ids,
                generated_ids,
                log_probs,
                vision,
                is_forced_final=False,
                response_mask=response_mask,
            )

            turn_record = {
                "turn_index": len(diagnostic_turns),
                "forced_final": False,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(generated_ids),
                "response_token_budget": turn_budget,
                "image_count": len(vision.get("images") or []),
                "generated_text": generated_text,
                "thought": self._extract_thought(generated_text),
                "code": self._extract_code(generated_text),
                "generation_protocol": "thought_code",
            }

            if self._response_is_truncated(generated_text, len(generated_ids), turn_budget):
                # Never execute a genuinely unfinished Python block. A closed
                # block at exactly the boundary is executed and audited below.
                truncated_normal_action = True
                termination_reason = "response_budget"
                last_observation = "The last exploration action was truncated and was not executed."
                turn_record.update(
                    {
                        "observation": last_observation,
                        "tool_trace": [],
                        "spatial_memory": recorder.memory.to_dict(),
                        "truncated": True,
                    }
                )
                diagnostic_turns.append(turn_record)
                break

            trace_size = len(recorder.trace)
            code = self._extract_code(generated_text)
            observation = self._execute_turn(code, executable, state, recorder)
            last_observation = observation
            turn_record.update(
                {
                    "observation": observation,
                    "tool_trace": to_jsonable(recorder.trace[trace_size:]),
                    "spatial_memory": recorder.memory.to_dict(),
                    "truncated": False,
                }
            )
            diagnostic_turns.append(turn_record)

            if recorder.final_answer is not None:
                termination_reason = "model_final"
                break

            memory_text = recorder.memory.serialize_with_relations()
            if memory_text:
                observation = f"{observation}\n\n{memory_text}"
            new_view = None
            for trace_step in reversed(recorder.trace[trace_size:]):
                if trace_step["action_type"] == "Navigate" and (
                    trace_step.get("image_path_after") != trace_step.get("image_path_before")
                ):
                    new_view = trace_step.get("image_path_after")
                    break
            observation_message = {
                "role": "user",
                # There are deliberately no assistant messages in persistent
                # history. Previous Observations and their new camera views do
                # remain available to the next turn.
                "content": self._user_content(f"Observation: {observation}", new_view),
            }
            messages.append(observation_message)
            user_turns += 1

        if recorder.final_answer is None:
            if termination_reason == "unknown":
                termination_reason = (
                    "turn_limit" if assistant_turns >= self.max_assistant_turns else "response_budget"
                )
            forced_final = True
            # Forced-final generation has its own per-turn budget. Earlier
            # Thought-Code actions are not persistent context and therefore do
            # not consume the final turn's allowance.
            remaining_tokens = self.response_length
            generated_text = ""
            if remaining_tokens > 32:
                observation_excerpt = last_observation.strip()
                if len(observation_excerpt) > 800:
                    observation_excerpt = observation_excerpt[-800:]
                force_text = (
                    f"Observation: {observation_excerpt}\n\n" if observation_excerpt else ""
                ) + (
                    "The exploration budget is exhausted. Do not call another navigation or perception tool. "
                    "Using the evidence already collected, choose the best option A, B, C, or D now. Your Code "
                    "block MUST contain exactly one call such as final_answer(\"B\")."
                )
                if sample.get("answer_setting") == "open":
                    force_text = (
                        f"Observation: {observation_excerpt}\n\n"
                        "The exploration budget is exhausted. Do not call another navigation or perception tool. "
                        "Use the evidence already collected to give a concise natural-language answer. "
                        'Your Code block MUST contain exactly one call: final_answer("your answer"). '
                        "Do not output an option letter."
                    )
                force_message = {"role": "user", "content": force_text}
                force_messages = [*messages, force_message]
                vision = await self.process_vision_info(force_messages)
                prompt_ids = await self.apply_chat_template(
                    force_messages, images=vision.get("images"), videos=vision.get("videos")
                )
                if len(prompt_ids) > self.prompt_length:
                    # Preserve the task, initial view, latest evidence, and
                    # spatial memory when the full Observation history fills
                    # the model context window.
                    memory_text = recorder.memory.serialize_with_relations()
                    compact_force = {
                        "role": "user",
                        "content": (
                            f"Latest Observation: {last_observation[-800:]}\n\n{memory_text}\n\n"
                            "Context budget exhausted. Call final_answer with the best choice A, B, C, or D now."
                        ),
                    }
                    if sample.get("answer_setting") == "open":
                        compact_force["content"] = (
                            f"Latest Observation: {last_observation[-800:]}\n\n{memory_text}\n\n"
                            "Context budget exhausted. Call final_answer with a concise natural-language answer now."
                        )
                    force_messages = [messages[0], messages[1], compact_force]
                    vision = await self.process_vision_info(force_messages)
                    prompt_ids = await self.apply_chat_template(
                        force_messages, images=vision.get("images"), videos=vision.get("videos")
                    )
                if len(prompt_ids) <= self.prompt_length:
                    user_turns += 1
                    force_params = dict(sampling_params)
                    force_params["stop"] = ["<end_action>", "<end action>", "Observation:"]
                    force_params.pop("max_new_tokens", None)
                    force_params["max_tokens"] = min(
                        self.forced_final_max_tokens,
                        remaining_tokens,
                    )
                    with simple_timer("generate_sequences", metrics):
                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=prompt_ids,
                            sampling_params=force_params,
                            image_data=vision.get("images"),
                            video_data=vision.get("videos"),
                        )
                    preempted = output.num_preempted if output.num_preempted is not None else -1
                    metrics["num_preempted"] = max(int(metrics.get("num_preempted", -1)), 0) + max(preempted, 0)
                    assistant_turns += 1
                    generated_ids = output.token_ids[:remaining_tokens]
                    generated_text = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    )
                    generated_token_count += len(generated_ids)
                    record_training_turn(
                        prompt_ids,
                        generated_ids,
                        output.log_probs,
                        vision,
                        is_forced_final=True,
                    )

                    trace_size = len(recorder.trace)
                    forced_observation = ""
                    try:
                        forced_code = self._extract_code(generated_text)
                        forced_observation = self._execute_code(forced_code, executable, state)
                    except Exception as error:
                        if len(recorder.trace) == trace_size:
                            recorder.record_invalid_code(error, self._extract_code(generated_text))
                        forced_observation = f"Code execution failed: {type(error).__name__}: {error}"
                    diagnostic_turns.append(
                        {
                            "turn_index": len(diagnostic_turns),
                            "forced_final": True,
                            "prompt_tokens": len(prompt_ids),
                            "response_tokens": len(generated_ids),
                            "image_count": len(vision.get("images") or []),
                            "generated_text": generated_text,
                            "thought": self._extract_thought(generated_text),
                            "code": self._extract_code(generated_text),
                            "observation": forced_observation,
                            "tool_trace": to_jsonable(recorder.trace[trace_size:]),
                            "spatial_memory": recorder.memory.to_dict(),
                            "truncated": False,
                        }
                    )

            if recorder.final_answer is None:
                fallback_answer = self._best_effort_final_answer(generated_text, sample)
                executable["final_answer"](fallback_answer)
            if recorder.trace and recorder.trace[-1].get("action_type") == "FinalAnswer":
                recorder.trace[-1]["forced"] = True

        trajectory_id = uuid4().hex
        write_trajectory(
            {
                "schema_version": 1,
                "trajectory_id": trajectory_id,
                **trajectory_meta,
                "sample": sample,
                "task": task,
                "initial_image": initial_image,
                "initial_camera_state": to_jsonable(initial_camera_state),
                "turns": diagnostic_turns,
                "tool_trace": recorder.trace,
                "final_answer": recorder.final_answer,
                "spatial_memory": recorder.memory.to_dict(),
                "outcome": {
                    "forced_final": forced_final,
                    "termination_reason": termination_reason,
                    "truncated_normal_action": truncated_normal_action,
                    "response_tokens_used": generated_token_count,
                    "assistant_turns": assistant_turns,
                    "user_turns": user_turns,
                    "runtime_limits": {
                        "prompt_length": self.prompt_length,
                        "response_length": self.response_length,
                        "model_context_length": self.model_context_length,
                        "normal_turn_max_tokens": self.normal_turn_max_tokens,
                        "generation_protocol": "thought_code",
                    },
                },
            }
        )
        trace_path = trajectory_path(trajectory_id, trajectory_meta.get("global_step"))

        if selected_turn is None:
            # This can happen only when the configured generation budget is
            # smaller than the minimum turn. Keep the output structurally
            # valid while the missing-final reward makes it strictly bad.
            vision = await self.process_vision_info(messages)
            prompt_ids = await self.apply_chat_template(
                messages, images=vision.get("images"), videos=vision.get("videos")
            )
            selected_turn = {
                "turn_index": -1,
                "prompt_ids": prompt_ids[: self.prompt_length],
                "response_ids": [],
                "response_mask": [],
                "response_logprobs": None,
                "multi_modal_data": vision,
            }

        return AgentLoopOutput(
            prompt_ids=selected_turn["prompt_ids"],
            response_ids=selected_turn["response_ids"],
            response_mask=selected_turn.get(
                "response_mask", [1] * len(selected_turn["response_ids"])
            ),
            response_logprobs=selected_turn["response_logprobs"],
            multi_modal_data=selected_turn["multi_modal_data"],
            num_turns=user_turns + assistant_turns + 1,
            metrics=AgentLoopMetrics.model_validate(metrics),
            extra_fields={
                "tooleqa_trace": recorder.trace,
                "trajectory_id": trajectory_id,
                "global_step": trajectory_meta.get("global_step"),
                "trajectory_log_path": str(trace_path) if trace_path is not None else None,
                "tooleqa_final_answer": recorder.final_answer,
                "sample_info": sample,
                "spatial_memory": recorder.memory.to_dict(),
                "forced_final": forced_final,
                "termination_reason": termination_reason,
                "response_tokens_used": generated_token_count,
                "truncated_normal_action": truncated_normal_action,
                "generation_protocol": "thought_code",
                "training_turn_index": selected_turn["turn_index"],
                "trajectory_turn_count": len(turn_context_stats),
                "turn_context_stats": turn_context_stats,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )

    @staticmethod
    def _close_simulators(toolbox: list[Any]) -> None:
        for tool in toolbox:
            backend = getattr(tool, "eqa_modeling", None)
            simulator = getattr(backend, "simulator", None) if backend is not None else None
            if simulator is not None:
                try:
                    simulator.close()
                except Exception:
                    pass
                backend.simulator = None
