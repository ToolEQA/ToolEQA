from __future__ import annotations

import asyncio
import json
import os
import re
import time
from argparse import Namespace
from io import StringIO
from typing import Any, Callable
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, FunctionCall, ToolAgentLoop
from verl.utils.profiler import simple_timer

from src.tools.tool_box import get_tool_box, show_tool_descriptions


@register("tooleqa_tool_agent")
class ToolEQAToolAgentLoop(ToolAgentLoop):
    """Thin extension of verl's built-in ToolAgentLoop."""

    _action_json_pattern = re.compile(r"\{[\s\S]*\"action_type\"[\s\S]*\}")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accumulated_grid_thw: list[torch.Tensor] = []
        self._accumulated_pixel_values: list[torch.Tensor] = []

    def _reset_vision_accumulators(self):
        self._accumulated_grid_thw = []
        self._accumulated_pixel_values = []

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        prompt_ids = await super().apply_chat_template(
            messages, tools=tools, images=images, videos=videos,
            remove_system_prompt=remove_system_prompt,
        )
        # Capture initial-template vision inputs immediately (always accepted).
        # Observation-template grids are captured in _handle_processing_tools_state
        # only after the overflow check passes.
        if not remove_system_prompt:
            vis = self._last_vision_inputs
            if vis.get("image_grid_thw") is not None:
                self._accumulated_grid_thw.append(vis["image_grid_thw"])
            if vis.get("pixel_values") is not None:
                self._accumulated_pixel_values.append(vis["pixel_values"])
        return prompt_ids

    def _extract_action_json_fallback(self, text: str) -> list[FunctionCall]:
        """Parse ToolEQA-style controller JSON into tool calls.

        The dataset prompt asks the model to emit:
        {"action_type": "...", "args": {...}}
        which is not the same as verl's native <tool_call> format. When the
        model follows the dataset instruction, convert that JSON directly into a
        FunctionCall so the normal tool execution path still runs.
        """
        if not text:
            return []

        match = self._action_json_pattern.search(text)
        if match is None:
            return []

        try:
            payload = json.loads(match.group(0))
        except Exception:
            return []

        action_type = payload.get("action_type")
        args = payload.get("args", {})
        if not isinstance(action_type, str) or not isinstance(args, dict):
            return []

        return [FunctionCall(name=action_type, arguments=json.dumps(args, ensure_ascii=False))]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        self._reset_vision_accumulators()
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        extra_info = kwargs.get("extra_info", {}) or {}

        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = extra_info["interaction_kwargs"]
            interaction_name = interaction_kwargs["name"]
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        agent_data.extra_fields["sample_info"] = dict(extra_info)

        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        # Store precomputed vision inputs so _compute_multi_modal_inputs
        # can skip text-based recomputation (which causes image_grid_thw
        # mismatches for multi-turn Qwen2.5-VL sequences).
        if self._accumulated_grid_thw:
            multi_modal_data["_image_grid_thw"] = torch.cat(self._accumulated_grid_thw, dim=0)
            images_seqlens = torch.repeat_interleave(
                multi_modal_data["_image_grid_thw"][:, 1] * multi_modal_data["_image_grid_thw"][:, 2],
                multi_modal_data["_image_grid_thw"][:, 0],
            )
            multi_modal_data["_images_seqlens"] = images_seqlens
        if self._accumulated_pixel_values:
            multi_modal_data["_pixel_values"] = torch.cat(self._accumulated_pixel_values, dim=0)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        result = await super()._handle_processing_tools_state(agent_data)
        if result == AgentState.GENERATING:
            # Observation was accepted (no overflow). Capture the vision inputs
            # that apply_chat_template computed but did not auto-capture because
            # remove_system_prompt=True.
            vis = self._last_vision_inputs
            if vis.get("image_grid_thw") is not None:
                self._accumulated_grid_thw.append(vis["image_grid_thw"])
            if vis.get("pixel_values") is not None:
                self._accumulated_pixel_values.append(vis["pixel_values"])
        return result

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        next_state = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        if next_state == AgentState.TERMINATED and not agent_data.tool_calls:
            text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            fallback_calls = self._extract_action_json_fallback(text)
            if fallback_calls:
                agent_data.tool_calls = fallback_calls
                return AgentState.PROCESSING_TOOLS
        return next_state


class _RecordedTool:
    def __init__(self, name: str, fn: Callable[..., Any], recorder: "_ToolExecutionRecorder"):
        self.name = name
        self.fn = fn
        self.recorder = recorder

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.fn(*args, **kwargs)
        self.recorder.record(self.name, args, kwargs, result, time.perf_counter() - start)
        return result


class _ToolExecutionRecorder:
    def __init__(self, toolbox: list[Any]):
        self.toolbox = toolbox
        self.trace: list[dict[str, Any]] = []
        self.final_answer: Any = None

    def _normalize_args(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if name == "GoNextPointTool":
            direction = kwargs.get("direction")
            if direction is None and args:
                direction = args[0]
            return "Navigate", {"direction": direction}
        if name == "ObjectLocation2D":
            return "ObjectLocation2D", {
                "object": kwargs.get("object", args[0] if args else None),
                "image_path": kwargs.get("image_path", args[1] if len(args) > 1 else None),
            }
        if name == "ObjectLocation3D":
            return "ObjectLocation3D", {
                "object": kwargs.get("object", args[0] if args else None),
                "image_path": kwargs.get("image_path", args[1] if len(args) > 1 else None),
            }
        if name == "ObjectCrop":
            return "ObjectCrop", {
                "bounding_box": kwargs.get("bounding_box", args[0] if args else None),
                "image_path": kwargs.get("image_path", args[1] if len(args) > 1 else None),
            }
        if name == "VisualQATool":
            return "VisualQA", {
                "question": kwargs.get("question", args[0] if args else None),
                "image_path": kwargs.get("image_path", ""),
                "image_paths": kwargs.get("image_paths", ""),
            }
        if name in {"FinalAnswerTool", "final_answer"}:
            return "FinalAnswer", {"answer": kwargs.get("answer", args[0] if args else None)}
        return name, {"args": list(args), **kwargs}

    def record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any, latency: float) -> None:
        action_type, normalized_args = self._normalize_args(name, args, kwargs)
        if action_type == "FinalAnswer":
            self.final_answer = result
        self.trace.append(
            {
                "action_type": action_type,
                "args": normalized_args,
                "result": result,
                "latency_s": latency,
            }
        )


@register("tooleqa_react_code_agent")
class ToolEQAReactCodeAgentLoop(AgentLoopBase):
    _code_block_pattern = re.compile(r"Code:\s*```(?:py|python)?\n(.*?)```", re.DOTALL)
    _fallback_code_pattern = re.compile(r"```(?:py|python)?\n(.*?)```", re.DOTALL)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self._shared_toolbox: list[Any] | None = None
        self._tool_gpu_id: int | None = None
        self._accumulated_grid_thw: list[torch.Tensor] = []
        self._accumulated_pixel_values: list[torch.Tensor] = []
        prompt_path = os.path.join(os.getcwd(), "data/ToolTrajectory/prompts/react_system_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.react_system_prompt_template = f.read()

    def _reset_vision_accumulators(self):
        self._accumulated_grid_thw = []
        self._accumulated_pixel_values = []

    def _capture_vision_inputs(self):
        """Store the last vision inputs from apply_chat_template."""
        vis = self._last_vision_inputs
        if vis.get("image_grid_thw") is not None:
            self._accumulated_grid_thw.append(vis["image_grid_thw"])
        if vis.get("pixel_values") is not None:
            self._accumulated_pixel_values.append(vis["pixel_values"])

    def _resolve_tool_gpu_id(self) -> int:
        if self._tool_gpu_id is not None:
            return self._tool_gpu_id

        configured = self.rollout_config.agent.get("tool_gpu_id", None)
        if configured is None:
            configured = os.environ.get("TOOLEQA_TOOL_GPU_ID", "0")
        self._tool_gpu_id = int(configured)
        return self._tool_gpu_id

    def _build_toolbox(self, sample_info: dict[str, Any]) -> tuple[list[Any], dict[str, Callable[..., Any]]]:
        gpu_id = self._resolve_tool_gpu_id()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices.strip() in {"", "-1"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if self._shared_toolbox is None:
            args = Namespace(cfg="config/react-eqa.yaml", open_vocab=True)
            self._shared_toolbox = get_tool_box(debug=False, gpu_id=gpu_id, args=args)
        toolbox = self._shared_toolbox
        for tool in toolbox:
            if hasattr(tool, "initialize"):
                tool.initialize(sample_info)

        recorder = _ToolExecutionRecorder(toolbox)
        exec_tools: dict[str, Callable[..., Any]] = {}
        for tool in toolbox:
            exec_tools[tool.name] = _RecordedTool(tool.name, tool, recorder)
        return toolbox, exec_tools | {"_recorder": recorder}

    def _build_task(self, raw_prompt: list[dict[str, Any]], extra_info: dict[str, Any]) -> str:
        question = extra_info.get("question")
        if not question and raw_prompt:
            question = raw_prompt[0].get("content", "")
        plan = extra_info.get("plan", "")
        proposals = extra_info.get("proposals") or []
        lines = [str(question).strip()]
        if plan:
            lines.extend(["", str(plan).strip()])
        if proposals:
            lines.extend(["", "Choices:"])
            for idx, proposal in enumerate(proposals):
                lines.append(f"{chr(ord('A') + idx)}. {proposal}")
        return "\n".join(lines).strip()

    def _build_system_prompt(self, toolbox: list[Any]) -> str:
        tool_desc = show_tool_descriptions(toolbox)
        return (
            self.react_system_prompt_template.replace("<<tool_descriptions>>", tool_desc).replace(
                "<<authorized_imports>>", "math, json, re"
            )
        )

    def _extract_code(self, text: str) -> str:
        match = self._code_block_pattern.search(text) or self._fallback_code_pattern.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        allowed = {"math", "json", "re"}
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(f"Import '{name}' is not allowed.")
        return __import__(name, globals, locals, fromlist, level)

    def _execute_code(
        self,
        code: str,
        exec_tools: dict[str, Callable[..., Any]],
        state: dict[str, Any],
    ) -> str:
        stdout = StringIO()

        def captured_print(*args, **kwargs):
            print(*args, file=stdout, **kwargs)

        tool_names = {k for k in exec_tools.keys() if not k.startswith("_")}
        safe_builtins = {
            "__import__": self._safe_import,
            "print": captured_print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
        }
        env = {"__builtins__": safe_builtins, **{k: v for k, v in state.items() if k not in tool_names}, **exec_tools}
        exec(code, env, env)
        for key, value in env.items():
            if key.startswith("__") or key in tool_names or key == "_recorder":
                continue
            state[key] = value
        observation = stdout.getvalue().strip()
        return observation or "No observation found from the code execution. You should use print() to expose tool results."

    def _current_image(self, toolbox: list[Any]) -> str | None:
        for tool in toolbox:
            if getattr(tool, "name", "") == "GoNextPointTool":
                return getattr(tool, "cur_rgb_path", None)
        return None

    def _build_user_content(self, text: str, image_path: str | None):
        if image_path:
            return [{"type": "image", "image": image_path}, {"type": "text", "text": text}]
        return text

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        extra_info = dict(kwargs.get("extra_info", {}) or {})
        toolbox, exec_tools = self._build_toolbox(extra_info)
        recorder: _ToolExecutionRecorder = exec_tools["_recorder"]
        try:
            return await self._run_inner(
                toolbox, exec_tools, recorder, raw_prompt, extra_info, sampling_params
            )
        finally:
            self._close_simulators(toolbox)

    async def _run_inner(
        self,
        toolbox: list[Any],
        exec_tools: dict[str, Any],
        recorder: Any,
        raw_prompt: list[dict[str, Any]],
        extra_info: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:
        self._reset_vision_accumulators()
        task = self._build_task(raw_prompt, extra_info)
        system_prompt = self._build_system_prompt(toolbox)
        initial_image = self._current_image(toolbox)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_user_content(task, initial_image)},
        ]

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        initial_prompt_ids = await self.apply_chat_template(
            messages, images=images, videos=videos
        )
        self._capture_vision_inputs()  # grid for initial prompt images

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        state: dict[str, Any] = {}
        assistant_turns = 0
        user_turns = 0

        while len(response_mask) < self.response_length and assistant_turns < self.max_assistant_turns:
            images = (await self.process_vision_info(messages)).get("images")
            videos = (await self.process_vision_info(messages)).get("videos")
            prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)
            step_sampling_params = dict(sampling_params)
            step_sampling_params["stop"] = ["<end_action>", "Observation:"]
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=step_sampling_params,
                    image_data=images,
                    video_data=videos,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            assistant_turns += 1
            generated_ids = output.token_ids
            generated_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            messages.append({"role": "assistant", "content": generated_text})
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if output.log_probs:
                response_logprobs.extend(output.log_probs)

            code = self._extract_code(generated_text)
            try:
                observation = self._execute_code(code, exec_tools, state)
            except Exception as e:
                observation = f"Code execution failed due to the following error:\n{e}"

            current_image = self._current_image(toolbox)
            observation_message = {
                "role": "user",
                "content": self._build_user_content(f"Observation: {observation}", current_image),
            }
            obs_mm = await self.process_vision_info([observation_message])
            obs_ids = await self.apply_chat_template(
                [observation_message],
                images=obs_mm.get("images"),
                videos=obs_mm.get("videos"),
                remove_system_prompt=True,
            )
            # If appending this observation would overflow response_length,
            # skip it entirely. Otherwise response truncation at the end of
            # the loop can slice through image tokens.
            if len(response_mask) + len(obs_ids) > self.response_length:
                break
            if obs_mm.get("images"):
                self._capture_vision_inputs()  # grid for observation images
            if obs_mm.get("images"):
                multi_modal_data.setdefault("images", []).extend(obs_mm["images"])
            if obs_mm.get("videos"):
                multi_modal_data.setdefault("videos", []).extend(obs_mm["videos"])
            messages.append(observation_message)
            response_ids.extend(obs_ids)
            response_mask.extend([0] * len(obs_ids))
            response_logprobs.extend([0.0] * len(obs_ids))
            user_turns += 1

            if recorder.final_answer is not None:
                break

        # Store precomputed vision inputs so _compute_multi_modal_inputs
        # can skip text-based recomputation (which causes image_grid_thw
        # mismatches for multi-turn Qwen2.5-VL sequences).
        if self._accumulated_grid_thw:
            multi_modal_data["_image_grid_thw"] = torch.cat(self._accumulated_grid_thw, dim=0)
            images_seqlens = torch.repeat_interleave(
                multi_modal_data["_image_grid_thw"][:, 1] * multi_modal_data["_image_grid_thw"][:, 2],
                multi_modal_data["_image_grid_thw"][:, 0],
            )
            multi_modal_data["_images_seqlens"] = images_seqlens
        if self._accumulated_pixel_values:
            multi_modal_data["_pixel_values"] = torch.cat(self._accumulated_pixel_values, dim=0)

        extra_fields = {
            "tooleqa_trace": recorder.trace,
            "tooleqa_final_answer": recorder.final_answer,
            "sample_info": extra_info,
            "turn_scores": [],
            "tool_rewards": [],
        }
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            num_turns=user_turns + assistant_turns + 1,
            metrics=AgentLoopMetrics.model_validate(metrics),
            extra_fields=extra_fields,
        )

    def _close_simulators(self, toolbox: list[Any]) -> None:
        """Explicitly close Habitat simulators in the current (EGL-enabled) thread.

        Without this, the simulators would be closed during GC in an arbitrary
        thread that may lack an OpenGL context, causing SIGABRT.
        """
        for tool in toolbox:
            eqa = getattr(tool, "eqa_modeling", None)
            if eqa is None:
                continue
            sim = getattr(eqa, "simulator", None)
            if sim is not None:
                try:
                    sim.close()
                except Exception:
                    pass
                eqa.simulator = None
