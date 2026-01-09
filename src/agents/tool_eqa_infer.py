from transformers.agents.python_interpreter import evaluate_python_code, LIST_SAFE_MODULES
from typing import Callable, List, Optional, Dict, Any
from transformers.agents.tools import Tool, DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from transformers.agents.agents import AgentGenerationError, AgentParsingError, AgentError, AgentMaxIterationsError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError
from transformers.agents import ReactCodeAgent
from transformers.agents.llm_engine import MessageRole
from src.tools.tool_box import get_tool_box
from src.llm_engine.qwen import QwenEngine
from src.llm_engine.gpt import GPTEngine

import random
import json
import jsonlines
import argparse
import multiprocessing as mp
import os
import time
import uuid
import ast

def save_data(data, path):
    with jsonlines.open(path, mode="a") as writer:
        writer.write(data)

class AgentToleranceError(AgentError):
    pass

class EQAReactAgent(ReactCodeAgent):
    def __init__(self, 
        tools: List[Tool],
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = LIST_SAFE_MODULES,
        planning_interval: int = None,
        error_tolerance_count: int = -1, 
        device: int = 0,
        **kwargs,
    ):
        super().__init__(tools=tools, 
                        llm_engine=llm_engine, 
                        system_prompt=system_prompt, 
                        tool_description_template=tool_description_template, 
                        additional_authorized_imports=additional_authorized_imports, 
                        planning_interval=planning_interval,
                        **kwargs
                        )
        self.image = []
        self.gpu_id = device
        self.error_tolerance_count = error_tolerance_count
        self.letter = ["A", "B", "C", "D"]
        self.thought = []

    def set_image_path(self, image, max_length=1):
        if image in self.image:
            self.image.remove(image)
        self.image.append(image)
        if len(self.image) > max_length:
            self.image = self.image[-max_length:]

    def evaluate_python_code_modify(
        self,
        code: str,
        static_tools: Optional[Dict[str, Callable]] = None,
        custom_tools: Optional[Dict[str, Callable]] = None,
        state: Optional[Dict[str, Any]] = None,
        authorized_imports: List[str] = None,
    ):
        print('authorized_imports', authorized_imports)
        result = evaluate_python_code(
            code,
            static_tools,
            custom_tools,
            state,
            authorized_imports
        )
        if state is not None and "print_outputs" in state and type(state["print_outputs"]) is str:
            state["print_outputs"] = state["print_outputs"] if len(state["print_outputs"]) > 0 else "No observation found from the code execution. You should use `print` function if need some information from the code execution."
        return result
    
    def direct_run(self, task: str):
        """
        Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.
        """
        final_answer = None
        iteration = 0
        error_count = 0
        
        # error_tolerance_count <= 0 disable this function
        while final_answer is None and iteration < self.max_iterations:
            if self.error_tolerance_count > 0 and error_count == self.error_tolerance_count:
                break
            try:
                if self.planning_interval is not None and iteration % self.planning_interval == 0 and iteration == 0:
                    self.planning_step(task, is_first_step=(iteration == 0), iteration=iteration)
                step_logs = self.step()
                if "final_answer" in step_logs:
                    final_answer = step_logs["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                self.logs[-1]["error"] = e
                error_count += 1
            finally:
                iteration += 1

        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
        elif final_answer is None and error_count == self.error_tolerance_count:
            error_message = f"Reached max execution exception. Max exception tolerance: {self.error_tolerance_count}."
            final_step_log = {"error": AgentToleranceError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer

        return final_answer
    
    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        if hasattr(self.toolbox._tools["GoNextPointTool"], "cur_rgb_path"):
            self.set_image_path(self.toolbox._tools["GoNextPointTool"].cur_rgb_path)

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"], image_paths=self.image)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output
        try:
            code_action = parse_code_blob(raw_code_action)
            print(raw_code_action)
            self.thought.append({"thought": rationale, "code": code_action})
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code. {raw_code_action}"
            raise AgentParsingError(error_msg)

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {"tool_name": "code interpreter", "tool_arguments": code_action}

        # Execute
        self.log_rationale_code_action(rationale, code_action)
        try:
            self.logger.info(f'authorized_imports {self.authorized_imports}')
            result = self.python_evaluator(
                code_action,
                static_tools={
                    **BASE_PYTHON_TOOLS.copy(),
                    **self.toolbox.tools,
                },
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            information = self.state["print_outputs"]
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information
        except Exception as e:
            error_msg = f"Code execution failed due to the following error:\n{str(e)}\ncode: {code_action}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += f"\nYou get this error because you passed a dict as input for one of the arguments instead of a string. "
            raise AgentExecutionError(error_msg)

        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result
        return current_step_logs

    def initialize_for_run(self, data):
        super().initialize_for_run()

        self.thought = []
        for tool_name in self.toolbox._tools:
            tool = self.toolbox._tools[tool_name]
            if hasattr(tool, "initialize"):
                tool.initialize(data)

        if "GoNextPointTool" in self.toolbox._tools:
            # Set the current image path if available
            if hasattr(self.toolbox._tools["GoNextPointTool"], "cur_rgb_path"):
                self.set_image_path(self.toolbox._tools["GoNextPointTool"].cur_rgb_path)
                max_explore_step = self.toolbox._tools["GoNextPointTool"].eqa_modeling.max_step
                self.max_iterations = 60 if max_explore_step > 60 else max_explore_step

    def run(self, data = None, reset: bool = True, **kwargs):
        oepn_vocab = kwargs.get("oepn_vocab", True)
        if oepn_vocab:
            self.task = data["question"]
        else:
            proposals = data["proposals"]
            if isinstance(proposals, str) and proposals != "":
                proposals = ast.literal_eval(proposals)
            self.task = data["question"] + str([f"{self.letter[i]}. {p}" for i, p in enumerate(proposals)])
        # if len(kwargs) > 0:
        #     self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        # self.state = kwargs.copy()
        self.state = {}
        if reset:
            self.initialize_for_run(data)
        else:
            self.logs.append({"task": self.task})

        return self.direct_run(self.task)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="config", type=str, default="./config/react-eqa.yaml")
    parser.add_argument("--scene", help="scene", type=str, default="00000-kfPV7w3FaU5")
    parser.add_argument("--init_pos", help="initial position", type=str, default="[4.446746, 0.18328714, 1.9121389]")
    parser.add_argument("--question", help="eqa question", type=str, default="In which corner of the room is the white radiator located?")
    parser.add_argument("--output", help="output direction", type=str, default="./results/inference/")
    parser.add_argument("--gpus", help="Comma-separated GPU IDs to use (e.g., '0,1,2')", type=str, default="0")
    args = parser.parse_args()

    # 设置当前进程可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    
    system_prompt = open("prompt/system_prompt.txt", "r").read()
    eqa_react_agent = EQAReactAgent(
        tools=get_tool_box(gpu_id=args.gpus, args=args),
        llm_engine=QwenEngine("/mynvme0/models/Qwen/Qwen2.5-VL-7B-Instruct", device=f"cuda:{args.gpus}"),
        # llm_engine=GPTEngine("gpt-4o"),
        system_prompt=system_prompt,
        add_base_tools=False,
        planning_interval=1,
        device=args.gpus
    )

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 基于时间的随机数作为sample_id
    data = {
        "question": args.question,
        "sample_id": str(uuid.uuid4()),
        "scene": args.scene,
        "init_pos": ast.literal_eval(args.init_pos),
        "summary": {
            "final_answer": "",
            "react": ""
        }
    }

    final_answer = eqa_react_agent.run(data, oepn_vocab=True)
    result = data.copy()
    result['summary'] = {}
    result['summary']['final_answer'] = final_answer
    result['summary']['react'] = eqa_react_agent.thought
    save_data(result, os.path.join(output_dir, f"output.jsonl"))
