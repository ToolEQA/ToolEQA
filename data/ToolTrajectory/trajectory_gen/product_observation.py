import os
import json
from transformers.agents.python_interpreter import evaluate_python_code, LIST_SAFE_MODULES
from typing import Callable, List, Optional, Dict, Any
from transformers.agents.tools import Tool, DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from transformers.agents.agents import AgentGenerationError, AgentParsingError, AgentError, AgentMaxIterationsError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError
from transformers.agents import ReactCodeAgent
from src.tools.tool_box import get_tool_box
from src.llm_engine.qwen import QwenEngine

class ObservationProductor():
    def __init__(self, data_path: str, tools: dict = None):
        self._tools = {tool.name: tool for tool in tools}
        self.data_path = data_path
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.templete = "```py\n{}\n```<end_action>"

    def run_code(self, raw_code: str):
        if not raw_code:
            return None
        try:
            raw_code = self.templete.format(raw_code)
            code = parse_code_blob(raw_code)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)
        
        if not code:
            return None
        
        try:
            result = evaluate_python_code(
                code,
                static_tools={
                    **BASE_PYTHON_TOOLS.copy(),
                    **self._tools,
                },
                custom_tools={},
                state=None,
                authorized_imports=LIST_SAFE_MODULES
            )
        except (AgentGenerationError, AgentParsingError, AgentExecutionError, AgentMaxIterationsError, AgentError) as e:
            print(f"Error evaluating code: {e}")
            return None
        return result
    
    def run_sample(self, sample: dict):
        trajectory = sample["trajectory"]
        for traj in trajectory:
            react = traj["react"]
            for i in range(len(react)):
                code = react[i]["code"]
                # if "turn_left" in code:
                #     code = code.replace("turn_left", "\"turn_left\"")
                # elif "turn_right" in code:
                #     code = code.replace("turn_right", "\"turn_right\"")
                # elif "move_forward" in code:
                #     code = code.replace("move_forward", "\"move_forward\"")

                if "ObjectCrop" in code:
                    continue
                
                obs = self.run_code(code)
                react[i]["observation"] = obs

                if "ObjectLocation3D" in code and i + 1< len(react):
                    code = react[i+1]["code"]
                    obs = self.run_code(code)
                    react[i+1]["observation"] = obs

        return sample

    def run(self):
        output = []
        for item in self.data:
            result = self.run_sample(item)
            output.append(result)
            break
        return output


if __name__=="__main__":
    data_path = "tmp/size_output_ans_with_plan_3.json"
    op = ObservationProductor(
        data_path,
        get_tool_box(debug=True)
    )
    res = op.run()
    print(res[0]["trajectory"]["react"][0]["observation"])