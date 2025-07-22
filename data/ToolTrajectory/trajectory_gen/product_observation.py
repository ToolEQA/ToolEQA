import os
import json
from transformers.agents.python_interpreter import evaluate_python_code, LIST_SAFE_MODULES
from typing import Callable, List, Optional, Dict, Any
from transformers.agents.tools import Tool, DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from transformers.agents.agents import AgentGenerationError, AgentParsingError, AgentError, AgentMaxIterationsError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError
from transformers.agents import ReactCodeAgent
from src.tools.tool_box import get_tool_box
from src.llm_engine.qwen import QwenEngine
import re
import ast

class ObservationProductor():
    def __init__(self, data_path: str, tools: dict = None):
        self._tools = {tool.name: tool for tool in tools}
        self.data_path = data_path
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.state = {}
        self.templete = "```py\n{}\n```<end_action>"

    def init(self, data):
        for tool_name in self._tools:
            tool = self._tools[tool_name]
            if hasattr(tool, "initialize"):
                tool.initialize(data)


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
                state=self.state,
                authorized_imports=LIST_SAFE_MODULES
            )
            if "print_outputs" in self.state.keys():
                output = self.state["print_outputs"]
        except (AgentGenerationError, AgentParsingError, AgentExecutionError, AgentMaxIterationsError, AgentError) as e:
            print(f"Error evaluating code: {e}")
            return None
        
        for line in code.split("\n"):
            if line[: len("FinalAnswerTool")] == "FinalAnswerTool":
                return result
            if line[: len("final_answer")] == "final_answer":
                return result

        return output
    
    def run_sample(self, sample: dict):
        trajectory = sample["trajectory"]
        for traj in trajectory:
            react = traj["react"]
            for i in range(len(react)):
                code = react[i]["code"]
                if "ObjectCrop" in code:
                    continue
                if "Location3D" in code:
                    continue

                obs = self.run_code(code)
                # import pdb; pdb.set_trace()

                react[i]["observation"] = obs

                if "ObjectLocation2D" in code and i + 1< len(react):
                    pattern = r'\{.*\}'
                    match = re.search(pattern, obs)
                    if match:
                        dict_str = match.group()
                        bboxs = ast.literal_eval(dict_str)['bboxes_2d']
                    code = react[i+1]["code"]
                    if "ObjectCrop" in code:
                        code = code.replace("[x1,y1,x2,y2]", str(bboxs))
                        code = code.replace("[x1, y1, x2, y2]", str(bboxs))
                        obs = self.run_code(code)
                        react[i+1]["code"] = code
                        react[i+1]["observation"] = obs

        return sample

    def run(self):
        output = []
        for item in self.data:
            self.init(item)
            result = self.run_sample(item)
            output.append(result)
            
        return output

if __name__=="__main__":
    data_path = "tmp/data/size_output_ans_with_plan_nonkey-toolnormal.json"
    op = ObservationProductor(
        data_path,
        get_tool_box()
    )
    res = op.run()
    
    with open("tmp/data/size-obs.json", "w") as f:
        json.dump(res, f, indent=4)