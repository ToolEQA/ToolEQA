from transformers.agents.python_interpreter import evaluate_python_code, LIST_SAFE_MODULES
from typing import Callable, List, Optional, Dict, Any
from transformers.agents.tools import Tool, DEFAULT_TOOL_DESCRIPTION_TEMPLATE
from transformers.agents.agents import AgentGenerationError, AgentParsingError, AgentError, AgentMaxIterationsError, parse_code_blob, BASE_PYTHON_TOOLS, AgentExecutionError
from transformers.agents import ReactCodeAgent
from src.tools.tool_box import get_tool_box
from src.llm_engine.qwen import QwenEngine

class AgentToleranceError(AgentError):
    pass

class EQAReactAgent(ReactCodeAgent):
    def __init__(self, 
        tools: List[Tool],
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        additional_authorized_imports: Optional[List[str]] = LIST_SAFE_MODULES,
        planning_interval: int = 1,
        error_tolerance_count: int = -1, 
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
        self.image = None
        self.error_tolerance_count = error_tolerance_count
        
    def set_image_path(self, image):
        self.image = image

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
                if self.planning_interval is not None and iteration % self.planning_interval == 0:
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
        print(final_answer)

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
        import pdb; pdb.set_trace()
        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
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
            error_msg = f"Code execution failed due to the following error:\n{str(e)}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result
        return current_step_logs
    

if __name__=="__main__":
    system_prompt = open("data/ToolTrajectory/prompts/react_system_prompt.txt", "r").read()
    eqa_react_agent = EQAReactAgent(get_tool_box(),
                  QwenEngine("/mynvme0/models/Qwen2-VL/Qwen2-VL-7B-Instruct/"),
                  system_prompt,
                  )
    print(eqa_react_agent.run("如何使用微波炉？"))