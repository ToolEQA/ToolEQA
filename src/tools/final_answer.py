from transformers import Tool

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.debug = kwargs.get("debug", False)
        if self.debug:
            return

    def forward(self, answer=None, **kwargs) -> str:
        if answer is None:
            answer = kwargs.get("final_answer", None)
        if isinstance(answer, dict):
            if 'answer' in answer:
                answer = answer['answer']
        elif isinstance(answer, str):
            answer = answer
        return answer