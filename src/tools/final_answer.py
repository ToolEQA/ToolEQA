from transformers import Tool

class FinalAnswerTool(Tool):
    name = "FinalAnswerTool"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = kwargs.get("debug", False)
        if self.debug:
            return

    def forward(self, answer) -> str:
        if self.debug:
            return "This is a debug answer."
        
        return answer