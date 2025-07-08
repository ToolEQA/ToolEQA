from transformers import Tool

class FinalAnswerTool(Tool):
    name = "FinalAnswerTool"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer) -> str:
        return answer