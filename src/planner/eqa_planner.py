from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


PLAN_SYSTEM_PROMPT = """You are an assistant that generates observation plans to answer questions about objects in a 3D indoor scene.

Given a question, your task is to:
1. Identify the key objects mentioned in the question.
2. Infer which room or area each object is likely located in.
3. Generate a concise, step-by-step plan to locate and observe the relevant objects.

Each step should describe a concrete exploration action (e.g., go to a room, find an object, check a property).
Keep the plan to 2-5 steps. Be specific about what to observe at each step.

Output format:
Plan:
1. <step 1>
2. <step 2>
..."""


class EQAPlanner:
    """Independent LLM planner that decomposes EQA questions into structured sub-goal plans.

    At inference time, generates a plan from the question using an LLM.
    At training time, plans are loaded directly from the training data.
    """

    def __init__(self, llm_engine: Optional[Callable] = None) -> None:
        self.llm_engine = llm_engine

    def generate_plan(self, question: str) -> str:
        """Generate a structured plan using the LLM engine.

        Args:
            question: The EQA question to plan for.

        Returns:
            A structured plan string starting with "Plan:".
        """
        if self.llm_engine is None:
            raise RuntimeError("EQAPlanner requires an llm_engine to generate plans.")

        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nGenerate a plan:"},
        ]
        response = self.llm_engine(messages, stop_sequences=["<end_action>"])
        return response.strip()

    @staticmethod
    def format_plan_for_prompt(plan_text: str) -> str:
        """Format a plan string for insertion into the task prompt.

        Handles both raw plan text and plan text already prefixed with 'Plan:'.
        """
        plan_text = plan_text.strip()
        if not plan_text:
            return ""
        if not plan_text.startswith("Plan:"):
            plan_text = f"Plan:\n{plan_text}"
        return plan_text

    @staticmethod
    def load_plan_from_data(data: Dict[str, Any]) -> str:
        """Extract and format the plan from a training data sample."""
        plan = data.get("plan", "")
        return EQAPlanner.format_plan_for_prompt(plan)
