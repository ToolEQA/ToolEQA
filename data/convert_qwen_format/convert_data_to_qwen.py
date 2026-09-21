"""Convert EQA-RT trajectories following the paper's controller-only SFT setup."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.memory.spatial_memory import SpatialMemory

SYSTEM_PROMPT = '''You are an expert embodied AI agent with the ability to perceive and interact with a virtual environment. You need to first explore the environment and collect information related to the problem, and when there is enough information, answer the question.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_action>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need. DO NOT generate a code which does not call 'print()' because you will lose this information. You can assume all tools must have a return that can be printed. 
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
You will save all intermediate file outputs to a folder by the relative path '.cache/'.
In the end you have to return a final answer using the `final_answer` tool. 

Here are a few examples using notional tools:

---
Task: "Is the height of the curtain framing the doorway near the pink toy storage unit greater than the height of the woven basket under a framed picture near the doorway?"

Thought: As I look around the room, I notice that directly to my left, there is a wooden door slightly ajar. The way it stands out with its rich, warm color suggests that it leads to another room, and doorways often indicate spaces where objects like curtains might be found, especially if the adjacent room has windows. In contrast, looking straight ahead presents more walls and appliances that are less likely to house the target object. \n\nGiven this specific clue, the opening of the door captures my attention because it hints at an unexplored area where the curtain might be positioned. Moving toward that space seems more promising than any other direction. Therefore, I choose to turn left.
Code: 
```py
path = GoNextPointTool("turn_left")
print(f'In this point, the current landscape is saved in {path}.')
```<end_action>
Observation: In this point, the current landscape is saved in ./cache/next_point_1.jpg.

Thought: As I survey the surroundings, I notice that the pathway ahead opens up into another room. The light peeking through the doorway and the clear space suggest that this area is likely more expansive, providing a better opportunity to locate the curtain. Additionally, I can see some furniture positioned against the walls, which may obstruct views in other directions. The walls in this area also appear to have a more distinct layout compared to the more closed-in feeling of the areas to my left and right.\n\nCrucially, the door frame in front of me leads into a space where I can make out more potential surfaces where a curtain might be hung, such as windows that may be present. In contrast, moving to the left or right only reveals walls and obscured spaces that may not contain the target object.\n\nConsidering the promising clues of light and spaciousness along with the likelihood of windows in the next room, I choose to move forward into this area, as it seems the most viable direction for finding the curtain.
Code: 
```py
path = GoNextPointTool("move_forward")
print(f'In this point, the current landscape is saved in {path}.')
```<end_action>
Observation: In this point, the current landscape is saved in ./cache/next_point_2.jpg.

Thought: Found the curtain object near the pink toy storage unit. I will locate it precisely and retrieve its bounding box to confirm its dimensions.
Code: 
```py
position, size = ObjectLocation3D(object='curtain', image_path='./cache/next_point_2.jpg')
print(f'The information of curtain is: position is {position},  size (Length, width, height) is {size}.')
```<end_action>
Observation: The information of curtain is: position is [5.503, 4.301, 4.479], size (Length, width, height) is [0.201, 1.83, 0.494].

Thought: Found the object stair. I will now locate it precisely to give its bounding box.
Code: 
```py
bbox = ObjectLocation2D(object='stair', image_path='/mynvme1/EQA-Traj-0720/syUktnn2RIWFX07abGVZLA/0-8.png')
print(f'The bounding box of stair is {bbox}.')
```<end_action>
Observation: The bounding box of stair is {'bboxes_2d': [[171, 266, 420, 639]], 'labels': ['stair'], 'text': 'stair'}.

Thought: Crop and save this objects region for further comparison.
Code: 
```py
path = ObjectCrop(bounding_box = [[171, 266, 420, 639]], image_path='/mynvme1/EQA-Traj-0720/syUktnn2RIWFX07abGVZLA/0-8.png')
print(f'The cropped result of stair is saved in {path}.')
```<end_action>
Observation: The cropped result of stair is saved in ['./cache/0-8_obj_0.jpg'].

Thought: Proceed to next point to find remaining objects.
Code:
```py
path = GoNextPointTool("turn_left")
print(f'In this point, the current landscape is saved in {path}.')
```<end_action>
Observation: In this point, the current landscape is saved in ./cache/next_point_3.jpg.

Thought: As I scan the area, I notice that in front of me through the doorway, the space seems to open up and gives off a sense of brightness compared to the other directions. There\u2019s a faint hint of color peeking from around the corner\u2014perhaps it\u2019s a piece of furniture or something that could be the target object that I\u2019m searching for. The walls in this direction are more exposed, and the path is clearer, enticing me to explore further. \n\nIn contrast, to my left and right, the paths are narrower and less inviting, with more doors that could lead to closed-off areas, making my search less efficient. The inviting glow and the potential view of something interesting ahead draw me in.\n\nTherefore, I choose to move_forward and investigate that promising space.
Code: 
```py
path = GoNextPointTool("move_forward")
print(f'In this point, the current landscape is saved in {path}.')
```<end_action>
Observation: In this point, the current landscape is saved in ./cache/next_point_4.jpg.

Thought: Found the object picture. I will locate it precisely and give its bounding box to confirm its dimensions.
Code: 
```py
position, size = ObjectLocation3D(object='picture', image_path='/mynvme1/EQA-Traj-0720/c_QsabMzTJu6pc3ZvLu9ZA/1-4.png')
print(f'The information of picture is: position is {position},  size (Length, width, height) is {size}.')
```<end_action>
Observation: The information of picture is: position is [7.724, 4.863, -0.12], size (Length, width, height) is [0.019, 0.47, 0.776].

Thought: The stair object has been processed and saved in ./cache/next_point_4.jpg. Now I will compare the saved object to answer the user's question about the railing material.
Code: 
```py
question = "What material is used for the railing on the staircase leading down to the entrance?"
answer = VisualQATool(question=question, images=["./cache/next_point_3.jpg", "./cache/next_point_4.jpg"])
print(question, answer)
```<end_action>
Observation: What material is used for the railing on the staircase leading down to the entrance?. The railing on the staircase leading down to the entrance appears to be made of metal.

Thought: The height of the curtain is given as 0.494, and the height of the picture is given as 0.776. Now we compare the heights: Height of curtain = 0.494. Height of picture = 0.776. Since 0.494 < 0.776, the curtain is indeed shorter than the picture.
Code: 
```py
final_answer(\"No, the picture is taller.\")
```<end_action>
Observation: No, the picture is taller.

Above example were using notional tools that might not exist for you. You ONLY have access to those tools, and do not use tools that have not been mentioned:

- VisualQATool: A tool that can answer questions about attached images.
    Takes inputs: {'question': {'description': 'the question to answer', 'type': 'string'}, 'image_paths': {'description': 'The path to the image on which to answer the question', 'type': 'string'}}
    Returns an output of type: string

- ObjectLocation2D: A tool that can localize objects in given images, outputing the bounding boxes of the objects.
    Takes inputs: {'object': {'description': 'the object that need to be localized', 'type': 'string'}, 'image_path': {'description': 'The path to the image on which to localize objects.', 'type': 'string'}}
    Returns an output of type: any

- ObjectLocation3D: Localize 3D objects in the scene and return their 3D bounding boxes and center coordinates.
    Takes inputs: {'object': {'description': 'the object that need to be localized', 'type': 'string'}, 'image_path': {'description': 'List of the pathes to the images on which to localize 3D objects.', 'type': 'string'}}
    Returns an output of type: any

- GoNextPointTool: the agent conitnue explore next point and obtain next observation (rgb image).
    Takes inputs: {'query': {'description': 'Next exploration direction, ONLY [move_forward, turn_left, turn_right] are supported.', 'type': 'string'}}
    Returns an output of type: image

- SegmentInstanceTool: A tool that can do instance segmentation on the given image.
    Takes inputs: {'image_path': {'description': 'The path of image that the tool can read.', 'type': 'string'}, 'prompt': {'description': "The bounding box that you want this model to segment. The bounding boxes could be from user input or tool `objectlocation`. You can set it as None or empty list to enable 'Segment Anything' mode.", 'type': 'any'}}
    Returns an output of type: string

- final_answer: Provides a final answer to the given problem.
    Takes inputs: {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    Returns an output of type: any

- ObjectCrop: Given the bounding boxes of objects, crop and save the relevant objects from the image.
    Takes inputs: {'bound_boxes': {'description': 'the bounding boxes of objects', 'type': 'number'}, 'image_path': {'description': 'The path to the image on which to crop objects.', 'type': 'string'}}
    Returns an output of type: string

You also can perform computations in the Python code that you generate.

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_action>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = final_answer({'answer': "Yes, it is."})', but use the arguments directly as in 'answer = final_answer(answer="Yes, it is.")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: <<authorized_imports>>
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.'''


def _has_final_answer_call(code):
    """Recognize terminal calls without treating strings/comments as calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Some stored snippets are not valid standalone Python. Fail closed
        # for terminal-looking snippets rather than supervising their answer.
        return bool(re.search(r"\b(?:final_answer|FinalAnswerTool)\s*\(", code))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in {"final_answer", "FinalAnswerTool"}:
                return True
    return False


def _format_action(react):
    return f"Thought: {react['thought']}\n\nCode:\n```py\n{react['code']}\n```<end_action>"


def convert_sample_to_qwen_style(sample):
    """Return one supervised controller response per nonterminal ReAct turn.

    History is user-side context so the existing Qwen label mask supervises
    only the current action, once. Plans and reference answers are not used.
    Memory is reconstructed from preceding observations, never future ones.
    The entire final-answer turn (including its thought) is excluded.
    """
    memory = SpatialMemory()
    history = []
    examples = []
    turn_index = 0
    for step_index, step in enumerate(sample["trajectory"]):
        for react_index, react in enumerate(step["react"]):
            if _has_final_answer_call(react["code"]):
                return examples
            image_path = str(step["image_path"]).strip()
            if not image_path:
                raise ValueError(f"Missing image for sample {sample['sample_id']}, step {step_index}")
            context = [f"<image>\nTask: {sample['question']}"]
            if history:
                context.append("[Interaction History]\n" + "\n\n".join(history))
            context.append(memory.serialize() or "[Spatial Memory]\nNo evidence collected yet.")
            context.append(f"Current observation image: {image_path}\nGenerate the next Thought and Code.")
            action = _format_action(react)
            examples.append({
                "id": f"{sample['sample_id']}:{step_index}:{react_index}",
                "source_id": sample["sample_id"],
                "image": [image_path],
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": "\n\n".join(context)},
                    {"from": "gpt", "value": action},
                ],
            })
            # Update only after constructing this turn's input/target pair.
            turn_index += 1
            observation = str(react.get("observation", ""))
            history.append(f"[Turn {turn_index}]\n{action}\nObservation:\n{observation}")
            memory.update_from_observation(react["code"], observation, turn_index)
    return examples


def iter_samples(path):
    """Read JSONL or stream a JSON array without loading all trajectories."""
    with Path(path).open(encoding="utf-8") as handle:
        if Path(path).suffix == ".jsonl":
            for line in handle:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buffer = ""
        started = False
        while True:
            chunk = handle.read(1024 * 1024)
            buffer += chunk
            cursor = 0
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if not started:
                    if cursor == len(buffer):
                        break
                    if buffer[cursor] != "[":
                        raise ValueError("Input must be a JSON array or a .jsonl file")
                    started = True
                    cursor += 1
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor == len(buffer):
                    break
                if buffer[cursor] == "]":
                    return
                try:
                    sample, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield sample
                cursor = end
            buffer = buffer[cursor:]
            if not chunk:
                raise ValueError("Unterminated input JSON array")


def main():
    parser = argparse.ArgumentParser(description="Convert EQA trajectories to controller-only SFT examples.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/ToolTrajectory/trainval.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("qwen_output_trainval.json"))
    parser.add_argument("--limit", type=int, help="Convert only this many source trajectories for inspection.")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_count = example_count = 0
    with args.output.open("w", encoding="utf-8") as output:
        output.write("[\n")
        for sample in iter_samples(args.input):
            for example in convert_sample_to_qwen_style(sample):
                if example_count:
                    output.write(",\n")
                json.dump(example, output, ensure_ascii=False)
                example_count += 1
            source_count += 1
            if args.limit is not None and source_count >= args.limit:
                break
        output.write("\n]\n")
    print(f"Converted {source_count} trajectories into {example_count} controller examples: {args.output}")


if __name__ == "__main__":
    main()
