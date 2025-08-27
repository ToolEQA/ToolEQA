import json
import os
import re
import copy
import random
from tqdm import tqdm

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

SYSTEM_PROMPT_FACTS = """Below I will present you a task.

You will now build a comprehensive preparatory survey of which facts we have at our disposal and which ones we still need.
To do so, you will have to read the task and identify things that must be discovered in order to successfully complete it.
Don't make any assumptions. For each item, provide a thorough reasoning. Here is how you will structure this survey:

---
### 1. Facts given in the task
List here the specific facts given in the task that could help you (there might be nothing here).

### 2. Facts to look up
List here any facts that we may need to look up.
Also list where and how to find each of these infomation, for instance living room, bed room... - maybe the task contains some sources that you should re-use here.

### 3. Facts to derive
List here anything that we want to derive from the above by logical reasoning, for instance computation or simulation.

Keep in mind that "facts" will typically be specific names, dates, values, etc. Your answer should use the below headings:
### 1. Facts given in the task
### 2. Facts to look up
### 3. Facts to derive
Do not add anything else."""

import re
import os

_IMG_RE = re.compile(r'(?P<path>(?:\.?/)?[A-Za-z0-9_\-./]+?\.(?:png|jpg|jpeg|webp|bmp))', re.I)

def str_to_bool(s: str) -> bool:
    return s.strip().lower() == "true"

def _pick_image_for_react(obs_text):
    if not obs_text:
        return None

    cands = [m.group('path') for m in _IMG_RE.finditer(obs_text)]
    if not cands:
        return None

    norm = lambda p: p.replace("\\", "/")

    non_cache = [norm(p) for p in cands if "cache/" not in norm(p)]
    chosen = (non_cache or [norm(p) for p in cands])[-1]

    # 归一化
    chosen = chosen.lstrip("./")
    if os.path.isabs(chosen) or re.match(r'^[A-Za-z]:[\\/]', chosen):
        return chosen

    if not chosen.startswith("data/"):
        chosen = f"data/{chosen}"
    return chosen


def is_success(threshold: float) -> bool:
    """
    给定阈值，随机生成一个 [0,1) 之间的浮点数，
    如果小于阈值则判断为成功。
    
    参数：
        threshold (float): 阈值，范围在 [0,1] 之间
    返回：
        bool: 是否成功
    """
    rand_val = random.random()  # 生成 [0,1) 之间的随机浮点数
    return rand_val < threshold

def convert_sample_to_qwen_style(sample):
    """
        一条样本数据可以产生多个多轮对话
        1. plan
        2. 关键步骤（每一个关键步骤都将产生一个多轮对话，上下文为之前的所有关键步骤信息）
        3. 非关键步骤（从非关键步骤中采样与关键步骤相同数量的非关键步骤作为一个多轮对话）
    """
    question = sample["question"]

    sample_conversations = []

    # # first: plan
    # conversations = []
    # conversations.append({"from": "system", "value": SYSTEM_PROMPT_FACTS})
    # conversations.append({"from": "human", "value": f"Here is the task:```{question}```Now begin!"})
    # conversations.append({"from": "gpt", "value": sample["plan"]})

    # sample_conversations.append(
    #     {
    #         "id": sample["sample_id"],
    #         "conversations": conversations
    #     }
    # )
    
    # second: 关键步骤
    conversations = []
    conversations.append({"from": "system", "value": SYSTEM_PROMPT})
    conversations.append({"from": "human", "value": f"<image>\nTask:{question}"})

    need_key = False
    for step_idx, step in enumerate(sample["trajectory"]):
        step_id = step["step"]
        img_path = step["image_path"]
        is_key = str_to_bool(step["is_key"])

        if not is_key and not need_key:
            if is_success(0.5):
                need_key = True
                react = step["react"][-1]
                code_block = f"Thought: {react['thought']}\n\nCode:\n```py\n{react['code']}\n```<end_action>"
                conversations.append({"from": "gpt", "value": code_block})

                user_content = f"Observation:\n{react['observation']}\n\n"
                conversations.append({"from": "human", "value": user_content})

                sample_conversations.append(
                    {
                        "id": sample["sample_id"],
                        "image": [img_path],
                        "conversations": copy.deepcopy(conversations[:-1])
                    }
                )

        elif is_key:
            need_key = False
            for react_idx, react in enumerate(step["react"]):
                code_block = f"Thought: {react['thought']}\n\nCode:\n```py\n{react['code']}\n```<end_action>"
                conversations.append({"from": "gpt", "value": code_block})

                # if step_idx != len(sample["trajectory"]) - 1 or react_idx != len(step["react"]) - 1:
                user_content = f"Observation:\n{react['observation']}\n\n"
                conversations.append({"from": "human", "value": user_content})

                sample_conversations.append(
                    {
                        "id": sample["sample_id"],
                        "image": [img_path],
                        "conversations": copy.deepcopy(conversations[:-1])
                    }
                )

    return sample_conversations

if __name__ == "__main__":
    with open("/home/zml/data/EQA-Traj-0720/trainval.json", "r", encoding="utf-8") as f:
        input_samples = json.load(f)
    output_sample = []
    for sample in tqdm(input_samples):
        output_sample.extend(convert_sample_to_qwen_style(sample))

    with open("qwen_output_trainval.json", "w", encoding="utf-8") as f:
        json.dump(output_sample, f, ensure_ascii=False)

    print("Done")