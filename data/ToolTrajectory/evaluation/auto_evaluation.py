import os
import json
import jsonlines
from tqdm import tqdm
from data.ToolTrajectory.generator_deerapi import requests_api

class Evaluation():
    def __init__(self, args, image_root, prompt_pathes):
        self.args = args
        self.image_root = image_root
        self.data = []
        # for data_path in args.file:
        with jsonlines.open(args.file, 'r') as reader:
            self.data = [item for item in reader]
        prompt_pathes = {k: open(v).read() for k, v in prompt_pathes.items()}
        self.__dict__.update(prompt_pathes)

    def _str_to_bool(self, s: str) -> bool:
        return s.strip().lower() in ("true", "1", "yes", "y", "t")

    def reasonable(self, item):
        # 3. 最终答案应当由 thought 和 observation 推理得到
        trajectory = item['trajectory']

        choice_letter = ['A', 'B', 'C', 'D']
        question = item['question']
        answer = item['answer']
        proposals = item['proposals']
        gt_answer = proposals[choice_letter.index(answer)]

        context = ""
        for i, step_i in enumerate(trajectory):
            if not self._str_to_bool(step_i['is_key']):
                continue
            for react_i in enumerate(step_i['react']):
                obs = react_i['observation']
                thought = react_i['thought']
                context += f"[STEP {i}] Thought: {thought}\nObservation: {obs}\n"

        prompt = self.prompt_reasonable
        prompt = prompt.replace("<<context>>", context)
        prompt = prompt.replace("<<question>>", question)
        prompt = prompt.replace("<<answer>>", gt_answer)

        response = requests_api(None, prompt)
        if response.lower() == "yes":
            return True
        else:
            return False

    def dehallucination(self, item):
        # 2. 非关键步骤去除幻觉（图片和thought的内容对应）
        trajectory = item['trajectory']
        result = []
        related_obj = item['related_objects']
        for i, step_i in enumerate(trajectory):
            step_id = step_i["step"]
            if self._str_to_bool(step_i['is_key']):
                continue
            image_path = os.path.join(self.image_root, step_i['image_path'])
            for react_i in step_i['react']:
                thought = react_i['thought']
                input_prompt = self.prompt_dehuallucination
                input_prompt = input_prompt.replace("<<thought>>", thought)
                response = requests_api(image_path, input_prompt)
                result.append(int(response))
                
                if float(response) <= 3.0:
                    obj_idx = int(step_id.split("-")[0])
                    obj = related_obj[obj_idx]['name']

                    if len(step_i['action']) > 0:
                        direct = step_i['action'][0][0]
                    else:
                        continue
                    
                    nonkey_thought = self.prompt_nonkey_thought
                    nonkey_thought = nonkey_thought.replace("<<object_name>>", obj)
                    nonkey_thought = nonkey_thought.replace("<<action_name>>", direct)
                    response = requests_api(None, nonkey_thought)
                    react_i['thought'] = response
                    # print(f"Revise thought: {thought} --> {response}")
        
        return item

    def answer_consist(self, item):
        # 1. 答案和真实答案是否一致（语义上）。
        """
        Evaluate if the answers are consistent based on the provided prompt.
        
        Args:
            data (list): List of dictionaries containing answers to evaluate.
            prompt (str): The prompt to use for evaluation.
            
        Returns:
            list: A list of dictionaries with evaluation results.
        """
        choice_letter = ['A', 'B', 'C', 'D']
        question = item['question']
        answer = item['answer']
        proposals = item['proposals']
        gt_answer = proposals[choice_letter.index(answer)]

        # get final answer
        final_code = item['trajectory'][-1]['react'][-1]['code']
        if "final_answer" in final_code:
            final_answer = item['trajectory'][-1]['react'][-1]['observation']
        else:
            return False

        if final_answer.lower() == gt_answer.lower():
            return True

        # prompt
        input_prompt = self.prompt_answer_consist
        input_prompt = input_prompt.replace("<<question>>", question)
        input_prompt = input_prompt.replace("<<answer>>", final_answer)
        input_prompt = input_prompt.replace("<<gt_answer>>", gt_answer)

        response = requests_api(None, input_prompt)
        if response.lower() == "consistent":
            return True
        else:
            return False

    def run_answer_consist(self):
        output_dict = {}
        for item in tqdm(self.data):
            sample_id = item['sample_id']
            response = self.answer_consist(item)
            if not response:
                step = item['trajectory'][-1]['step']
                output_dict[sample_id] = [step]
        return output_dict

    def run_dehallucination(self):
        output_list = []
        already_list = []
        if os.path.exists(self.args.output_file):
            with jsonlines.open(self.args.output_file, mode="r") as reader:
                for item in reader:
                    already_list.append(item["sample_id"])
        for item in tqdm(self.data):
            sample_id = item["sample_id"]
            if sample_id in already_list:
                continue
            new_item = self.dehallucination(item)
            output_list.append(new_item)

            with jsonlines.open(self.args.output_file, mode="a") as writer:
                writer.write(new_item)

        return output_list
        
    def run_reasonable(self):
        pass

    def run(self):
        if self.args.function == "answer_consist":
            output_dict = self.run_answer_consist()

            with open(self.args.output_file) as f:
                data = json.load(f)
            for key in data.keys():
                if key not in output_dict:
                    output_dict[key] = data[key]
                else:
                    step_data = list(set(data[key] + output_dict[key]))
                    output_dict[key] = step_data
            with open(self.args.output_file, "w") as f:
                json.dump(output_dict, f, indent=4)

        elif self.args.function == "dehallucination":
            self.run_dehallucination()

        elif self.args.function == "reasonable":
            self.run_reasonable()

        return

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="读取 JSONL 文件并逐行处理")
    parser.add_argument("--function", type=str, required=True) # [answer_consist, dehallucination, reasonable]
    parser.add_argument("--file", type=str, required=False, help="JSONL 文件路径")
    parser.add_argument('--output_file', type=str, required=False, help="输出的文件路径")
    args = parser.parse_args()

    image_root = "/home/zml/algorithm/ReactEQA/data/EQA-Traj-0720"
    prompt_path = {
        "prompt_dehuallucination": "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/evaluation/prompts/dehuallucination.txt",
        "prompt_answer_consist": "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/evaluation/prompts/answer_consist.txt",
        "prompt_nonkey_thought": "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/evaluation/prompts/nonkey_thought.txt",
        "prompt_reasonable": "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/evaluation/prompts/reasonable.txt",
    }

    evaluation = Evaluation(args, image_root, prompt_path)

    evaluation.run()