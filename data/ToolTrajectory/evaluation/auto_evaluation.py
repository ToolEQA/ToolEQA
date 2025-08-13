import os
import json
import jsonlines
from tqdm import tqdm
from data.ToolTrajectory.generator_deerapi import requests_api

class Evaluation():
    def __init__(self, data_paths, image_root, prompt_pathes):
        self.image_root = image_root
        self.data = []
        for data_path in data_paths:
            with jsonlines.open(data_path, 'r') as reader:
                self.data = [item for item in reader]
        prompt_pathes = {k: open(v).read() for k, v in prompt_pathes.items()}
        self.__dict__.update(prompt_pathes)

    def _str_to_bool(self, s: str) -> bool:
        return s.strip().lower() in ("true", "1", "yes", "y", "t")

    def dehallucination(self, item):
        trajectory = item['trajectory']
        result = []
        related_obj = item['related_objects']
        for i, step_i in enumerate(trajectory):
            if self._str_to_bool(step_i['is_key']):
                continue
            image_path = os.path.join(self.image_root, step_i['image_path'])
            for react_i in step_i['react']:
                thought = react_i['thought']
                input_prompt = self.prompt_dehuallucination
                input_prompt = input_prompt.replace("<<thought>>", thought)
                response = requests_api(image_path, input_prompt)
                result.append(int(response))
                
                if float(response) <= 3:
                    obj = related_obj[i]['name']
                    direct = step_i['action'][0][0]
                    nonkey_thought = self.prompt_nonkey_thought
                    nonkey_thought = nonkey_thought.replace("<<object_name>>", obj)
                    nonkey_thought = nonkey_thought.replace("<<action_name>>", direct)
                    response = requests_api(None, nonkey_thought)
                    react_i['thought'] = response
                    # print(f"Revise thought: {thought} --> {response}")
        
        return item

    def answer_consist(self, item):
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
        final_answer = None
        for step_i in item['trajectory']:
            if final_answer is not None:
                break
            for react_i in step_i['react']:
                if "final_answer" in react_i['code']:
                    final_answer = react_i['code']
                    start = final_answer.find("final_answer")
                    end = final_answer.find(")", start)
                    final_answer = final_answer[start + 12:end].strip("()'\"")
                    break

        if final_answer.lower() == gt_answer.lower():
            return "Consistent"

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

    def run(self):
        new_data = []
        for item in tqdm(self.data):
            print(item['sample_id'])
            new_item = self.dehallucination(item)
            new_data.append(new_item)
            break

        # save json
        with open("data/ToolTrajectory/evaluation/output/dehallucination.json", 'w') as f:
            json.dump(new_data, f, indent=4)
        exit()
        output_list = []
        count = 0
        for item in tqdm(self.data):
            count += 1
            if count > 100:
                break
            sample_id = item['sample_id']
            response = self.answer_consist(item)
            if not response:
                output_list.append(sample_id)
        
        # save
        with open("data/ToolTrajectory/evaluation/output/answer_consist.txt", 'w') as f:
            for sample_id in output_list:
                f.write(f"{sample_id}\n")
            
            


if __name__=="__main__":
    image_root = "data/EQA-Traj-0720"
    data_path = ["data/ToolTrajectory/trajectory_gen/status/output/status.jsonl",]
    prompt_path = {
        "prompt_dehuallucination": "data/ToolTrajectory/evaluation/prompts/dehuallucination.txt",
        "prompt_answer_consist": "data/ToolTrajectory/evaluation/prompts/answer_consist.txt",
        "prompt_nonkey_thought": "data/ToolTrajectory/evaluation/prompts/nonkey_thought.txt"
    }

    evaluation = Evaluation(data_path, image_root, prompt_path)

    evaluation.run()