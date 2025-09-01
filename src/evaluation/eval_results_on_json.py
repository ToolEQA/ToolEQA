import os
import json
import numpy as np
import ast
import math
import jsonlines
from tqdm import tqdm
from src.vlm.generator_deerapi import requests_api

def answer_rating(answer, predict_answer):
    """
    调用大模型，对答案进行评分
    """
    prompt = f"""
    Please evaluate the answer according to the following criteria, with a maximum score of 5:

    The answer is completely correct and detailed — score 5.
    The answer is mostly correct but lacks some details — score 4.
    The answer is partially correct but has obvious omissions or errors — score 3.
    The answer is mostly incorrect or irrelevant — score 2.
    The answer is completely incorrect or does not address the question — score 1.

    Please return only an integer score, without any explanation or additional information.
    Question: {answer}
    Answer: {predict_answer}
    """
    result = requests_api(None, prompt, max_retries=10)
    rating = result.strip()
    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            rating = 3
    except:
        rating = 3

    return rating

def load_jsons(files_path):
    """
    读取 JSON 文件。

    :param file_path: 文件路径
    :return: JSON 数据
    """
    all_data = []
    for file_path in files_path:
        print(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            file_ext = os.path.splitext(file_path)[-1]
            if file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_data.extend(json.load(file))
            elif file_ext == ".jsonl":
                with jsonlines.open(file_path, mode="r") as reader:
                    for item in reader:
                        all_data.append(item)
    
    return all_data

def can_see_object(cam_pos, angle, obj_pos, fov_deg=120, max_distance=15):
    """
    判断相机在某一步是否能看到物体

    参数:
        cam_pos: list or np.array, 相机位置 [x, y, z]
        angle: float, 相机朝向角度 (弧度制, 绕y轴, 0表示朝 -z)
        obj_pos: list or np.array, 物体位置 [x, y, z]
        fov_deg: float, 视场角 (默认60度)
        max_distance: float, 可见的最大距离 (默认50)

    返回:
        True / False
    """
    cam_pos = np.array(cam_pos, dtype=float)
    obj_pos = np.array(obj_pos, dtype=float)

    # 相机 -> 物体 向量
    vec_to_obj = obj_pos - cam_pos
    distance = np.linalg.norm(vec_to_obj)
    if distance > max_distance:
        return False
    vec_to_obj /= distance  # 归一化

    # 相机前向向量: 初始为 (0,0,-1)，绕 y 轴旋转 angle
    cam_forward = np.array([np.sin(angle), 0, -np.cos(angle)])

    # 计算夹角
    dot = np.dot(cam_forward, vec_to_obj)
    dot = np.clip(dot, -1.0, 1.0)  # 防止数值误差
    angle_between = np.arccos(dot)

    # 判断是否在FOV内
    fov_rad = np.deg2rad(fov_deg)
    return angle_between < fov_rad / 2

def compute_weighted_recall(steps, objs, max_distance=15, fov_deg=120):
    """
    计算距离加权的召回率

    参数:
        steps: list of (pos, angle)，相机位置和角度
        objs: list of pos，物体位置
        fov_deg: float，视场角
        max_distance: float，最大可见距离

    返回:
        weighted_recall: float，0~1
    """
    n = len(objs)
    max_weights = np.zeros(n, dtype=float)

    for cam_pos, angle in steps:
        cam_pos = np.array(cam_pos)
        for j, obj_pos in enumerate(objs):
            obj_pos = np.array(obj_pos)
            vec_to_obj = obj_pos - cam_pos
            distance = np.linalg.norm(vec_to_obj)
            if distance > max_distance:
                continue
            # 判断是否在FOV内
            if not can_see_object(cam_pos, angle, obj_pos, fov_deg, max_distance):
                continue
            # 权重 = 距离越近越大
            weight = 1 - distance / max_distance
            if weight > max_weights[j]:
                max_weights[j] = weight

    weighted_recall = max_weights.mean()
    return weighted_recall


def evaluate(files_path):
    samples = load_jsons(files_path)
    results = []
    choices = ['A', 'B', 'C', 'D']

    last_length = 0

    for data in tqdm(samples):

        meta_data = data['meta']
        summary_data = data["summary"]

        shortest_length = meta_data['shortest_length']
        path_length = summary_data["shorest_path"] - last_length
        last_length = summary_data["shorest_path"]

        related_objs = meta_data['related_objs']
        if isinstance(related_objs, str):
            related_objs = ast.literal_eval(meta_data['related_objs'])

        result = {}

        step_data = data['step']
        # max_step = meta_data['max_steps']
        answer = meta_data['proposals'][choices.index(meta_data['answer'])]

        predict_answer = summary_data["final_answer"]

        steps = []
        for step in step_data:
            steps.append((step['pts'], step['angle']))
        objs = []
        for related_obj in related_objs:
            objs.append(related_obj['pos'])

        len_step = math.sqrt(len(steps))
        if len_step == 0.0:
            len_step = 1.0
        obj_recall_at5 = compute_weighted_recall(steps, objs, 5) / len_step
        obj_recall_at10 = compute_weighted_recall(steps, objs, 10) / len_step
        obj_recall_at15 = compute_weighted_recall(steps, objs, 15) / len_step
        
        # predict = 1 if predict_answer == answer else 0
        predict = answer_rating(answer, predict_answer) / 5.0
        # predict = 0.4

        # e_path_at5 = predict * obj_recall_at5 * (shortest_length / max(path_length, shortest_length))
        # e_path_at10 = predict * obj_recall_at10 * (shortest_length / max(path_length, shortest_length))
        # e_path_at15 = predict * obj_recall_at15 * (shortest_length / max(path_length, shortest_length))

        e_path_at5 = predict * obj_recall_at5 * math.exp(shortest_length / max(path_length, shortest_length))
        e_path_at10 = predict * obj_recall_at10 * math.exp(shortest_length / max(path_length, shortest_length))
        e_path_at15 = predict * obj_recall_at15 * math.exp(shortest_length / max(path_length, shortest_length))

        result["len_steps"] = len(steps)
        result["predict"] = predict
        result["obj_recall"] = {"@5": obj_recall_at5, "@10": obj_recall_at10, "@15": obj_recall_at15}
        result["e_path"] = {"@5": e_path_at5, "@10": e_path_at10, "@15": e_path_at15}

        results.append(result)

    results_num = len(results)
    norm_steps = 0
    norm_early_steps = 0
    early_count = 0
    success_count = 0

    avg_obj_recall_at5 = 0
    avg_obj_recall_at10 = 0
    avg_obj_recall_at15 = 0
    avg_e_path_at5 = 0
    avg_e_path_at10 = 0
    avg_e_path_at15 = 0
    avg_predict = 0
    avg_steps = 0

    for result in results:
        avg_steps += result.get("len_steps")

        avg_obj_recall_at5 += result.get("obj_recall").get("@5")
        avg_e_path_at5 += result.get("e_path").get("@5")

        avg_obj_recall_at10 += result.get("obj_recall").get("@10")
        avg_e_path_at10 += result.get("e_path").get("@10")

        avg_obj_recall_at15 += result.get("obj_recall").get("@15")
        avg_e_path_at15 += result.get("e_path").get("@15")
        avg_predict += result.get("predict")

    print("========================================")
    print("avg step: ", avg_steps / results_num)
    print("========================================")
    print("avg obj_recall@5: ", avg_obj_recall_at5 / results_num)
    print("avg obj_recall@10: ", avg_obj_recall_at10 / results_num)
    print("avg obj_recall@15: ", avg_obj_recall_at15 / results_num)
    print("========================================")
    print("avg e_path@5: ", avg_e_path_at5 / results_num)
    print("avg e_path@10: ", avg_e_path_at10 / results_num)
    print("avg e_path@15: ", avg_e_path_at15 / results_num)
    print("========================================")
    print("success: ", avg_predict / results_num)

if __name__ == '__main__':
    files_path = [
        "results/gpt4omini.unseen.0829/result_0.jsonl",
        "results/gpt4omini.unseen.0829/result_1.jsonl",
        "results/gpt4omini.unseen.0829/result_2.jsonl",
        "results/gpt4omini.unseen.0829/result_3.jsonl",
        "results/gpt4omini.unseen.0829/result_4.jsonl",
        "results/gpt4omini.unseen.0829/result_5.jsonl",
        "results/gpt4omini.unseen.0829/result_6.jsonl",
        "results/gpt4omini.unseen.0829/result_7.jsonl",
    ]
    evaluate(files_path)
