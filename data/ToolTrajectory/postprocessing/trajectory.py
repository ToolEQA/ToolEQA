# Author: zml
# 该脚本用于在Habitat-sim中生成路径并保存为视频
# 并将路径点转换为动作序列

import os
import numpy as np
import pandas as pd
import quaternion
import habitat_sim
import json
import re
from tqdm import tqdm
import cv2
from src.utils.habitat import make_simple_cfg
from data.ToolTrajectory.postprocessing.extract_object_location import get_image_from_id
import subprocess
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
import uuid
import base64
import random
from src.utils.navigate import path_to_actions

save_dir = "/mynvme1/EQA-Traj-0611"


def short_uuid():
    u = uuid.uuid4()
    return base64.urlsafe_b64encode(u.bytes).rstrip(b'=').decode('ascii')

def look_at(cur_point, target_point):
    """
        UP = Vector(0, 1, 0)
        GRAVITY = Vector(-0, -1, -0)
        FRONT = Vector(-0, -0, -1)
        BACK = Vector(0, 0, 1)
        LEFT = Vector(-1, 0, 0)
        RIGHT = Vector(1, 0, 0)
    """
    direction = np.array(target_point) - np.array(cur_point)
    direction = direction / np.linalg.norm(direction)  # 归一化

    angle_radians = np.arctan2(direction[0], direction[2]) # x/z 平面上的角度 [-pi, pi]
    angle_radians += np.pi # [0, 2*pi]

    # forward_vector = np.array([0.0, 0.0, -1.0])
    # cos_theta = np.dot(forward_vector, direction)
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # angle_radians = np.arccos(cos_theta) # [0, pi]
    # cross_product = np.cross(forward_vector, direction)
    # if cross_product[1] < 0:  # 如果目标在左侧（x坐标为负）
    #     angle_radians += np.pi  # 转换为 [0, 2pi]

    return quat_to_coeffs(quat_from_angle_axis(angle_radians, np.array([0, 1, 0]))).tolist()


def get_navigable_path_safe(simulator, start_pos, goal_pos, verbose=False,
                            search_radius=0.5, sample_count=100):
    """
    尝试从起点到终点规划路径，如果失败则在终点周围采样多个点寻找可达路径。

    参数：
        simulator: habitat_sim.Simulator 实例
        start_pos: 起点 [x, y, z]
        goal_pos: 终点 [x, y, z]
        verbose: 是否打印调试信息
        search_radius: 终点附近的采样半径（米）
        sample_count: 尝试的最大采样数量

    返回：
        path_points: 路径点列表（如果找不到则为空）
        geodesic_distance: 路径长度（float）
        final_goal: 实际使用的终点（如果与输入 goal_pos 不同）
    """
    pathfinder = simulator.pathfinder

    def snap(pos):
        return pathfinder.snap_point(pos)

    def is_valid(pos):
        return pathfinder.is_navigable(pos)

    def try_path(start, end):
        req = habitat_sim.ShortestPath()
        req.requested_start = start
        req.requested_end = end
        found = pathfinder.find_path(req)
        if found and req.geodesic_distance < float('inf') and len(req.points) > 0:
            return req.points, req.geodesic_distance
        return [], float('inf')

    # Snap 起点和终点
    start = snap(start_pos)
    goal = snap(goal_pos)

    if verbose:
        print(f"起点 snap 后: {start}, 是否可通行: {is_valid(start)}")
        print(f"终点 snap 后: {goal}, 是否可通行: {is_valid(goal)}")

    # 尝试直接路径
    path, dist = try_path(start, goal)
    if dist < float('inf'):
        if verbose:
            print(f"✅ 直接路径可达，长度: {dist:.3f}")
        return path, dist, goal

    if verbose:
        print("⚠️ 直接路径失败，开始在目标附近搜索可达点...")

    # 尝试在目标附近采样多个点
    for _ in range(sample_count):
        offset = np.random.uniform(-1, 1, size=3)
        offset[1] = 0  # 不改变高度
        offset = offset / np.linalg.norm(offset) * random.uniform(0.05, search_radius)
        sampled_goal = goal + offset

        if not is_valid(sampled_goal):
            continue

        path, dist = try_path(start, sampled_goal)
        if dist < float('inf'):
            if verbose:
                print(f"✅ 找到备选目标点 {sampled_goal}，路径长度: {dist:.3f}")
            return path, dist, sampled_goal

    if verbose:
        print("❌ 所有尝试都失败，无法规划路径")
    return [], float('inf'), None


def objstr2list(objs_str):
    pattern = r'^(?P<name>.+?)\s+\((?P<id>\d+)\):\s+\[(?P<position>[-\d.,\s]+)\]'
    objs_list = objs_str.split("; ")
    objs_data = []
    for obj in objs_list:
        match = re.match(pattern, obj.strip())
        if match:
            name = match.group('name').strip()
            id = match.group('id')
            pos = [float(p) for p in match.group('position').split(", ")]
            if len(pos) != 3:
                return None
            objs_data.append({
                "name": name,
                "id": int(id),
                "pos": [pos[0], pos[2], -pos[1]],
            })

    if len(objs_list) == len(objs_data):
        return objs_data
    return None


def save_obs_on_path(sim, agent, order, path_points, start_rotation, save_root="./"):
    steps_data = []
    agent_state = habitat_sim.AgentState()
    for i in range(len(path_points)):
        step_data = {}
        if i == len(path_points) - 1:
            # 最后一个点使用终点旋转
            cur_position = agent_state.position
            start_rotation = look_at(cur_position, path_points[-1])
            action = path_to_actions([cur_position, cur_position], agent_state.rotation, start_rotation)
            agent_state.position = cur_position
            agent_state.rotation = start_rotation
            agent.set_state(agent_state)
        elif i == 0:
            # 第一个点使用起点旋转
            agent_state.position = path_points[i]
            agent_state.rotation = start_rotation
            agent.set_state(agent_state)
            continue
        else:
            # 中间点的方向使用当前点和下一点之间的方向
            point = path_points[i]
            target_point = path_points[i + 1]
            start_rotation = look_at(point, target_point)
            action = path_to_actions([agent_state.position, point], agent_state.rotation, start_rotation)
            # 设置代理状态
            agent_state.position = point
            agent_state.rotation = start_rotation
            agent.set_state(agent_state)

        obs = sim.get_sensor_observations()
        save_path = os.path.join(save_root, f"{order}-{i}.png")
        image = obs["color_sensor"][..., :3][..., ::-1]
        cv2.imwrite(save_path, image)

        step_data["step"] = f"{order}-{i}"
        step_data["thought"] = ""
        step_data["code"] = ""
        step_data["observation"] = ""
        step_data["action"] = action
        step_data["position"] = path_points[i].tolist()
        step_data["rotation"] = start_rotation
        step_data["is_key"] = True if i + 1 == len(path_points) - 1 else False
        step_data["image_path"] = save_path

        steps_data.append(step_data)

    return steps_data


def get_sample_obs(scene_dir, objs_data, init_pos, init_rot, save_root="./"):
    trajectory = [{}]

    scene_id_order = scene_dir.strip("/").split("/")[-1]
    scene_id = scene_dir.strip("/").split("-")[-1]
    scene_mesh_file = os.path.join(scene_dir, scene_id + ".basis" + ".glb")
    navmesh_file = os.path.join(scene_dir, scene_id + ".basis" + ".navmesh")

    sim_settings = {
        "scene": scene_mesh_file,
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 640,
        "height": 640,
        "hfov": 120,
    }

    sim_cfg = make_simple_cfg(sim_settings)
    simulator = habitat_sim.Simulator(sim_cfg)

    simulator.pathfinder.load_nav_mesh(navmesh_file)
    # 初始化代理
    agent = simulator.initialize_agent(0)

    # 加载导航网格
    if not os.path.exists(navmesh_file):
        print(f"Navmesh file {navmesh_file} not found!")

    # 计算最短路径
    path = habitat_sim.ShortestPath()

    # 保存初始位置obs
    agent_state = habitat_sim.AgentState()
    agent_state.position = init_pos
    agent_state.rotation = init_rot
    agent.set_state(agent_state)
    obs = simulator.get_sensor_observations()
    save_path = os.path.join(save_root, "0-0.png")
    image = obs["color_sensor"][..., :3][..., ::-1]
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    cv2.imwrite(save_path, image)

    trajectory[0]["step"] = f"0-0"
    trajectory[0]["thought"] = ""
    trajectory[0]["code"] = ""
    trajectory[0]["observation"] = ""
    trajectory[0]["action"] = ""
    trajectory[0]["position"] = init_pos.tolist()
    trajectory[0]["rotation"] = init_rot
    trajectory[0]["is_key"] = False
    trajectory[0]["image_path"] = save_path

    traj_length = 0

    print(objs_data)
    # 初始位置到第一个关键点
    # path_points, path_length, real_goal = get_navigable_path_safe(simulator, init_pos, objs_data[0]["pos"])
    # traj_length += path_length
    # print(path_points)

    path.requested_start = init_pos
    path.requested_end = objs_data[0]["pos"]
    found_path = simulator.pathfinder.find_path(path)
    path_points = path.points
    traj_length += path.geodesic_distance
    print(f"Path found with {len(path.points)} points")
    print(f"Path length: {path.geodesic_distance}")

    # print(f"Path found with {len(path_points)} points")
    # print(f"Path length: {path_length}")

    start_rotation = init_rot
    temp_traj = save_obs_on_path(simulator, agent, 0, path_points, start_rotation, save_root)
    trajectory.extend(temp_traj)

    # # 保存终点位置obs
    # target_image = get_image_from_id(scene_id_order, objs_data[0]["id"])
    # if target_image is None:
    #     return 
    # else:
    #     target_image = target_image[0]
    # end_name = objs_data[0]["name"]
    # # 复制到指定位置
    # subprocess.run(["cp", target_image, f"{save_dir}/{0}-{end_name}-end.png"], check=True)

    if len(objs_data) > 1:
        for i in range(1, len(objs_data)):
            start_position = objs_data[i-1]["pos"]
            end_position = objs_data[i]["pos"]

            # path_points, path_length, real_goal = get_navigable_path_safe(simulator, start_position, end_position)
            # print(path_points)
            # traj_length += path_length
            
            nearest_point = simulator.pathfinder.snap_point(end_position)
            path.requested_start = start_position
            path.requested_end = nearest_point
            # 计算最短路径
            found_path = simulator.pathfinder.find_path(path)
            traj_length += path.geodesic_distance
            if not found_path:
                print("No valid path found between start and end positions!")

            print(f"Path found with {len(path.points)} points")
            print(f"Path length: {path.geodesic_distance}")

            # 使用 path.points 获取路径点
            path_points = path.points

            # 中间点图像
            temp_traj = save_obs_on_path(simulator, agent, i, path_points, start_rotation, save_root)
            trajectory.extend(temp_traj)

            # # 保存终点位置obs
            # target_image = get_image_from_id(scene_id_order, objs_data[i]["id"])[0]
            # end_name = objs_data[i]["name"]
            # # 复制到指定位置
            # subprocess.run(["cp", target_image, f"{save_dir}/{i}-{end_name}-end.png"], check=True)

    try:
        simulator.close()
    except:
        pass

    return trajectory, traj_length


if __name__ == "__main__":
    scene_root = "data/HM3D"
    question_files = [
        "data/ToolTrajectory/questions/final_question/attribute/color.csv",
        "data/ToolTrajectory/questions/final_question/attribute/size.csv",
        "data/ToolTrajectory/questions/final_question/attribute/special.csv",
        "data/ToolTrajectory/questions/final_question/counting/counting.csv",
        "data/ToolTrajectory/questions/final_question/distance/distance.csv",
        "data/ToolTrajectory/questions/final_question/location/location.csv",
        "data/ToolTrajectory/questions/final_question/location/special.csv",
        "data/ToolTrajectory/questions/final_question/relationship/relationship.csv",
        "data/ToolTrajectory/questions/final_question/status/status.csv"
    ]
    init_file = "data/HM3D/scene_init_poses_with_floor.csv"
    floor_file = "data/HM3D/scene_floor_info.json"

    with open(floor_file, "r") as f:
        floor_dict = json.load(f)

    init_dict = {}
    init_data = pd.read_csv(init_file)
    for index, row in init_data.iterrows():
        init_dict[row["scene_floor"]] = row

    samples_trajectory = []

    if os.path.exists(os.path.join(save_dir, "trajectory.jsonl")):
        with open(os.path.join(save_dir, "trajectory.jsonl"), "r", encoding='utf-8') as f:
            samples_trajectory = [json.loads(line) for line in f]
        print(f"已经生成了: {len(samples_trajectory)}. 现在继续生成...")

    count = 0
    for question_file in question_files:
        question_cate = question_file.split("/")[-2]
        question_sub_cat = question_file.split("/")[-1].split(".")[0]
        # print(f"Processing {question_cate} - {question_sub_cat}...")
        
        csv_data = pd.read_csv(question_file)
        for index, row in csv_data.iterrows():
            count += 1
            if count < len(samples_trajectory):
                continue

            sample_trajectory = {}

            question_id = short_uuid()
            print(f"================== {count}: {question_id} ==================")

            scene_id = row["scene_id"]
            scene_dir = os.path.join(scene_root, row["scene_id"])
            objs_data = objstr2list(row["locations"])
            if objs_data is None:
                continue
            try:
                floor = list(floor_dict[scene_id.split("-")[-1]].keys()).index(str(row["floor_id"]))
            except:
                continue
            if scene_id + f"_{floor}" not in init_dict.keys():
                continue
            init_info = init_dict[scene_id + f"_{floor}"]

            start_position = np.array([init_info["init_x"], init_info["init_y"], init_info["init_z"]])
            start_rotation = quat_to_coeffs(quat_from_angle_axis(init_info["init_angle"], np.array([0, 1, 0]))).tolist()
            # start_rotation = quaternion.from_float_array(yaw_to_quaternion(init_info["init_angle"]))

            trajectory, traj_length = get_sample_obs(scene_dir, objs_data, start_position, start_rotation, os.path.join(save_dir, question_id))

            print(scene_id)
            print(traj_length)

            if traj_length == float('inf'):
                continue

            sample_trajectory["sample_id"] = question_id
            sample_trajectory["scene"] = scene_id
            sample_trajectory["question"] = row["question"]
            sample_trajectory["proposals"] = eval(row["choices"])
            sample_trajectory["answer"] = row["answer"]
            sample_trajectory["question_type"] = f"{question_cate}-{question_sub_cat}"
            sample_trajectory["floor"] = row["floor_id"]
            sample_trajectory["floor_index"] = floor
            sample_trajectory["init_pos"] = [init_info["init_x"], init_info["init_y"], init_info["init_z"]]
            sample_trajectory["init_rot"] = init_info["init_angle"]
            sample_trajectory["related_objects"] = objs_data
            sample_trajectory["traj_length"] = traj_length
            sample_trajectory["trajectory"] = trajectory

            samples_trajectory.append(sample_trajectory)

            # 保存最后的json
            with open(os.path.join(save_dir, "trajectory.jsonl"), "a") as f:
                json.dump(sample_trajectory, f)
                f.write('\n')

    # 保存最后的json
    with open(os.path.join(save_dir, "trajectory.json"), "w") as f:
        json.dump(samples_trajectory, f, indent=4)


    # {
    #     "sample_id": "question_00001",
    #     "scene": "00695-BJovXQkqbC3",
    #     "question": "What is the color of the chair?",
    #     "proposals": ["The chair is black.", "The chair is red.", "The chair is blue.", "The chair is green."],
    #     "answer": "The chair is black.",
    #     "question_type": "color",
    #     "floor": 0,
    #     "init_pos": [-4.031486511230469, 0.07823586463928223, -3.293611526489258],
    #     "init_rot": [0.0, 0.417904810818124, 0.0, 0.9084908194886],        
    #     "related_objects": [
    #         {
    #             "category_name": "chair", 
    #             "region": "living room", 
    #             "position": [0.0, 0.0, 0.0]
    #         }
    #     ],
    #     "trajectory": {
    #         "step_0": {
    #             "thought": "thought",
    #             "code": "code",
    #             "observation": "observation",
    #             "action": "action",
    #             "position": [-4.031486511230469, 0.07823586463928223, -3.293611526489258],
    #             "rotation": [0.0, 0.417904810818124, 0.0, 0.9084908194886],
    #             "is_key": false, 
    #             "image_path": ""
    #         },
    #         "step_1": {
    #             "thought": "thought",
    #             "code": "code",
    #             "observation": "observation",
    #             "action": "action",
    #             "position": [0, 0, 0],
    #             "rotation": [1, 0.0, 0.0, 0.0],
    #             "is_key": true,
    #             "image_path": ""
    #         }
    #     }
    # }