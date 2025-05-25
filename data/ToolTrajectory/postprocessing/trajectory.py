# Author: zml
# 该脚本用于在Habitat-sim中生成路径并保存为视频
# 并将路径点转换为动作序列

import os
import numpy as np
import pandas as pd
import quaternion
import habitat_sim
import json
from tqdm import tqdm
import cv2
from src.utils.habitat import make_simple_cfg
from data.ToolTrajectory.postprocessing.extract_object_location import get_image_from_id
import subprocess
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis

def two_points_direction(point, prior_point):
    direction = point - prior_point
    direction = direction / np.linalg.norm(direction)
    horizontal_direction = np.array([direction[0], direction[2]])
    yaw_rad = np.arctan2(horizontal_direction[0], horizontal_direction[1])
    yaw_deg = np.degrees(yaw_rad)

    rotation = quat_to_coeffs(quat_from_angle_axis(yaw_deg, np.array([0, 1, 0]))).tolist()
    return rotation


def objstr2list(objs_str):
    objs_list = objs_str.split("; ")
    objs_data = []
    for obj in objs_list:
        info = obj.split(" ")
        name = info[0]
        id = info[1].strip(" ():")
        pos = [float(p.strip("[],")) for p in info[2:]]
        objs_data.append({
            "name": name,
            "id": int(id),
            "pos": [pos[0], pos[2], -pos[1]],
        })
    return objs_data


def save_obs_on_path(sim, agent, order, path_points, start_rotation, save_root="./"):
    steps_data = []
    agent_state = habitat_sim.AgentState()
    for i in range(len(path_points) - 1):
        step_data = {}
        # if i == len(path_points) - 1:
        #     # 最后一个点使用终点旋转
        #     cur_position = agent_state.position
        #     end_rotation = two_points_direction(cur_position, path_points[i])
        #     agent_state.position = path_points[i]
        #     agent_state.rotation = end_rotation
        if i == 0:
            # 第一个点使用起点旋转
            agent_state.position = path_points[i]
            agent_state.rotation = start_rotation
            continue
        else:
            # 中间点的方向使用当前点和下一点之间的方向
            point = path_points[i]
            prior_point = path_points[i + 1]
            start_rotation = two_points_direction(point, prior_point)
            # 设置代理状态
            agent_state.position = path_points[i]
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
        step_data["action"] = ""
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

    try:
        simulator.close()
    except:
        pass

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

    # 初始位置到第一个关键点
    path.requested_start = init_pos
    path.requested_end = objs_data[-1]["pos"]
    found_path = simulator.pathfinder.find_path(path)
    path_points = path.points
    print(f"Path found with {len(path.points)} points")
    print(f"Path length: {path.geodesic_distance}")
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
    # subprocess.run(["cp", target_image, f"{0}-{end_name}-end.png"], check=True)

    if len(objs_data) > 1:
        for i in range(1, len(objs_data)):
            start_position = objs_data[i-1]["pos"]
            end_position = objs_data[i]["pos"]

            path.requested_start = start_position
            path.requested_end = end_position

            # 计算最短路径
            found_path = simulator.pathfinder.find_path(path)
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
            # subprocess.run(["cp", target_image, f"{i}-{end_name}-end.png"], check=True)

    return trajectory


if __name__ == "__main__":
    scene_root = "data/HM3D"
    question_files = ["data/ToolTrajectory/questions/attribute/color.csv"]
    init_file = "data/HM3D/scene_init_poses_with_floor.csv"
    floor_file = "data/HM3D/scene_floor_info.json"

    with open(floor_file, "r") as f:
        floor_dict = json.load(f)

    init_dict = {}
    init_data = pd.read_csv(init_file)
    for index, row in init_data.iterrows():
        init_dict[row["scene_floor"]] = row

    samples_trajectory = []
    for question_file in question_files:
        question_cate = question_file.split("/")[-1].split(".")[0]
        csv_data = pd.read_csv(question_file)
        for index, row in csv_data.iterrows():
            sample_trajectory = {}
            if index == 0:
                scene_id = row["scene_id"]
                scene_dir = os.path.join(scene_root, row["scene_id"])
                objs_data = objstr2list(row["locations"])
                floor = list(floor_dict[scene_id.split("-")[-1]].keys()).index(str(row["floor_id"]))
                init_info = init_dict[scene_id + f"_{floor}"]

                start_position = np.array([init_info["init_x"], init_info["init_y"], init_info["init_z"]])
                start_rotation = quat_to_coeffs(quat_from_angle_axis(init_info["init_angle"], np.array([0, 1, 0]))).tolist()
                # start_rotation = quaternion.from_float_array(yaw_to_quaternion(init_info["init_angle"]))

                trajectory = get_sample_obs(scene_dir, objs_data, start_position, start_rotation, f"./data/EQA-Traj/{index}")

                sample_trajectory["sample_id"] = index
                sample_trajectory["scene"] = scene_id
                sample_trajectory["question"] = row["question"]
                sample_trajectory["proposals"] = row["choices"]
                sample_trajectory["answer"] = row["answers"]
                sample_trajectory["question_type"] = row["label"] + "_" + question_cate
                sample_trajectory["floor"] = row["floor_id"]
                sample_trajectory["floor_index"] = floor
                sample_trajectory["init_pos"] = [init_info["init_x"], init_info["init_y"], init_info["init_z"]]
                sample_trajectory["init_rot"] = init_info["init_angle"]
                sample_trajectory["related_objects"] = objs_data
                sample_trajectory["trajectory"] = trajectory

            samples_trajectory.append(sample_trajectory)

    # 保存最后的json
    with open("data/EQA-Traj/trajectory.json", "w") as f:
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