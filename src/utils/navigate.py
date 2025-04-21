import os
import numpy as np
import quaternion
from typing import List
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_angle_axis
from habitat_sim.utils import viz_utils as vut
import magnum as mn
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import cv2
import math
from src.utils.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)


def _quaternion_to_forward_vector(q):
    """将单位四元数转换为旋转后的前向向量 (-Z轴)"""
    w, x, y, z = q.w, q.x, q.y, q.z
    # 初始前向向量 (0, 0, 1)
    vx = 2 * (x*z - w*y)
    vy = 2 * (y*z + w*x)
    vz = 1 - 2 * (x*x + y*y)
    return np.array([vx, vy, -vz])


def _rotation_to_deg(rotation, direction):
    forward = np.quaternion(0, 0, 0, -1)
    # 使用四元数旋转向量
    rotated_forward = rotation * forward * rotation.conj()
    current_forward = np.array([rotated_forward.x, rotated_forward.y, rotated_forward.z])

    current_forward[1] = 0
    direction[1] = 0

    # 归一化
    if np.linalg.norm(current_forward) < 1e-6 or np.linalg.norm(direction) < 1e-6:
        return

    current_forward = current_forward / np.linalg.norm(current_forward)
    direction = direction / np.linalg.norm(direction)

    # 计算旋转角度 (带符号)
    cross = np.cross(current_forward, direction)
    dot = np.dot(current_forward, direction)

    # 计算旋转角度 (度)
    angle_deg = np.degrees(np.arctan2(np.linalg.norm(cross), dot))

    # 确定旋转方向 (使用叉积的y分量符号)
    if cross[1] > 0:
        angle_deg = -angle_deg

    return angle_deg

def _rotate_actions(current_rot, angle_deg, rotation_step):
    actions = []
    # 分解旋转为多个小步
    if abs(angle_deg) > 1e-3:  # 忽略极小角度
        num_rot_steps = int(np.ceil(abs(angle_deg) / rotation_step))
        actual_step = angle_deg / num_rot_steps

        for _ in range(num_rot_steps):
            if actual_step > 0:
                actions.append(("turn_right", min(rotation_step, abs(actual_step))))
            else:
                actions.append(("turn_left", min(rotation_step, abs(actual_step))))
            
            # 更新当前旋转 (使用四元数乘法)
            delta_rot = np.quaternion(np.cos(np.radians(-actual_step)/2), 
                                    0, np.sin(np.radians(-actual_step)/2), 0)
            current_rot = current_rot * delta_rot
    return actions, current_rot


def _move_actions(distance, move_step):
    actions = []
    # 分解移动为多个小步
    if distance > 1e-3:  # 忽略极小距离
        num_move_steps = int(np.ceil(distance / move_step))
        actual_step = distance / num_move_steps
        
        for _ in range(num_move_steps):
            actions.append(("move_forward", min(move_step, actual_step)))
    return actions

# 将路径转换为前进、左转、右转动作
def path_to_actions(path, start_rotation, end_rotation, rotation_step=10, move_step=0.1):
    """
    将路径点转换为动作序列
    
    参数:
        path: 路径点列表，每个点是3D坐标 (x,y,z)
        start_rotation: 起始旋转 (np.quaternion类型)
        end_rotation: 结束旋转 (np.quaternion类型)
        rotation_step: 每次旋转的角度 (度)
        move_step: 每次移动的距离

    返回:
        动作列表，每个动作是 ("move_forward", distance) 或 ("turn_left"/"turn_right", angle)
    """
    actions = []
    if len(path) < 2:
        return actions
    
    current_rot = start_rotation
    current_pos = np.array(path[0], dtype=np.float64)
    
    for i in range(1, len(path)):
        target_pos = np.array(path[i], dtype=np.float64)
        
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:  # 忽略极小距离
            continue

        direction = direction / distance
        
        angle_deg = _rotation_to_deg(current_rot, direction)
        
        retation_actions, current_rot = _rotate_actions(current_rot, angle_deg, rotation_step)
        actions.extend(retation_actions)

        move_actions = _move_actions(distance, move_step)
        actions.extend(move_actions)

        current_pos = target_pos
    
    # 对齐最后的朝向
    angle_deg = _rotation_to_deg(current_rot, _quaternion_to_forward_vector(end_rotation))
    retation_actions, current_rot = _rotate_actions(current_rot, angle_deg, rotation_step)
    actions.extend(retation_actions)
    
    return actions


# 执行动作并记录视频
def execute_actions(sim, actions):
    observations: List[np.ndarray] = []
    agent = sim.agents[0]
    
    for action, amount in tqdm(actions):
        # 更新动作参数
        agent.agent_config.action_space[action].actuation.amount = amount
        # 执行动作
        agent.act(action)
        
        # 获取观察并写入视频
        obs = sim.get_sensor_observations()
        observations.append({"color": obs["color"]})

    return observations


def navigation_video(sim, agent, pathes, output_video="navigation.mp4"):
    agent_state = habitat_sim.AgentState()

    shortest_path = habitat_sim.ShortestPath()

    # agent_state.position = pathes[-1]["position"]
    # agent_state.rotation = pathes[-1]["rotation"]
    # agent.set_state(agent_state)
    # obs = sim.get_sensor_observations()
    # cv2.imwrite("end_rgb.jpg", obs["color"][..., :3])

    # 设置起始位置和旋转
    agent_state.position = pathes[0]["position"]
    agent_state.rotation = pathes[0]["rotation"]
    agent.set_state(agent_state)

    observations = []
    for i in range(len(pathes)):
        if i == 0:
            continue
        start_point = pathes[i-1]
        end_point = pathes[i]

        shortest_path.requested_start = start_point["position"]
        shortest_path.requested_end = end_point["position"]

        found_path = sim.pathfinder.find_path(shortest_path)
        if not found_path:
            print("No valid path found between start and end positions!")
            return
        print(f"Path found with {len(shortest_path.points)} points")
        print(f"Path length: {shortest_path.geodesic_distance}")
        actions = path_to_actions(shortest_path.points, start_point["rotation"], end_point["rotation"])

        # obs = sim.get_sensor_observations()
        # cv2.imwrite("start_rgb.jpg", obs["color"][..., :3])
        # 设置起始位置和旋转

        observations.extend(execute_actions(sim, actions))

    if len(observations) < 2:
        print("Not enough frames captured for video!")
        return

    # 创建视频录制器
    vut.make_video(
        observations=observations,
        primary_obs="color",
        primary_obs_type="color",
        video_file=output_video,
        fps=24,
        open_vid=False
    )

if __name__ == "__main__":
    # 配置参数
    scene_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb"  # 替换为你的场景GLB文件路径
    navmesh_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.navmesh"  # 替换为你的导航网格文件路径
    output_video = "output.mp4"  # 输出视频文件名
    
    # 起点和终点的位置和旋转(四元数)
    position1 = np.array([3.34482479095459, 0.050354525446891785, 9.988510131835938]) # 替换为实际起点坐标
    rotation1 = quaternion.from_float_array([0.06694663545449525, 0.0, 0.997756557483499, 0.0])  # wxyz

    position2 = np.array([-1.5373001098632812, 0.050354525446891785, 16.74457359313965]) # 替换为实际终点坐标
    rotation2 = quaternion.from_float_array([0.015153784750062174, 0.0, 0.9998851748114626, 0.0])  # wxyz

    position3 = np.array([-8.596624374389648, 0.0503545999526978, 15.994026184082031])
    rotation3 = quaternion.from_float_array([0.443950753937056, 0.0, 0.8960511860818664, 0.0])

    pathes = [
        {"position": position1, "rotation": rotation1},
        {"position": position2, "rotation": rotation2},
        {"position": position3, "rotation": rotation3},
    ]

    # 初始化模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = False  # 不需要物理引擎

    # 2. 配置RGB传感器
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [480, 640]
    sensor_spec.hfov = 120.0  # 水平视场角
    sensor_spec.position = mn.Vector3(0, 1.5, 0)  # 相对于代理的位置
    sensor_spec.orientation = mn.Vector3(0, 0, 0)
    
    action_space_config = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", 
            habitat_sim.ActuationSpec(amount=0.1)  # 默认步长
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=10.0)  # 默认旋转角度
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=10.0)  # 默认旋转角度
        )
    }

    # 代理配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    agent_cfg.action_space = action_space_config
    
    # 创建模拟器
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    # 加载导航网格
    if not os.path.exists(navmesh_file):
        print(f"Navmesh file {navmesh_file} not found!")
        exit()
    
    sim.pathfinder.load_nav_mesh(navmesh_file)
    
    # 初始化代理
    agent = sim.initialize_agent(0)

    navigation_video(sim, agent, pathes)