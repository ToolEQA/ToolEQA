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
from src.utils.geom import (
    get_cam_intr,
    get_scene_bnds
)
from PIL import Image, ImageFont, ImageDraw
import textwrap
from src.planner.tsdf import TSDFPlanner
import matplotlib.pyplot as plt


def concatenate_images_horizontally(image1, image2):
    # 确保两张图片高度相同（可选：自动调整高度）
    if isinstance(image1, np.ndarray):
        image1 = Image.fromarray(image1)
    if isinstance(image2, np.ndarray):
        image2 = Image.fromarray(image2)
    if image1.height != image2.height:
        # 计算新高度（取较大者或较小者，这里以较大者为例）
        new_height = max(image1.height, image2.height)
        
        # 调整图片高度（保持宽高比）
        def resize_with_aspect_ratio(img, height):
            ratio = height / float(img.height)
            new_width = int(float(img.width) * ratio)
            return img.resize((new_width, height), Image.Resampling.LANCZOS)
            
        image1 = resize_with_aspect_ratio(image1, new_height)
        image2 = resize_with_aspect_ratio(image2, new_height)
    
    # 创建新图片（宽度为两张图片宽度之和，高度为共同高度）
    new_width = image1.width + image2.width
    new_image = Image.new('RGB', (new_width, image1.height))
    
    # 将两张图片粘贴到新图片中
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    
    return new_image

def add_draw_point(image, point=None, trajectory=None, radius=18):
    # print(type(image))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    rgb_im_draw = image.copy()
    draw = ImageDraw.Draw(rgb_im_draw)
    if point is not None:
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill=(200, 200, 200, 255),
            outline=(0, 0, 255, 255),
            width=5,
        )
    if trajectory is not None:
        draw.line(trajectory, fill=(255, 0, 0), width=2, joint="curve")

    return rgb_im_draw


def draw_text_fill_region(image, text, 
                         region_width=None, region_height=None,
                         font_path=None, 
                         max_font_size=100, min_font_size=8,
                         text_color=(255, 255, 255), 
                         bg_color=(0, 0, 0, 128),
                         padding=10,
                         position='top',
                         max_lines=None):
    """
    在图像上绘制自适应字体大小和自动换行的文字，尽量填满文字区域
    
    参数:
        image: PIL Image对象或图像路径
        text: 要绘制的文本
        region_width: 文字区域宽度(默认图像宽度)
        region_height: 文字区域高度(默认图像高度的20%)
        font_path: 字体文件路径(可选)
        max_font_size: 最大字体大小
        min_font_size: 最小字体大小
        text_color: 文字颜色
        bg_color: 背景颜色(含透明度)
        padding: 内边距
        position: 文字位置('top', 'center', 'bottom')
        max_lines: 最大行数限制(可选)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 设置默认文字区域大小
    if region_width is None:
        region_width = image.width
    if region_height is None:
        region_height = image.height * 0.2
    
    # 计算实际可用的文字绘制区域(减去padding)
    text_area_width = region_width - 2 * padding
    text_area_height = region_height - 2 * padding
    
    # 初始化字体大小
    font_size = max_font_size
    
    # 最佳结果变量
    best_font = None
    best_lines = []
    best_font_size = min_font_size
    
    # 二分查找最佳字体大小
    low = min_font_size
    high = max_font_size
    
    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid) if font_path else ImageFont.load_default(mid)
        
        # 分割文本为多行
        lines = []
        if font.getlength(text) <= text_area_width:
            lines = [text]
        else:
            # 计算每行大致字符数
            avg_char_width = font.getlength("a")  # 近似字符宽度
            chars_per_line = int(text_area_width / avg_char_width)
            lines = textwrap.wrap(text, width=chars_per_line)
            
            # 如果设置了最大行数，截断多余行
            if max_lines and len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = lines[-1][:30] + "..."  # 添加省略号
        
        # 计算总文本高度
        line_height = font.size * 1.2  # 行高为字体大小的1.2倍
        total_height = len(lines) * line_height
        
        # 检查是否适合
        if total_height <= text_area_height:
            # 可以容纳，尝试更大的字体
            best_font = font
            best_lines = lines
            best_font_size = mid
            low = mid + 1
        else:
            # 太大，尝试更小的字体
            high = mid - 1
    
    # 如果没有找到合适的字体(文本太长)，使用最小字体
    if not best_font:
        best_font_size = min_font_size
        best_font = ImageFont.truetype(font_path, best_font_size) if font_path else ImageFont.load_default(best_font_size)
        
        # 强制分割文本
        avg_char_width = best_font.getlength("a")
        chars_per_line = int(text_area_width / avg_char_width)
        best_lines = textwrap.wrap(text, width=chars_per_line)
        
        # 如果设置了最大行数，截断多余行
        if max_lines and len(best_lines) > max_lines:
            best_lines = best_lines[:max_lines]
            best_lines[-1] = best_lines[-1][:30] + "..."  # 添加省略号
    
    # 创建新的图像，包含文字区域
    result = Image.new('RGBA', (image.width, image.height + int(region_height)), (0, 0, 0, 0))
    result.paste(image, (0, int(region_height) if position == 'top' else 0))
    
    # 创建文字背景
    text_bg = Image.new('RGBA', (region_width, int(region_height)), bg_color)
    
    # 计算文字绘制位置
    if position == 'top':
        result.paste(text_bg, (0, 0), text_bg)
        draw_y = padding
    elif position == 'bottom':
        result.paste(text_bg, (0, image.height), text_bg)
        draw_y = image.height + padding
    else:  # center
        center_y = (result.height - int(region_height)) // 2
        result.paste(text_bg, (0, center_y), text_bg)
        draw_y = center_y + padding
    
    # 绘制文本
    draw = ImageDraw.Draw(result)
    line_height = best_font.size * 1.2
    
    for i, line in enumerate(best_lines):
        # 计算文本宽度和位置(居中)
        text_width = best_font.getlength(line)
        x = (region_width - text_width) / 2
        y = draw_y + i * line_height
        
        draw.text((x, y), line, font=best_font, fill=text_color)
    
    return result


def _quaternion_to_forward_vector(q):
    """将单位四元数转换为旋转后的前向向量 (-Z轴)"""
    if isinstance(q, np.quaternion):
        w, x, y, z = q.w, q.x, q.y, q.z
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]
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
def execute_actions(sim, actions, args, trajectory, frame_rate=24, planner=None, show_answer=False):
    observations: List[np.ndarray] = []
    agent = sim.agents[0]
    
    question = args["question"]
    answer = args["answer"]
    next_direction = args["direction"]
    
    cam_intr = get_cam_intr(120, 480, 640)
    for action, amount in tqdm(actions):
        # 更新动作参数
        agent.agent_config.action_space[action].actuation.amount = amount
        # 执行动作
        agent.act(action)
        
        # 获取观察并写入视频
        obs = sim.get_sensor_observations()
        img = obs["color_sensor"]
        depth = obs["depth_sensor"]

        sensor = agent.get_state().sensor_states["depth_sensor"]
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(sensor.rotation)
        cam_pose[:3, 3] = sensor.position
        cam_pose_normal = pose_habitat_to_normal(cam_pose)
        cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

        if planner is not None:
            planner.integrate(
                color_im=img,
                depth_im=depth,
                cam_intr=cam_intr,
                cam_pose=cam_pose_tsdf,
                obs_weight=1.0,
                margin_h=int(0.6 * 480),
                margin_w=int(0.25 * 640),
            )
            pts_normal = pos_habitat_to_normal(sensor.position)
            island, unoccupied = planner.get_island_around_pts(pts_normal, height=0.4)
            cur_point = planner.world2vox(pts_normal)
            island_image = Image.fromarray(np.where(island, 255, 0).astype('uint8')).convert("RGB")
            trajectory.append((cur_point[1], cur_point[0]))
            island_image = add_draw_point(island_image, trajectory[-1], trajectory, 5)

        img = concatenate_images_horizontally(img, island_image)
        img = draw_text_fill_region(img, question)
        observations.append({"color": img})

    next_direction = None
    if next_direction is not None:
        obs = sim.get_sensor_observations()
        img = obs["color_sensor"]
        for frame in range(frame_rate):
            point_img = add_draw_point(img, next_direction)
            point_img = concatenate_images_horizontally(point_img, island_image)
            point_img = draw_text_fill_region(point_img, question)
            observations.append({"color": point_img})

    if answer is not None and show_answer:
        obs = sim.get_sensor_observations()
        img = obs["color_sensor"]
        for frame in range(frame_rate):
            answer_img = concatenate_images_horizontally(img, island_image)
            answer_img = draw_text_fill_region(answer_img, answer)
            observations.append({"color": answer_img})

    return observations, trajectory


def navigation_video(sim, agent, pathes, frame_rate=24.0, output_video="eqa.mp4"):
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
    pts_normal = pos_habitat_to_normal(pathes[0]["position"])
    floor_height = pts_normal[-1]
    tsdf_bnds, scene_size = get_scene_bnds(sim.pathfinder, floor_height)

    tsdf_planner = TSDFPlanner(
        vol_bnds=tsdf_bnds,
        voxel_size=0.1,
        floor_height_offset=0,
        pts_init=pts_normal,
        init_clearance=1,
    )
    trojectory = []
    for i in range(len(pathes)):
        if i == 0:
            continue
        start_point = pathes[i-1]
        end_point = pathes[i]

        shortest_path.requested_start = start_point["position"]
        shortest_path.requested_end = end_point["position"]

        found_path = sim.pathfinder.find_path(shortest_path)
        if not found_path:
            print("No valid path be founded between start and end positions!")
            return
        print(f"Path found with {len(shortest_path.points)} points")
        print(f"Path length: {shortest_path.geodesic_distance}")
        actions = path_to_actions(shortest_path.points, start_point["rotation"], end_point["rotation"])

        # obs = sim.get_sensor_observations()
        # cv2.imwrite("start_rgb.jpg", obs["color"][..., :3])
        # 设置起始位置和旋转
        if i < len(pathes) - 1:
            observation, trojectory = execute_actions(sim, actions, pathes[i], trojectory, planner=tsdf_planner)
        else:
            observation, trojectory = execute_actions(sim, actions, pathes[i], trojectory, planner=tsdf_planner, show_answer=True)
        observations.extend(observation)

    if len(observations) < 2:
        print("Not enough frames captured for video!")
        return

    # 创建视频录制器
    vut.make_video(
        observations=observations,
        primary_obs="color",
        primary_obs_type="color",
        video_file=output_video,
        fps=frame_rate,
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
        {"position": position1, "rotation": rotation1, "question": "QUESTION: Where is the TV in the bed room?", "answer": "ANSWER", "confidence": "YES", "direction": [320, 240]},
        {"position": position2, "rotation": rotation2, "question": "QUESTION: Where is the TV in the bed room?", "answer": "ANSWER", "confidence": "YES", "direction": [320, 240]},
        {"position": position3, "rotation": rotation3, "question": "QUESTION: Where is the TV in the bed room?", "answer": "ANSWER: The TV is on the cabinet next to the window.", "confidence": "YES", "direction": [320, 240]},
    ]

    # 初始化模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = False  # 不需要物理引擎

    # 2. 配置RGB传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [480, 640]
    rgb_sensor_spec.hfov = 120.0  # 水平视场角
    rgb_sensor_spec.position = mn.Vector3(0, 1.5, 0)  # 相对于代理的位置
    rgb_sensor_spec.orientation = mn.Vector3(0, 0, 0)

    # 3. 配置DEPTH传感器
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]
    depth_sensor_spec.hfov = 120.0
    depth_sensor_spec.position = mn.Vector3(0, 1.5, 0)
    depth_sensor_spec.orientation = mn.Vector3(0, 0, 0)
    
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
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
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

    navigation_video(sim, agent, pathes, output_video=output_video)