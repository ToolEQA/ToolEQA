#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time
from dataclasses import dataclass, field
from typing import List, Literal, Dict, Set, Tuple, Optional, Callable, Any

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt

# ================== 类型定义 ==================

Action = Literal["move_forward", "turn_left", "turn_right"]
ObstacleRegion = Literal["LEFT", "CENTER", "RIGHT"]


@dataclass
class Pose:
    x: float
    y: float
    yaw: float  # 弧度


@dataclass
class Node:
    id: str
    visit_count: int = 0
    explored_directions: Set[int] = field(default_factory=set)


@dataclass
class ExplorerState:
    pose: Pose = Pose(0.0, 0.0, 0.0)

    history_actions: List[Action] = field(default_factory=list)
    last_action: Optional[Action] = None
    oscillation_counter: int = 0
    max_oscillation: int = 2

    graph: Dict[str, Node] = field(default_factory=dict)
    grid_resolution: float = 0.5
    revisit_threshold: int = 3

    recent_node_ids: List[str] = field(default_factory=list)
    recent_poses: List[Pose] = field(default_factory=list)

    full_poses: List[Pose] = field(default_factory=list)


@dataclass
class ObstacleInfo:
    blocked: Dict[ObstacleRegion, bool]
    nearest_distance: Dict[ObstacleRegion, Optional[float]]


# ================== 相机参数（基于你提供的信息） ==================

# 分辨率：宽 1280，高 720
IMG_W = 1280
IMG_H = 720

# 内参（如果有标定结果，直接替换为你的 fx, fy, cx, cy）
# CAM_FX = 900.0
# CAM_FY = 900.0
# CAM_CX = IMG_W / 2.0      # 640.0
# CAM_CY = IMG_H * 0.55     # 大约 396，略偏下

CAM_FX = 528.7459106445312 
CAM_FY = 528.7459106445312 
CAM_CX = 652.0160522460938 
CAM_CY = 364.0149841308594

# 相机安装信息
CAM_HEIGHT = 0.5          # 相机离地 0.5 米
CAM_PITCH = 0.0           # 俯仰角 0°，平行于地面（此处仅保留符号，不在投影中旋转）


# ================== 通用工具函数 ==================

def _distance(p1: Pose, p2: Pose) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def discretize_pose_to_node_id(pose: Pose, res: float) -> str:
    ix = round(pose.x / res)
    iy = round(pose.y / res)
    return f"{ix},{iy}"


def quaternion_to_euler_xyz(q):
    qw, qx, qy, qz = q
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qz)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def pose_from_obs(position, rotation) -> Pose:
    px, py, pz = position
    _, _, yaw = quaternion_to_euler_xyz(rotation)
    return Pose(x=px, y=py, yaw=yaw)


# ================== 深度障碍检测（左右中） ==================

def detect_obstacles_from_depth(
    depth: np.ndarray,
    obstacle_thresh: float = 0.5,
    invalid_value: float = 0.0,
) -> ObstacleInfo:
    """
    利用深度图识别左右中是否有障碍，并估计最近障碍距离。
    depth: 2D array, 单位为米。
    """
    if depth.ndim != 2:
        raise ValueError("depth must be 2D (H, W)")

    H, W = depth.shape
    d = depth.astype(np.float32).copy()
    d[(d == invalid_value) | (~np.isfinite(d))] = 0.0

    # 使用下半部分
    h0 = H // 4
    h1 = H

    w_left = (0, W // 3)
    w_center = (W // 3, 2 * W // 3)
    w_right = (2 * W // 3, W)

    regions = {
        "LEFT": w_left,
        "CENTER": w_center,
        "RIGHT": w_right,
    }

    blocked: Dict[ObstacleRegion, bool] = {k: False for k in regions.keys()}  # type: ignore
    nearest_distance: Dict[ObstacleRegion, Optional[float]] = {k: None for k in regions.keys()}  # type: ignore

    for name, (w0, w1) in regions.items():
        region = d[h0:h1, w0:w1]
        if region.size == 0:
            continue

        valid = region[region > 0]
        if valid.size == 0:
            continue

        near_mask = valid < obstacle_thresh
        near_ratio = float(np.mean(near_mask))

        blocked[name] = near_ratio > 0.10
        nearest_distance[name] = float(np.min(valid))

    return ObstacleInfo(blocked=blocked, nearest_distance=nearest_distance)


# ================== 深度投影到地面坐标 ==================

def project_depth_obstacles_to_world_xy(
    depth: np.ndarray,
    robot_pose: Pose,
    obstacle_thresh: float = 0.5,
    invalid_value: float = 0.0,
    max_points: int = 400,
) -> np.ndarray:
    """
    从深度图中抽样“近距离障碍点”，投影到世界坐标平面 (x, y)。

    假设：
      - 相机看向正前方，平行地面；
      - 世界/机器人坐标： x 前, y 左, z 上；
      - 深度为“相机前方方向”的距离（典型 RGB-D 相机标定）。

    近似做法：
      - 只使用图像下半部分的一条/几条水平带；
      - 将这些像素的深度 1:1 当作“前向距离”（Z_cam），并用针孔模型计算左右偏移；
      - 将相机视为安装在机器人头部，z=CAM_HEIGHT；
      - 将 (X_cam, Z_cam) 视为机器人坐标中的 (右, 前)，再转换成世界坐标 (x,y)。
    """
    H, W = depth.shape
    d = depth.astype(np.float32).copy()
    d[(d == invalid_value) | (~np.isfinite(d))] = 0.0

    # 只看靠近底部的一块区域（比如最后 1/3 高度）
    h0 = int(H * 2 / 3)
    h1 = H

    # 近距离障碍
    mask_obstacle = (d > 0) & (d < obstacle_thresh)
    mask_region = np.zeros_like(mask_obstacle, dtype=bool)
    mask_region[h0:h1, :] = True
    final_mask = mask_obstacle & mask_region

    ys, xs = np.where(final_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # 抽样
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    zs = d[ys, xs]  # 看作沿前向的距离 Z_cam

    # 像素 -> 相机坐标 (X_cam, Y_cam, Z_cam)
    #  X_cam：右正，Y_cam：下正，Z_cam：前正（典型 RGB-D）
    Xc = (xs - CAM_CX) * zs / CAM_FX   # 左右偏移
    Yc = (ys - CAM_CY) * zs / CAM_FY   # 上下偏移（暂不使用）
    Zc = zs

    # 相机 -> 机器人坐标
    # 这里近似认为相机坐标与机器人坐标差一个平移：
    #   机器人坐标：Xr 前, Yr 左, Zr 上
    #   相机坐标：  Zc 前, Xc 右 => 左 = -右
    Xr = Zc                # 前
    Yr = -Xc               # 左
    Zr = CAM_HEIGHT - Yc * 0.0  # 粗略忽略 Yc 对高度的影响，直接认为障碍高度 ≈ 相机高度以下

    # 机器人 -> 世界坐标
    cos_yaw = math.cos(robot_pose.yaw)
    sin_yaw = math.sin(robot_pose.yaw)

    Xw = robot_pose.x + cos_yaw * Xr - sin_yaw * Yr
    Yw = robot_pose.y + sin_yaw * Xr + cos_yaw * Yr

    world_points = np.stack([Xw, Yw], axis=-1).astype(np.float32)
    return world_points


# ================== 俯视图画前方障碍扇形 ==================

def _draw_obstacle_sector_on_ax(
    ax,
    pose: Pose,
    nearest_front_distance: float,
    max_draw_dist: float = 2.0,
    fov_deg: float = 60.0,
    color: str = "red",
    alpha: float = 0.25,
):
    import numpy as np

    if nearest_front_distance is None or nearest_front_distance <= 0:
        return

    r = min(nearest_front_distance, max_draw_dist)

    half_fov = math.radians(fov_deg / 2.0)
    angles = np.linspace(pose.yaw - half_fov, pose.yaw + half_fov, 30)
    xs = pose.x + r * np.cos(angles)
    ys = pose.y + r * np.sin(angles)

    poly_x = [pose.x] + xs.tolist() + [pose.x]
    poly_y = [pose.y] + ys.tolist() + [pose.y]

    ax.fill(poly_x, poly_y, color=color, alpha=alpha, edgecolor=None)


# ================== DFSExplorer 主体 ==================

class DFSExplorer:
    def __init__(
        self,
        call_vlm_fn: Callable[[str, Any], str],
        action_fn: Callable[[Action], None],
        get_observation_fn: Callable[[], Tuple[Any, Any, List[float], List[float]]],
        grid_resolution: float = 0.5,
        revisit_threshold: int = 3,
        max_oscillation: int = 2,
        history_limit: int = 50,
        stuck_window: int = 20,
        stuck_radius: float = 0.8,
        escape_steps: int = 8,
        move_eps: float = 0.10,
        prefer_turn: Action = "turn_left",
        max_turn_attempts: int = 6,
        depth_invalid_value: float = 0.0,
        obstacle_distance_thresh: float = 0.5,
    ) -> None:

        self.call_vlm_fn = call_vlm_fn
        self.action_fn = action_fn
        self.get_observation_fn = get_observation_fn
        self.global_obstacle_points: List[np.ndarray] = []

        self.state = ExplorerState(
            grid_resolution=grid_resolution,
            revisit_threshold=revisit_threshold,
            max_oscillation=max_oscillation,
        )
        self.history_limit = history_limit

        self.stuck_window = stuck_window
        self.stuck_radius = stuck_radius
        self.default_escape_steps = escape_steps
        self.escape_steps_remaining = 0

        self.MOVE_EPS = move_eps
        self.prefer_turn = prefer_turn
        self.max_turn_attempts = max_turn_attempts

        self.depth_invalid_value = depth_invalid_value
        self.obstacle_distance_thresh = obstacle_distance_thresh

        self.last_obstacle_info: Optional[ObstacleInfo] = None
        self.last_depth: Optional[np.ndarray] = None  # 用于投影障碍

    # ---------- 观测封装 ----------

    def _get_current_obs_and_pose(self) -> Tuple[Any, np.ndarray, Pose]:
        obs = self.get_observation_fn()
        image = obs["color_sensor"]
        depth = obs["depth_sensor"]
        position = obs["highstate"]["position"]
        rotation = obs["highstate"]["imu_quaternion"]

        depth_np = np.array(depth)
        # 你的深度是毫米：uint16
        if depth_np.dtype == np.uint16 or depth_np.dtype == np.int32 or depth_np.dtype == np.int16:
            depth_np = depth_np.astype(np.float32) / 1000.0  # 毫米 -> 米
        else:
            depth_np = depth_np.astype(np.float32)

        pose = pose_from_obs(position, rotation)
        return image, depth_np, pose

    # ---------- 节点 & stuck 状态维护 ----------

    def _update_node_with_pose(self, pose: Pose) -> Node:
        self.state.pose = pose

        node_id = discretize_pose_to_node_id(
            self.state.pose, self.state.grid_resolution
        )
        node = self.state.graph.get(node_id)
        if node is None:
            node = Node(id=node_id, visit_count=0)
            self.state.graph[node_id] = node
        node.visit_count += 1

        self.state.recent_node_ids.append(node_id)
        self.state.recent_poses.append(Pose(pose.x, pose.y, pose.yaw))
        if len(self.state.recent_node_ids) > self.stuck_window:
            self.state.recent_node_ids = self.state.recent_node_ids[-self.stuck_window:]
            self.state.recent_poses = self.state.recent_poses[-self.stuck_window:]

        self.state.full_poses.append(Pose(pose.x, pose.y, pose.yaw))

        return node

    def _is_stuck(self) -> bool:
        if len(self.state.recent_node_ids) < self.stuck_window:
            return False

        unique_nodes = set(self.state.recent_node_ids)
        if len(unique_nodes) > 2:
            return False

        xs = [p.x for p in self.state.recent_poses]
        ys = [p.y for p in self.state.recent_poses]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        radius = max(dx, dy)

        return radius < self.stuck_radius

    # ---------- escape 模式 ----------

    def _plan_escape_action(self) -> Action:
        used_steps = self.default_escape_steps - self.escape_steps_remaining
        if used_steps < 2:
            return "turn_left"
        else:
            return "move_forward"

    # ---------- 历史/振荡过滤 ----------

    def _trim_history(self):
        if len(self.state.history_actions) > self.history_limit:
            self.state.history_actions = self.state.history_actions[-self.history_limit:]

    def _filter_action(self, vlm_action: Action, in_escape: bool) -> Action:
        last = self.state.last_action

        if not in_escape:
            if last is not None:
                is_osc = (
                    (last == "turn_left" and vlm_action == "turn_right") or
                    (last == "turn_right" and vlm_action == "turn_left")
                )
                if is_osc:
                    self.state.oscillation_counter += 1
                else:
                    self.state.oscillation_counter = 0
            else:
                self.state.oscillation_counter = 0

            if self.state.oscillation_counter >= self.state.max_oscillation:
                chosen: Action = "move_forward"
                self.state.oscillation_counter = 0
            else:
                chosen = vlm_action
        else:
            chosen = vlm_action

        self.state.last_action = chosen
        self.state.history_actions.append(chosen)
        self._trim_history()
        return chosen

    # ---------- 几何优先决策 ----------

    def _geometry_pre_decision(self, depth: np.ndarray) -> Tuple[Optional[Action], ObstacleInfo]:
        obs = detect_obstacles_from_depth(
            depth,
            obstacle_thresh=self.obstacle_distance_thresh,
            invalid_value=self.depth_invalid_value,
        )

        self.last_obstacle_info = obs

        front_blocked = obs.blocked["CENTER"]
        left_blocked = obs.blocked["LEFT"]
        right_blocked = obs.blocked["RIGHT"]

        # front_blocked |= front_blocked_by_map
        # left_blocked  |= left_blocked_by_map
        # right_blocked |= right_blocked_by_map

        front_free = not front_blocked
        left_free = not left_blocked
        right_free = not right_blocked

        if front_free and not left_free and not right_free:
            return "move_forward", obs

        if not front_free and left_free and not right_free:
            return "turn_left", obs

        if not front_free and not left_free and right_free:
            return "turn_right", obs

        return None, obs

    # ---------- 前进 + 恢复 ----------

    def _try_forward_with_recovery(self) -> Action:
        _, _, start_pose = self._get_current_obs_and_pose()

        try:
            self.action_fn("move_forward")
        except Exception as e:
            print("[DFSExplorer] action_fn move_forward failed:", e)
            return "move_forward"

        time.sleep(0.05)
        _, _, new_pose = self._get_current_obs_and_pose()
        move_dist = _distance(start_pose, new_pose)

        self.state.pose = new_pose
        self.state.history_actions.append("move_forward")
        self.state.last_action = "move_forward"
        self._trim_history()

        if move_dist >= self.MOVE_EPS:
            return "move_forward"

        for _ in range(self.max_turn_attempts):
            try:
                self.action_fn(self.prefer_turn)
            except Exception as e:
                print("[DFSExplorer] action_fn turn failed:", e)
                break

            time.sleep(0.05)
            _, _, turned_pose = self._get_current_obs_and_pose()
            self.state.pose = turned_pose
            self.state.history_actions.append(self.prefer_turn)
            self.state.last_action = self.prefer_turn
            self._trim_history()

            try:
                self.action_fn("move_forward")
            except Exception as e:
                print("[DFSExplorer] action_fn move_forward failed:", e)
                break

            time.sleep(0.05)
            _, _, after_forward_pose = self._get_current_obs_and_pose()
            move_dist = _distance(turned_pose, after_forward_pose)

            self.state.pose = after_forward_pose
            self.state.history_actions.append("move_forward")
            self.state.last_action = "move_forward"
            self._trim_history()

            if move_dist >= self.MOVE_EPS:
                return "move_forward"

        return self.state.last_action or "move_forward"

    # ---------- Prompt 生成 ----------

    def _build_prompt(
        self,
        current_node: Node,
        rgb_image_description: str,
        in_stuck: bool,
        obs: ObstacleInfo,
    ) -> str:
        history_str = (
            ", ".join(self.state.history_actions[-10:])
            if self.state.history_actions
            else "none"
        )

        if in_stuck:
            extra = (
                "The robot seems to be trapped in a small area and keeps revisiting the same places. "
                "You MUST prefer actions that clearly move the robot forward and out of this region, "
                "even if it means ignoring minor side branches. Avoid turning in place unless forward is clearly blocked."
            )
        else:
            extra = ""

        if current_node.visit_count >= self.state.revisit_threshold:
            node_comment = (
                f"You have already visited a very similar location about "
                f"{current_node.visit_count} times. "
                "Try to choose actions that help you leave this small region and explore new areas."
            )
        else:
            node_comment = (
                f"This location has been visited {current_node.visit_count} times. "
                "You can continue exploring around, but avoid unnecessary turning."
            )

        def fmt_dist(d: Optional[float]) -> str:
            return "unknown" if d is None else f"{d:.2f} m"

        geometry_text = f"""
Depth-based geometry info (approximate):
- Nearest obstacle in FRONT direction: {fmt_dist(obs.nearest_distance['CENTER'])}
- Nearest obstacle on the LEFT side: {fmt_dist(obs.nearest_distance['LEFT'])}
- Nearest obstacle on the RIGHT side: {fmt_dist(obs.nearest_distance['RIGHT'])}

If the distance is below {self.obstacle_distance_thresh:.2f} m, you should treat that direction as blocked.
"""

        prompt = f"""
You are controlling a quadruped robot to explore an unknown environment using ONLY three discrete actions:
- move_forward: move forward about 0.5 meter
- turn_left: rotate about 45 degrees to the left (in place)
- turn_right: rotate about 45 degrees to the right (in place)

Your goal is to explore as much NEW area as possible, similar to a depth-first search strategy:
- At each place, prefer to keep moving forward if the path is safe, instead of turning left or right.
- Avoid oscillating between left and right turns (like LEFT, RIGHT, LEFT, RIGHT).
- If you have already visited a place many times, strongly prefer actions that lead you to new, unexplored space.

{extra}

Current first-person observation:
{rgb_image_description}

{geometry_text}

Recent actions (oldest to newest):
{history_str}

Location information:
{node_comment}

Decision rules:
1. If the way forward is not blocked by an obstacle (according to geometry info), PREFER choosing move_forward.
2. Only choose turn_left or turn_right if there is a clear reason (e.g., obstacle directly ahead, or a clearly open side-corridor).
3. DO NOT alternate between left and right turns without moving forward in between.

Now, based on the current first-person view and the above rules, output ONE action from:
move_forward, turn_left, turn_right

You must answer with exactly one of these three tokens, without any additional text.
"""
        return prompt.strip()

    # ---------- 单步 ----------

    def run_step(
        self,
        image: Any,
        depth: np.ndarray,
        pose: Pose,
        rgb_image_description: str = "Current first-person RGB image.",
        command: Optional[str] = None,
    ) -> Optional[Action]:
        self.last_depth = depth.copy()  # 保存本帧深度用于可视化

        # 2) 投影本帧障碍点到世界坐标，并累积到全局列表
        obs_pts = project_depth_obstacles_to_world_xy(
            depth,
            robot_pose=pose,
            obstacle_thresh=self.obstacle_distance_thresh,
            invalid_value=self.depth_invalid_value,
            max_points=400,
        )
        if obs_pts.shape[0] > 0:
            self.global_obstacle_points.append(obs_pts)

        current_node = self._update_node_with_pose(pose)
        self.visualize_path_with_obstacles(save_path="visualize.png")

        in_stuck = self._is_stuck()
        if in_stuck and self.escape_steps_remaining == 0:
            self.escape_steps_remaining = self.default_escape_steps
            print("[DFSExplorer] Detected STUCK, entering escape mode for",
                  self.default_escape_steps, "steps")

        in_escape = self.escape_steps_remaining > 0

        if in_escape:
            planned = self._plan_escape_action()
            final = self._filter_action(planned, in_escape=True)
            self.escape_steps_remaining -= 1

            try:
                if final == "move_forward":
                    real_action = self._try_forward_with_recovery()
                    return real_action
                else:
                    self.action_fn(final)
                    _, _, new_pose = self._get_current_obs_and_pose()
                    self.state.pose = new_pose
                    return final
            except Exception as e:
                print("[DFSExplorer] escape action_fn failed:", e)
                return None

        geom_action, obs = self._geometry_pre_decision(depth)
        if geom_action is not None:
            final_action = self._filter_action(geom_action, in_escape=False)
            try:
                if final_action == "move_forward":
                    real_action = self._try_forward_with_recovery()
                    return real_action
                else:
                    self.action_fn(final_action)
                    _, _, new_pose = self._get_current_obs_and_pose()
                    self.state.pose = new_pose
                    return final_action
            except Exception as e:
                print("[DFSExplorer] geometry action_fn failed:", e)
                return None
        else:
            obs_info = obs

        prompt = self._build_prompt(
            current_node=current_node,
            rgb_image_description=rgb_image_description,
            in_stuck=in_stuck,
            obs=obs_info,
        )

        try:
            raw = self.call_vlm_fn(prompt, image)
            raw_action_str = raw.strip() if isinstance(raw, str) else str(raw).strip()
        except Exception as e:
            print("[DFSExplorer] VLM call failed:", e)
            return None

        if raw_action_str not in ("move_forward", "turn_left", "turn_right"):
            print("[DFSExplorer] Invalid action from VLM:", raw_action_str)
            return None

        vlm_action: Action = raw_action_str
        final_action = self._filter_action(vlm_action, in_escape=False)

        try:
            if final_action == "move_forward":
                real_action = self._try_forward_with_recovery()
                return real_action
            else:
                self.action_fn(final_action)
                _, _, new_pose = self._get_current_obs_and_pose()
                self.state.pose = new_pose
                return final_action
        except Exception as e:
            print("[DFSExplorer] action_fn failed:", e)
            return None

    # ---------- 循环 ----------

    def run_loop(
        self,
        rgb_desc_func: Optional[Callable[[Any], str]] = None,
        step_delay: float = 0.0,
    ):
        while True:
            image, depth, pose = self._get_current_obs_and_pose()

            if rgb_desc_func is not None:
                desc = rgb_desc_func(image)
            else:
                desc = "Current first-person RGB image."

            action = self.run_step(
                image,
                depth,
                pose,
                rgb_image_description=desc,
            )
            if action is None:
                print("[DFSExplorer] step returned None, continue...")

            if step_delay > 0:
                time.sleep(step_delay)

    # ---------- 路径 + 障碍 在同一张图中可视化 ----------

    def visualize_path_with_obstacles(
        self,
        use_full_path: bool = True,
        save_path: str = "path_and_obstacles.png",
    ):
        poses = (
            self.state.full_poses
            if use_full_path and len(self.state.full_poses) > 0
            else self.state.recent_poses
        )

        if not poses:
            print("[DFSExplorer] No poses to visualize yet.")
            return

        xs = [p.x for p in poses]
        ys = [p.y for p in poses]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal', 'box')
        ax.set_title("Path + Projected Obstacles")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # 1) 访问栅格
        for node_id, node in self.state.graph.items():
            ix_str, iy_str = node_id.split(',')
            ix, iy = int(ix_str), int(iy_str)
            cell_x = ix * self.state.grid_resolution
            cell_y = iy * self.state.grid_resolution
            ax.add_patch(
                plt.Rectangle(
                    (cell_x - self.state.grid_resolution / 2,
                     cell_y - self.state.grid_resolution / 2),
                    self.state.grid_resolution,
                    self.state.grid_resolution,
                    facecolor='lightgray',
                    edgecolor='none',
                    alpha=min(0.1 + 0.1 * node.visit_count, 0.6),
                )
            )

        # 2) 路径
        ax.plot(xs, ys, color='blue', linewidth=2, label="Path")
        start_x, start_y = xs[0], ys[0]
        end_x, end_y = xs[-1], ys[-1]
        ax.scatter(start_x, start_y, color='green', s=50, label="Start")
        ax.scatter(end_x, end_y, color='red', s=50, label="Current")

        # 当前朝向箭头
        current_pose = poses[-1]
        arrow_len = self.state.grid_resolution * 0.8
        ax.arrow(
            current_pose.x,
            current_pose.y,
            arrow_len * math.cos(current_pose.yaw),
            arrow_len * math.sin(current_pose.yaw),
            head_width=self.state.grid_resolution * 0.4,
            head_length=self.state.grid_resolution * 0.4,
            fc='red',
            ec='red',
        )

        # 3) 前方障碍扇形
        if self.last_obstacle_info is not None:
            nf = self.last_obstacle_info.nearest_distance["CENTER"]
            _draw_obstacle_sector_on_ax(
                ax,
                pose=current_pose,
                nearest_front_distance=nf if nf is not None else 0.0,
                max_draw_dist=2.0,
                fov_deg=60.0,
                color="red",
                alpha=0.20,
            )

        # 4) 投影障碍点
        if len(self.global_obstacle_points) > 0:
            all_pts = np.concatenate(self.global_obstacle_points, axis=0)
            ax.scatter(
                all_pts[:, 0], all_pts[:, 1],
                s=5,
                c='black',
                alpha=0.4,
                label="Obstacle points (all frames)",
            )

        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.3)

        margin = self.state.grid_resolution * 2
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[DFSExplorer] Path+Obstacle visualization saved to: {save_path}")
