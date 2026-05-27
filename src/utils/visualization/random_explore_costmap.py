"""
随机探索场景，积分 TSDF 得到 costmap（占据网格图），并绘制墙壁边界。

用法:
  python src/utils/visualization/random_explore_costmap.py \
    --scene data/HM3D/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb \
    --steps 500 --output ./tsdf_costmap.png
"""

import numpy as np
import habitat_sim
import matplotlib.pyplot as plt
import quaternion
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.planner.tsdf import TSDFPlanner
from src.utils.geom import get_scene_bnds, get_cam_intr
from src.utils.habitat import pos_habitat_to_normal, pose_habitat_to_normal, pose_normal_to_tsdf


def make_sensor_cfg(scene_path, width=640, height=480, hfov=70, sensor_height=1.2):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.position = [0.0, sensor_height, 0.0]
    rgb_spec.hfov = hfov

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, sensor_height, 0.0]
    depth_spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_cam_pose_tsdf(sim):
    sensor_state = sim.get_agent(0).get_state().sensor_states["color_sensor"]
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    cam_pose[:3, 3] = sensor_state.position
    cam_pose_normal = pose_habitat_to_normal(cam_pose)
    return pose_normal_to_tsdf(cam_pose_normal)


def get_cur_position(sim):
    return sim.get_agent(0).get_state().position


def random_explore_and_integrate(scene_path, num_steps=500, voxel_size=0.05,
                                  width=640, height=480, hfov=70):
    cfg = make_sensor_cfg(scene_path, width, height, hfov)
    sim = habitat_sim.Simulator(cfg)

    if not sim.pathfinder.is_loaded:
        sim.close()
        raise RuntimeError("Pathfinder 未加载")

    floor_height = sim.pathfinder.get_bounds()[0][1]
    tsdf_bnds, scene_size = get_scene_bnds(sim.pathfinder, floor_height)
    print(f"TSDF bounds: {tsdf_bnds}")
    print(f"Scene size: {scene_size:.1f} m2")

    init_pt = sim.pathfinder.get_random_navigable_point()
    print(f"初始位置: {init_pt}")

    init_pt_normal = pos_habitat_to_normal(init_pt)
    planner = TSDFPlanner(
        vol_bnds=tsdf_bnds,
        voxel_size=voxel_size,
        floor_height_offset=floor_height,
        pts_init=init_pt_normal,
        init_clearance=0.5,
    )

    cam_intr = get_cam_intr(hfov, height, width)

    agent = sim.get_agent(0)
    init_state = agent.get_state()
    init_state.position = init_pt
    agent.set_state(init_state)

    obs = sim.get_sensor_observations()
    cam_pose_tsdf = get_cam_pose_tsdf(sim)
    planner.integrate(obs["color_sensor"], obs["depth_sensor"], cam_intr, cam_pose_tsdf,
                      obs_weight=1.0)
    integrated = 1

    stuck_count = 0
    last_pos = get_cur_position(sim)

    for step in range(num_steps):
        action_choice = np.random.random()

        if action_choice < 0.65:
            sim.step("move_forward")
        elif action_choice < 0.85:
            sim.step("turn_left")
        else:
            sim.step("turn_right")

        cur_pos = get_cur_position(sim)
        if np.linalg.norm(cur_pos - last_pos) < 0.01:
            stuck_count += 1
        else:
            stuck_count = 0
        last_pos = cur_pos

        if stuck_count > 20:
            new_pt = sim.pathfinder.get_random_navigable_point()
            state = agent.get_state()
            state.position = new_pt
            agent.set_state(state)
            stuck_count = 0

        cam_pose_tsdf = get_cam_pose_tsdf(sim)
        obs = sim.get_sensor_observations()
        planner.integrate(obs["color_sensor"], obs["depth_sensor"], cam_intr, cam_pose_tsdf,
                          obs_weight=1.0)
        integrated += 1

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{num_steps}, integrated {integrated} frames")

    sim.close()

    # --- 从 TSDF 体积提取 2D costmap ---
    tsdf_vol = planner._tsdf_vol_cpu
    explored_3d = planner._explore_vol_cpu

    explored = np.sum(explored_3d, axis=-1) > 0

    # 自由空间：已探索且 TSDF 最小值 < 0
    unoccupied = np.zeros(explored.shape, dtype=bool)
    unoccupied[explored] = np.min(tsdf_vol[explored], axis=-1) < 0

    # 障碍物：已探索且 TSDF 最大值 > 0
    occupied = np.zeros(explored.shape, dtype=bool)
    occupied[explored] = np.max(tsdf_vol[explored], axis=-1) > 0

    # 保留膨胀前的二值图用于墙壁检测
    binary_free = unoccupied.copy()
    binary_occupied = occupied.copy()

    # costmap: 0=free, 100=obstacle, -1=unknown
    costmap = np.full(explored.shape, -1, dtype=np.int32)
    costmap[unoccupied] = 0
    costmap[occupied] = 100

    # 障碍物膨胀
    from scipy.ndimage import distance_transform_edt
    inflation_radius = 0.3
    inflation_pixels = int(inflation_radius / voxel_size)
    if inflation_pixels > 0:
        dist_to_free = distance_transform_edt(costmap == 0)
        for px in range(1, inflation_pixels + 1):
            mask = (dist_to_free <= px) & (costmap != 100) & (costmap != 0)
            cost = 100.0 * (1.0 - px / (inflation_pixels + 1))
            costmap[mask] = np.maximum(costmap[mask].astype(float), cost)

    costmap = costmap.astype(np.int32)

    return costmap, planner, explored, binary_free, binary_occupied


def display_costmap(costmap, explored_mask, binary_free, binary_occupied,
                     output_path="./tsdf_costmap.png", planner=None):
    from scipy.ndimage import binary_dilation

    plt.style.use("seaborn-v0_8-white")

    # 墙壁像素 = 障碍物与自由空间的交界处（取障碍物紧邻自由的那一层像素作为墙壁）
    wall_mask = binary_occupied & binary_dilation(binary_free, iterations=1)

    light_green = np.array([0.56, 0.83, 0.60, 1.0])
    wall_color  = np.array([0.0,  0.0,  0.0,  1.0])  # 纯黑墙壁
    dark_obs    = np.array([0.15, 0.15, 0.15, 1.0])

    rgba = np.zeros(costmap.shape + (4,), dtype=np.float32)
    rgba[binary_free] = light_green
    rgba[binary_occupied] = dark_obs
    rgba[wall_mask] = wall_color  # 墙壁覆盖在障碍物边界上
    # unknown 保持 [0,0,0,0] 透明

    wall_px_count = wall_mask.sum()
    print(f"墙壁像素: {wall_px_count}")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgba, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none", transparent=True)
    print(f"costmap 已保存: {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str,
                        default="data/HM3D/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="./tsdf_costmap.png")
    args = parser.parse_args()

    costmap, planner, explored, binary_free, binary_occupied = random_explore_and_integrate(
        args.scene, num_steps=args.steps, voxel_size=args.voxel_size
    )

    display_costmap(costmap, explored, binary_free, binary_occupied, args.output, planner)
    print(f"自由空间: {(costmap == 0).sum()}, 障碍物: {(costmap == 100).sum()}, "
          f"未知: {(costmap == -1).sum()}")
