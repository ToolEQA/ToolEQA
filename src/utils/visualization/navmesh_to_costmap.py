"""
从 navmesh 生成 costmap（占据网格代价图）。
- 白色 = 自由空间 (cost=0)
- 黑色 = 致命障碍物 (cost=100)
- 灰度渐变 = 障碍物膨胀区域 (cost=1~99)
"""

import numpy as np
import habitat_sim
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from pathlib import Path


def navmesh_to_costmap(scene_glb_path, meters_per_pixel=0.05, inflation_radius=0.3):
    """
    从 navmesh 生成 costmap。

    Args:
        scene_glb_path: HM3D .glb 场景文件路径
        meters_per_pixel: 地图分辨率（米/像素）
        inflation_radius: 障碍物膨胀半径（米）

    Returns:
        costmap: 2D numpy array, 值域 0-100
    """
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_glb_path
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    if not sim.pathfinder.is_loaded:
        raise RuntimeError("Navmesh 未加载")

    bounds = sim.pathfinder.get_bounds()
    print(f"NavMesh bounds: {bounds}")

    height = bounds[0][1]  # 使用导航网格最低点作为切片高度
    topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    sim.close()

    # topdown_map: True = 可通行, False = 障碍物
    # 转为 costmap: 自由=0, 障碍=100
    costmap = np.where(topdown_map, 0, 100).astype(np.float32)

    print(f"costmap 尺寸: {costmap.shape}, "
          f"自由空间: {(costmap == 0).sum()}, 障碍物: {(costmap == 100).sum()}")

    # 障碍物膨胀 —— 距离变换
    inflation_pixels = int(inflation_radius / meters_per_pixel)
    if inflation_pixels > 0:
        dist_to_obstacle = distance_transform_edt(costmap == 0)
        # 在膨胀半径内，cost 从 100（紧邻障碍物）线性递减到 1（膨胀边界）
        for px in range(1, inflation_pixels + 1):
            mask = (dist_to_obstacle <= px) & (costmap != 100)
            cost = 100.0 * (1.0 - px / (inflation_pixels + 1))
            costmap[mask] = np.maximum(costmap[mask], cost)

    # 将障碍物设为最高代价
    costmap = np.clip(costmap.astype(np.int32), 0, 100)

    return costmap


def display_costmap(costmap, output_path="./costmap.png"):
    """可视化 costmap 并保存"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：灰度 costmap
    ax1 = axes[0]
    im1 = ax1.imshow(costmap, cmap="gray", vmin=0, vmax=100, origin="lower")
    ax1.set_title("Costmap (gray)", fontsize=14)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, label="Cost")

    # 右图：彩色 costmap（模仿 ROS costmap 配色）
    from matplotlib.colors import ListedColormap

    colors = ["white", "lightgray"]  # 0=free, 1-99=inflated
    # 在 inflated 区域加渐变
    n_inflated = 99
    gray_gradient = [(i / 100, i / 100, i / 100) for i in range(100, 0, -1)]
    # 0: white, 1-99: gray gradient, 100: black
    full_colors = [(1, 1, 1)] + gray_gradient + [(0, 0, 0)]
    cmap = ListedColormap(full_colors)

    ax2 = axes[1]
    im2 = ax2.imshow(costmap, cmap=cmap, vmin=-0.5, vmax=100.5, origin="lower")
    ax2.set_title("Costmap (ROS style)", fontsize=14)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, label="Cost")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"costmap 已保存: {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", type=str,
        default="data/HM3D/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb",
        help="HM3D 场景 .glb 文件路径"
    )
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="地图分辨率 米/像素")
    parser.add_argument("--inflation", type=float, default=0.3,
                        help="障碍物膨胀半径 米")
    parser.add_argument("--output", type=str, default="./costmap.png")
    args = parser.parse_args()

    costmap = navmesh_to_costmap(
        args.scene,
        meters_per_pixel=args.resolution,
        inflation_radius=args.inflation,
    )
    display_costmap(costmap, args.output)
