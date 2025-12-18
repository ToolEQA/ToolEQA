import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from open3d.pipelines import odometry, registration, integration

import numpy as np
import os
import matplotlib.pyplot as plt


class RGBDMapper:
    def __init__(self,
                 fx, fy, cx, cy,
                 width, height,
                 depth_scale=1000.0,
                 depth_trunc=4.0,
                 voxel_length=0.02,
                 sdf_trunc=0.04,
                 loop_distance_thresh=1.0,
                 min_loop_index_gap=30):
        """
        fx, fy, cx, cy: 相机内参
        width, height: 图像宽高
        depth_scale: 深度图比例（真实深度[m] = depth_value / depth_scale）
        depth_trunc: 深度截断（超过该距离的深度不使用）
        voxel_length: TSDF 体素大小
        sdf_trunc: TSDF 截断距离
        loop_distance_thresh: 回环候选的空间距离阈值（米）
        min_loop_index_gap: 回环帧索引最小间隔（避免近邻帧被误认为回环）
        """

        # 相机内参
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # 深度和 TSDF 相关参数
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc

        # 回环检测参数
        self.loop_distance_thresh = loop_distance_thresh
        self.min_loop_index_gap = min_loop_index_gap

        # 数据缓存
        self.positions = []      # List[np.array(3,)]
        self.quaternions = []    # List[np.array(4,)]  (w,x,y,z)
        self.rgb_images = []     # List[H,W,3] uint8
        self.depth_images = []   # List[H,W] float/uint16

        # 轨迹（初始 IMU -> Camera 假设对齐）
        self.T_world_cam_imu = []        # 初始位姿（4x4）
        self.T_world_cam_optimized = []  # 优化后的位姿（4x4）

        # PoseGraph
        self.pose_graph = None

        # 地图（点云）
        self.map_pcd = None

    # ========= 公共接口 =========

    def add_frame(self, position, quaternion, rgb_image, depth_image):
        """
        添加一帧原始数据：
        - position: np.array(3,)  [x, y, z]
        - quaternion: np.array(4,) [w, x, y, z]
        - rgb_image: np.uint8, HxWx3
        - depth_image: HxW（float32 或 uint16）
        """
        self.positions.append(np.asarray(position, dtype=float))
        self.quaternions.append(np.asarray(quaternion, dtype=float))
        self.rgb_images.append(rgb_image)
        self.depth_images.append(depth_image)

    def run_slam(self, use_loop_closure=True):
        """
        执行整个 SLAM/建图流程：
        1. 根据 (pos,quat) 生成初始位姿 T_world_cam_imu（假设 IMU=Camera）
        2. 构建 PoseGraph（相邻约束）
        3. 可选：简单回环检测 + 回环约束
        4. 全局位姿图优化
        5. TSDF 融合生成地图
        """
        self._build_initial_poses()
        self._build_pose_graph()
        if use_loop_closure:
            self._add_loop_closures()
        self._optimize_pose_graph()
        self._integrate_tsdf()

    def get_trajectory(self):
        """
        返回优化后的轨迹（N, 4, 4）数组（世界到相机的变换）
        若尚未优化，则返回空或 IMU 初始轨迹。
        """
        if len(self.T_world_cam_optimized) > 0:
            return np.stack(self.T_world_cam_optimized, axis=0)
        elif len(self.T_world_cam_imu) > 0:
            return np.stack(self.T_world_cam_imu, axis=0)
        else:
            return np.empty((0, 4, 4))

    def get_map_pointcloud(self):
        """
        返回融合后的地图点云（open3d.geometry.PointCloud）
        """
        return self.map_pcd

    def visualize_map_and_trajectory(self, prefix: str = "slam_result"):
        """
        无 GUI 环境：导出地图和轨迹到文件，在本地可视化。
        会生成：
        - {prefix}_map.ply
        - {prefix}_traj_points.ply
        - （可选）{prefix}_traj_lines.ply（如果你的 Open3D 版本支持 write_line_set）
        """

        if self.map_pcd is None:
            print("地图为空，请先运行 run_slam() 完成 TSDF 融合。")
            return

        # 选择使用的位姿列表
        if not self.T_world_cam_optimized:
            print("尚未进行位姿优化，使用 IMU 初始轨迹。")
            T_list = self.T_world_cam_imu
        else:
            T_list = self.T_world_cam_optimized

        # 1. 保存地图点云
        map_path = f"{prefix}_map.ply"
        o3d.io.write_point_cloud(map_path, self.map_pcd)
        print(f"[导出] 地图已保存到: {os.path.abspath(map_path)}")

        # 如果没有位姿，只导出地图就结束
        if T_list is None or len(T_list) == 0:
            print("[导出] 轨迹为空，只导出地图。")
            return

        # 2. 提取轨迹点（相机位置）
        line_points = [np.asarray(T[:3, 3], dtype=np.float64).reshape(3,) for T in T_list]
        points_np = np.asarray(line_points, dtype=np.float64)

        print("trajectory points shape:", points_np.shape)

        # 3. 构造线段索引：0-1, 1-2, 2-3, ...
        n_pts = points_np.shape[0]
        if n_pts >= 2:
            trajectory_lines = np.column_stack([
                np.arange(0, n_pts - 1, dtype=np.int32),
                np.arange(1, n_pts,     dtype=np.int32)
            ])
        else:
            trajectory_lines = np.zeros((0, 2), dtype=np.int32)

        print("trajectory_lines shape:", trajectory_lines.shape)
        print("trajectory_lines:", trajectory_lines)

        # 4. 构造 LineSet（代码里可能还要用）
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_np)
        line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)

        # 5. 导出轨迹为点云（所有工具都能识别）
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(points_np)
        traj_points_path = f"{prefix}_traj_points.ply"
        o3d.io.write_point_cloud(traj_points_path, traj_pcd)
        print(f"[导出] 轨迹点已保存到: {os.path.abspath(traj_points_path)}")

        # 6. （可选）如果你的 Open3D 版本支持写 LineSet，可以额外保存带连线的轨迹
        # 注意：老版本可能没有这个接口，若报错就注释掉这一段
        try:
            traj_lines_path = f"{prefix}_traj_lines.ply"
            o3d.io.write_line_set(traj_lines_path, line_set)
            print(f"[导出] 轨迹线已保存到: {os.path.abspath(traj_lines_path)}")
        except AttributeError:
            print("[导出] 当前 Open3D 版本不支持 write_line_set，仅导出轨迹点云。")

        print(
            "\n请在本地用 Open3D / MeshLab / CloudCompare 等工具打开：\n"
            f"  - {os.path.abspath(map_path)}\n"
            f"  - {os.path.abspath(traj_points_path)}\n"
            "（如果有 *_traj_lines.ply，也可以一起加载）\n"
            "即可查看地图 + 轨迹。\n"
        )

    def plot_trajectory_matplotlib(self):
        """
        使用 matplotlib 画出 3D 轨迹曲线（仅用于检查）
        """
        T_list = self.T_world_cam_optimized if self.T_world_cam_optimized else self.T_world_cam_imu
        if not T_list:
            print("没有轨迹数据。")
            return

        traj = np.array([T[:3, 3] for T in T_list])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory')
        plt.savefig("trajectory.png")

    # ========= 内部功能 =========

    def _build_initial_poses(self):
        """
        将 (pos, quat) 转成初始 4x4 T_world_cam_imu
        暂时假设 IMU 坐标系与相机坐标系对齐。
        若不对齐，应在此处加入 T_cam_imu 外参。
        """
        self.T_world_cam_imu = []
        for pos, quat in zip(self.positions, self.quaternions):
            T = self._pose_from_pos_quat(pos, quat)
            self.T_world_cam_imu.append(T)

    def _build_pose_graph(self):
        """
        构建仅包含“相邻帧约束”的 PoseGraph，并为每一帧提供初始位姿估计。
        """
        N = len(self.positions)
        if N == 0:
            raise RuntimeError("没有任何帧数据，请先调用 add_frame()")

        print("构建相邻帧约束 PoseGraph...")

        self.pose_graph = registration.PoseGraph()
        # 第 0 帧：世界坐标 = 第0帧相机坐标
        self.pose_graph.nodes.append(registration.PoseGraphNode(np.eye(4)))

        for i in range(N - 1):
            rgbd_i = self._o3d_rgbd_from_np(self.rgb_images[i], self.depth_images[i])
            rgbd_ip1 = self._o3d_rgbd_from_np(self.rgb_images[i + 1], self.depth_images[i + 1])

            # IMU 提供的相对位姿当初值
            T_i_world_imu = self.T_world_cam_imu[i]
            T_ip1_world_imu = self.T_world_cam_imu[i + 1]
            T_i_ip1_init = np.linalg.inv(T_i_world_imu) @ T_ip1_world_imu

            success, T_i_ip1, info = self._compute_relative_pose_rgbd(
                rgbd_i, rgbd_ip1, init=T_i_ip1_init
            )
            if not success:
                T_i_ip1 = T_i_ip1_init
                info = np.eye(6) * 1e-3
                print(f"Frame {i}->{i+1}: RGBD odometry 失败，退回 IMU 相对位姿")

            # 邻接约束 edge
            edge = registration.PoseGraphEdge(
                i, i + 1,
                T_i_ip1,
                info,
                uncertain=False
            )
            self.pose_graph.edges.append(edge)

            # 计算 i+1 帧的初始世界位姿：
            # 若当前 nodes[i].pose = T_cam_world_i（相机到世界），
            # 则 T_world_cam_i = inverse(T_cam_world_i)
            T_cam_world_prev = self.pose_graph.nodes[i].pose
            T_world_cam_prev = np.linalg.inv(T_cam_world_prev)
            T_world_cam_ip1 = T_world_cam_prev @ T_i_ip1

            # 注意：PoseGraphNode 存的是 T_cam_world（相机到世界）
            self.pose_graph.nodes.append(
                registration.PoseGraphNode(np.linalg.inv(T_world_cam_ip1))
            )

    def _add_loop_closures(self):
        """
        简单回环检测：
        - 使用 IMU 初始位姿在空间中找距离较近的帧对 (j, i)
        - 用 RGBD odometry 拟合 T_j_i
        - 若成功，则作为不确定（uncertain=True）的回环边加入 PoseGraph
        """
        N = len(self.positions)
        print("执行简单回环检测...")

        for i in range(N):
            T_w_i = self.T_world_cam_imu[i]
            p_i = T_w_i[:3, 3]

            for j in range(0, i - self.min_loop_index_gap):
                T_w_j = self.T_world_cam_imu[j]
                p_j = T_w_j[:3, 3]
                if np.linalg.norm(p_i - p_j) < self.loop_distance_thresh:
                    # 候选回环 (j, i)
                    rgbd_j = self._o3d_rgbd_from_np(self.rgb_images[j], self.depth_images[j])
                    rgbd_i = self._o3d_rgbd_from_np(self.rgb_images[i], self.depth_images[i])

                    T_j_i_init = np.linalg.inv(T_w_j) @ T_w_i
                    success, T_j_i, info = self._compute_relative_pose_rgbd(
                        rgbd_j, rgbd_i, init=T_j_i_init
                    )
                    if success:
                        print(f"检测到回环: {j} <-> {i}")
                        edge = registration.PoseGraphEdge(
                            j, i,
                            T_j_i,
                            info,
                            uncertain=True
                        )
                        self.pose_graph.edges.append(edge)

    def _optimize_pose_graph(self):
        """
        使用 Open3D 对 PoseGraph 进行全局位姿图优化，
        结果保存在 self.T_world_cam_optimized
        """
        if self.pose_graph is None:
            raise RuntimeError("PoseGraph 尚未构建")

        print("开始全局位姿图优化...")
        option = registration.GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=2.0,
            reference_node=0
        )
        method = registration.GlobalOptimizationLevenbergMarquardt()
        criteria = registration.GlobalOptimizationConvergenceCriteria()

        registration.global_optimization(
            self.pose_graph,
            method,
            criteria,
            option
        )
        print("位姿图优化完成。")

        # 提取优化后的 T_world_cam
        self.T_world_cam_optimized = []
        for node in self.pose_graph.nodes:
            # node.pose = T_cam_world（相机到世界）
            T_cam_world = node.pose
            T_world_cam = np.linalg.inv(T_cam_world)
            self.T_world_cam_optimized.append(T_world_cam)

    def _integrate_tsdf(self):
        """
        使用优化后的位姿和所有 RGBD 帧进行 TSDF 融合，并提取点云地图
        """
        N = len(self.positions)
        if N == 0:
            raise RuntimeError("没有帧数据")

        if not self.T_world_cam_optimized:
            print("警告：尚未进行位姿优化，使用 IMU 初始位姿进行 TSDF 融合。")
            T_list = self.T_world_cam_imu
        else:
            T_list = self.T_world_cam_optimized

        print("开始 TSDF 融合建图...")
        volume = integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=integration.TSDFVolumeColorType.RGB8
        )

        for k in range(N):
            rgbd_k = self._o3d_rgbd_from_np(self.rgb_images[k], self.depth_images[k])
            T_world_cam = T_list[k]
            T_cam_world = np.linalg.inv(T_world_cam)
            volume.integrate(
                rgbd_k,
                self.intrinsic,
                T_cam_world
            )
            if (k + 1) % 50 == 0 or k == N - 1:
                print(f"已融合 {k + 1}/{N} 帧")

        print("提取点云地图...")
        self.map_pcd = volume.extract_point_cloud()

    # ========= 工具函数（静态/内部） =========

    @staticmethod
    def _pose_from_pos_quat(pos, quat):
        """
        pos: (3,), quat: (w,x,y,z) -> 4x4 变换矩阵
        """
        w, x, y, z = quat
        r = R.from_quat([x, y, z, w])  # scipy: [x,y,z,w]
        R_mat = r.as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = pos
        return T

    def _o3d_rgbd_from_np(self, rgb_np, depth_np):
        """
        将 numpy 的 rgb, depth 转为 Open3D 的 RGBDImage
        """
        color_o3d = o3d.geometry.Image(rgb_np.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )
        return rgbd

    def _compute_relative_pose_rgbd(self, rgbd_src, rgbd_dst, init=np.eye(4)):
        """
        使用 Open3D 的 RGBD odometry 计算 src->dst 相对位姿
        """
        option = odometry.OdometryOption()
        odo_method = odometry.RGBDOdometryJacobianFromHybridTerm()
        success, T_src_dst, info = odometry.compute_rgbd_odometry(
            rgbd_src,
            rgbd_dst,
            self.intrinsic,
            init,
            odo_method,
            option,
        )
        return success, T_src_dst, info


# =========================
# 使用示例（伪代码）
# =========================

from src.runs.go2_driver import Go2Driver

def example_usage():
    # 假设你已经有以下数据（这里只是示意）：

    H, W, _ = 720, 1280, 3

    fx = 528.7459106445312 
    fy = 528.7459106445312 
    cx = 652.0160522460938 
    cy = 364.0149841308594

    depth_scale = 1000.0   # 例如深度为毫米

    mapper = RGBDMapper(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=W, height=H,
        depth_scale=depth_scale,
        depth_trunc=4.0,
        voxel_length=0.02,
        sdf_trunc=0.04,
        loop_distance_thresh=1.0,
        min_loop_index_gap=30
    )

    go2 = Go2Driver()

    while True:
        # 获取rgb, depth, pos, quat数据
        observation = go2.get_observation() # {"color_sensor": rgb, "depth_sensor": depth, "highstate": highstate}
        rgb = observation["color_sensor"]
        depth = observation["depth_sensor"]
        pos = observation["highstate"]["position"]
        quat = observation["highstate"]["imu_quaternion"]

        mapper.add_frame(pos, quat, rgb, depth)

        # 跑完整的 SLAM + 建图
        mapper.run_slam(use_loop_closure=True)

        # 得到轨迹和地图
        traj = mapper.get_trajectory()         # (N,4,4)
        map_pcd = mapper.get_map_pointcloud()  # open3d.geometry.PointCloud

        # 可视化
        mapper.plot_trajectory_matplotlib()
        mapper.visualize_map_and_trajectory()

        import pdb; pdb.set_trace()

if __name__ == "__main__":
    example_usage()
