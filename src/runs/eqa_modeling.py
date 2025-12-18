"""
Run EQA in Habitat-Sim with VLM exploration.

"""

import os
import json

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.utils.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.utils.navigate import navigation_video
from src.utils.geom import get_cam_intr, get_scene_bnds
from src.planner.tsdf import TSDFPlanner
from src.llm_engine.qwen import QwenEngine
from src.llm_engine.gpt import GPTEngine
from omegaconf import OmegaConf

import matplotlib.pyplot as plt

class EQA_Modeling():
    def __init__(self, cfg, device):
        self.cfg = cfg

        # sensor
        self.camera_tilt = cfg.camera_tilt_deg * np.pi / 180
        self.img_height = cfg.img_height
        self.img_width = cfg.img_width
        self.cam_intr = get_cam_intr(cfg.hfov, self.img_height, self.img_width)

        # vlm
        self.vlm_question = ""
        self.vlm_pred_candidates = ["A", "B", "C", "D"]

        # state
        self.max_step = 0
        self.cur_step = 0

        # Load dataset
        # with open(cfg.question_data_path) as f:
        #     self.questions_data = json.load(f)
        # logging.info(f"Loaded {len(self.questions_data)} questions.")

        self.letters = ["A", "B", "C", "D"]  # always four
        self.fnt = ImageFont.truetype("data/Open_Sans/static/OpenSans-Regular.ttf", 30,)

        self.shortest_path = habitat_sim.ShortestPath()
        self.path_length = 0

        # Load VLM
        self.vlm = QwenEngine(cfg.vlm.model_name_or_path, device=f"cuda:{device}")
        # self.vlm = GPTEngine("gpt-4o-mini")

        self.fnt = ImageFont.truetype(
            "data/Open_Sans/static/OpenSans-Regular.ttf",
            30,
        )

    def _display_sample(self, rgb, depth, save_path="sample.png"):
        # 创建一个包含3列的子图
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))

        # 显示RGB图像
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')  # 关闭坐标轴

        # 显示深度图像
        axes[1].imshow(depth, cmap='jet')  # 使用 'jet' 配色方案
        axes[1].set_title("Depth Image")
        axes[1].axis('off')  # 关闭坐标轴

        # 调整子图布局
        plt.tight_layout()

        # 保存图像为PNG文件
        plt.savefig(save_path, format="png")

        plt.close()


    def _init_data(self, data):
        self.question_id = data["sample_id"]
        self.floor = data.get("floor", 0)
        self.question = data["question"]
        self.choices = data["proposals"]
        self.answer = data["answer"]

        self.pts = data["init_pos"]
        # self.pts = data["goal_position"]
        self.angle = data["init_rot"]

        meta_data = {
            "sample_id": self.question_id,
            "question": self.question,
            "related_objs": data["related_objects"] if "related_objects" in data.keys() else "",
            "shortest_length": data["traj_length"] if "traj_length" in data.keys() else 0,
            "answer": self.answer,
            "scene": self.scene,
            "proposals": data["proposals"]
        }

        self.result = {
            "meta": meta_data,
            "step": [],
            "summary": {}
        }

        self.path_length = 0
        self.episode_data_dir = os.path.join(self.cfg.output_dir, str(self.question_id))
        os.makedirs(self.episode_data_dir, exist_ok=True)

        self.pts_normal = pos_habitat_to_normal(np.array(self.pts))
        self.floor_height = self.pts_normal[-1]

        logging.info("Finished initializing data.")

    def _init_sim(self):
        # Set up scene in Habitat
        try:
            self.simulator.close()
        except:
            pass
        scene_data_path = ""
        for scene_path in self.cfg.scene_data_path:
            if os.path.exists(os.path.join(scene_path, self.scene)):
                scene_data_path = scene_path
                break

        if "scene" in self.scene:
            scene_mesh_dir = os.path.join(
                scene_data_path, self.scene, self.scene + "_vh_clean_2" + ".glb"
            )
            navmesh_file = os.path.join(
                scene_data_path, self.scene, self.scene + "_vh_clean_2" + ".navmesh"
            )
        else:
            scene_mesh_dir = os.path.join(
                scene_data_path, self.scene, self.scene[6:] + ".basis" + ".glb"
            )
            navmesh_file = os.path.join(
                scene_data_path, self.scene, self.scene[6:] + ".basis" + ".navmesh"
            )

        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": self.cfg.camera_height,
            "width": self.img_width,
            "height": self.img_height,
            "hfov": self.cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(self.cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_file)
        if not self.pathfinder.is_loaded:
            print("Not loaded .navmesh file yet. Please check file path {}.".format(navmesh_file))

        self.tsdf_bnds, self.scene_size = get_scene_bnds(self.pathfinder, self.floor_height)
        self.max_step = int(math.sqrt(self.scene_size) * self.cfg.max_step_room_size_ratio)

        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])
        self.agent_state = habitat_sim.AgentState()

        logging.info(
            f"Scene size: {self.scene_size} Floor height: {self.floor_height} Steps: {self.max_step}"
        )
        logging.info("Finished initializing simulation.")

    def _init_planner(self):
        # Initialize TSDF Planner
        self.planner = TSDFPlanner(
            vol_bnds=self.tsdf_bnds,
            voxel_size=self.cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=self.pts_normal,
            init_clearance=self.cfg.init_clearance * 2,
        )
        logging.info("Finished initializing planner.")

    def initialize(self, data):
        self.pts_pixs = np.empty((0, 2))
        self.cur_step = 0

        self.scene = data["scene"]
        self.vlm_question = data["question"]

        self._init_data(data)
        self._init_sim()
        self._init_planner()

        self.agent_state.position = self.pts
        self.agent_state.rotation = quat_to_coeffs(quat_from_angle_axis(self.angle, np.array([0, 1, 0]))).tolist()
        self.agent.set_state(self.agent_state)

        # save current obs
        obs = self.simulator.get_sensor_observations()
        self.cur_rgb = obs["color_sensor"]
        self.cur_depth = obs["depth_sensor"]

        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position

        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0
        cam_pose_normal = pose_habitat_to_normal(cam_pose)
        self.cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

        if self.cfg.save_obs:
            self._display_sample(self.cur_rgb, self.cur_depth, os.path.join(self.episode_data_dir, "init.png"))
            # 保存初始rgb图像
            save_path = os.path.join(self.episode_data_dir, "init_rgb.jpg")
            cv2.imwrite(os.path.join(self.episode_data_dir, "init_rgb.jpg"), cv2.cvtColor(self.cur_rgb, cv2.COLOR_RGB2BGR))

        return save_path

    def _get_mesh(self, planner, save_path):
        return planner.get_mesh(save_path)

    def _draw_point(self, im, points_pix, texts):
        rgb_im_draw = im.copy()
        draw = ImageDraw.Draw(rgb_im_draw)
        for prompt_point_ind, point_pix in enumerate(points_pix):
            draw.ellipse(
                (
                    point_pix[0] - self.cfg.visual_prompt.circle_radius,
                    point_pix[1] - self.cfg.visual_prompt.circle_radius,
                    point_pix[0] + self.cfg.visual_prompt.circle_radius,
                    point_pix[1] + self.cfg.visual_prompt.circle_radius,
                ),
                fill=(200, 200, 200, 255),
                outline=(0, 0, 0, 255),
                width=3,
            )
            draw.text(
                tuple(point_pix.astype(int).tolist()),
                texts[prompt_point_ind],
                font=self.fnt,
                fill=(0, 0, 0, 255),
                anchor="mm",
                font_size=12,
            )
        rgb_im_draw_path = os.path.join(self.episode_data_dir, f"{self.cur_step}_draw.png")
        rgb_im_draw.save(rgb_im_draw_path)
        return rgb_im_draw, rgb_im_draw_path

    def go_next_point(self, command):
        """
            Go next point in the environment.
            Input: 
                agent: the object of agent
                planner: the object of planner
                weight: 
            Output: 
                obversation(rgb image)
        """
        self.planner.integrate(
            color_im=self.cur_rgb,
            depth_im=self.cur_depth,
            cam_intr=self.cam_intr,
            cam_pose=self.cam_pose_tsdf,
            obs_weight=1.0,
            margin_h=int(self.cfg.margin_h_ratio * self.img_height),
            margin_w=int(self.cfg.margin_w_ratio * self.img_width),
        )

        rgb_im = Image.fromarray(self.cur_rgb, mode="RGBA").convert("RGB")

        # Get VLM prediction
        # prompt_question = (self.vlm_question + "\nAnswer with the option's letter from the given choices directly.")
        # message = [{"role": "user", "content": prompt_question}]
        # response_pred = self.vlm.call_vlm(message, rgb_im)[0].strip(".")

        # smx_vlm_pred = np.zeros(len(self.vlm_pred_candidates))
        # for i in range(len(self.vlm_pred_candidates)):
        #     if response_pred == self.vlm_pred_candidates[i]:
        #         smx_vlm_pred[i] = 1
        # logging.info(f"Pred - Prob: {smx_vlm_pred}")

        # Get frontier candidates
        prompt_points_pix = []
        if self.cfg.use_active:
            prompt_points_pix, fig = (
                self.planner.find_prompt_points_within_view(
                    self.pts_normal,
                    self.img_width,
                    self.img_height,
                    self.cam_intr,
                    self.cam_pose_tsdf,
                    **self.cfg.visual_prompt,
                )
            )
            fig.tight_layout()
            plt.savefig(os.path.join(self.episode_data_dir, "prompt_points.png"))
            plt.close()

        # Visual prompting
        actual_num_prompt_points = len(prompt_points_pix)
        if actual_num_prompt_points >= self.cfg.visual_prompt.min_num_prompt_points:
            rgb_im_draw, draw_path = self._draw_point(rgb_im, prompt_points_pix, self.letters)

            # get VLM reasoning for exploring
            if self.cfg.use_lsv:
                proposal_point = self.letters[:actual_num_prompt_points]
                direction = command.split("_")[-1]
                prompt_lsv = f"\nConsider the question: '{self.vlm_question}', and you will explore {direction} the environment for answering it.\nWhich direction (black letters on the image {proposal_point}) would you explore then? Answer with a single letter."
                message = [{"role": "user", "content": prompt_lsv}]
                response_lsv = self.vlm.call_vlm(message, image_paths=[draw_path])[0]
                lsv = np.zeros(actual_num_prompt_points)
                for i in range(actual_num_prompt_points):
                    if response_lsv == self.letters[i]:
                        lsv[i] = 1
                if lsv.sum() >= 0:
                    lsv *= actual_num_prompt_points / 3
                else:
                    print("error lsv reponse: ", response_lsv)
            else:
                lsv = (
                    np.ones(actual_num_prompt_points) / actual_num_prompt_points
                )

            # base - use image without label
            if self.cfg.use_gsv:
                prompt_gsv = f"\nConsider the question: '{self.vlm_question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                message = [{"role": "user", "content": prompt_gsv}]
                rgb_im_path = os.path.join(self.episode_data_dir, "cur_rgb.png")
                rgb_im.save(rgb_im_path)
                response_gsv = self.vlm.call_vlm(message, image_paths=[rgb_im_path])[0].strip(".")
                gsv = np.zeros(2)
                if response_gsv == "Yes":
                    gsv[0] = 1
                else:
                    gsv[1] = 1
                gsv = (np.exp(gsv[0] / self.cfg.gsv_T) / self.cfg.gsv_F)  # scale before combined with lsv
            else:
                gsv = 1
            sv = lsv * gsv
            logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

            # Integrate semantics only if there is any prompted point
            self.planner.integrate_sem(
                sem_pix=sv,
                radius=1.0,
                obs_weight=1.0,
            )  # voxel locations already saved in tsdf class

        self.pts_normal, self.angle, _, pts_pix, fig = self.planner.find_next_pose(
                    pts=self.pts_normal,
                    angle=self.angle,
                    cam_pose=self.cam_pose_tsdf,
                    flag_no_val_weight=self.cur_step < self.cfg.min_random_init_steps,
                    **self.cfg.planner,
                )
        self.pts_pixs = np.vstack((self.pts_pixs, pts_pix))
        self.pts_normal = np.append(self.pts_normal, self.floor_height)

        pts = pos_normal_to_habitat(self.pts_normal)
        rotation = quat_to_coeffs(quat_from_angle_axis(self.angle, np.array([0, 1, 0]))).tolist()

        # Add path to ax5, with colormap to indicate order
        ax5 = fig.axes[4]
        ax5.plot(self.pts_pixs[:, 1], self.pts_pixs[:, 0], linewidth=5, color="black")
        ax5.scatter(self.pts_pixs[0, 1], self.pts_pixs[0, 0], c="white", s=50)
        fig.tight_layout()
        plt.savefig(
            os.path.join(self.episode_data_dir, f"map_{self.cur_step}.png")
        )
        plt.close()

        self.shortest_path.requested_start = self.agent_state.position
        self.shortest_path.requested_end = pts
        found = self.pathfinder.find_path(self.shortest_path)
        if found:
            self.path_length += self.shortest_path.geodesic_distance

        self.result['step'].append({
            "step": self.cur_step,
            "pts": pts.tolist(),
            "angle": self.angle,
            "image": os.path.join(self.episode_data_dir, f"{self.cur_step}.png")
        })

        # update state
        self.agent_state.position = pts
        self.agent_state.rotation = rotation
        self.agent.set_state(self.agent_state)

        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position

        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0
        cam_pose_normal = pose_habitat_to_normal(cam_pose)
        self.cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

        obs = self.simulator.get_sensor_observations()
        self.cur_rgb = obs["color_sensor"]
        self.cur_depth = obs["depth_sensor"]

        # visualize
        if self.cfg.save_obs:
            self._display_sample(self.cur_rgb, self.cur_depth, os.path.join(self.episode_data_dir, f"{self.cur_step}.png"))

        self.cur_step += 1


    def run(self):
        for cnt_step in range(self.max_step):
            logging.info(f"\n== step: {cnt_step}")
            self.go_next_point()


if __name__ == "__main__":
    cfg = OmegaConf.load("config/react-eqa.yaml")
    OmegaConf.resolve(cfg)
    eqa_modeling = EQA_Modeling(cfg)
    eqa_modeling.initialize(eqa_modeling.questions_data[2])
    eqa_modeling.run()