# 连接unitree go2并提供驱动机器狗接口
import json
import cv2
import os
import logging
import requests
import yaml
import time
import numpy as np

from src.planner.dfs import DFSExplorer
from src.server_wrapper.go2_server import send_request, ServerMixin, host_model, str_to_image, string_to_numpy, json_to_pointcloud2
from src.llm_engine.gpt import GPTEngine
from src.llm_engine.qwen import QwenEngine

def show_depth_cv2(image):
    # image = np.clip(image, 0, 4000)
    depth_norm = (image / image.max() * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imwrite("depth_viz.png", depth_color)

def show_rgb_cv2(image):
    cv2.imwrite("rgb_viz.png", image)

class Go2Driver:
    def __init__(self, cfg=None, **kwargs):
        # 初始化连接到Go2机器狗的接口
        self.cfg = cfg

        self.publish_url = kwargs.get('publish_url', 'http://10.24.4.195:10401/robot_server')
        self.subscribe_url = kwargs.get('subscribe_url', 'http://10.24.4.195:10305/')

        # self.llm_engine = GPTEngine(model="gpt-4o")
        self.llm_engine = QwenEngine("/mynvme0/models/Qwen/Qwen2.5-VL-7B-Instruct", device=f"cuda:0")
        self.explorer = DFSExplorer(
            call_vlm_fn=self.call_vlm,
            action_fn=self.execute_action_queue,
            get_observation_fn=self.get_observation,
        )

        self.frame_number = 0
        self.max_step = 9999999

        self.cur_rgb = self.get_observation()["color_sensor"]  # 初始观察
        self.action_queue = []

    def call_vlm(self, prompt, image):
        message = [{'role': "user", "content": prompt}]
        # image_path = os.path.join(self.cfg.output_dir, "temp_vlm_image.jpg")
        image_path = "./temp_vlm_image.jpg"
        cv2.imwrite(image_path, image)
        response = self.llm_engine(message, image_paths=[image_path])
        return response

    def initialize(self, data):
        # 初始化机器狗状态和位置
        # 无需实现
        self.question_id = data["sample_id"]
        self.episode_data_dir = os.path.join(self.cfg.output_dir, str(self.question_id))

        save_path = os.path.join(self.episode_data_dir, "init_rgb.jpg")

        os.makedirs(self.episode_data_dir, exist_ok=True)

        if self.cfg.save_obs:
            cv2.imwrite(save_path, self.cur_rgb)
        return os.path.abspath(save_path)

    def _move(self, vx, vy, vyaw):
        return send_request(self.publish_url, cmd_id="obstacle_avoid_move",
                            cmd_value={"linear_x": vx, "linear_y": vy, "angular_z": vyaw})

    def _velocity_move(self, vx, vy, vyaw, duration):
        return send_request(self.publish_url, cmd_id="obstacle_avoid_move",
                            cmd_value={"linear_x": vx, "linear_y": vy, "angular_z": vyaw, "duration": duration})

    def _fetch_rgbd_with_highstate(self, camera_id="zed2", save_data=False):
        highstate = None
        frame = None
        frame_number = self.frame_number
        url = self.subscribe_url + f"yield_{camera_id}_with_highstate"
        stream = requests.get(url, stream=True, timeout=10)

        bytes_data = b''
        json_buffer = b''  # 用于积累 JSON 数据

        for chunk in stream.iter_content(chunk_size=1024):
            if not chunk:
                continue
            bytes_data += chunk

            # 调试输出每个块的长度
            # print(f"Received chunk of size {len(chunk)}")

            # 寻找JPEG帧
            start_img = bytes_data.find(b'Content-Type: rgb\r\n\r\n') + len(b'Content-Type: rgb\r\n\r\n')
            end_img = bytes_data.find(b'--rgb_end')

            start_depth = bytes_data.find(b'Content-Type: depth\r\n\r\n') + len(b'Content-Type: depth\r\n\r\n')
            end_depth = bytes_data.find(b'--depth_end')

            end_of_data = bytes_data.find(b'--end\r\n')

            # 处理 RGB 图像
            # print("rgb: ", start_img, end_img)
            if start_img != -1 and end_img != -1:
                png_data = bytes_data[start_img: end_img + 2]
                bytes_data = bytes_data[end_img + 2:]
                rgb = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR)

                if rgb is not None and save_data:
                    frame_filename = f"{frame_number:08d}.png"  # 保存为 .png
                    cv2.imwrite(frame_filename, rgb)  # 保存 .png 文件
                    print(f"Saved {frame_filename}")

            # 处理深度图像
            # print("depth: ", start_depth, end_depth)
            if start_depth != -1 and end_depth != -1:
                depth_data = bytes_data[start_depth: end_depth + 2]
                bytes_data = bytes_data[end_depth + 2:]

                # 将深度数据解码为原始格式 (保持16位深度信息)
                depth = np.frombuffer(depth_data, np.uint16)[:-1,].reshape((720, 1280))  # 假设深度图像大小为720x1280

                if depth is not None and save_data:
                    depth_filename = f"{frame_number:08d}.png"  # 保存为 .png
                    cv2.imwrite(depth_filename, depth)  # 保存 .png 文件，保留16位深度信息
                    print(f"Saved {depth_filename}")
                    self.frame_number += 1

            json_start = bytes_data.find(b'Content-Type: application/json\r\n\r\n')
            if json_start != -1:
                # 提取出 JSON 数据的部分，直到 "end" 标记
                json_data_start = json_start + len(b'Content-Type: application/json\r\n\r\n')
                end_json = bytes_data.find(b"\r\n--end\r\n")

                # 如果找到 JSON 的结尾，提取并解析
                if end_json != -1:
                    json_buffer = bytes_data[json_data_start:end_json]

                    # 解析 JSON 数据
                    highstate = json.loads(json_buffer.decode('utf-8'))

                    # 清空已处理的数据
                    bytes_data = bytes_data[end_json + len(b"\r\n--end\r\n"):]

                    if highstate is not None and save_data:
                        json_filename = f"{frame_number:08d}.json"
                        with open(json_filename, 'w') as f:
                            json.dump(highstate, f, indent=4)

            # print(end_of_data)
            if end_of_data != -1:
                break
        # show_cv2(depth)
        # exit()
        return {"color_sensor": rgb, "depth_sensor": depth, "highstate": highstate}

    def execute_action(self, action_name):
        # "Lay Down": self.sport_client.StandDown, # 1005
        # "Stand Up": self.sport_client.RecoveryStand, # 1006

        # "Sit": self.sport_client.Sit, # 1009
        # "RiseSit": self.sport_client.RiseSit, #1010
        # "Hello" : self.sport_client.Hello, # 1016
        # "Stretch": self.sport_client.Stretch, # 1017
        # "Wallow": self.sport_client.Wallow, # 1021
        # "Dance1": self.sport_client.Dance1, # 1022
        # "Dance2": self.sport_client.Dance2, # 1023

        # "Scrape": self.sport_client.Scrape, # 1029
        # "FrontFlip": self.sport_client.FrontFlip, # 1030
        # "FrontJump": self.sport_client.FrontJump, # 1031
        # "FrontPounce": self.sport_client.FrontPounce, # 1032
        # "WiggleHips": self.sport_client.WiggleHips, # 1033
        # "Heart": self.sport_client.Heart, # 1036

        # #below API only works while in advanced mode
        # "HandStand": self.sport_client.HandStand, #1301
        return send_request(self.publish_url, cmd_id="command", cmd_value=action_name)

    def move_forward(self, distance):
        # 控制机器狗向前移动指定距离
        if distance > 0:
            vx = 0.5
        else:
            vx = -0.5
        duration = distance / vx
        self._velocity_move(vx, 0.0, 0.0, duration)
        time.sleep(duration + 1)
        
    def turn(self, angle, offset=10):
        # 控制机器狗转动指定角度
        angle = (angle + offset) if angle > 0 else (angle - offset)
        angle_rad = np.deg2rad(angle)
        if angle_rad > 0:
            vyaw = 0.5
        else:
            vyaw = -0.5
        duration = angle_rad / vyaw
        self._velocity_move(0.0, 0.0, vyaw, duration)
        time.sleep(duration + 1)

    def stop(self):
        # 停止机器狗的运动
        pass

    def get_observation(self):
        # 获取机器狗当前的观察（如RGB图像）
        observation = self._fetch_rgbd_with_highstate()
        self.cur_rgb = observation["color_sensor"]
        show_rgb_cv2(observation["color_sensor"])
        show_depth_cv2(observation["depth_sensor"])
        # print(observation["highstate"]["position"])
        # print(observation["highstate"]["imu_quaternion"])
        # exit()
        return observation

    def go_next_point(self, command, stride_forward=0.5, stride_turn=45):
        # self.action_queue.append(command)
        # if command == "move_forward":
        #     self.move_forward(stride_forward)
        # elif command == "turn_left":
        #     self.turn(stride_turn)
        # elif command == "turn_right":
        #     self.turn(-stride_turn)
        # elif command == "turn_around":
        #     self.turn(180)
        # else:
        #     raise ValueError("Unsupported command: {}".format(command))


        image, depth, pose = self.explorer._get_current_obs_and_pose()
        self.explorer.run_step(image, depth, pose, command)

    def execute_action_queue(self, action):
        if action == "move_forward":
            self.move_forward(1)
        elif action == "turn_left":
            self.turn(45)
        elif action == "turn_right":
            self.turn(-45)
        elif action == "turn_around":
            self.turn(180)
        else:
            raise ValueError("Unsupported command: {}".format(action))
        # update observation
        self.get_observation()
    
    def run_loop(self):
        explorer = DFSExplorer(
            call_vlm_fn=self.call_vlm,
            action_fn=self.go_next_point,
            get_observation_fn=self.get_observation,
            )
        explorer.run_loop(
            step_delay=0.5,
        )
        

if __name__ == "__main__":
    go2_driver = Go2Driver()
    go2_driver.run_loop()

    # while True:
    #     obs = go2_driver.get_observation()
    #     depth = obs["depth_sensor"]
    #     depth_image = (depth.astype(np.float32) / depth.max()) * 255
    #     depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)
    #     depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

    #     cv2.imwrite("depth_raw_16bit.png", depth_image)
