import pybullet as p
import numpy as np
import cv2
from pathlib import Path
import shutil


class Camera:
    def __init__(self,
                 width,
                 height,
                 robot_id,
                 rgb_link_id,
                 fov=90,
                 aspect=1.0,
                 near=0.1,
                 far=100,
                 physicsClientId=0,
                 output_folder="camera",
                 has_depth=False
                 ):
        self.width = width
        self.height = height
        self.robot_id = robot_id
        self.rgb_link_id = rgb_link_id
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.physicsClientId = physicsClientId
        self.output_folder = output_folder
        self.has_depth = has_depth

        output_path = Path(self.output_folder)
        if output_path.exists():
            shutil.rmtree(self.output_folder)

        self.rgb_path = output_path / 'rgb'
        self.rgb_path.mkdir(parents=True)

        self.depth_path = output_path / 'depth'
        self.depth_path.mkdir()

        self.image_count = 0
        self.image_id_width = 4

    def takePicture(self, target_pos):
        camera_state = p.getLinkState(
                self.robot_id,
                self.rgb_link_id,
                computeForwardKinematics=True,
                physicsClientId=self.physicsClientId)
        camera_pos = camera_state[0]

        img = p.getCameraImage(
                width=self.width,
                height=self.height,
                viewMatrix=p.computeViewMatrix(
                    cameraEyePosition=camera_pos,
                    cameraTargetPosition=target_pos,
                    cameraUpVector=[0, 0, 1]),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    self.fov,
                    self.aspect,
                    self.near,
                    self.far),
                physicsClientId=self.physicsClientId
                )

        image_id = f"{self.image_count:0{self.image_id_width}d}"
        color_image_name = f"rgb_{image_id}.png"
        self.image_count += 1
        rgb_image = img[2]  # RGB image
        rgb_image = np.array(rgb_image, dtype=np.uint8)
        # Opencv uses BGR
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.rgb_path / color_image_name, bgr_image)

        if (self.has_depth):
            depth_image_name = f"depth_{image_id}.png"
            depth_image = np.array(img[3])
            # Scale depth to 0-255 for saving as image
            depth_image = (depth_image * 255).astype(np.uint8)
            cv2.imwrite(self.depth_path / depth_image_name, depth_image)
