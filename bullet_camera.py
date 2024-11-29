import pybullet as p
import numpy as np
import cv2
from pathlib import Path
import shutil


class Camera:
    def __init__(self,
                 width,
                 height,
                 fov=90,
                 aspect=1.0,
                 near=0.1,
                 far=100,
                 physicsClientId=0,
                 output_folder="camera"
                 ):
        self.width = width
        self.height = height
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.physicsClientId = physicsClientId
        self.output_folder = output_folder

        output_path = Path(self.output_folder)
        if output_path.exists():
            shutil.rmtree(self.output_folder)

        self.rgb_path = output_path / 'rgb'
        self.rgb_path.mkdir(parents=True)

        self.depth_path = output_path / 'depth'
        self.depth_path.mkdir()

    def takePicture(self, camera_pos, target_pos):
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
        rgb_image = img[2]  # RGB image
        rgb_image = np.array(rgb_image, dtype=np.uint8)
        # Opencv uses BGR
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.rgb_path / "rgb_test.png", bgr_image)

        depth_image = np.array(img[3])
        # Scale depth to 0-255 for saving as image
        depth_image = (depth_image * 255).astype(np.uint8)
        cv2.imwrite(self.depth_path / "depth_test.png", depth_image)
