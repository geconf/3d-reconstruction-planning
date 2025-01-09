import numpy as np
import json
import cv2
import open3d as o3d
import stitcher
import time
import pyrealsense2 as rs
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal
import math
from enum import Enum
import rtde_control
import rtde_receive

class MoveMode(Enum):
    JOINT = "joint"
    CARTESIAN = "cartesian"

class RTDE:
    def __init__(self, robot_ip: str = "192.168.1.102"):
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    def get_joint_values(self) -> List[float]:
        """Get the joint positions in radians."""
        return self.rtde_r.getActualQ()

    def get_joint_speed(self) -> List[float]:
        """Get the joint speed in radians per second."""
        return self.rtde_r.getActualQd()

    def get_tool_pose(self) -> List[float]:
        """Get the pose of the Tool Center Point (TCP) in Cartesian space."""
        return self.rtde_r.getActualTCPPose()

    def get_tool_speed(self) -> List[float]:
        """Get the speed of the Tool Center Point (TCP) in Cartesian space."""
        return self.rtde_r.getActualTCPSpeed()

    def move_joint(self, joint_values: List[float], speed: float = 1.05, 
                  acceleration: float = 1.4, asynchronous: bool = False):
        """Move the robot to the target joint positions."""
        self.rtde_c.moveJ(joint_values, speed, acceleration, asynchronous)

    def move_tool(self, pose: List[float], speed: float = 0.25,
                 acceleration: float = 1.2, asynchronous: bool = False):
        """Move the robot to the tool position."""
        self.rtde_c.moveL(pose, speed, acceleration, asynchronous)

    def stop(self, acceleration: float = 2.0, asynchronous: bool = False):
        """Stop the robot."""
        self.rtde_c.stopJ(acceleration, asynchronous)

class RealSenseCamera:
    def __init__(self, config_file: str = "realsense_config.json"):
        """Initialize RealSense camera with custom configuration."""
        # Initialize context and pipeline
        self.ctx = rs.context()
        self.pipeline = rs.pipeline(self.ctx)
        self.config = rs.config()
        
        # Find D435f device
        devices = self.ctx.query_devices()
        self.device = None
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == '336522303434':
                self.device = dev
                break
                
        if not self.device:
            raise RuntimeError("No RealSense D435f device found!")
        
        # Enable advanced mode and load config
        self.advanced_mode = rs.rs400_advanced_mode(self.device)
        if not self.advanced_mode.is_enabled():
            self.advanced_mode.toggle_advanced_mode(True)
            time.sleep(2)
        
        # Load JSON configuration
        with open(config_file, 'r') as f:
            config_json = f.read()
            self.advanced_mode.load_json(config_json)
            print("Loaded RealSense advanced configuration")
            
        # Enable streams
        self.config.enable_device(self.device.get_info(rs.camera_info.serial_number))
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get and store depth scale
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # Create align object
        self.align = rs.align(rs.stream.color)
        
        # Wait for camera to stabilize
        time.sleep(2)
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get aligned color and depth frames."""
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align frames
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get frames from RealSense camera")
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
    
    def get_intrinsics(self) -> Dict:
        """Get camera intrinsics for both streams."""
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        return {
            "depth": {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "scale": self.depth_scale
            },
            "color": {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy
            }
        }
    
    def release(self):
        """Stop the RealSense pipeline."""
        self.pipeline.stop()

class RobotController:
    def __init__(self, robot_ip: str = "192.168.1.102"):
        self.robot = RTDE(robot_ip)
    
    def move_to_target(self, 
                      target: List[float], 
                      mode: MoveMode,
                      target_num, 
                      speed: float = 0.5,
                      acceleration: float = 0.5) -> bool:
        """Move robot to target position using specified mode."""
        try:
            if mode == MoveMode.JOINT:
                self.robot.move_joint(target, speed, acceleration)
            else:
                self.robot.move_tool(target, speed, acceleration)
            return True
        except Exception as e:
            print(f"Movement failed: {str(e)}")
            return False
    
    def get_current_state(self) -> Dict:
        """Get current robot state."""
        return {
            "joint_positions": self.robot.get_joint_values(),
            "tool_pose": self.robot.get_tool_pose()
        }
        
class DataCollector:
    def __init__(self, robot_ip: str = "192.168.1.102", config_file: str = "realsense_config.json"):
        # Initialize robot and camera
        self.robot = RobotController(robot_ip)
        self.camera = RealSenseCamera(config_file)
        
        # Create output directory structure
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"robot_data_{self.timestamp}")
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.rgb_dir.mkdir(exist_ok=True)
        self.depth_dir.mkdir(exist_ok=True)
        
        # Store camera intrinsics
        self.camera_intrinsics = self.camera.get_intrinsics()
        
        # Storage for collected data
        self.collected_data = []
    
    def move_to_target(self, 
                      target: List[float], 
                      mode: MoveMode,
                      target_num, 
                      wait_for_input: bool = False) -> bool:
        """Move robot to target position and collect data."""
        # Move robot to position
        if not self.robot.move_to_target(target, mode, target_num):
            print("Failed to reach target position")
            return False
        
        if wait_for_input:
            print("\nPress Enter to capture data at current position or 'q' to quit...")
            while True:
                if keyboard.is_pressed('enter'):
                    break
                elif keyboard.is_pressed('q'):
                    return False
                time.sleep(0.1)
        
        #if target_num % 50 != 0:
        #    return True
        # Get robot state
        robot_state = self.robot.get_current_state()
        tool_pose = robot_state["tool_pose"]
        joint_pos = robot_state["joint_positions"]
        
        # Get camera frames
        color_frame, depth_frame = self.camera.get_frames()
        
        # Generate frame ID and filenames
        frame_id = len(self.collected_data)
        rgb_filename = f"{frame_id:04d}.jpg"
        depth_filename = f"{frame_id:04d}.npy"
        
        # Save frames
        cv2.imwrite(str(self.rgb_dir / rgb_filename), color_frame)
        np.save(str(self.depth_dir / depth_filename), depth_frame)
        
        # Create data point
        data_point = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "move_mode": mode.value,
            "target": {
                "values": target,
                "type": mode.value
            },
            "achieved_state": {
                "tool_pose": {
                    "position": {
                        "x": tool_pose[0],
                        "y": tool_pose[1],
                        "z": tool_pose[2]
                    },
                    "rotation": {
                        "rx": tool_pose[3],
                        "ry": tool_pose[4],
                        "rz": tool_pose[5]
                    }
                },
                "joint_angles": {
                    f"joint_{i+1}": angle for i, angle in enumerate(joint_pos)
                }
            },
            "images": {
                "rgb": f"rgb/{rgb_filename}",
                "depth": f"depth/{depth_filename}"
            }
        }
        
        # Store data point
        self.collected_data.append(data_point)
        print(f"Captured data point {frame_id}")
        print(f"Tool position: x={tool_pose[0]:.3f}, y={tool_pose[1]:.3f}, z={tool_pose[2]:.3f}")
        print(f"Tool rotation: rx={tool_pose[3]:.3f}, ry={tool_pose[4]:.3f}, rz={tool_pose[5]:.3f}")
        print(f"Joint angles: {', '.join(f'j{i+1}={j:.3f}' for i, j in enumerate(joint_pos))}")
        
        return True
    
    def collect_data_from_targets(self, targets: List[Dict[str, Union[List[float], str]]]):
        """Collect data from a list of targets with specified movement modes."""
        try:
            for i, target in enumerate(targets):
                position = target['position']
                mode = MoveMode(target['mode'])
                
                print(f"\nMoving to target {i+1}/{len(targets)}")
                print(f"Mode: {mode.value}")
                print(f"Target: {position}")
                
                if not self.move_to_target(position, mode, i):
                    print("\nData collection interrupted")
                    break
                    
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        finally:
            # Save metadata
            metadata = {
                "collection_info": {
                    "timestamp": self.timestamp,
                    "total_frames": len(self.collected_data),
                },
                "camera_intrinsics": self.camera_intrinsics,
                "frames": self.collected_data
            }
            
            with open(self.output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.camera.release()
            
            print(f"\nData collection complete!")
            print(f"Collected data for {len(self.collected_data)} positions")
            print(f"Output directory: {self.output_dir}")

def main():
    # Robot IP and camera config
    ROBOT_IP = "192.168.1.102"
    CAMERA_CONFIG = "realsense_config.json"
    
    # Example targets list with mixed joint and cartesian positions
    '''
    targets = [
        # Cartesian positions [x, y, z, rx, ry, rz]
        {
            'position': [0.4, -0.2, 0.5, 0, 3.14, 0],
            'mode': 'cartesian'
        },
        # Joint positions [j1, j2, j3, j4, j5, j6]
        {
            'position': [1.57, -1.57, 1.57, 0, 1.57, 0],
            'mode': 'joint'
        },
        {
            'position': [0.4, 0.2, 0.5, 0, 3.14, 0],
            'mode': 'cartesian'
        },
        {
            'position': [0, -1.57, 1.57, 0, 1.57, 0],
            'mode': 'joint'
        }
    ]
    '''
    targets = get_targets('ctraj.txt')
    
    try:
        # Create collector
        collector = DataCollector(
            robot_ip=ROBOT_IP,
            config_file=CAMERA_CONFIG
        )
        
        # Start data collection
        collector.collect_data_from_targets(targets)

        # Stitch clouds
        stitch(collector.rgb_dir, collector.depth_dir)
        
    except Exception as e:
        print(f"Error during data collection: {str(e)}")


def stitch(rgb_folder, depth_folder):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480,
        fx=615.6707153320312, fy=615.962158203125,
        cx=326.0557861328125, cy=240.55592346191406)

    '''
    # Create stitcher
    rgbd_stitcher = stitcher.RGBDStitcher(intrinsic)

    color_images, depth_images = rgbd_stitcher.load_dataset_realsense(
        rgb_folder,
        depth_folder
    )

    # Perform stitching
    combined_cloud = rgbd_stitcher.stitch_sequence(color_images, depth_images)

    # Visualize result
    o3d.visualization.draw_geometries([combined_cloud])
    '''

def get_targets(filename):
    joint_positions = read_joint_positions(filename)
    targets = []
    for joint_pos in joint_positions:
        target = {
            'position': joint_pos,
            'mode': 'joint'
        }
        targets.append(target)
    return targets


def read_joint_positions(filename):
    joint_positions = []

    with open(filename, 'r') as file:
        i = 0
        for line in file:
            i += 1
            if i % 20 != 0:
                continue
            # Split the line at the first comma to separate the scalar from the list
            scalar, joint_pos_str = line.strip().split(',', 1)

            # Remove the square brackets from the list part
            if joint_pos_str.startswith('[') and joint_pos_str.endswith(']'):
                joint_pos_str = joint_pos_str[1:-1].strip()  # Remove the brackets and trim spaces

            # Convert the space-separated values into a Python list of floats
            joint_pos = list(map(float, joint_pos_str.split()))

            # Add pi/2 to offset the curve (this is a workaround)
            joint_pos[0] += math.pi*0.35

            # Normalize all values to be within [-pi, pi]
            normalized_pos = [normalize_to_pi(val) for val in joint_pos]

            # Append the list to the result list
            joint_positions.append(normalized_pos)

    return joint_positions

def normalize_to_pi(value):
    """Normalize the value to be within the range [-pi, pi]"""
    return (value + math.pi) % (2 * math.pi) - math.pi

if __name__ == "__main__":
    main()
