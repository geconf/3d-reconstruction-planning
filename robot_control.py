import os
import math
import time
import math
import numpy as np
import stitcher
import open3d as o3d

import sys
sys.path.append(os.path.abspath('./UR10_RTDE'))
from rtde.rtde import RTDE

import ast

def normalize_to_pi(value):
    """Normalize the value to be within the range [-pi, pi]"""
    return (value + math.pi) % (2 * math.pi) - math.pi

def read_joint_positions(filename):
    joint_positions = []
    
    with open(filename, 'r') as file:
        for line in file:
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

def main():

    rtde = RTDE('192.168.1.102')
    input("Execute?")

    home = [ 1.57, -1.7, 2, -1.87, -1.57, 3.14 ]

    joint_positions = read_joint_positions("ctraj.txt") 
    traj = [joint_positions[i] + [0.15, 0.15, 0.02] for i in range(len(joint_positions))]

    try:
        # Get current joint angle and tool pose
        curr_joint = rtde.get_joint_values()
        print("Joint values: ", curr_joint)

        # Test joint trajectory
        rtde.move_joint_trajectory(traj)

        # Stop the robot
        rtde.stop()

    finally:
        rtde.stop_script()

    input("Finish")

    # Stitching
    # Initialize camera intrinsic parameters (example for Intel RealSense D435)
    '''
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480,
        fx=615.6707153320312, fy=615.962158203125,
        cx=326.0557861328125, cy=240.55592346191406)

    rgbd_stitcher = stitcher.RGBDStitcher(intrinsic)

    rgb_images, depth_images = rgbd_stitcher.load_default()

    # Perform stitching
    combined_cloud = rgbd_stitcher.stitch_sequence(rgb_images, depth_images)

    # Visualize result
    o3d.visualization.draw_geometries([combined_cloud])
    '''


def grr_plan(grr, workspace_path):
    """Plan pushing with GRR"""

    # config_path = [
    #     grr.solve(waypoint[0] + waypoint[1], none_on_fail=True)
    #     for waypoint in workspace_path
    # ]

    TrackArray = []
    config_path = []
    count = 0
    for waypoint in workspace_path:
        # Print the position and orientation before calling grr.solve
        # print("Position (waypoint[0]):", waypoint[0])
        # print("Orientation (waypoint[1]):", waypoint[1])

        # Call grr.solve and add the result to config_path
        if count == 0:
            config_path.append(grr.solve(waypoint[0] + waypoint[1], none_on_fail=True, TrackArray=TrackArray))
        else:
            config_path.append(grr.solve(waypoint[0] + waypoint[1], curr_config=config_path[count-1], regular_ik=False, none_on_fail=True, TrackArray=TrackArray))
        count = count + 1

        # config_path.append(grr.solve(waypoint[0] + waypoint[1], none_on_fail=True))

        # config_path.append(grr.solve(waypoint[0], none_on_fail=True))

    # # Debug
    # for config in config_path:
    #     print(config)

    # TODO 0
    # Valid solution check

    # print(TrackArray)
    with open("trackarr.txt", "w") as file:
        for entry in TrackArray:
            file.write(f"{entry}\n")

    for conf in config_path:
        if conf is None:
            print("\nInvalid configuration found\n")
            return config_path

    # Filter out any None entries
    # config_path = [conf for conf in config_path if conf is not None]
    #
    # Check if there are any invalid configurations left
    # if len(config_path) < len(workspace_path):
    #    print("\nSome waypoints could not be solved. Invalid configurations were skipped.\n")
    return config_path


if __name__ == "__main__":
    main()
