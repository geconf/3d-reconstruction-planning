import os
import math
import pybullet as p
import time
import numpy as np
import pybullet_data
import bullet_camera
import stitcher
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append(os.path.abspath('./Expansion-GRR'))
from bullet_api.loader import load_grr
from grr.utils import euler_to_quat, quat_to_euler
from grr.robot import KinematicChain, Kinova, UR10

def main():
    # Connect to PyBullet in GUI mode
    physics_client = p.connect(p.GUI)

    # Reset the simulation to a clean state
    p.resetSimulation()

    dir = os.path.dirname(os.path.abspath(__file__))
    urdf = dir + "/Expansion-GRR/data/robots/ur10.urdf"
    robot_id = p.loadURDF(urdf, useFixedBase=True, physicsClientId=physics_client)
    p.setGravity(0, 0, -9.8)

    # Set camera position for better visibility
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=2.0,             # Distance from the center of the scene
    #     cameraYaw=45,                   # Horizontal rotation
    #     cameraPitch=-30,                # Vertical angle
    #     cameraTargetPosition=[0, 0, 0]  # Focus point
    # )
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,             # Distance from the center of the scene
        cameraYaw=135,                  # Horizontal rotation
        cameraPitch=-15,                # Vertical angle
        cameraTargetPosition=[0, 0, 0]  # Focus point
    )

    grr = load_grr(urdf, "ur10", "rot_free")
    ObjectPoint = [0.75, 0.75, 0]
    banana_urdf = './011_banana/banana.urdf'
    banana_pos = ObjectPoint
    banana_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # Need to load these after the robot as joint_ids are 0-based
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    # p.loadURDF("plane.urdf", physicsClientId=physics_client)
    build_house()
    p.loadURDF(banana_urdf,
               basePosition=banana_pos,
               baseOrientation=banana_orientation,
               physicsClientId=physics_client,
               useFixedBase=True)

    d435_rgb_link = 15  # d435 color frame link
    d435 = bullet_camera.Camera(
            width=640,
            height=480,
            robot_id=robot_id,
            rgb_link_id=d435_rgb_link,
            physicsClientId=physics_client,
            has_depth=True)

    circRad = 0.25
    circRad = 0.3
    circHeight = 0.15

    workspace_path = []

    path_disc = 500

    y_const = ObjectPoint[1]
    y_const = ObjectPoint[1]-0.3

    z_const = ObjectPoint[2] + circHeight
    x_arr = np.linspace(
            ObjectPoint[0]-circRad,
            ObjectPoint[0]+circRad, path_disc)
    y_arr = np.linspace(y_const, y_const, path_disc)
    z_arr = np.linspace(z_const, z_const, path_disc)

    circArr = np.linspace(0, np.pi, path_disc)

    x_arr = ObjectPoint[0] + circRad*np.cos(circArr)
    z_arr = circHeight + ObjectPoint[2] + circRad*np.sin(circArr)
    y_arr = y_arr

    z_arr = circHeight + ObjectPoint[2] + circRad*np.sin(circArr)
    x_arr = ObjectPoint[0] + circRad*np.cos(circArr)

    x_arr = ObjectPoint[0] - \
        (0.15)*(np.cos(np.pi/4)) + circRad*np.cos(circArr)*np.cos(3*np.pi/4)
    y_arr = ObjectPoint[1] - \
        (0.15)*(np.cos(np.pi/4)) + circRad*np.cos(circArr)*np.sin(3*np.pi/4)
    z_arr = circHeight + ObjectPoint[2] + circRad*np.sin(circArr)

    for i in range(len(x_arr)):
        x_curr = x_arr[i]
        y_curr = y_arr[i]
        z_curr = z_arr[i]

        vVec = [
                ObjectPoint[0]-x_curr,
                ObjectPoint[1]-y_curr,
                ObjectPoint[2]-z_curr
                ]
        vNorm = np.sqrt(vVec[0]**2 + vVec[1]**2 + vVec[2]**2)
        vVecN = [vVec[0]/vNorm, vVec[1]/vNorm, vVec[2]/vNorm]

        z_axis = vVecN
        arbit_vec = np.array([1, 0, 0]) \
            if not np.allclose(z_axis, [1, 0, 0]) else np.array([0, 1, 0])
        x_axis = np.cross(arbit_vec, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        rotation_matrix = rotation_matrix.T

        r = R.from_matrix(rotation_matrix)

        euler_calc = r.as_euler('ZYX')
        quat_calc = euler_to_quat([0, euler_calc[1], euler_calc[2]], seq='ZYX')

        new_point = [
                [x_curr, y_curr, z_curr],
                [quat_calc[0],
                 quat_calc[1],
                 quat_calc[2],
                 quat_calc[3]]
                ]
        workspace_path.append(new_point)

    with open('wtraj_input.txt', 'w') as f:
        for row in workspace_path:
            f.write(','.join(map(str, row)) + '\n')

    # Solve the Cartesian path with GRR
    config_path = grr_plan(grr, workspace_path)

    maneuver_time = 10
    time_path = np.linspace(0, maneuver_time, len(config_path))
    traj = [(t, q) for t, q in zip(time_path, config_path)]

    with open('ctraj.txt', 'w') as f:
        for row in traj:
            f.write(','.join(map(str, row)) + '\n')

    robot = UR10("ur10",
                 [[-1, 1], [-1, 1], [-0.5, 1]], [[0, 0], [0, 0], [0, 0]])

    ws_path = []
    for currConf in traj:
        fksolved = robot.solve_fk(currConf[1])
        ws_path.append([fksolved[0][6], fksolved[1][6]])

    ws_real = [(t, q) for t, q in zip(time_path, ws_path)]

    with open('wtraj.txt', 'w') as f:
        for row in ws_real:
            f.write(','.join(map(str, row)) + '\n')

    p.addUserDebugLine(
        lineFromXYZ=ObjectPoint,
        lineToXYZ=[
            ObjectPoint[0]+0.01, ObjectPoint[1]+0.01, ObjectPoint[2]+0.01
            ],
        lineColorRGB=[0, 0, 1],  # Red color for the path
        lineWidth=10,  # Adjust line thickness if needed
    )

    print("Visualizing workspace path...")
    print(workspace_path[0][0])
    for i in range(len(workspace_path) - 1):
        start_point = workspace_path[i][0]  # Extract the xyz coordinate
        end_point = workspace_path[i + 1][0]  # Extract the next xyz coordinate
        p.addUserDebugLine(
            lineFromXYZ=start_point,
            lineToXYZ=end_point,
            lineColorRGB=[1, 0, 0],  # Red color for the path
            lineWidth=5,  # Adjust line thickness if needed
        )

    # for i, joint_id in enumerate([0, 1, 2, 3, 4, 5]):
    #     p.resetJointStateMultiDof(
    #         0, joint_id, targetValue=[0], physicsClientId=physics_client
    # )

    startInd = 0

    for waypoint in traj:
        if waypoint[1] is not None:
            startInd = startInd
        else:
            startInd = startInd+1
    print("Start Ind: ", startInd)

    for i, joint_id in enumerate([0, 1, 2, 3, 4, 5]):
        p.resetJointStateMultiDof(
            0,
            joint_id,
            targetValue=[traj[startInd][1][i]],
            physicsClientId=physics_client
            )

    img_total = 8
    img_count = 0

    t0 = time.time()
    slowdown_factor = 1.0

    input("Execute?")

    for waypoint in traj:
        if waypoint[1] is not None:
            for i, joint_id in enumerate([0, 1, 2, 3, 4, 5]):
                p.setJointMotorControl2(
                    bodyIndex=0,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=waypoint[1][i],
                    physicsClientId=physics_client,
                )
            while time.time() - t0 < waypoint[0]*slowdown_factor:
                if (1+math.floor(((img_total-1)/(maneuver_time-0))*(time.time() - t0)) > img_count) and img_count < img_total:
                    d435.takePicture(banana_pos)
                    img_count = img_count + 1
                    print("Took a picture, image id: ", img_count-1)
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1 / 240.0)

    input("Finish")
    p.disconnect()

    # Stitching
    # Initialize camera intrinsic parameters (example for Intel RealSense D435)
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


def build_house():
    # Define wall dimensions
    wall_length = 10  # Length of walls
    wall_height = 5   # Height of walls
    wall_thickness = 0.5  # Thickness of the walls

    # Define the colors for each wall
    wall_colors = [
        [1, 0, 0, 1],  # Red for front wall
        [0, 1, 0, 1],  # Green for back wall
        [0, 0, 1, 1],  # Blue for right wall
        [0.5, 0, 0.5, 1]  # Purple
    ]

    # Positioning the walls (front, back, left, and right)
    wall_positions = [
        [0, wall_length / 2, wall_height / 2],  # Front wall
        [0, -wall_length / 2, wall_height / 2], # Back wall
        [wall_length / 2, 0, wall_height / 2],  # Right wall
        [-wall_length / 2, 0, wall_height / 2], # Left wall
    ]

    # Rotation for the right and left walls (90 degrees around the Y-axis)
    wall_orientations = [
        p.getQuaternionFromEuler([0, 0, 0]),  # Front wall (no rotation)
        p.getQuaternionFromEuler([0, 3.14159, 0]),  # Back wall (no rotation)
        p.getQuaternionFromEuler([0, 0, 1.5708]),  # Right wall (90 degrees rotation)
        p.getQuaternionFromEuler([0, 0, -1.5708]), # Left wall (-90 degrees rotation)
    ]

    # Load the walls (using cubes)
    wall_ids = []
    for i, position in enumerate(wall_positions):
        # Create the wall collision shape (box)
        wall_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2])
        # Create the wall body and set the position
        wall_body = p.createMultiBody(baseCollisionShapeIndex=wall_id, basePosition=position, baseOrientation=wall_orientations[i])
        # Change the color of the wall (each wall has a different color)
        p.changeVisualShape(wall_body, -1, rgbaColor=wall_colors[i])
        wall_ids.append(wall_body)

    # Create the roof (another box)
    roof_height = 0.5  # Thickness of roof
    roof_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_length / 2, roof_height / 2])
    roof_position = [0, 0, wall_height + roof_height / 2]
    roof_body = p.createMultiBody(baseCollisionShapeIndex=roof_id, basePosition=roof_position)
    p.changeVisualShape(roof_body, -1, rgbaColor=[0.5, 0.5, 0.5, 1])  # Gray color for the roof

    build_floor(wall_length)


def build_floor(wall_length):
    # Create the floor (ground)
    floor_height = 0.1
    floor_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_length / 2, floor_height / 2])
    floor_position = [0, 0, -floor_height / 2]
    floor_body = p.createMultiBody(baseCollisionShapeIndex=floor_id, basePosition=floor_position)
    p.changeVisualShape(floor_body, -1, rgbaColor=[0.3, 0.3, 0.3, 1])  # Dark ground color


if __name__ == "__main__":
    main()
