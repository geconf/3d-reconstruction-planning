"""A demo of the Bullet API."""

import os
import numpy as np
import time
import pybullet as p
import pybullet_data
import bullet_camera

import sys
sys.path.append(os.path.abspath('./Expansion-GRR'))
from bullet_api.loader import load_grr


def main():
    """Build a demo"""
    # Load a GRR resolution
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Expansion-GRR')
    print(dir)
    # urdf = dir + "/data/robots/ur10_bullet.urdf"
    urdf = dir + "/data/robots/ur10.urdf"
    grr = load_grr(urdf, "ur10", "rot_variable_yaw")

    # TODO for the users
    # Define workspace path in the robot base frame
    # example - move along y axis
    ys = np.linspace(-0.5, 0.5, num=20)
    workspace_path = [
        ([0.5, y, 0.05], [0.7071068, 0.7071068, 0, 0]) for y in ys
    ]

    # Solve the Cartesian path with GRR
    config_path = grr_plan(grr, workspace_path)
    print(config_path)

    # TODO
    # If you need to build a time-parametrized trajectory,
    # include the time path to generate a trajectory
    # It would be better to include velocity and acceleration as well
    time_path = np.linspace(0, 2, len(config_path))
    traj = [(t, q) for t, q in zip(time_path, config_path)]

    # Display Results
    print("\nTrajectory:")
    for waypoint in traj:
        print(f"Time {waypoint[0]}: {waypoint[1]}")

    # Run the simulation
    sim = p.connect(p.GUI)
    robot_id = p.loadURDF(urdf, useFixedBase=True, physicsClientId=sim)

    banana_urdf = './011_banana/banana.urdf'
    banana_pos = [0.5, 0.5, 0.0]
    banana_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # Need to load these after the robot as joint_ids are 0-based
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.loadURDF("plane.urdf",
               physicsClientId=sim)
    p.loadURDF(banana_urdf,
               basePosition=banana_pos,
               baseOrientation=banana_orientation,
               physicsClientId=sim,
               useFixedBase=True)

    # Initialize camera
    d435 = bullet_camera.Camera(640, 480, physicsClientId=sim)

    # Init
    for i, joint_id in enumerate([0, 1, 2, 3, 4, 5]):
        p.resetJointStateMultiDof(
            0, joint_id, targetValue=[traj[0][1][i]], physicsClientId=sim
        )
    input("Execute?")

    t0 = time.time()
    for waypoint in traj:
        for i, joint_id in enumerate([0, 1, 2, 3, 4, 5]):
            # Fake execution
            # p.resetJointStateMultiDof(
            #     0, joint_id, targetValue=[waypoint[1][i]], physicsClientId=sim
            # )
            # Real execution
            p.setJointMotorControl2(
                bodyIndex=0,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=waypoint[1][i],
                physicsClientId=sim,
            )
        while time.time() - t0 < waypoint[0]:
            p.stepSimulation(physicsClientId=sim)
            time.sleep(1 / 240.0)

    # Take picture
    d435_index = 13
    d435_state = p.getLinkState(
            robot_id,
            d435_index,
            computeForwardKinematics=True,
            physicsClientId=sim)
    d435_world_position = d435_state[0]
    d435.takePicture(d435_world_position, banana_pos)

    input("Finish")
    p.disconnect()


def grr_plan(grr, workspace_path):
    """Plan pushing with GRR"""
    config_path = [
        grr.solve(waypoint[0] + waypoint[1], none_on_fail=True)
        for waypoint in workspace_path
    ]

    # # Debug
    # for config in config_path:
    #     print(config)

    # TODO 0
    # Valid solution check
    for conf in config_path:
        if conf is None:
            print("\nInvalid configuration found\n")
            return config_path

    # TODO 1
    # Collision checking with obstacles is not implimented

    # TODO 2
    # Continuity checking is not performed,
    # although this should be guranteed by GRR.
    # One corner case is that base joint believes 0 and 2pi are the same,
    # but the physical robot will not think so.

    return config_path


if __name__ == "__main__":
    main()
