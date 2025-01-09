from rtde.rtde import RTDE
import numpy as np

def test():
    rtde = RTDE("192.168.1.102")

    # Test joint control functions
    # Set home first
    home = [1.57, -1.7, 2, -1.87, -1.57, 3.14]
    
    filename = "ctraj.txt"

    target1 = [1.5, -1.6, 1.9, -1.8, -1.5, 3]
    target2 = [1.3, -1.5, 1.8, -1.7, -1.4, 2.9]
    target3 = [1.1, -1.4, 1.5, -1.8, -1.5, 3.14]
    traj2 = [home ,target1, target2, target3]
    
    #print(traj2)
    
    #traj = [home, target1, target2, target3]
    traj = []
    traj.append(home)
    
    traj2 = []
    traj2.append(home)
    traj2.append(target1)
    traj2.append(home)

    with open(filename, "r") as file:
        for line in file:
            time_str, joint_values_str = line.split(",[")
            
            joint_values = list(map(float, joint_values_str.strip("]\n").split()))
            
            joint_values_wrapped = []

            for i in range(len(joint_values)):
                offAng = np.arctan(-1/0) #For Object placed at 0,-1 
                q1Disp = pi + offAng - pi/4
                
                if(i==0):
                    currAng = joint_values[i]+q1Disp
                else:
                    currAng = joint_values[i]
                roundFact = round(currAng/(2*np.pi))
                #print(roundFact)
                wrappedAng = currAng - 2*np.pi*roundFact
                joint_values_wrapped.append(wrappedAng)

            #traj.append(joint_values)
            traj.append(joint_values_wrapped)
    #print(traj)
    
    traj.append(home)
    print("Traj length: ",len(traj))
    print(traj[0])
    print(traj[1])

    traj = [traj[i] + [0.1, 0.1, 0.02] for i in range(len(traj))]
    traj2 = [traj2[i] + [0.1, 0.1, 0.02] for i in range(len(traj2))]

    try:
        # Move to home
        rtde.move_joint(home)

        # Get current joint angle and tool pose
        curr_joint = rtde.get_joint_values()
        print("Joint values: ", curr_joint)

        # Test joint controls
        #rtde.move_joint(target1)

        # Test joint trajectory
        #rtde.move_joint(home)
        rtde.move_joint_trajectory(traj)

        # Stop the robot
        rtde.stop()

    finally:
        rtde.stop_script()


if __name__ == "__main__":
    test()
