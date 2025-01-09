"""Main file to run global redundancy resolution to find a solution for a robot"""

import os
import sys
import time
import pickle
import numpy as np

from grr.json_utils import load_json
from grr.robot import KinematicChain, Kinova, UR10
from grr.resolution import RedundancyResolution

from experiment.roadmap_quality import evaluate_roadmap


def main(opts, objPos, load_existed_ws_graph=False, load_existed_solver_graph=False):
    """Run global redundancy resolution"""
    RobotClass = getattr(sys.modules[__name__], opts["robot_class"])
    robot = RobotClass(
        opts["robot_name"],
        opts["domain"],
        opts["rotation_domain"],
        opts["fixed_rotation"],
    )

    # Graph folder path
    graph_folder = (
        "graph/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    # Global redundancy resolution
    resolution = RedundancyResolution(robot)

    # Build workspace graph
    if not load_existed_ws_graph:
        resolution.sample_workspace(
            objPos,
            n_pos_points=opts["number_of_position_points"],
            n_rot_points=opts["number_of_rotation_points"],
            sampling_method="random",
        )
        resolution.save_workspace_graph(
            graph_folder + "graph_workspace.pickle",
            graph_folder + "nn_workspace.pickle",
        )
    else:
        resolution.load_workspace_graph(
            graph_folder + "graph_workspace.pickle",
            graph_folder + "nn_workspace.pickle",
        )
    # resolution.visualize_workspace_graph()

    # Code to sample valid initial configurations from current workspace graph goes here
    
    with open(graph_folder+"graph_workspace.pickle", "rb") as graph_file_IC:
        graph_IC = pickle.load(graph_file_IC)
    # Iterate over all nodes and their attributes
    for node_id, attributes in graph_IC.nodes(data=True):
        if 'point' in attributes:
            point = attributes['point']  # Extract the 6D point
            #print(f"Node {node_id}: 6D Point = {point}")
        else:
            print(f"Node {node_id} has no 'point' attribute")

    new_init_configs = np.zeros((8,6))
    new_init_wspts = np.zeros((8,7))
    space_res = len(graph_IC.nodes)/10
    space_out = 1
    last_space = 0;
    new_q_ind = 0;
        
    for node_id, attributes in graph_IC.nodes(data=True):
        point = attributes['point']
        #print(point)
        #q = resolution.robot.solve_ik(point)
        q = resolution.robot.solve_ik(point, max_iters=100, tolerance=1e-2)
        #print(q)
        
        jointCont = 1
  
        if (new_q_ind!=0) and q is not None:
            if (resolution.robot.distance(new_init_configs[new_q_ind-1],q)<4):
                jointCont = 1
            else:
                jointCont = 0
        
        if q is not None and space_out and (new_q_ind<8) and jointCont:
  
            print("New Initial Config Found: ", new_q_ind+1)
            new_init_configs[new_q_ind] = q
            new_init_wspts[new_q_ind] = point
            new_q_ind = new_q_ind + 1

            last_space = node_id
            space_out = 0
            
        if abs(node_id - last_space)>space_res:
            space_out = 1
    
    
    print(new_init_configs)     
    print(new_init_wspts)
    
    init_conf_arr = []
    for i in range(len(new_init_configs)):
        if(i==0):
            diff = 0
        else:
            diff = resolution.robot.distance(new_init_configs[i-1], new_init_configs[i])
        init_conf_arr.append(diff)
    print(init_conf_arr)
    

    # Timer
    start_time = time.time()

    # Build configuration space graph
    if not load_existed_solver_graph:
        #resolution.global_expansion(opts["init_configs"])
        resolution.global_expansion(new_init_configs)
        resolution.save_solver_graph(graph_folder + "graph_solver.pickle")
    else:
        resolution.load_solver_graph(graph_folder + "graph_solver.pickle")

    # Optimization
    resolution.fix_boundary(n_neighbor_layer=1, n_iter=2)
    # TODO Future Improvement
    # resolution.optimize_resolution(n_length_optimization=2)

    # Timer
    print("Total Computation Time:", time.time() - start_time)

    # Build resolution graph
    resolution.build_resolution_graph_and_nn()
    resolution.save_solver_graph(graph_folder + "graph_solver.pickle")
    resolution.save_resolution_graph(
        graph_folder + "graph_resolution.pickle",
        graph_folder + "nn_resolution.pickle",
    )
    # resolution.load_resolution_graph(
    #     graph_folder + "graph_resolution.pickle",
    #     graph_folder + "nn_resolution.pickle",
    # )

    # Evaluate the roadmap
    evaluate_roadmap(resolution)


if __name__ == "__main__":
    # Default json file
    robot_name = "ur10"
    json_file_name = "rot_free"
    
    objPos = [0.75, 0.75, 0]; #Hard coding an obstacle position
    
    # Override with system arguments if provided
    if len(sys.argv) == 2:
        print("Need to specify both the robot and the json file")
        print("python redundancy.py <robot> <json>")
    elif len(sys.argv) > 2:
        robot_name = sys.argv[1]
        json_file_name = sys.argv[2]

    opts = load_json(robot_name, json_file_name)
    main(opts, objPos, False, False)
