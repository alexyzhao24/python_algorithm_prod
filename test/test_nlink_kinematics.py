import test
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from python_algo_production.robotics.nlink_kinematics import forward_kinematics, forward_points, inverse_kinematics, distance_to_goal, ang_diff

def get_random_goal():
    from random import random
    SAREA = 15.0
    return [SAREA * random() - SAREA / 2.0,
            SAREA * random() - SAREA / 2.0]


## Special test with zero distanace to goal
def test_invkin_zero():
        # creat n-link with n=7 with parameter initializatio
    N_LINKS = 10
    link_lengths = np.array([1.0] * N_LINKS)      # link_lengths = np.array([5., 3., 2., 1., 1., 0.5, 0.25])
    joint_angles = np.array([0.0] * N_LINKS)
    origin = [0, 0]
    # creat n-link with n=7 with parameter initializatio
    link_points = np.array([[0.,0.] for __ in range(N_LINKS + 1)])
    link_points = forward_points(link_points, link_lengths, joint_angles, N_LINKS)
    print(f"link_points\n: {link_points}")
    #  run forward points with the orign given
    end_point = forward_kinematics(link_lengths, joint_angles, N_LINKS, origin[0], origin[1])
    cur_pos = end_point
    print(f"cur_goal: {end_point}")

    # kinematics related param
    Kp = 1    # proportional gain for inverse kinematics
    dt = 0.1  # time step for inverse kinematics
    # special test of zero distance
    goal_pos = cur_pos
    link_points[0][0], link_points[0][1] = origin[0], origin[1]

    print(f"Running inverse Kinematics with Kp {Kp}: cur_pos={cur_pos}, goal_pos={goal_pos}")
    joint_goal_angles, solution_found, distance = inverse_kinematics(link_lengths, joint_angles, N_LINKS, goal_pos, Kp, origin)
    # update cur_pos after inverse kinematics
    updated_pos = forward_kinematics(link_lengths, joint_goal_angles, N_LINKS, origin[0], origin[1])
    link_points = forward_points(link_points, link_lengths, joint_goal_angles, N_LINKS)
    print(f"updated pos={updated_pos}, end_points={link_points[-1,:]}, goal_pos={goal_pos}\n")


## Let's try different Kp to see the impact on convergence with random goals
def test_invkin_gain():
    # creat n-link with n=7 with parameter initializatio
    N_LINKS = 10
    link_lengths = np.array([1.0] * N_LINKS)      # link_lengths = np.array([5., 3., 2., 1., 1., 0.5, 0.25])
    joint_angles = np.array([0.0] * N_LINKS)
    origin = [0, 0]
    # creat n-link with n=7 with parameter initializatio
    link_points = np.array([[0.,0.] for __ in range(N_LINKS + 1)])
    link_points = forward_points(link_points, link_lengths, joint_angles, N_LINKS)
    print(f"link_points\n: {link_points}")
    #  run forward points with the orign given
    end_point = forward_kinematics(link_lengths, joint_angles, N_LINKS, origin[0], origin[1])
    cur_pos = end_point
    print(f"cur_goal: {end_point}")

    for Kp in list([1.0, 0.25, 0.5]):
        # perturb the goal
        for iter in range(10):
            # reinitiate
            goal_pos = get_random_goal()
            joint_angles = np.array([0.0] * N_LINKS)

            print(f"Running inverse Kinematics with Kp {Kp}: cur_pos={cur_pos}, goal_pos={goal_pos}")
            joint_goal_angles, solution_found, distance = inverse_kinematics(link_lengths, joint_angles, N_LINKS, goal_pos, Kp, origin)
            # update cur_pos after inverse kinematics
            updated_pos = forward_kinematics(link_lengths, joint_goal_angles, N_LINKS, origin[0], origin[1])
            # Test the difference between forwar_points (used in NLinkArm) and forward_kinematics
            link_points = forward_points(link_points, link_lengths, joint_goal_angles, N_LINKS)
            if solution_found:
                print(f"Found solution: updated pos={updated_pos}, end_points={link_points[-1,:]}, goal_pos={goal_pos}\n")
            else:
                print(f"Solution Not Found: updated pos={updated_pos}, end_points={link_points[-1,:]}, goal_pos={goal_pos}\n")


## Test a particular goal position
def test_invkin_particular():
    # creat n-link with n=7 with parameter initializatio
    N_LINKS = 10
    link_lengths = np.array([1.0] * N_LINKS)      # link_lengths = np.array([5., 3., 2., 1., 1., 0.5, 0.25])
    joint_angles = np.array([0.0] * N_LINKS)
    origin = [0, 0]
    # creat n-link with n=7 with parameter initializatio
    link_points = np.array([[0.,0.] for __ in range(N_LINKS + 1)])
    link_points = forward_points(link_points, link_lengths, joint_angles, N_LINKS)
    print(f"link_points\n: {link_points}")
    #  run forward points with the orign given
    end_point = forward_kinematics(link_lengths, joint_angles, N_LINKS, origin[0], origin[1])
    cur_pos = end_point
    print(f"cur_goal: {end_point}")

    goal_pos = [1.28528226, -4.80194805]
    Kp = 0.5
    joint_angles = np.array([0.0] * N_LINKS)
    print(f"\nRunning inverse Kinematics with Kp {Kp}: cur_pos={cur_pos}, goal_pos={goal_pos}")
    joint_goal_angles, solution_found, distance = inverse_kinematics(link_lengths, joint_angles, N_LINKS, goal_pos, Kp, origin)
    # update cur_pos after inverse kinematics
    updated_pos = forward_kinematics(link_lengths, joint_goal_angles, N_LINKS, origin[0], origin[1])
    link_points = forward_points(link_points, link_lengths, joint_goal_angles, N_LINKS)
    print(f"Found solution: updated pos={updated_pos}, end_points={link_points[-1,:]}, goal_pos={goal_pos}\n")

