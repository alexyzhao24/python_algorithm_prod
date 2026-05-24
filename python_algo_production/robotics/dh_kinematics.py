# Copied and modified from Python/Robotics git:
# https://atsushisakai.github.io/PythonRobotics/

import numpy as np
import math

# DH Denavit-Hartenberg Parameters
# https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
class Link:
    def __init__(self, dh_params):
        self._dh_params = dh_params

    # transformation matrix given DH parameters
    def transformation_matrix(self):
        theta = self._dh_params[0]  # joint angle is the angle from x_i-1 to x_i, measured about z_i
        alpha = self._dh_params[1]  # link twist is the angle between z_i-1 to z_i, measured about x_i-1
        a = self._dh_params[2]      # aka r, link length, assuming a revolute joint, this is the radius about previous z
        d = self._dh_params[3]      # link offset is the distance from the intersection of x_i-1 and z_i to the origin of the link-i frame

        st = math.sin(theta)
        ct = math.cos(theta)
        sa = math.sin(alpha)
        ca = math.cos(alpha)

        # transformation matrix for homogeneous coord: [ R  T; 0 0 0 1]
        trans = np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d],
                          [0, 0, 0, 1]])
        return trans

    # function for calculating basic jacobian (per link)
    '''
        trans_prev: The transformation matrix of the previous joint
        ee_pos: The position of the end-effector
    '''
    @staticmethod
    def jacobian(trans_prev, ee_pos):
        # Extracts the position of the previous joint from its transformation matrix
        pos_prev = np.array(
            [trans_prev[0, 3], trans_prev[1, 3], trans_prev[2, 3]])

        # Extracts the Z-axis of the previous joint's coordinate frame
        z_axis_prev = np.array(
            [trans_prev[0, 2], trans_prev[1, 2], trans_prev[2, 2]])

        # Calculates the Jacobian column for a revolute joint
        # Linear velocity part: np.cross(z_axis_prev, ee_pos - pos_prev)
        #    Cross product of the joint's rotation axis and the vector from joint to end-effector
        # Angular velocity part: z_axis_prev
        #    The joint's rotation axis
        jacobian = np.hstack(
            (np.cross(z_axis_prev, ee_pos - pos_prev), z_axis_prev))

        return jacobian

# function to convert transformation matrix [R T; 0 1] into euler angles
# alpha / beta / gamma that create the R in the transformation matrix
def trans_to_euler_angle(trans):
    alpha = math.atan2(trans[1][2], trans[0][2])
    if not (-math.pi / 2 <= alpha <= math.pi / 2):
        alpha = math.atan2(trans[1][2], trans[0][2]) + math.pi
    if not (-math.pi / 2 <= alpha <= math.pi / 2):
        alpha = math.atan2(trans[1][2], trans[0][2]) - math.pi
    beta = math.atan2(
        trans[0][2] * math.cos(alpha) + trans[1][2] * math.sin(alpha),
        trans[2][2])
    gamma = math.atan2(
        -trans[0][0] * math.sin(alpha) + trans[1][0] * math.cos(alpha),
        -trans[0][1] * math.sin(alpha) + trans[1][1] * math.cos(alpha))

    return alpha, beta, gamma


# Calculate the cascaded transformation matrix:
''' [T]=[Z1][X1][Z2][X2]…[Xn−1][Zn][Xn] '''
def dh_transformation_matrix(link_list):
    trans = np.identity(4)
    for i in range(len(link_list)):
        trans = np.dot(trans, link_list[i].transformation_matrix())
    return trans


# forwarad kimeatics become simple using DH parameters rturing ee_pos with [x, y, z; alpha, beta, gamma]
def dh_forward_kinematics(link_list):
    trans = dh_transformation_matrix(link_list)

    x = trans[0, 3]
    y = trans[1, 3]
    z = trans[2, 3]
    alpha, beta, gamma = trans_to_euler_angle(trans)

    # Position and Orientation of the end-effector
    return [x, y, z, alpha, beta, gamma]


# Jacobian become simple using DH parameters: one column with basic jacobian per joint
def dh_jacobian(link_list):
    ee_pos = dh_forward_kinematics(link_list)[0:3]
    jacobian_mat = []

    trans = np.identity(4)
    for i in range(len(link_list)):
        jacobian_mat.append(link_list[i].jacobian(trans, ee_pos))
        trans = np.dot(trans, link_list[i].transformation_matrix())

    return np.array(jacobian_mat).T

# numerical/iterative inverse kinematics to update theta/alpha. Note that d/a are typically fixed!
def dh_inverse_kinematics(link_list, ref_ee_pose, dist_threshold=0.1):
    for cnt in range(500):
        diff_pose = get_diff_pose(link_list, ref_ee_pose)

        dist = diff_pose_to_distance(diff_pose)

        if dist < dist_threshold:
            return get_joint_angle_list(link_list), True,  dist
        else:
            # print(f"dh_inverse_kinematics(): iter {cnt} diff_pose {diff_pose}")

            # Jacobian matrix
            jacobian_mat = dh_jacobian(link_list)
            alpha, beta, gamma = trans_to_euler_angle(dh_transformation_matrix(link_list))

            K_zyz = np.array(
                [[0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
                    [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
                    [1, 0, math.cos(beta)]])
            K_alpha = np.identity(6)
            K_alpha[3:, 3:] = K_zyz

            # use inverse jacobian matrix to update joint angles theta (r and d are fixed)
            theta_dot = np.dot(
                np.dot(np.linalg.pinv(jacobian_mat), K_alpha),
                np.array(diff_pose))
            update_joint_angle_list(link_list, theta_dot / 100.)

    return get_joint_angle_list(link_list), False, dist

# distance given diff pose: how to combine distance and angular errors?
def diff_pose_to_distance(diff_pose):
    weights = [0.9, 0.1]  # weights: linear vs angular
    distance = math.sqrt(sum((np.array(diff_pose[0:2])*weights[0] + np.array(diff_pose[0:2])*weights[1])**2))
    return distance

# calculate diff in pose
def get_diff_pose(link_list, ref_ee_pose):
    ee_pose = dh_forward_kinematics(link_list)
    diff_pose = [ref_ee_pose[i] - ee_pose[i] for i in range(len(ref_ee_pose))]
    return diff_pose

# get joint angles (thetas) from DH parameters
def get_joint_angle_list(link_list):
    joint_angle_list = []
    for i in range(len(link_list)):
        joint_angle_list.append(link_list[i]._dh_params[0])
    return joint_angle_list

# set DH params using zero joint_angles: theta = 0
def zero_joint_angle_list(link_list):
    for i in range(len(link_list)):
        link_list[i]._dh_params[0] = 0.0

# set joint_angle_list of the DH params: [theta]
def set_joint_angle_list(link_list, joint_angle_list):
    for i in range(len(link_list)):
        link_list[i]._dh_params[0] = joint_angle_list[i]

# update joint_angle_list of the DH params: [theta]
def update_joint_angle_list(link_list, diff_joint_angle_list):
    for i in range(len(link_list)):
        link_list[i]._dh_params[0] += diff_joint_angle_list[i]
