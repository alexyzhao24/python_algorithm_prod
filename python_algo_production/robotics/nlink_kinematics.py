# Copied and modified from Python/Robotics git:
# https://atsushisakai.github.io/PythonRobotics/

'''
    <n-link arm with these mechanics>
    origin                                  point 1 =   origin
    angle 1     x      link 1               point 2 =   point 1 + [l1 * cos(angle1), l1 * sin(angle1)]
    angle 2     x      link 2               point 3 =   point 2 + [l2 * cos(angle1+angle2), l2 * sin(angle1+angle2)]
    ......
    angle n-1   x      link n-1             point n =   point n-2 + [ln-1 * cos(angle1+...+angle n-1)]
    angle n     x      end link n           point n+1 = point n-1 + [ln * cos(angle1+...+angle n)]
'''

import numpy as np

# forward kinematics from joint angles to Euclidian (x, y)
def forward_kinematics(link_lengths, joint_angles, N_LINKS, origin_x=0, origin_y=0):
    x, y = origin_x, origin_y
    for i in range(1, N_LINKS + 1):
        # incremental joint angles --> absolute angles: x=l1*cos(theta1)+l2*cos(theta1+theta2)+l3*cos(theta1+theta2+theta3)...
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))  # not inclusive: [:i] means [0], [1], ..., [i-1]
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T

# forward points from joint angles to Euclidian (x, y)
# Please noticed that the same points are both input and ouput
def forward_points(points, link_lengths, joint_angles, N_LINKS, origin_x=0, origin_y=0):
    points[0][0], points[0][1] = origin_x, origin_y
    for i in range(1, N_LINKS + 1):
        # incremental joint angles --> absolute angles: x=l1*cos(thejointa1)+l2*cos(theta1+theta2)+l3*cos(theta1+theta2+theta3)...
        points[i][0] = points[i - 1][0] + link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        points[i][1] = points[i - 1][1] + link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return points

# derivative of the forward-kinematics 2xn matrix : [dx/dtheta_i, dy/dtheta_i]
# x=l1*cos(theta1) + l2*cos(theta1+theta2) + l3*cos(theta1+theta2+theta3)... + ln*cos(theta1+...+theta n)
# y=l1*sin(theta1) + l2*sin(theta1+theta2) + l3*sin(theta1+theta2+theta3)... + ln*sin(theta1+...+theta n)
def jacobian(link_lengths, joint_angles, N_LINKS):
    J = np.zeros((2, N_LINKS))
    for i in range(0, N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        # travese forward through all the links
        for j in range(i, N_LINKS):
            # derivative: dx/dtheta1 = -l1*sin(theta1) -l2*sin(theta1+theta2) -l3*sin(theta1+theta2+theta3) ...
            # derivative: dx/dtheta2 =                 -l2*sin(theta1+theta2) -l3*sin(theta1+theta2+theta3) ...
            # derivative: dx/dtheta3 =                                        -l3*sin(theta1+theta2+theta3) ...
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j+1]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j+1]))  # not inclusive: [:j+1] means [0], [1], ..., [j]
    return J

# derivative of the inverse-kinematics nx2 matrix: [dtheta_i/dx, dtheta_i/dy]
# pseudo-inverse due to the rank-deficiency of jacobian
def jacobian_inverse(link_lengths, joint_angles, N_LINKS):
    # call jacobian
    J = jacobian(link_lengths, joint_angles, N_LINKS)
    # return pseudo inverse as the Jacobian inverse: jacobian likely to be rank-deficient (when n>2)
    return np.linalg.pinv(J)


'''
Tunability: The Kp factor acts as a control gain, allowing you to tune the speed and accuracy of the
inverse kinematics solution.
Damping: Using Kp effectively implements a damped least squares approach, which helps avoid large joint
velocities near singularities.

The typical formulation using Kp is:
    Δθ = Kp * J^+ * e

    Where:
    Δθ is the joint angle update
    Kp is the gain factor (typically between 0 and 1)
    J^+ is the pseudoinverse of the Jacobian
    e is the error between current and desired end-effector position
'''
# iterative inverse kinematics from Euclidian (x, y) to joint angles
# Please noticed that the same joint_angles are both input and ouput
def inverse_kinematics(link_lengths, joint_angles, N_LINKS, goal_pos, Kp=1.0, origin = [0.0, 0.0], dist_threshold=0.05, MAX_ITER=1000):
    for iter in range(MAX_ITER):
        # calculate the error and derivative
        current_pos = forward_kinematics(link_lengths, joint_angles, N_LINKS, origin[0], origin[1])
        err, distance = distance_to_goal(current_pos, goal_pos)
        # print(f"inverse_kinematics(): cur_pos {current_pos}, distance {distance}, joint_angles {joint_angles}")
        if distance < dist_threshold:
            print(f"inverse kinematics(): solution found in {iter} iterations")
            return joint_angles, True, distance
        J_inv = jacobian_inverse(link_lengths, joint_angles, N_LINKS)
        # let's update the joint angle based on inverse of the derivative with a gain Kp for forward kinematics
        # https://nrsyed.com/2017/12/10/inverse-kinematics-using-the-jacobian-inverse-part-2/
        joint_angles += Kp*np.matmul(J_inv, err)
    return joint_angles, False, distance


# calculate the difference from current_pos to goal_pos
def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)

# We need to handle angle difference due to 2pi round up
def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    x = theta1 - theta2
    return (x + np.pi) % (2 * np.pi) - np.pi