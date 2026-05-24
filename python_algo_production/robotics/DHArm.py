## Basic Class for Descibing a robot arm with n links each of which was described by length and joint angle
# Copied and modified from Python/Robotics git:
# https://atsushisakai.github.io/PythonRobotics/

import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dh_kinematics import Link, set_joint_angle_list, get_joint_angle_list, zero_joint_angle_list, \
                                      get_diff_pose, diff_pose_to_distance


class NLinkDHArm:
    def __init__(self, dh_params_list, obstacle_list=None, show_animation=None):
        self.show_animation = show_animation
        self._link_list = []
        self._goal_pose = None
        self._obstacle_list = obstacle_list
        for i in range(len(dh_params_list)):
            self._link_list.append(Link(dh_params_list[i]))

        if self.show_animation: # pragma: no cover
            # turn on interactive model
            plt.ion()
            self.fig = plt.figure()
            # The auto_add_to_figure=False parameter prevents the axes from being automatically added to the figure.
            self.ax = Axes3D(self.fig, auto_add_to_figure=False)
            # now add the ax to the figure
            self.fig.add_axes(self.ax)
            self.plotArm()

            # set up events: key press/release event
            self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)


    # call-back function when event triggered: set up new goal and plot
    def on_key_release(self, event):
        if event.key == 'q':
            self.close()
            exit(0) # exit program

    # for external app to start Matplotlib event loop
    # which keeps the program running and responsive to user input.
    def run(self):
        plt.show()

    def close(self):
        plt.close(self.fig)


    # basic plot functions to be called by applications
    def update_title(self, fig_name):
        self.ax.set_title(f'{fig_name}')
        self.fig.canvas.draw()

    # purely internal function
    def _get_arm_joints(self):
        x_list = []
        y_list = []
        z_list = []

        trans = np.identity(4)

        x_list.append(trans[0, 3])
        y_list.append(trans[1, 3])
        z_list.append(trans[2, 3])
        for i in range(len(self._link_list)):
            trans = np.dot(trans, self._link_list[i].transformation_matrix())
            x_list.append(trans[0, 3])
            y_list.append(trans[1, 3])
            z_list.append(trans[2, 3])
        return x_list, y_list, z_list

    def plotArm(self, titleText=None, specialPoint=None, obstacle_list=None):
        self.ax.cla() # fig.clf() clear entire figure for re-draw / ax.cla() if you want to clear only the current axes and redraw on it.

        # update arm and transformation matrix
        x_list, y_list, z_list = self._get_arm_joints()

        self.ax.plot(x_list, y_list, z_list, "o-", color="#00aa00", ms=4, mew=0.5)
        self.ax.plot([0], [0], [0], "o")

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        if specialPoint is not None:
            self.ax.scatter(specialPoint[0], specialPoint[1], specialPoint[2], s=50, c='r', marker="*")
        if titleText is not None:
            self.fig.suptitle(titleText)
        if obstacle_list is not None:
            for (ox, oy, oz, size) in obstacle_list:
                self.plot_sphere(ox, oy, oz, size=size, color='0.25') # 0.25: dark gray, 0.75: light gray

        self.ax.axis("equal")
        self.fig.canvas.draw()
        # the following statment is critical not blocking the event while drawing the figure
        self.fig.canvas.flush_events()


    # Animation from current DH to the goal DH
    def animate_move_to_goal(self, joint_angle_list, obstacle_list=None, goal=None, KpDt=None):
        distance = np.inf
        step = 1
        dist_threshold = 0.01
        Kp_move = 2
        dt = 0.1
        if KpDt is None:
            KpDt = Kp_move*dt
        if goal is None:
                goal = self.get_goal_pose()
        joint_move_angle_list = get_joint_angle_list(self.get_link_list())
        while distance > dist_threshold and step < 50:
            for i in range(len(joint_angle_list)):
                joint_move_angle_list[i] += KpDt * (joint_angle_list[i] - joint_move_angle_list[i])
            self.set_joint_angles(joint_move_angle_list)
            diff_pose = get_diff_pose(self.get_link_list(), goal)
            distance = diff_pose_to_distance(diff_pose)
            self.plotArm(titleText=f"Moving to goal [{step}]: distance {distance:0.3f}", specialPoint=self.get_goal_pose(), obstacle_list=obstacle_list)
            plt.pause(0.1)
            step += 1

    # utility function of plotting spheric obstacles
    def plot_sphere(self, x, y, z, size=1, color="k"): # p.iragma: no cover
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xl = x+size*np.cos(u)*np.sin(v)
        yl = y+size*np.sin(u)*np.sin(v)
        zl = z+size*np.cos(v)
        self.ax.plot_wireframe(xl, yl, zl, color=color)

    # getter methods
    def get_link_list(self):
        return self._link_list

    def get_goal_pose(self):
        return self._goal_pose

    def get_obstacle_list(self):
        return self._obstacle_list

    def set_obstacle_list(self, obstacleList):
        self._obstacle_list = obstacleList

    # setter methods
    def set_joint_angles(self, joint_angle_list):
        set_joint_angle_list(self.get_link_list(), joint_angle_list)

    # set DH params using zero joint_angles: theta = 0
    def zero_joint_angles(self):
        zero_joint_angle_list(self.get_link_list())

    def set_goal_pose(self, goal_pose):
        self._goal_pose = goal_pose

    def get_plot_ax(self):
        return self.ax

    # input new joint angle_list and output link lists
    def get_arm_joints(self, joint_angle_list):
        self.set_joint_angles(joint_angle_list)

        return self._get_arm_joints()