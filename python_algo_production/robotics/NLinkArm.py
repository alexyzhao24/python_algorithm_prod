## Basic Class for Descibing a robot arm with n links each of which was described by length and joint angle
# Copied and modified from Python/Robotics git:
# https://atsushisakai.github.io/PythonRobotics/

import numpy as np
import matplotlib.pyplot as plt
from nlink_kinematics import forward_points

### Base class for robotic arm with n links (link lengh, and joint angles)
## Default mode: animation with internal event handling and origin at [0, 0].
## It does allow for external application to handle event via passing the event handler
class NLinkArm:
    def __init__(self, link_lengths, joint_angles, show_animation, event_handler=None, origin=[0.0, 0.0]):
        self.show_animation = show_animation
        self.event_handler = event_handler  # this is pass by reference!
        self.dist_threshold = 0.05
        if len(link_lengths) != len(joint_angles):
            raise ValueError

        # Allocate memory for internal members rather than reference to the input variables/memory!
        # create list of points [x, y] for easy operation: from the origin [x, y] of 1st link to the tips of all the links
        points = [origin]   # the origin is critical
        points += [[0.0,0.0] for __ in range(len(link_lengths))] # other points not important as it will updated based on origin and joint_angles

        # use _var to indicate we want to keep them 'private'
        self._points = np.array(points) # list to array
        self._n_links = len(link_lengths)
        self._link_lengths = np.array(link_lengths, copy=True)
        self._joint_angles = np.array(joint_angles, copy=True)
        self._origin = np.array([origin[0], origin[1]]) # works even when origin is an array
        self._end_effector = np.array(points[-1])
        self._goal = np.array([np.inf, np.inf]) # special value
        self._lim = sum(link_lengths)

        # calculate initial points at CS based on initial joint angles
        self.update_arm(joint_angles)

        if self.show_animation: # pragma: no cover
            if self.event_handler is None:  # handle event internally
                # turn on interactive model
                plt.ion()
                self.fig, self.ax = plt.subplots()
                # create an interactive fig by adding event connection: All Matplotlib events inherit from the base
                # class matplotlib.backend_bases.Event.# non-blocking call
                # plot the arm at initial status
                self.plotArm()

                # set up events: mouse click and key press/release event
                self.fig.canvas.mpl_connect("button_press_event", self.on_click)
                self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
            else:
                # turn on interactive model
                self.fig, self.ax = plt.subplots()
                # create an interactive fig by adding event connection: All Matplotlib events inherit from the base
                # class matplotlib.backend_bases.Event.# non-blocking call
                # plot the arm at initial status
                self.plotArm()

                self.fig.canvas.mpl_connect("button_press_event", self.handle_event)
                self.fig.canvas.mpl_connect("key_release_event", self.handle_event)

    # define the api for applicaion to implement or call
    def handle_event(self, event):
        self.event_handler(event, self)


    # for external app to start Matplotlib event loop
    # which keeps the program running and responsive to user input.
    def run(self):
        plt.show()

    def close(self):
        plt.close(self.fig)

    # mouse click event call-back function when event triggered: set up new goal and plot
    def on_click(self, event):
        self.set_goal(event.xdata, event.ydata)
        print(f"on_click(): arm.goal={self.get_goal()}")
        self.plotArm()

    # call-back function when event triggered: set up new goal and plot
    def on_key_release(self, event):
        if event.key == 'escape':
            self.close()
            exit(0) # exit program

    # basic plot functions to be called by applications
    def update_title(self, fig_name):
        self.ax.set_title(f'{fig_name}')
        self.fig.canvas.draw()

    # update joint angles and Euclidean points
    def update_arm(self, joint_angles):
        # using slice: to assign individual numbers without creating new memory
        self._joint_angles[:] = np.array(joint_angles)
        self._update_points()

    # internal function to use forward_kinematics to update points in Euclidean CS from joint CS
    def _update_points(self):
        # run forward kinematics for all the joints/points
        self._points[:] = forward_points(self.get_points(), self.get_link_lengths(), self.get_joint_angles(),
                            self.get_n_links(), self.get_origin()[0], self.get_origin()[1])
        # using slice: to assign individual numbers without creating new memory
        self._end_effector[:] = self.get_points()[-1]

   # set the goal for the arm to reach
    def set_goal(self, x, y):
        self._goal[:] = np.array([x, y])

    def plotArm(self):  # pragma: no cover
        self.ax.cla() # fig.clf() clear entire figure for re-draw / ax.cla() if you want to clear only the current axes and redraw on it.

        for i in range(self.get_n_links() + 1):
            if i is not self.get_n_links():
                self.ax.plot([self._points[i][0], self._points[i + 1][0]],
                         [self._points[i][1], self._points[i + 1][1]], 'r-') # plot a line
            self.ax.plot(self._points[i][0], self._points[i][1], 'ko')        # plot a point

        # plot the goal point identified by on_click event
        if not np.isnan(self.get_goal()).any():
           self.ax.plot([self._end_effector[0], self._goal[0]], [self._end_effector[1], self._goal[1]], 'g--') # plot a line
           self.ax.plot(self._goal[0], self._goal[1], 'gx')  # plot a point

        self.ax.set_xlim([-self._lim, self._lim])  # with plt, you have plt.xlim()
        self.ax.set_ylim([-self._lim, self._lim])
        self.ax.axis("equal")
        self.fig.canvas.draw()
        # the following statment is critical not blocking the event while drawing the figure
        self.fig.canvas.flush_events()

      # getter methods
    def get_origin(self):
        return self._origin

    def get_goal(self):
        return self._goal

    def get_end_effector(self):
        return self._end_effector

    def get_n_links(self):
        return self._n_links

    def get_lim(self):
        return self._lim

    def get_joint_angles(self):
        return self._joint_angles

    def get_link_lengths(self):
        return self._link_lengths

    def get_points(self):
        return self._points