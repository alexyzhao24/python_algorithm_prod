# Copied and modified from Python/Robotics git:
# https://atsushisakai.github.io/PythonRobotics/


''' From Modern Robotics Book 10.5.1
# Algorithm 10.3 RRT algorithm.
    initialize search tree T with x start
    while T is less than the maximum tree size do
        x samp ← sample from X
        x nearest ← nearest node in T to x samp
        employ a local planner to find a motion from x nearest to x new in
            the direction of x samp
        if the motion is collision-free then
            add x new to T with an edge from x nearest to x new
            if x new is in X goal then
                return SUCCESS and the motion to x new
            end if
        end if
    end while
    return FAILURE

* The Sampler: x samp ← sample from X
    The most obvious sampler is one that samples randomly from a uniform distri-
    bution over X . This is straightforward for Euclidean C-spaces R^n , as well as
    for n-joint robot C-spaces T^n = S^1 × · · · × S&1 (n times), where we can choose
    a uniform distribution over each joint angle; and for the C-space R^2 × S^1 for a
    mobile robot in the plane, where we can choose a uniform distribution over R^2
    and S^1 individually.

    For dynamic systems, a uniform distribution over the state space
    can be defined as the cross product of a uniform distribution over C-space and a uniform
    distribution over a bounded velocity set.

* Defining the Nearest Node:  x nearest ← nearest node in T to x samp
    Finding the “nearest” node depends on a definition of distance on X . For an
    unconstrained kinematic robot on C = R^n , a natural choice for the distance
    between two points is simply the Euclidean distance. For other spaces, the choice is less obvious.

    As an example, for a car-like robot with a C-space R^2 × S^1 , which configuration is
    closest to the configuration x_samp : one that is rotated 20 degrees relative
    to x_samp , one that is 2 meters straight behind it, or one that is 1 meter straight
    to the side of it (Figure 10.18)? Since the motion constraints prevent spinning
    in place or moving directly sideways, the configuration that is 2 meters straight
    behind is best positioned to make progress toward x_samp . Thus defining a notion
    of distance requires
        • combining components of different units (e.g., degrees, meters, degrees/s,
        meters/s) into a single distance measure; and
        • taking into account the motion constraints of the robot.
    A simple choice of a distance measure from x to x samp is the weighted sum
    of the distances along the different components of x_samp − x. The weights
    express the relative importance of the different components. If more is known
    about the set of states that the robot can reach from a state x in limited time, this
    information can be used in determining the nearest node.

* The Local Planner: find a motion from x_nearest to x_new in the direction of x_samp
    The job of the local planner is to find a motion from x_nearest to some point
    x_new which is closer to x_samp . The planner should be simple and it should run
    quickly. Three examples are as follows.
    1) A straight-line planner. The plan is a straight line to x_new , which may be
    chosen at x_samp or at a fixed distance d from x_nearest on the straight line to
    x_samp . This is suitable for kinematic systems with no motion constraints.
    2) Discretized controls planner. For systems with motion constraints, such as
    wheeled mobile robots or dynamic systems, the controls can be discretized
    into a discrete set {u_1 , u_2 , . . .}, as in the grid methods with motion con-
    straints (Section 10.4.2 and FiguresFalsege 10.14 and 10.16). Each control is
    integrated from x_nearest for a fixed time ∆t using ẋ = f (x, u). Among the
    new states reached without collision, the state that is closest to x_samp is
    chosen as x_new .
    3) Wheeled robot planners. For a wheeled mobile robot, local plans can be
    found using Reeds–Shepp curves, as described in Section 13.3.3.
'''

import math
import random
import copy
import matplotlib.pyplot as plt
import numpy as np


class RRT:
    """
    Class for RRT planning
    """
    # initialization of RRT
    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,           # area for generating random node
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,  # 5% to sample goal as the rnd_node, 95% sample a random point
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 show_animation=False
                 ):
        """
        Parameters for RRT

        start:              Start Position [x,y] or [x,y,z]
        goal:               Goal Position [x,y] or [x, y, z]
        obstacleList:       obstacle Positions [[x,y,size],...] or [x, y, z, size]
        randArea:           Rrandom Sampling Area [min,max]
        play_area:          stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius:       robot body modeled as circle with given radius

        """
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.dimension = len(start)   # automatic detecion of dimension 2d/3d
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.animation = show_animation

        if show_animation:
            if self.dimension == 2:
                self.fig, self.ax = plt.subplots()
            elif self.dimension >= 3:
                self.fig = plt.figure()
                # The auto_add_to_figure=False parameter prevents the axes from being automatically added to the figure.
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.animation = False

    class Node:
        """
        RRT Node:
        1) Its own position
        2) Its path: list of feasible positions from its parent node to this node
        3) Tree parent node

        pt stands for a (positional) variable: [x,y] in 2D Euclidean and [x, y, z] in 3D Euclidiean
        """
        def __init__(self, pt):
            # let's make sure the pt (x,y) or (x,y,z) can not be shared!
            self.pt = np.array(pt, copy=True, dtype=float)
            self.path_pt = []
            self.parent = None

    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    # core of RRT algorithm by 1) generating a random node, 2) fining its nearest node, and 3) generating
    # a new node between them to add to the tree
    def planning(self, animation=True, check_collision_func=None, Arm=None):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        # default is to use internal check_conflict
        if check_collision_func is None:
            check_collision_func = self.check_collision_func
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # call local planner to create a new node
            steer_new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if Arm is None:
                if self.check_if_outside_play_area(steer_new_node, self.play_area) and \
                check_collision_func(steer_new_node, self.obstacle_list, self.robot_radius):
                    self.node_list.append(steer_new_node)   # list of reference
            else:
                if self.check_if_outside_play_area(steer_new_node, self.play_area) and \
                check_collision_func(steer_new_node, self.obstacle_list, Arm):
                    self.node_list.append(steer_new_node)   # list of reference

            # debugging print
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            # self.print_path_from_end(self.node_list[-1])
            # self.print_node_list(self.node_list)
            if animation and i % 3 == 0 and self.dimension >= 2:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].pt) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision_func(final_node, self.obstacle_list, self.robot_radius):
                    # debugging print
                    # self.print_path_from_end(self.node_list[-1])
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    # Local planner: generating a new node between the from_node and to_node, with from_node as its parent
    def steer(self, from_node, to_node, extend_length=float("inf")):

        # create a new node
        new_node = self.Node(from_node.pt)
        if self.dimension == 2:
            d, _ = self.calc_distance_and_angle(new_node, to_node, self.dimension)
        elif self.dimension >= 3:
             d, _, _ = self.calc_distance_and_angle(new_node, to_node, self.dimension)

        # unpack pt into x, y and z as constant, hence no reference to be modified un-intentionally
        new_node.path_pt = [[*new_node.pt]]
        if extend_length > d:
            extend_length = d

        # how many positions are used to represent the path from from_node to this new_node
        n_expand = math.floor(extend_length / self.path_resolution)

        v = to_node.pt - from_node.pt  # node.pt is array
        u = v / (np.sqrt(np.sum(v ** 2)))
        for _ in range(n_expand):
            new_node.pt += u * self.path_resolution
            # let's make sure we copy the pt not just append the reference!
            new_node.path_pt.append([*new_node.pt])

        # to account for the remaining distance
        if self.dimension == 2:
            d, _ = self.calc_distance_and_angle(new_node, to_node, self.dimension)
            if d <= self.path_resolution:
                new_node.path_pt.append([*to_node.pt])
                # let's make sure the pt (x,y) or (x,y,z) can not be shared! new_node.pt = to_node.pt
                new_node.pt = np.array(to_node.pt, copy=True, dtype=float)
        elif self.dimension == 3:
            d, _, _ = self.calc_distance_and_angle(new_node, to_node, self.dimension)
            if d <= self.path_resolution:
                new_node.path_pt.append([*to_node.pt])
                new_node.pt = np.array(to_node.pt, copy=True, dtype=float)

        new_node.parent = from_node

        return new_node

    # back track from goal to its starting point through parent relationship
    def generate_final_course(self, goal_ind):
        path = [[*self.end.pt]]
        node = self.node_list[goal_ind]
        # BUG: somehow node.parent became none!!!!
        while node.parent is not None:
            path.append([*node.pt])
            node = node.parent
            # debugging print of node list: still working
            # user_input = input(f"generate_finale_course: type to contniue finding the path")
            # self.print_node_list(self.node_list)
        path.append([*node.pt])
        # reverse the path from start to end
        path.reverse()
        return path

    def calc_dist_to_goal(self, pt):
        distance = np.linalg.norm(np.array(pt) - np.array(self.end.pt))
        return distance

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(np.random.uniform(self.min_rand, self.max_rand, self.dimension))
        else:  # goal point sampling
            rnd = self.Node(self.end.pt)
        return rnd

    def draw_graph(self, rnd=None):
        self.ax.cla()
        # for stopping simulation with the esc key.
        self.fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        if  self.dimension <= 1:
            return self.ax
        elif self.dimension >= 3:
            self.ax.grid(True)
            if rnd is not None:
                self.ax.plot([rnd.pt[0]], [rnd.pt[1]], [rnd.pt[2]], "^k")
            for (ox, oy, oz, size) in self.obstacle_list:
                self.plot_sphere(self.ax, ox, oy, oz, size=size, color='0.25') # 0.25: dark gray, 0.75: light gray
            for node in self.node_list:
                if node.parent:
                    path = np.array(node.path_pt)  # list of 2d lists to nx2 array
                    self.ax.plot(path[:, 0], path[:, 1], path[:, 2], "-g")
            self.ax.plot(self.start.pt[0], self.start.pt[1],
                            self.start.pt[2], "xr")
            self.ax.plot(self.end.pt[0], self.end.pt[1], self.end.pt[2], "xr")
            self.ax.axis("equal")
        elif self.dimension == 2:
            if rnd is not None:
                self.ax.plot(rnd.pt[0], rnd.pt[1], "^k")
                if self.robot_radius > 0.0:
                    self.plot_circle(self.ax, rnd.pt[0], rnd.pt[1], self.robot_radius, '-r')
            for node in self.node_list:
                if node.parent:
                    path_pt_x, path_pt_y = zip(*node.path_pt)

                    self.ax.plot(path_pt_x, path_pt_y, "-g")
            for (ox, oy, size) in self.obstacle_list:
                self.plot_circle(self.ax, ox, oy, size)
            if self.play_area is not None:
                self.ax.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")
            self.ax.plot(self.start.pt[0], self.start.pt[1], "or")
            self.ax.plot(self.end.pt[0], self.end.pt[1], "xr")
            self.ax.axis("equal")
            self.ax.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
            self.ax.grid(True)

        self.fig.canvas.draw()
        # the following statment is critical not blocking the event while drawing the figure
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        return self.ax

    # API for input/change obstaccle list
    def set_obstacle_list(self, obstacleList):
        self.obstacle_list = obstacleList

    def get_obstacle_list(self):
            return self.obstacle_list

    # staticmethod:
    # 1) Cannot access or modify class state
    # 2) Useful for utility functions related to the class but not self.node_listdependent on instance-specific data
    @staticmethod
    def print_path_from_end(end_node):
        path = [*end_node.pt]
        node = end_node
        while node.parent is not None:
            node = node.parent
            path.append([*node.pt])
        # reverse the path from start to end
        path.reverse()
        print(f"Path from end: {path}")


    @staticmethod
    def print_node_list(node_list):
        nodeId = 0
        for node in node_list:
            if node.parent:
                print(f"Node {nodeId}: {node.pt} / parent {node.parent.pt}")
            else:
                print(f"Node {nodeId}: {node.pt} /parent None: ")
            for pt in node.path_pt:
                if len(node.pt) == 2:
                    print(f"[{pt[0]}, {pt[1]}]")
                else:
                    print(f"[{pt[0]}, {pt[1]}, {pt[2]}]")
            nodeId += 1
        print(f"\n")

    @staticmethod
    def plot_circle(ax, x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        ax.plot(xl, yl, color)


    @staticmethod
    def plot_sphere(ax, x, y, z, size=1, color="k"): # p.iragma: no cover
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xl = x+size*np.cos(u)*np.sin(v)
        yl = y+size*np.sin(u)*np.sin(v)
        zl = z+size*np.cos(v)
        ax.plot_wireframe(xl, yl, zl, color=color)


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [np.sum((np.array(node.pt) - np.array(rnd_node.pt))**2)
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.pt < play_area.xmin or node.pt > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision_func(node, obstacleList, robot_radius):

        if node is None:
            return False
        elif len(node.pt) == 2:
            path_pt_x, path_pt_y = zip(*node.path_pt)
            for (ox, oy, size) in obstacleList:
                dx_list = [ox - x for x in path_pt_x]
                dy_list = [oy - y for y in path_pt_y]
                d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
                if min(d_list) <= (size+robot_radius)**2:
                    return False  # collision
        elif len(node.pt) == 3:
            for (ox, oy, oz, size) in obstacleList:
                path_pt_x, path_pt_y, path_pt_z = zip(*node.path_pt)
                dx_list = [ox - x for x in path_pt_x]
                dy_list = [oy - y for y in path_pt_y]
                dz_list = [oz - z for z in path_pt_z]
                d_list = [dx * dx + dy * dy + dz * dz
                          for (dx, dy, dz) in zip(dx_list,
                                                  dy_list,
                                                  dz_list)]
                if min(d_list) <= (size+robot_radius)**2:
                    return False  # collision
        return True  # safe


    @staticmethod
    def calc_distance_and_angle(from_node, to_node, dimension=None):
        dx = to_node.pt[0] - from_node.pt[0]
        dy = to_node.pt[1] - from_node.pt[1]
        if dimension is not None and dimension >= 3:
            dz = to_node.pt[2] - from_node.pt[2]
            d = np.sqrt(np.sum((np.array(to_node.pt) - np.array(from_node.pt))**2))
            phi = math.atan2(dy, dx)
            theta = math.atan2(math.hypot(dx, dy), dz)
            return d, phi, theta
        else:
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            return d, theta

