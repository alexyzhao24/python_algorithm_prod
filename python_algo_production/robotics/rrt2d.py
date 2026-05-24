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
    straints (Section 10.4.2 and Figures 10.14 and 10.16). Each control is
    integrated from x_nearest for a fixed time ∆t using ẋ = f (x, u). Among the
    new states reached without collision, the state that is closest to x_samp is
    chosen as x_new .
    3) Wheeled robot planners. For a wheeled mobile robot, local plans can be
    found using Reeds–Shepp curves, as described in Section 13.3.3.
'''

import math
import random

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class RRT:
    """
    Class for RRT planning
    """
    # initialization of RRT
    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 ):
        """
        Parameters for RRT

        start:              Start Position [x,y]
        goal:               Goal Position [x,y]
        obstacleList:       obstacle Positions [[x,y,size],...]
        randArea:           Rrandom Sampling Area [min,max]
        play_area:          stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius:       robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
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


    ## For a new RRT node added to the tree: 1) it has a parent (the "nearest node"), 2) it has an edge/branch from its parent
    ## Key note here about RRT:
    # If the nearest node has been used before, meaning, it already has child nodes, this is completely normal and expected.
    # In RRT, nodes in the tree can and very often are used multiple times as parents for new branches. There is no restriction
    # that a particular node can only be extended once.
    # In fact, the tree structure naturally grows with many nodes having multiple children, as the tree "rapidly explores" the
    # space in many directions from the same nod
    class Node:
        """
        RRT Node:
        1) Its own position
        2) Its path: list of feasible positions from its parent node to this node
        3) Tree parent node
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    # core of RRT algorithm by 1) generating a random node, 2) fining its nearest node, and 3) generating
    # a new node between them to add to the tree
    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # Use the steer to add the rnd_node (nearest_node as its parent) and its path of new nodes from the existing nearest_node to the rnd_node
            # call local planner: from 1) the existing, nearest_node of rnd node to 2) the rnd_node
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            # debugging print
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            # self.print_node_list(self.node_list)
            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(
                        final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    # Local planner: generating a new node between the from_node and to_node, with from_node as its parent
    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # how many positions are used to represent the path from from_node to this new_node
        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        # reverse the path from start to end
        path.reverse()
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    # Get a random node in the defined area, or the goal node with a small probability
    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:  # random sampling 95% of the time
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling for less than 5% of the time
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.x, self.start.y, "or")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)


    # staticmethod:
    # 1) Cannot access or modify class state
    # 2) Useful for utility functions related to the class but not dependent on instance-specific data
    @staticmethod
    def print_node_list(node_list):
        nodeId = 0
        for node in node_list:
            if node.parent:
                print(f"Node {nodeId}: [{node.x, node.y}] / parent [{node.parent.x}, {node.parent.y}]: ")
            else:
                print(f"Node {nodeId}: [{node.x, node.y}] / parent [None]: ")
            for x, y in zip(node.path_x, node.path_y):
                print(f"[{x}, {y}]")
            nodeId += 1
        print(f"\n")


    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # Define obstacle list: [x, y, radius]
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-3, 15],
        obstacle_list=obstacleList,
        #play_area=[-2, 15, -2, 16],
        robot_radius=0.8
        )

    # running RRT algo to find the goal
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # plot the final path
        print(f"Final path:")
        for (x, y) in path:
            print(f"[{x}, {y}]")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', linewidth=5.0)
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
