# Copied and modified from Python/Robotics git:
# https://atsushisakai.github.io/PythonRobotics/

''' From Modern Robotics Book 10.5.4
    The basic RRT algorithm returns SUCCESS once a motion to X goal is
    found. An alternative is to continue running the algorithm and to terminate the
    search only when another termination condition is reached (e.g., a maximum
    running time or a maximum tree size). Then the motion with the minimum
    cost can be returned. In this way, the RRT solution may continue to improve as
    time goes by. Because edges in the tree are never deleted or changed, however,
    the RRT generally does not converge to an optimal solution.

    The RRT ∗ algorithm is a variation on the single-tree RRT that continually
    rewires the search tree to ensure that it always encodes the shortest path from
    x start to each node in the tree. The basic approach works for C-space path
    planning with no motion constraints, allowing exact paths from any node to
    any other node.

    To modify the RRT to the RRT ∗ , line "add x new to T with an edge from x nearest to
    x new" of the RRT algorithm, which inserts x new in T with an edge from x nearest to x new ,
    is replaced by a test of all the nodes x ∈ X near in T that are sufficiently near to
    x new . An edge to x new is created from the x ∈ X near by the local planner that (1)
    has a collision-free motion and (2) minimizes the total cost of the path from x start
    to x new , not just the cost of the added edge. The total cost is the cost to reach the
    candidate x ∈ X near plus the cost of the new edge.

    The next step is to consider each x ∈ X near to see whether it could be reached
    at lower cost by a motion through x new . If so, the parent of x is changed to x new .
    In this way, the tree is incrementally rewired to eliminate high-cost motions in
    favor of the minimum-cost motions available so far.

    Unlike the RRT, the solution provided by RRT ∗ approaches the optimal solution as the
    number of sample nodes increases. Like the RRT, the RRT ∗ algorithm is probabilistically
    complete. That is the probability of finding a solution using these RRT/RRT*, if
    one exists, tends to 1 as the planning time goes to infinity.
'''


import math
import sys
import matplotlib.pyplot as plt
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from rrt import RRT

show_animation = True

# RRTStar class with RRT as base Class
class RRTStar(RRT):
    """
    Class for RRT Star planning
    """
    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=1000,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 robot_radius=0.0,
                show_animation=False):

        """

        Additional Params for RRT*

        connnect_circle_dist:

        Parameters for RRT

        start:              Start Position [x,y]
        goal:               Goal Position [x,y]
        obstacleList:       obstacle Positions [[x,y,size],...]
        randArea:           Rrandom Sampling Area [min,max]
        robot_radius:       robot body modeled as circle with given radius

        """

        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter,
                         robot_radius=robot_radius, show_animation=show_animation)
        self.goal_node = self.Node(goal)
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter

    # RRT* node has additional property: cost of
    class Node(RRT.Node):
        def __init__(self, pt):
            super().__init__(pt)
            self.cost = 0.0

    def planning(self, animation=True, check_collision_func=None):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """

        # default is to use internal check_conflict
        if check_collision_func is None:
            check_collision_func = self.check_collision_func
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)

            # call local planner from RRT
            new_node = self.steer(self.node_list[nearest_ind], rnd_node,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]

            # additional step beyond RRT to calculate the cost
            new_node.cost = near_node.cost + \
                    sum((new_node.pt - near_node.pt)**2)

            if check_collision_func(
                    new_node, self.obstacle_list, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds, check_collision_func)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds, check_collision_func)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            # debugging print
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            # self.print_node_list(self.node_list)

            if animation:
                self.draw_graph(rnd_node)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                last_index = self.search_best_goal_node(check_collision_func)
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node(check_collision_func)
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds, check_collision_func):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and check_collision_func(
                    t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self, check_collision_func):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.pt) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if check_collision_func(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [self.node_list[i].cost +
                           self.calc_dist_to_goal(self.node_list[i].pt)
                           for i in safe_goal_inds]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [sum((node.pt - new_node.pt)**2)
                        for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds, check_collision_func):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = check_collision_func(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


