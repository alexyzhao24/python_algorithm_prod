import test
import random
import sys
from python_algo_production.robotics.a_star import *

def test_a_star_basic():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


def test_a_star_rrt_obstacles():

    print(__file__ + " start!!")

    ### Set Initial parameters for rrt
    # rrt start and goal position
    rrt_gx=6.0
    rrt_gy=10.0
    rrt_sx=0.0
    rrt_sy=0.0
    # rrt obstacle list: [x, y, radius]
    rrt_obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1), (6, 12, 1)]

    ## grid_size that will be used to convert from rrt to a_star
    conversion_size = 0.25  # resolution for rrt params

    ### COnvert RRT parameters to A* parameters
    # convert positions into grid positions
    sx = int(rrt_sx / conversion_size)  # [grid]
    sy = int(rrt_sy / conversion_size)  # [grid]
    gx = int(rrt_gx / conversion_size)  # [grid]
    gy = int(rrt_gy / conversion_size)  # [grid]

    # Convert RRT obstacle list to A* style obstacles with fixed grid size
    ox = []
    oy = [] # obstacle grids
    for (x, y, r) in rrt_obstacleList:
        # Calculate grid coordinates for the center of the circle
        grid_x = int(x / conversion_size)
        grid_y = int(y / conversion_size)
        # Calculate the radius in terms of grid size
        grid_radius = int(r / conversion_size)
        # Iterate over square bounding box around obstacle
        for ix in range(grid_x - grid_radius, grid_x + grid_radius + 1):
            for iy in range(grid_y - grid_radius, grid_y + grid_radius + 1):
                # Check if the cell center is inside the obstacle radius
                cell_x = ix * conversion_size + conversion_size / 2.0
                cell_y = iy * conversion_size + conversion_size / 2.0
                dist = ((cell_x - x)**2 + (cell_y - y)**2)**0.5
                if dist <= r:
                    ox.append(ix)
                    oy.append(iy)

    # need to create a closed boundary for A* to work
    for i in range(-20, 70):
        ox.append(i)
        oy.append(-20.0)
    for i in range(-20, 70):
        ox.append(i)
        oy.append(70.0)
    for i in range(-20, 70):
        oy.append(i)
        ox.append(-20.0)
    for i in range(-20, 70):
        oy.append(i)
        ox.append(70.0)

    # Set Initial parameters
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    grid_size = 1.0
    robot_radius = 2.0  # [m]
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()
