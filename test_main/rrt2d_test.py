import numpy as np
import matplotlib.pyplot as plt
from python_algo_production.robotics.rrt import RRT

def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # Define obstacle list: [x, y, radius]
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1), (6, 12, 1)]
    # Set Initial parameters
    show_animation = True
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-3, 15],
        obstacle_list=obstacleList,
        #play_area=[-2, 15, -2, 16],
        robot_radius=0.8,
        show_animation=show_animation
        )

    # running RRT* algo to find the goal
    path = rrt.planning(show_animation)

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
            rrt.ax.plot([x for (x, y) in path], [y for (x, y) in path], '-r', linewidth=5.0)
            rrt.ax.grid(True)
            plt.pause(0.01)  # Need for Mac


if __name__ == '__main__':
    main()
    plt.show()