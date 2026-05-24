
import numpy as np
import argparse  ## https://docs.python.org/3.5/library/argparse.html
import json
from datetime import datetime
import matplotlib.pyplot as plt
# import python_algo_production modules
from python_algo_production.utils.math_np_utils import *
from python_algo_production.robotics.pendulum_utils import plot_cart
from python_algo_production.robotics.mpc_control import mpc_control
from python_algo_production.robotics.lqr_control import lqr_control

# physical parameters
l_bar = 2.0  # length of bar
M = 1.0  # cart weight [kg]
m = 0.3  # thin bar weight [kg]
g = 9.8  # gravity [m/s^2]

# state parameters
nx = 4  # number of state:
nu = 1  # number of input
Q = np.diag([0.0, 1.0, 1.0, 0.0])  # state cost matrix --> r^T*Q*r with r = [dx/dt->0, theta --> 0]
R = np.diag([0.01])                # input cost matrix --> u^T*R*u


# Simulation of discrete steps
T = 30  # Horizon length for MPC
delta_t = 0.1  # time tick [s]
sim_time = 5.0  # simulation time [s]

# Here is the model matrix derived from the simplified physics
'''
Eq for stationary pivot point
d^2(theta)/d^2(t) = (g/l)*sin(theta)

Here are the lineaized 4-state (x, dx/dt, theta, d_theta/dt) and the differential eqs:
d(x)/dt           = dx/dt
d(dx/dt)/dt       = (m*g/M)*theta + M*(1/u)
d(theta)/dt       = d_theta/dt
d(d_theta/dt)/dt  = g*((M+m)/(M*l_bar))*theta + (1/(M*l_bar)**u

Here is the state model
dx/dt = A*x + B*u
'''
def get_model_matrix():
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A  ## discretized state model: dx/dt = [x(t+dealta_t) - x(t)] / dt

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B
'''
Here is the output model of y = [x, theta]
y =  C*x + D*u
'''
def get_output_matrix():
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    D = np.array([
        [0.0],
        [0.0]
    ])
    return C, D


def quadratic_cost(Q, x, R, u):
    cost = np.transpose(x) @ Q @ x + np.transpose(u) @ R @ u
    return cost[0,0]

def simulation_state(x, u, A, B):
    x = A @ x + B @ u

    return x


## Main function to run a chosen control algo
'''
For MPC/LQR optimal control, we have the following
Quadratic cost function: x^T*Q*x [0 to t] + u^T*R*u [0 to t-1]

For PID control,
https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlPID

The structure of the controller for this problem is a little different than the standard control problems you
may be used to. Since we are attempting to control the pendulum's position, which should return to the vertical
after the initial disturbance, the reference signal we are tracking should be zero: dx/dt --> 0; theta --> 0
This type of situation is often referred to as a Regulator problem.


https://math.stackexchange.com/questions/3201288/for-what-objective-function-is-pid-control-the-optimal-control
A PD/PID controller can be reformulated as a state-feedback controller.

'''

def main_process(choice, animationFlag=True):

    # given initial state x0
    x0 = np.array([
        [0.0],
        [0.0],
        [1.0], ## starting position with theta angle in radion
        [0.0]
    ])

    # algo initialization
    x = np.copy(x0)
    time = 0.0
    u = np.array([[0.0]])
    ref = np.zeros([2,1])
    # critical to have this starting point correct!
    last_err = np.zeros([2,1])
    err_IntDer =  np.zeros([2,1])
    y = vec_sqrt_sign(x)

    while sim_time > time:
        time += delta_t

        # Potential to handle time-varying A(t), B(t)
        A, B = get_model_matrix()

        # calc control input based on algo choice from user
        if choice == "mpc": # MPC control
            opt_input, opt_x, opt_delta_x, opt_theta, opt_delta_theta = mpc_control(x, A, B, R, Q, nx, nu, T)
            u = np.array([[opt_input[0]]])  # u: make it 1x1 array so B @ u makes sense
            print(f"MPC control u = {u[0]} @ {time}: cost: {quadratic_cost(Q, x, R, u)}")
        elif choice == "lqr":  # lqr control
            u = lqr_control(x, A, B, R, Q) # u: [[number]]
            print(f"LQR control u = {u[0]} @ {time}: cost: {quadratic_cost(Q, x, R, u)}")
        else:  # no control/force applied
            print(f"No control u = {u[0]} @ {time}: cost: {quadratic_cost(Q, x, R, u)}")

        # simulate next state and measurement
        x = simulation_state(x, u, A, B)

        if animationFlag:
            plt.clf()
            px = float(x[0, 0])
            theta = float(x[2, 0])
            plot_cart(px, theta, l_bar)
            plt.xlim([-8.0, 2.0])
            plt.pause(0.05)  # unit is second


# Can runt it as a script, not as a function
if __name__ == '__main__':

    # Create an argument parser object
    parser = argparse.ArgumentParser()

    # Prompt for user inputs
    parser.add_argument('--choice', default="mpc", choices=["mpc","lqr", "none"], help="control algo choice")

    # Parse the command-line arguments
    args = parser.parse_args()
    ## print acquired input arguments
    print(f"{args}")

    if (args.choice == 'lqr' or args.choice == 'mpc' or args.choice == 'none'):
        main_process(args.choice)
    else:
        print(f"Please choose a valid control algorithm: {args.choice}")


