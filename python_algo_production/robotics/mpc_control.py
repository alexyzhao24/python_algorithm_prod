import cvxpy
import time
import numpy as np
from python_algo_production.utils.math_np_utils import get_numpy_array_from_matrix

## Model Predictive COntrol
'''
Model predictive controllers rely on dynamic models of the process, most often
linear empirical models obtained by system identification. The main advantage of MPC
is the fact that it allows the current timeslot to be optimized, while keeping future
timeslots in account.

This is achieved by optimizing a finite time-horizon, but only implementing the current
timeslot and then optimizing again, repeatedly.
'''
def mpc_control(x0, A, B, R, Q, nx, nu, T):

    x = cvxpy.Variable((nx, T + 1))
    u = cvxpy.Variable((nu, T))

    cost = 0.0
    constr = []
    for t in range(T):
        ## Quadratic cost function: x^T*Q*x [0 to t] + u^T*R*u [0 to t-1]
        cost += cvxpy.quad_form(x[:, t + 1], Q)
        cost += cvxpy.quad_form(u[:, t], R)
        ## Constrain to the dynamic state model
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

    ## Constrain to the initial condition x0
    constr += [x[:, 0] == x0[:, 0]]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

    start = time.time()
    prob.solve(verbose=False)
    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if prob.status == cvxpy.OPTIMAL:
        ox = get_numpy_array_from_matrix(x.value[0, :])
        dx = get_numpy_array_from_matrix(x.value[1, :])
        theta = get_numpy_array_from_matrix(x.value[2, :])
        d_theta = get_numpy_array_from_matrix(x.value[3, :])

        ou = get_numpy_array_from_matrix(u.value[0, :])
    else:
        ox, dx, theta, d_theta, ou = None, None, None, None, None

    return ou, ox, dx, theta, d_theta