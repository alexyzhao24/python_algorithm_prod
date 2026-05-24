import time
import numpy as np
from python_algo_production.utils.math_np_utils import get_numpy_array_from_matrix
from numpy.linalg import inv, eig

## least quadratic Regulator Optimal Control
'''
Optimal control theory is a branch of control theory that deals with finding a control for
a dynamical system over a period of time such that an objective function is optimized.
Optimal control is an extension of the calculus of variations, and is a mathematical
optimization method for deriving control policies.

A special case of the general nonlinear optimal control problem is the linear quadratic (LQ)
optimal control problem. A particular form of the LQ problem that arises in many control system
problems is that of the linear quadratic regulator (LQR) where all of the matrices (A, B, Q and R)
are constant,the initial time is arbitrarily set to zero, and the terminal time is taken in the limit
tf→∞ (this last assumption is what is known as infinite horizon).
'''

def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    '''
    Solve a Discrete-time_Algebraic Riccati Equation (DARE)
    '''
    P = Q

    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ \
            inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (abs(Pn - P)).max() < eps:
            break
        P = Pn

    return Pn


def dlqr(A, B, Q, R):
    '''
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
   '''
    # first, try to solve the ricatti equation
    P = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    eigVals, eigVecs = eig(A - B @ K)
    return K, P, eigVals


def lqr_control(x, A, B, R, Q):

    start = time.time()
    K, _, _ = dlqr(A, B, Q, R)
    u = -K @ x
    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")
    return u