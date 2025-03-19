import numpy as np
import cvxpy as cp

from src.QP import solve_qp
from src.Miscellaneous import get_solvers,get_norm_types

# Example usage
scenarios = np.array([[2,1,-100],[3,2,-120],[-1,0,0],[0,-1,0]])
def A(deltas:np.ndarray):
    return np.array([[deltas[0],deltas[1]]])
def b(deltas:np.ndarray):
    return np.array([[deltas[2]]])
c = np.array([-5,-3])
T = 0.0
P = 0.0
norm_type = 2
print(get_solvers())
print(get_norm_types())
solver = cp.SCS
Q = np.array([[1,0],[0,1]])

optimal_x, optimal_s, optimal_cost = solve_qp(scenarios,A, b, c, Q, T, P, norm_type, solver)
print("Optimal x:\n", optimal_x)
print("Optimal s:\n", optimal_s)
print("Optimal cost:", optimal_cost)