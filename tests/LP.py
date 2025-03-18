import numpy
import numpy as np
import cvxpy as cp

from src.LP import solve_lp
from src.Miscellaneous import get_solvers,get_norm_types

# Example usage
deltas = np.array([[2,1,-100],[3,2,-120],[-1,0,0],[0,-1,0]])
def A(deltas:numpy.ndarray):
    return np.array([[deltas[0],deltas[1]]])
def b(deltas:numpy.ndarray):
    return np.array([[deltas[2]]])
c = np.array([-5,-3])
T = 0.0
P = 0.0
norm_type = 2
print(get_solvers())
print(get_norm_types())
solver = cp.SCS

optimal_x, optimal_s, optimal_cost = solve_lp(deltas,A, b, c, T, P, norm_type, solver)
print("Optimal x:\n", optimal_x)
print("Optimal s:\n", optimal_s)
print("Optimal cost:", optimal_cost)