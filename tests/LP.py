import numpy as np
import cvxpy as cp

from src.LP import solve_lp
from src.Miscellaneous import get_solvers

# Example usage
A = np.array([[2,1],[3,2],[-1,0],[0,-1]])
b = np.array([[-100],[-120],[0],[0]])
c = np.array([-5,-3])
T = 0.0
P = 0.0
norm_type = 2
print(get_solvers())
solver = cp.SCS

optimal_x, optimal_s, optimal_cost = solve_lp(A, b, c, T, P, norm_type, solver)
print("Optimal x:\n", optimal_x)
print("Optimal s:\n", optimal_s)
print("Optimal cost:", optimal_cost)