"""
Test for the SDP solver (solve_sdp) using a simple 2x2 LMI problem.

Problem: Find x in R^2 maximizing x_1 + x_2 subject to
  F_0(delta) + x_1*F_1 + x_2*F_2 << 0
where F_0(delta) = delta*I, F_1 = [[1,0],[0,0]], F_2 = [[0,0],[0,1]]

This gives diagonal LMI: [[delta+x_1, 0],[0, delta+x_2]] << 0,
so x_1 <= -delta and x_2 <= -delta for all scenarios.
Maximizing (c=[-1,-1]) pushes x towards 0; optimal is x = -delta_max.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.SDP import solve_sdp

# Scenario data: each delta is a scalar
np.random.seed(42)
N = 50
deltas = np.random.uniform(0.5, 2.0, size=(N, 1))

# F_d: F_0(delta) = delta[0]*I, F_1 = [[1,0],[0,0]], F_2 = [[0,0],[0,1]]
F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
F2 = np.array([[0.0, 0.0], [0.0, 1.0]])

def F_d(delta):
    return {
        '0': delta[0] * np.eye(2),
        '1': F1,
        '2': F2,
    }

# Hard constraint: none (empty dict)
E = {}

# Objective: maximize x_1 + x_2  =  minimize -x_1 - x_2
c = np.array([-1.0, -1.0])
Q = np.zeros((2, 2))

x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
    deltas=deltas, F_d=F_d, E=E, c=c, Q=Q, tau=0.0,
    x_ref=np.zeros(2), rho=0.0, norm_type=2, solver=None
)

delta_max = np.max(deltas)
expected_x = -delta_max
expected_cost = 2 * delta_max  # c'x = (-1)(-delta_max)*2 = 2*delta_max

print(f"Number of scenarios: {N_out}")
print(f"Optimal x: {x}")
print(f"Optimal cost: {cost:.6f}")
print(f"Max delta: {delta_max:.6f}")
print(f"Expected optimal: x = [{expected_x:.6f}, {expected_x:.6f}], cost = {expected_cost:.6f}")
print(f"Complexity k: {k}")
print(f"Degeneracy: {degeneracy}")

# Verify solution
assert abs(x[0] - expected_x) < 1e-4, f"x[0] should be ~{expected_x}, got {x[0]}"
assert abs(x[1] - expected_x) < 1e-4, f"x[1] should be ~{expected_x}, got {x[1]}"
assert abs(cost - expected_cost) < 1e-4, f"cost should be ~{expected_cost}, got {cost}"
assert N_out == N, f"N should be {N}, got {N_out}"

print("\nAll assertions passed!")
