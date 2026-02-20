"""
Test for the SDP solver (solve_sdp) using a simple 2x2 LMI problem.

Problem: Find x in R^2 maximizing x_1 + x_2 subject to
  F_0(delta) + x_1*F_1 + x_2*F_2 << 0
where F_0(delta) = delta*I, F_1 = [[1,0],[0,0]], F_2 = [[0,0],[0,1]]

This gives diagonal LMI: [[delta+x_1, 0],[0, delta+x_2]] << 0,
so x_1 <= -delta and x_2 <= -delta for all scenarios.
Maximizing (c=[-1,-1]) pushes x towards 0; optimal is x = -delta_max.
"""
import numpy as np

from src.SDP import solve_sdp

F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
F2 = np.array([[0.0, 0.0], [0.0, 1.0]])


def test_sdp_basic():
    np.random.seed(42)
    N = 50
    deltas = np.random.uniform(0.5, 2.0, size=(N, 1))

    def F_d(delta):
        return {
            "0": delta[0] * np.eye(2),
            "1": F1,
            "2": F2,
        }

    E = {}
    c = np.array([-1.0, -1.0])
    Q = np.zeros((2, 2))

    x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
        deltas=deltas, F_d=F_d, E=E, c=c, Q=Q,
        tau=0.0, x_ref=np.zeros(2), rho=0.0, norm_type=2, solver=None,
    )

    delta_max = np.max(deltas)
    expected_x = -delta_max
    expected_cost = 2 * delta_max

    assert abs(x[0] - expected_x) < 1e-4, f"x[0] should be ~{expected_x}, got {x[0]}"
    assert abs(x[1] - expected_x) < 1e-4, f"x[1] should be ~{expected_x}, got {x[1]}"
    assert abs(cost - expected_cost) < 1e-4, f"cost should be ~{expected_cost}, got {cost}"
    assert N_out == N, f"N should be {N}, got {N_out}"
