import numpy as np

from src.QP import solve_qp


def test_qp_basic():
    """QP with identity Q matrix using default solver."""
    scenarios = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])

    def A_d(deltas):
        return np.array([[deltas[0], deltas[1]]])

    def b_d(deltas):
        return np.array([[deltas[2]]])

    G = np.array([[0, 0]])
    h = np.array([[0]])
    c = np.array([[-5], [-3]])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])

    x, zeta, cost, N, complexity, constraints, degeneracy = solve_qp(
        deltas=scenarios, A_d=A_d, b_d=b_d, G=G, h=h, c=c, Q=Q,
        tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert x.shape == (2, 1)
    assert np.isfinite(cost)
    assert N == 4
    assert complexity >= 0
