import numpy as np
import pytest

from src.QP import solve_qp


SCENARIOS = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])


def _A_d(deltas):
    return np.array([[deltas[0], deltas[1]]])


def _b_d(deltas):
    return np.array([[deltas[2]]])


G_EMPTY = np.array([[0, 0]])
H_EMPTY = np.array([[0]])
C_VEC = np.array([[-5], [-3]])
Q_IDENTITY = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_qp_basic():
    """QP with identity Q matrix using default solver."""
    x, zeta, cost, N, complexity, constraints, degeneracy = solve_qp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G_EMPTY, h=H_EMPTY,
        c=C_VEC, Q=Q_IDENTITY,
        tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert x.shape == (2, 1)
    assert np.isfinite(cost)
    assert N == 4
    assert complexity >= 0


def test_qp_with_slack():
    """QP with rho > 0 activates slack variable creation."""
    x, zeta, cost, N, complexity, constraints, degeneracy = solve_qp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G_EMPTY, h=H_EMPTY,
        c=C_VEC, Q=Q_IDENTITY,
        tau=0.0, x_ref=np.array([0, 0]), rho=1.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    assert zeta is not None
    assert N == 4


def test_qp_with_regularization():
    """QP with tau > 0 exercises the regularization term."""
    x, zeta, cost, N, complexity, constraints, degeneracy = solve_qp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G_EMPTY, h=H_EMPTY,
        c=C_VEC, Q=Q_IDENTITY,
        tau=1.0, x_ref=np.array([1, 1]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    assert N == 4


def test_qp_with_hard_constraints():
    """QP with non-trivial hard constraints (G x + h <= 0)."""
    G = np.array([[1, 0], [0, 1]])
    h = np.array([[-10], [-10]])

    x, zeta, cost, N, complexity, constraints, degeneracy = solve_qp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G, h=h,
        c=C_VEC, Q=Q_IDENTITY,
        tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    assert x[0, 0] <= 10 + 1e-3
    assert x[1, 0] <= 10 + 1e-3


def test_qp_non_psd_q():
    """QP with non-positive-semidefinite Q raises AssertionError."""
    Q_bad = np.array([[1.0, 3.0], [3.0, 1.0]])  # eigenvalues: 4, -2

    with pytest.raises(AssertionError, match="positive semi-definite"):
        solve_qp(
            deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G_EMPTY, h=H_EMPTY,
            c=C_VEC, Q=Q_bad,
            tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
        )


def test_qp_non_symmetric_q():
    """QP with non-symmetric Q raises AssertionError."""
    Q_bad = np.array([[1.0, 0.5], [0.0, 1.0]])

    with pytest.raises(AssertionError, match="positive semi-definite"):
        solve_qp(
            deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G_EMPTY, h=H_EMPTY,
            c=C_VEC, Q=Q_bad,
            tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
        )
