import numpy as np
import pytest
from pathlib import Path

from src.LP import solve_lp


def test_lp_basic():
    """LP with inline scenario data using default solver."""
    scenarios = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])

    def A_d(deltas):
        return np.array([[deltas[0], deltas[1]]])

    def b_d(deltas):
        return np.array([[deltas[2]]])

    G = np.array([[0, 0]])
    h = np.array([[0]])
    c = np.array([[-5], [-3]])

    x, zeta, cost, N, complexity, constraints, degeneracy = solve_lp(
        deltas=scenarios, A_d=A_d, b_d=b_d, G=G, h=h, c=c,
        tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert x.shape == (2, 1)
    assert np.isfinite(cost)
    assert N == 4
    assert complexity >= 0


def test_lp_smallest_interval():
    """LP with CSV-loaded scenario data (1D smallest interval benchmark)."""
    csv_path = Path(__file__).parent / "upload_csvs_1d_smallest_interval" / "smallest_interval_1d.csv"
    scenarios = np.loadtxt(csv_path, delimiter=",")

    def A_d(deltas):
        return np.array([[-1, -1], [1, -1]])

    def b_d(deltas):
        return np.array([[deltas], [-deltas]])

    G = np.array([[0, 0]])
    h = np.array([[0]])
    c = np.array([[0], [1]])

    x, zeta, cost, N, complexity, constraints, degeneracy = solve_lp(
        deltas=scenarios, A_d=A_d, b_d=b_d, G=G, h=h, c=c,
        tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert x.shape == (2, 1)
    assert np.isfinite(cost)
    assert N == len(scenarios)
    assert complexity >= 0


# ── Slack variables ──────────────────────────────────────────────────────

SCENARIOS = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])


def _A_d(deltas):
    return np.array([[deltas[0], deltas[1]]])


def _b_d(deltas):
    return np.array([[deltas[2]]])


def test_lp_with_slack():
    """LP with rho > 0 activates slack variable creation."""
    x, zeta, cost, N, complexity, constraints, degeneracy = solve_lp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d,
        G=np.array([[0, 0]]), h=np.array([[0]]),
        c=np.array([[-5], [-3]]),
        tau=0.0, x_ref=np.array([0, 0]), rho=1.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    assert zeta is not None
    assert N == 4


def test_lp_with_regularization():
    """LP with tau > 0 exercises the regularization term."""
    x, zeta, cost, N, complexity, constraints, degeneracy = solve_lp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d,
        G=np.array([[0, 0]]), h=np.array([[0]]),
        c=np.array([[-5], [-3]]),
        tau=1.0, x_ref=np.array([1, 1]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    assert N == 4


def test_lp_with_hard_constraints():
    """LP with non-trivial hard constraints (G x + h <= 0)."""
    # Hard constraint: x1 <= 10, x2 <= 10
    G = np.array([[1, 0], [0, 1]])
    h = np.array([[-10], [-10]])

    x, zeta, cost, N, complexity, constraints, degeneracy = solve_lp(
        deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G, h=h,
        c=np.array([[-5], [-3]]),
        tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    # Hard constraints must be satisfied: x <= 10
    assert x[0, 0] <= 10 + 1e-3
    assert x[1, 0] <= 10 + 1e-3


def test_lp_infeasible():
    """LP with contradictory constraints raises ValueError."""
    # Hard constraint: x1 >= 1000 AND scenario constraints push x small
    G = np.array([[-1, 0]])
    h = np.array([[1000]])

    with pytest.raises(ValueError, match="did not solve to optimality"):
        solve_lp(
            deltas=SCENARIOS, A_d=_A_d, b_d=_b_d, G=G, h=h,
            c=np.array([[-5], [-3]]),
            tau=0.0, x_ref=np.array([0, 0]), rho=0.0, norm_type=2, solver="SCS",
        )
