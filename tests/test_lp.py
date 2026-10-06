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


# ── Interval LP (smallest enclosing interval) used by the regression tests below ──
_INTERVAL = np.array([0.1, 0.4, 0.9])


def _interval_A(delta):
    return np.array([[-1.0, -1.0], [1.0, -1.0]])


def test_lp_empty_deltas_raises_value_error():
    """No scenarios gives a clear ValueError (not an IndexError)."""
    with pytest.raises(ValueError, match="at least one row"):
        solve_lp(np.empty((0, 1)), _interval_A, lambda d: np.array([[d[0]], [-d[0]]]),
                 np.array([]), np.array([]), np.array([[0.0], [1.0]]), solver="CLARABEL")


def test_lp_one_dimensional_b_is_not_broadcast():
    """A 1-D b(δ) of shape (m,) gives the same solution as a (m, 1) column."""
    deltas = _INTERVAL.reshape(-1, 1)
    c = np.array([[0.0], [1.0]])
    x_col, *_ = solve_lp(deltas, _interval_A, lambda d: np.array([[d[0]], [-d[0]]]),
                         np.array([]), np.array([]), c, solver="CLARABEL")
    x_1d, *_ = solve_lp(deltas, _interval_A, lambda d: np.array([d[0], -d[0]]),
                        np.array([]), np.array([]), c, solver="CLARABEL")
    np.testing.assert_allclose(x_1d, x_col, atol=1e-6)
    np.testing.assert_allclose(x_col.ravel(), [0.5, 0.4], atol=1e-6)


def test_lp_without_regularization_is_a_pure_lp():
    """With τ = 0 no norm term is added, so an LP-only solver can solve it."""
    import cvxpy as cp
    if "SCIPY" not in cp.installed_solvers():
        pytest.skip("SCIPY not installed")
    x, zeta, cost, N, k, cons, degenerate = solve_lp(
        _INTERVAL.reshape(-1, 1), _interval_A, lambda d: np.array([[d[0]], [-d[0]]]),
        np.array([]), np.array([]), np.array([[0.0], [1.0]]), solver="SCIPY")
    np.testing.assert_allclose(x.ravel(), [0.5, 0.4], atol=1e-6)
    assert k == 2 and not degenerate


@pytest.mark.parametrize("p", [1, 1.5, 2, 3, "inf", "fro"])
def test_lp_regularization_accepts_any_vector_norm(p):
    """The regularization term is a vector norm, so any p >= 1, inf and fro work."""
    x, *_ = solve_lp(_INTERVAL.reshape(-1, 1), _interval_A, lambda d: np.array([[d[0]], [-d[0]]]),
                     np.array([]), np.array([]), np.array([[0.0], [1.0]]),
                     tau=0.01, x_ref=np.array([0.5, 0.4]), norm_type=float("inf") if p == "inf" else p,
                     solver="CLARABEL")
    np.testing.assert_allclose(x.ravel(), [0.5, 0.4], atol=1e-5)
