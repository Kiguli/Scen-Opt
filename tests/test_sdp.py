"""
Tests for the SDP solver (solve_sdp).

Basic test uses a simple 2x2 LMI problem:
  F_0(delta) + x_1*F_1 + x_2*F_2 << 0
where F_0(delta) = delta*I, F_1 = [[1,0],[0,0]], F_2 = [[0,0],[0,1]]
"""
import numpy as np
import pytest

from src.SDP import solve_sdp

F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
F2 = np.array([[0.0, 0.0], [0.0, 1.0]])


def _make_F_d(delta):
    return {
        "0": delta[0] * np.eye(2),
        "1": F1,
        "2": F2,
    }


def test_sdp_basic():
    np.random.seed(42)
    N = 50
    deltas = np.random.uniform(0.5, 2.0, size=(N, 1))

    x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
        deltas=deltas, F_d=_make_F_d, E={}, c=np.array([-1.0, -1.0]),
        Q=np.zeros((2, 2)),
        tau=0.0, x_ref=np.zeros(2), rho=0.0, norm_type=2, solver="SCS",
    )

    delta_max = np.max(deltas)
    expected_x = -delta_max
    expected_cost = 2 * delta_max

    assert abs(x[0] - expected_x) < 1e-4, f"x[0] should be ~{expected_x}, got {x[0]}"
    assert abs(x[1] - expected_x) < 1e-4, f"x[1] should be ~{expected_x}, got {x[1]}"
    assert abs(cost - expected_cost) < 1e-4, f"cost should be ~{expected_cost}, got {cost}"
    assert N_out == N, f"N should be {N}, got {N_out}"


def test_sdp_with_hard_constraints():
    """SDP with non-empty hard constraint E dict."""
    np.random.seed(42)
    N = 50
    deltas = np.random.uniform(0.5, 2.0, size=(N, 1))

    # Hard constraint: x_1*F1 + x_2*F2 << 5*I  (generous, should not bind)
    E = {
        "0": -5.0 * np.eye(2),
        "1": F1,
        "2": F2,
    }

    x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
        deltas=deltas, F_d=_make_F_d, E=E, c=np.array([-1.0, -1.0]),
        Q=np.zeros((2, 2)),
        tau=0.0, x_ref=np.zeros(2), rho=0.0, norm_type=2, solver="SCS",
    )

    delta_max = np.max(deltas)
    expected_x = -delta_max

    assert x is not None
    assert np.isfinite(cost)
    # Solution should still be driven by scenarios, not the hard constraint
    assert abs(x[0] - expected_x) < 1e-3
    assert abs(x[1] - expected_x) < 1e-3


def test_sdp_with_slack():
    """SDP with rho > 0 activates slack variable creation."""
    np.random.seed(42)
    N = 50
    deltas = np.random.uniform(0.5, 2.0, size=(N, 1))

    x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
        deltas=deltas, F_d=_make_F_d, E={}, c=np.array([-1.0, -1.0]),
        Q=np.zeros((2, 2)),
        tau=0.0, x_ref=np.zeros(2), rho=1.0, norm_type=2, solver="SCS",
    )

    assert x is not None
    assert np.isfinite(cost)
    assert zeta is not None
    assert N_out == N


def test_sdp_non_psd_q():
    """SDP with non-positive-semidefinite Q raises AssertionError."""
    np.random.seed(42)
    deltas = np.random.uniform(0.5, 2.0, size=(10, 1))
    Q_bad = np.array([[1.0, 3.0], [3.0, 1.0]])

    with pytest.raises(AssertionError, match="positive semi-definite"):
        solve_sdp(
            deltas=deltas, F_d=_make_F_d, E={}, c=np.array([-1.0, -1.0]),
            Q=Q_bad,
            tau=0.0, x_ref=np.zeros(2), rho=0.0, norm_type=2, solver="SCS",
        )
