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


def _lpv_problem():
    """LPV stability example (Example SDP1): x = (p11, p12, p22)."""
    def F_d(delta):
        d = delta[0]
        return {'0': np.zeros((2, 2)),
                '1': np.array([[0.0, 1.0], [1.0, 0.0]]),
                '2': np.array([[-4 + 0.6 * d, -1 + 0.1 * d], [-1 + 0.1 * d, 2.0]]),
                '3': np.array([[0.0, 0.3 * d - 2], [0.3 * d - 2, -2 + 0.2 * d]])}
    E = {'0': np.zeros((2, 2)), '1': -np.array([[1.0, 0], [0, 0]]),
         '2': -np.array([[0, 1.0], [1.0, 0]]), '3': -np.array([[0, 0], [0, 1.0]])}
    deltas = np.linspace(-0.2, 1.0, 7).reshape(-1, 1)
    c = np.array([[-1.0], [0.0], [-1.0]])
    return deltas, F_d, E, c, 0.1 * np.eye(3)


def test_sdp_integer_keys_match_string_keys():
    """Matrix dicts keyed 0, 1, ... give the same solution as keys '0', '1', ..."""
    deltas, F_d, E, c, Q = _lpv_problem()
    x_str, *_ = solve_sdp(deltas, F_d, E, c, Q, rho=1.0, solver="CLARABEL")
    F_int = lambda delta: {int(k): v for k, v in F_d(delta).items()}
    E_int = {int(k): v for k, v in E.items()}
    x_int, *_ = solve_sdp(deltas, F_int, E_int, c, Q, rho=1.0, solver="CLARABEL")
    np.testing.assert_allclose(x_int, x_str, atol=1e-5)


def test_sdp_empty_q_means_no_quadratic_term():
    """An empty Q is the same as a zero matrix."""
    deltas, F_d, E, c, _ = _lpv_problem()
    E_bounded = dict(E, **{'0': 0.0 * np.eye(2)})
    # Without Q, the regularization (τ = 2 > √2) is what keeps max trace(P) bounded.
    x_empty, *_ = solve_sdp(deltas, F_d, E_bounded, c, np.array([]), tau=2.0, rho=1.0, solver="CLARABEL")
    x_zero, *_ = solve_sdp(deltas, F_d, E_bounded, c, np.zeros((3, 3)), tau=2.0, rho=1.0, solver="CLARABEL")
    np.testing.assert_allclose(x_empty, x_zero, atol=1e-5)


def test_sdp_column_x_ref_matches_vector_x_ref():
    """x_ref given as a (d, 1) column gives the same solution as a 1-D vector."""
    deltas, F_d, E, c, Q = _lpv_problem()
    x_vec, *_ = solve_sdp(deltas, F_d, E, c, Q, tau=0.5, x_ref=np.array([1.0, 0.0, 1.0]), rho=1.0, solver="CLARABEL")
    x_col, *_ = solve_sdp(deltas, F_d, E, c, Q, tau=0.5, x_ref=np.array([[1.0], [0.0], [1.0]]), rho=1.0, solver="CLARABEL")
    np.testing.assert_allclose(x_col, x_vec, atol=1e-5)
