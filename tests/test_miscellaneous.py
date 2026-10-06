from src.Miscellaneous import get_solvers, get_norm_types, SOLVER_CAPABILITIES, _cvxpy_capabilities


def test_get_solvers():
    """get_solvers returns a dict with at least SCS and CLARABEL."""
    solvers = get_solvers()
    assert isinstance(solvers, dict)
    assert "SCS" in solvers
    assert "CLARABEL" in solvers


def test_get_solvers_capabilities():
    """Installed solver capabilities include expected problem types."""
    solvers = get_solvers()
    assert "LP" in solvers["SCS"]
    assert "QP" in solvers["SCS"]
    assert "SDP" in solvers["SCS"]


def test_get_norm_types():
    """get_norm_types returns the expected list of norms."""
    norms = get_norm_types()
    assert norms == [1, 2, "inf", "fro"]


def test_solver_capabilities_match_cvxpy():
    """Table entries never claim a program type CVXPY says the solver cannot handle."""
    for name, types in SOLVER_CAPABILITIES.items():
        assert set(types) <= set(_cvxpy_capabilities(name)), name


def test_unlisted_solvers_use_cvxpy_metadata():
    """A solver missing from the table gets its types from CVXPY, not all three."""
    assert _cvxpy_capabilities("CUOPT") == ["LP"]
    assert _cvxpy_capabilities("COPT") == ["LP", "QP", "SDP"]
    assert _cvxpy_capabilities("NOT_A_SOLVER") == []


def test_check_psd_rounds_tiny_negative_eigenvalues():
    """A Q accepted within the tolerance is returned exactly positive semidefinite."""
    import numpy as np
    from src.Miscellaneous import check_psd
    Q = check_psd(np.diag([1.0, -1e-12]))
    assert np.linalg.eigvalsh(Q).min() >= 0
    np.testing.assert_allclose(Q, np.diag([1.0, 0.0]), atol=1e-12)
