from src.Miscellaneous import get_solvers, get_norm_types


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
    assert norms == [1, 2, "inf", "fro", "nuc"]
