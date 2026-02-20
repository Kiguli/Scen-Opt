import numpy as np
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
