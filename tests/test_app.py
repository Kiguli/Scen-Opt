import io
import json
import numpy as np
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_route(client):
    """GET / returns 200 and contains solver info."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"SCS" in resp.data


def test_solve_lp_route(client):
    """POST /solve with LP form data returns optimal_x."""
    scenarios = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])
    csv_bytes = "\n".join(",".join(str(v) for v in row) for row in scenarios).encode()

    resp = client.post("/solve", data={
        "active_tab": "lp-tab",
        "A_d": '[["delta[0]", "delta[1]"]]',
        "b_d": '[["delta[2]"]]',
        "G": "",
        "h": "",
        "c": '[["-5"], ["-3"]]',
        "rho": "0",
        "tau": "0",
        "theta_bar": "0,0",
        "p": "2",
        "solver": "SCS",
        "confidence": "0.001",
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "optimal_x" in data
    assert data["optimal_x"][0] is not None


@pytest.mark.parametrize("theta_bar", ["", "null", '""', "[]"])
def test_solve_lp_route_empty_theta_bar(client, theta_bar):
    """An empty x_ref (as sent for a program JSON with "x_ref": null, "" or []) is treated as unset."""
    deltas = np.array([0.1, 0.4, 0.9])
    csv_bytes = "\n".join(str(v) for v in deltas).encode()

    resp = client.post("/solve", data={
        "active_tab": "lp-tab",
        "A_d": '[["-1", "-1"], ["1", "-1"]]',
        "b_d": '[["delta[0]"], ["-delta[0]"]]',
        "c": '[["0"], ["1"]]',
        "rho": "0",
        "tau": "0",
        "theta_bar": theta_bar,
        "p": "",
        "solver": "CLARABEL",
        "confidence": "1e-06",
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["errorcode"] == ["None"]
    np.testing.assert_allclose(np.ravel(data["optimal_x"][0]), [0.5, 0.4], atol=1e-6)


def test_solve_qp_route(client):
    """POST /solve with QP form data returns optimal_x."""
    scenarios = np.array([[2, 1, -100], [3, 2, -120], [-1, 0, 0], [0, -1, 0]])
    csv_bytes = "\n".join(",".join(str(v) for v in row) for row in scenarios).encode()

    resp = client.post("/solve", data={
        "active_tab": "qp-tab",
        "A_d": '[["delta[0]", "delta[1]"]]',
        "b_d": '[["delta[2]"]]',
        "G": "",
        "h": "",
        "c": '[["-5"], ["-3"]]',
        "Q": '[["1", "0"], ["0", "1"]]',
        "rho": "0",
        "tau": "0",
        "theta_bar": "0,0",
        "p": "2",
        "solver": "SCS",
        "confidence": "0.001",
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "optimal_x" in data
    assert data["optimal_x"][0] is not None


def test_solve_sdp_route(client):
    """POST /solve with SDP form data returns optimal_x."""
    np.random.seed(42)
    deltas = np.random.uniform(0.5, 2.0, size=(20, 1))
    csv_bytes = "\n".join(str(row[0]) for row in deltas).encode()

    resp = client.post("/solve", data={
        "active_tab": "sdp-tab",
        "F_d": '{"0": [["delta[0]", "0"], ["0", "delta[0]"]], '
               '"1": [["1", "0"], ["0", "0"]], '
               '"2": [["0", "0"], ["0", "1"]]}',
        "E": "",
        "c": '[["-1"], ["-1"]]',
        "C": "",
        "Q": '[["0", "0"], ["0", "0"]]',
        "rho": "0",
        "tau": "0",
        "theta_bar": "0,0",
        "p": "2",
        "solver": "SCS",
        "confidence": "0.001",
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "optimal_x" in data


def test_download_mat_route(client):
    """POST /download-mat returns a .mat file."""
    payload = {"x": [[1.0, 2.0], [3.0, 4.0]], "cost": [5.0]}
    resp = client.post(
        "/download-mat",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert resp.status_code == 200
    assert "matlab" in resp.content_type or "octet-stream" in resp.content_type


def _interval_form(**overrides):
    """Form fields for the smallest-enclosing-interval LP, as the browser sends them."""
    csv_bytes = "\n".join(str(v) for v in [0.1, 0.4, 0.9]).encode()
    data = {
        "active_tab": "lp-tab",
        "A_d": '[["-1", "-1"], ["1", "-1"]]',
        "b_d": '[["delta[0]"], ["-delta[0]"]]',
        "c": '[["0"], ["1"]]',
        "rho": "0",
        "tau": "0",
        "option": "robust",
        "solver": "CLARABEL",
        "confidence": "1e-06",
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }
    data.update(overrides)
    return data


def test_solve_accepts_plain_numbers(client):
    """Matrices with plain numbers (not strings) solve like their string form."""
    resp = client.post("/solve", data=_interval_form(c="[[0], [1]]", A_d='[[-1, -1], [1, -1]]'),
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    np.testing.assert_allclose(np.ravel(resp.get_json()["optimal_x"][0]), [0.5, 0.4], atol=1e-6)


@pytest.mark.parametrize("beta", ["", "0", "1", "abc"])
def test_solve_rejects_invalid_beta(client, beta):
    """A missing or out-of-range β is reported as an error, not treated as 0."""
    resp = client.post("/solve", data=_interval_form(confidence=beta), content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "confidence parameter" in resp.get_json()["error"]


def test_solve_robust_ignores_hidden_parameters(client):
    """With the robust option, hidden τ and ρ fields are neither applied nor swept."""
    resp = client.post("/solve", data=_interval_form(tau="5", rho="0.1, 1"), content_type="multipart/form-data")
    data = resp.get_json()
    assert data["n_attempts"] == 1 and data["tau_"] == [0.0] and data["rho_"] == [0.0]
    np.testing.assert_allclose(np.ravel(data["optimal_x"][0]), [0.5, 0.4], atol=1e-6)


def test_solve_sweep_confidence_and_total_constraints(client):
    """A sweep reports 1 − β·n_runs; Total Constraints counts the hard constraints too."""
    resp = client.post("/solve", data=_interval_form(option="relaxation", rho="0.5, 1", confidence="0.01",
                                                     G='[["0", "-1"]]', h='[["0"]]'),
                       content_type="multipart/form-data")
    data = resp.get_json()
    assert data["n_attempts"] == 2
    assert abs(data["conf"] - 0.98) < 1e-12
    assert data["tot_con"] == [4, 4]  # 3 scenario constraints + 1 row of G


def test_download_mat_with_a_failed_run(client):
    """A run without results (None) no longer breaks the .MAT download."""
    resp = client.post("/download-mat", json={"optimal_cost": [0.5, None], "risk": [[0.0, 0.2], []]})
    assert resp.status_code == 200
    assert resp.data[:6] == b"MATLAB"


@pytest.mark.parametrize("c", ['[["0"], ["1"]', '[["exp(1000)"], ["1"]]', '[["1/0"], ["1"]]'])
def test_solve_reports_unreadable_matrices(client, c):
    """A malformed or unevaluable matrix gives a readable 400 error, not a 500."""
    resp = client.post("/solve", data=_interval_form(c=c), content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"]
