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
