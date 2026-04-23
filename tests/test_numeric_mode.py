"""
Compare symbolic vs numeric mode for LP, QP, and SDP benchmarks.

For each benchmark:
1. Run in symbolic mode (expressions + delta scenario data)
2. Convert to numeric mode (pre-computed flattened [A_i | b_i] rows)
3. Verify both produce the same optimal_x and optimal_cost
"""
import io
import json
import math
import numpy as np
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── helpers ──────────────────────────────────────────────────────────

def _eval_matrix(expr_matrix, delta):
    """Evaluate a 2D list of expression strings at a given delta vector."""
    return np.array([
        [eval(expr, {"delta": delta, "math": math}) for expr in row]
        for row in expr_matrix
    ])


def _to_csv_bytes(arr):
    """Convert a 2D numpy array to CSV bytes."""
    lines = []
    for row in arr:
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines).encode()


# ── LP: half_width_2d ────────────────────────────────────────────────

def _lp_benchmark():
    with open("benchmarks/LP_half_width_2d/data/program_symbolic.json") as f:
        bm = json.load(f)
    scenarios_raw = np.loadtxt("benchmarks/LP_half_width_2d/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    return bm, scenarios_raw


def test_lp_symbolic_vs_numeric(client):
    """LP half_width_2d: symbolic and numeric modes produce same results."""
    bm, scenarios_raw = _lp_benchmark()

    # ── Symbolic request ──
    sym_csv = _to_csv_bytes(scenarios_raw)
    sym_resp = client.post("/solve", data={
        "active_tab": "lp-tab",
        "mode": "symbolic",
        "A_d": json.dumps(bm["A_d"]),
        "b_d": json.dumps(bm["b_d"]),
        "G": "",
        "h": "",
        "c": json.dumps(bm["c"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(sym_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert sym_resp.status_code == 200, f"Symbolic LP failed: {sym_resp.get_data(as_text=True)}"
    sym_data = sym_resp.get_json()
    assert "optimal_x" in sym_data, f"Symbolic LP no optimal_x: {sym_data}"

    # ── Build numeric scenarios: each row = [A_i_flat | b_i_flat] ──
    A_d_expr = bm["A_d"]  # 2x2
    b_d_expr = bm["b_d"]  # 2x1
    rows_A = len(A_d_expr)
    n_x = len(A_d_expr[0])

    numeric_rows = []
    for delta in scenarios_raw:
        A_i = _eval_matrix(A_d_expr, delta).flatten()
        b_i = _eval_matrix(b_d_expr, delta).flatten()
        numeric_rows.append(np.concatenate([A_i, b_i]))
    numeric_scenarios = np.array(numeric_rows)
    num_csv = _to_csv_bytes(numeric_scenarios)

    # ── Numeric request ──
    num_resp = client.post("/solve", data={
        "active_tab": "lp-tab",
        "mode": "numeric",
        "n_x": str(n_x),
        "rows_A": str(rows_A),
        "G": "",
        "h": "",
        "c": json.dumps(bm["c"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(num_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert num_resp.status_code == 200, f"Numeric LP failed: {num_resp.get_data(as_text=True)}"
    num_data = num_resp.get_json()
    assert "optimal_x" in num_data, f"Numeric LP no optimal_x: {num_data}"

    # ── Compare (response wraps results in lists for sweep support) ──
    sym_x = np.array(sym_data["optimal_x"][0])
    num_x = np.array(num_data["optimal_x"][0])
    np.testing.assert_allclose(sym_x, num_x, atol=1e-4,
                               err_msg="LP optimal_x mismatch between symbolic and numeric")

    sym_cost = sym_data["optimal_cost"][0]
    num_cost = num_data["optimal_cost"][0]
    assert abs(sym_cost - num_cost) < 1e-4, \
        f"LP cost mismatch: symbolic={sym_cost}, numeric={num_cost}"

    print(f"\nLP OK: x={sym_x.flatten()}, cost={sym_cost:.6f}")


# ── QP: Iris_minimal_3d ──────────────────────────────────────────────

def _qp_benchmark():
    with open("benchmarks/QP_Iris_minimal_3d/data/program_symbolic.json") as f:
        bm = json.load(f)
    scenarios_raw = np.loadtxt("benchmarks/QP_Iris_minimal_3d/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    return bm, scenarios_raw


def test_qp_symbolic_vs_numeric(client):
    """QP Iris_minimal_3d: symbolic and numeric modes produce same results."""
    bm, scenarios_raw = _qp_benchmark()

    # ── Symbolic request ──
    sym_csv = _to_csv_bytes(scenarios_raw)
    sym_resp = client.post("/solve", data={
        "active_tab": "qp-tab",
        "mode": "symbolic",
        "A_d": json.dumps(bm["A_d"]),
        "b_d": json.dumps(bm["b_d"]),
        "G": "",
        "h": "",
        "c": json.dumps(bm["c"]),
        "Q": json.dumps(bm["Q"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(sym_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert sym_resp.status_code == 200, f"Symbolic QP failed: {sym_resp.get_data(as_text=True)}"
    sym_data = sym_resp.get_json()
    assert "optimal_x" in sym_data, f"Symbolic QP no optimal_x: {sym_data}"

    # ── Build numeric scenarios ──
    A_d_expr = bm["A_d"]  # 1x3
    b_d_expr = bm["b_d"]  # 1x1
    rows_A = len(A_d_expr)
    n_x = len(A_d_expr[0])

    numeric_rows = []
    for delta in scenarios_raw:
        A_i = _eval_matrix(A_d_expr, delta).flatten()
        b_i = _eval_matrix(b_d_expr, delta).flatten()
        numeric_rows.append(np.concatenate([A_i, b_i]))
    numeric_scenarios = np.array(numeric_rows)
    num_csv = _to_csv_bytes(numeric_scenarios)

    # ── Numeric request ──
    num_resp = client.post("/solve", data={
        "active_tab": "qp-tab",
        "mode": "numeric",
        "n_x": str(n_x),
        "rows_A": str(rows_A),
        "G": "",
        "h": "",
        "c": json.dumps(bm["c"]),
        "Q": json.dumps(bm["Q"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(num_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert num_resp.status_code == 200, f"Numeric QP failed: {num_resp.get_data(as_text=True)}"
    num_data = num_resp.get_json()
    assert "optimal_x" in num_data, f"Numeric QP no optimal_x: {num_data}"

    # ── Compare ──
    sym_x = np.array(sym_data["optimal_x"][0])
    num_x = np.array(num_data["optimal_x"][0])
    np.testing.assert_allclose(sym_x, num_x, atol=1e-4,
                               err_msg="QP optimal_x mismatch between symbolic and numeric")

    sym_cost = sym_data["optimal_cost"][0]
    num_cost = num_data["optimal_cost"][0]
    assert abs(sym_cost - num_cost) < 1e-4, \
        f"QP cost mismatch: symbolic={sym_cost}, numeric={num_cost}"

    print(f"\nQP OK: x={sym_x.flatten()}, cost={sym_cost:.6f}")


# ── SDP: LPV_stability_3d ────────────────────────────────────────────

def _sdp_benchmark():
    with open("benchmarks/SDP_LPV_stability_3d/data/program_symbolic.json") as f:
        bm = json.load(f)
    scenarios_raw = np.loadtxt("benchmarks/SDP_LPV_stability_3d/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    return bm, scenarios_raw


def _eval_tensor(expr_dict, delta):
    """Evaluate a dict of matrix expression strings at a given delta vector."""
    return {
        key: np.array([
            [eval(expr, {"delta": delta, "math": math}) for expr in row]
            for row in expr_dict[key]
        ])
        for key in expr_dict
    }


def test_sdp_symbolic_vs_numeric(client):
    """SDP LPV_stability_3d: symbolic and numeric modes produce same results."""
    bm, scenarios_raw = _sdp_benchmark()

    # ── Symbolic request ──
    sym_csv = _to_csv_bytes(scenarios_raw)
    sym_resp = client.post("/solve", data={
        "active_tab": "sdp-tab",
        "mode": "symbolic",
        "F_d": json.dumps(bm["F_d"]),
        "E": json.dumps(bm["E"]),
        "c": json.dumps(bm["c"]),
        "Q": json.dumps(bm["Q"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(sym_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert sym_resp.status_code == 200, f"Symbolic SDP failed: {sym_resp.get_data(as_text=True)}"
    sym_data = sym_resp.get_json()
    assert "optimal_x" in sym_data, f"Symbolic SDP no optimal_x: {sym_data}"

    # ── Build numeric scenarios: each row = [F_0_flat | F_1_flat | ... | F_d_flat] ──
    F_d_expr = bm["F_d"]
    n_x = len(bm["c"])          # 3 decision variables
    lmi_size = len(F_d_expr["0"])  # 2x2 matrices

    numeric_rows = []
    for delta in scenarios_raw:
        F_dict = _eval_tensor(F_d_expr, delta)
        row = []
        for j in range(n_x + 1):
            row.append(F_dict[str(j)].flatten())
        numeric_rows.append(np.concatenate(row))
    numeric_scenarios = np.array(numeric_rows)
    num_csv = _to_csv_bytes(numeric_scenarios)

    # ── Numeric request ──
    num_resp = client.post("/solve", data={
        "active_tab": "sdp-tab",
        "mode": "numeric",
        "n_x": str(n_x),
        "lmi_size": str(lmi_size),
        "E": json.dumps(bm["E"]),
        "c": json.dumps(bm["c"]),
        "Q": json.dumps(bm["Q"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(num_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert num_resp.status_code == 200, f"Numeric SDP failed: {num_resp.get_data(as_text=True)}"
    num_data = num_resp.get_json()
    assert "optimal_x" in num_data, f"Numeric SDP no optimal_x: {num_data}"

    # ── Compare ──
    sym_x = np.array(sym_data["optimal_x"][0])
    num_x = np.array(num_data["optimal_x"][0])
    np.testing.assert_allclose(sym_x, num_x, atol=1e-3,
                               err_msg="SDP optimal_x mismatch between symbolic and numeric")

    sym_cost = sym_data["optimal_cost"][0]
    num_cost = num_data["optimal_cost"][0]
    assert abs(sym_cost - num_cost) < 1e-3, \
        f"SDP cost mismatch: symbolic={sym_cost}, numeric={num_cost}"

    print(f"\nSDP OK: x={sym_x.flatten()}, cost={sym_cost:.6f}")


# ── Detect Program: load JSON then solve ─────────────────────────────

def test_detect_program_lp(client):
    """Detect Program with LP benchmark JSON loads and solves correctly."""
    with open("benchmarks/LP_half_width_2d/data/program_symbolic.json") as f:
        bm = json.load(f)

    scenarios_raw = np.loadtxt("benchmarks/LP_half_width_2d/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    csv_bytes = _to_csv_bytes(scenarios_raw)

    # Simulate what Detect Program does: populate form fields from JSON
    resp = client.post("/solve", data={
        "active_tab": "lp-tab",
        "mode": "symbolic",
        "A_d": json.dumps(bm["A_d"]),
        "b_d": json.dumps(bm["b_d"]),
        "G": "",
        "h": "",
        "c": json.dumps(bm["c"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "optimal_x" in data
    assert data["optimal_cost"][0] is not None
    print(f"\nDetect Program LP OK: x={np.array(data['optimal_x'][0]).flatten()}, cost={data['optimal_cost'][0]:.6f}")


def test_detect_program_sdp(client):
    """Detect Program with SDP benchmark JSON loads and solves correctly."""
    with open("benchmarks/SDP_LPV_stability_3d/data/program_symbolic.json") as f:
        bm = json.load(f)

    scenarios_raw = np.loadtxt("benchmarks/SDP_LPV_stability_3d/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    csv_bytes = _to_csv_bytes(scenarios_raw)

    resp = client.post("/solve", data={
        "active_tab": "sdp-tab",
        "mode": "symbolic",
        "F_d": json.dumps(bm["F_d"]),
        "E": json.dumps(bm["E"]),
        "c": json.dumps(bm["c"]),
        "Q": json.dumps(bm["Q"]),
        "rho": bm.get("rho", "0"),
        "tau": bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": bm.get("confidence", "0.001"),
        "file": (io.BytesIO(csv_bytes), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "optimal_x" in data
    assert data["optimal_cost"][0] is not None
    print(f"\nDetect Program SDP OK: x={np.array(data['optimal_x'][0]).flatten()}, cost={data['optimal_cost'][0]:.6f}")


# ── Numeric SDP benchmark: pre-computed scenarios ────────────────────

def test_detect_program_sdp_numeric(client):
    """Detect Program with numeric SDP benchmark loads and matches symbolic results."""
    # Load numeric benchmark
    with open("benchmarks/SDP_LPV_stability_3d_numeric/data/program_symbolic.json") as f:
        num_bm = json.load(f)
    num_scenarios = np.loadtxt("benchmarks/SDP_LPV_stability_3d_numeric/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    num_csv = _to_csv_bytes(num_scenarios)

    num_resp = client.post("/solve", data={
        "active_tab": "sdp-tab",
        "mode": "numeric",
        "n_x": str(num_bm["n_x"]),
        "lmi_size": str(num_bm["lmi_size"]),
        "E": json.dumps(num_bm["E"]),
        "c": json.dumps(num_bm["c"]),
        "Q": json.dumps(num_bm["Q"]),
        "rho": num_bm.get("rho", "0"),
        "tau": num_bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": num_bm.get("confidence", "0.001"),
        "file": (io.BytesIO(num_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert num_resp.status_code == 200, f"Numeric SDP failed: {num_resp.get_data(as_text=True)}"
    num_data = num_resp.get_json()
    assert "optimal_x" in num_data, f"Numeric SDP no optimal_x: {num_data}"

    # Compare against symbolic benchmark
    with open("benchmarks/SDP_LPV_stability_3d/data/program_symbolic.json") as f:
        sym_bm = json.load(f)
    sym_scenarios = np.loadtxt("benchmarks/SDP_LPV_stability_3d/data/scenarios.csv",
                                delimiter=",", ndmin=2)
    sym_csv = _to_csv_bytes(sym_scenarios)

    sym_resp = client.post("/solve", data={
        "active_tab": "sdp-tab",
        "mode": "symbolic",
        "F_d": json.dumps(sym_bm["F_d"]),
        "E": json.dumps(sym_bm["E"]),
        "c": json.dumps(sym_bm["c"]),
        "Q": json.dumps(sym_bm["Q"]),
        "rho": sym_bm.get("rho", "0"),
        "tau": sym_bm.get("tau", "0"),
        "theta_bar": "",
        "p": "2",
        "solver": "SCS",
        "confidence": sym_bm.get("confidence", "0.001"),
        "file": (io.BytesIO(sym_csv), "scenarios.csv"),
    }, content_type="multipart/form-data")

    assert sym_resp.status_code == 200
    sym_data = sym_resp.get_json()

    sym_x = np.array(sym_data["optimal_x"][0])
    num_x = np.array(num_data["optimal_x"][0])
    np.testing.assert_allclose(sym_x, num_x, atol=1e-3,
                               err_msg="Numeric SDP benchmark vs symbolic mismatch")

    sym_cost = sym_data["optimal_cost"][0]
    num_cost = num_data["optimal_cost"][0]
    assert abs(sym_cost - num_cost) < 1e-3, \
        f"SDP cost mismatch: symbolic={sym_cost}, numeric={num_cost}"

    print(f"\nNumeric SDP benchmark OK: x={num_x.flatten()}, cost={num_cost:.6f}")
