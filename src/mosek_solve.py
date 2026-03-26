#!/usr/bin/env python3
"""Subprocess entry point for MOSEK solves.

Sets MOSEKLM_LICENSE_FILE *before* importing CVXPY so that MOSEK 11
reads the license path on first import. This enables concurrent
users to each supply their own license without conflicts.

Usage:
    echo '<json>' | python3 src/mosek_solve.py /tmp/mosek_<uuid>/mosek.lic
"""
import os
import sys

# CRITICAL: set license path before any CVXPY / MOSEK imports
if len(sys.argv) > 1:
    os.environ["MOSEKLM_LICENSE_FILE"] = sys.argv[1]

import json
import time
import numpy as np

# Now safe to import solve functions (which import CVXPY → MOSEK)
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Risk import quantify_risk
from src.parsing import (
    generate_matrix_function,
    generate_matrix,
    generate_tensor_function,
    generate_tensor,
    generate_numeric_A_b,
    generate_numeric_F,
)


def _safe_tolist(v):
    """Convert numpy types to JSON-serialisable Python types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def main():
    # Redirect stdout → stderr so that stray print() calls from solver
    # code (LP.py, Miscellaneous.py, etc.) don't corrupt the JSON output.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    input_data = json.loads(sys.stdin.read())
    form = input_data["form"]
    scenarios_raw = input_data.get("scenarios")

    scenarios = np.array(scenarios_raw, dtype=float) if scenarios_raw is not None else None

    active_tab = form.get("active_tab")

    # ---- build matrix functions / constants from raw form strings ----
    A_d = b_d = F_d = A_da = b_da = None

    mode = form.get("mode", "symbolic")
    n_x = int(form.get("n_x", 0)) if form.get("n_x") else 0
    rows_A = int(form.get("rows_A", 0)) if form.get("rows_A") else 0
    lmi_size = int(form.get("lmi_size", 0)) if form.get("lmi_size") else 0

    if mode == "numeric":
        if scenarios is None:
            raise ValueError("Numeric mode requires an uploaded scenario data file.")
        if active_tab in ("lp-tab", "qp-tab"):
            if n_x <= 0 or rows_A <= 0:
                raise ValueError("Numeric mode requires valid n_x and rows_A.")
            expected_cols = rows_A * n_x + rows_A
            if scenarios.shape[1] != expected_cols:
                raise ValueError(
                    f"Each scenario row must have {expected_cols} elements "
                    f"(rows_A*n_x + rows_A = {rows_A}*{n_x} + {rows_A}), "
                    f"but got {scenarios.shape[1]}.")
            A_d, b_d = generate_numeric_A_b(rows_A, n_x)
        elif active_tab == "sdp-tab":
            if n_x <= 0 or lmi_size <= 0:
                raise ValueError("Numeric mode requires valid n_x and lmi_size.")
            expected_cols = (n_x + 1) * lmi_size ** 2
            if scenarios.shape[1] != expected_cols:
                raise ValueError(
                    f"Each scenario row must have {expected_cols} elements "
                    f"((n_x+1)*lmi_size² = {n_x + 1}*{lmi_size}² = {expected_cols}), "
                    f"but got {scenarios.shape[1]}.")
            F_d = generate_numeric_F(lmi_size, n_x)
    else:
        if form.get("A_d"):
            A_d = generate_matrix_function(form["A_d"])
        elif active_tab in ("lp-tab", "qp-tab"):
            raise ValueError("\\(A(\\delta)\\) is ill-defined")

        if form.get("b_d"):
            b_d = generate_matrix_function(form["b_d"])
        elif active_tab in ("lp-tab", "qp-tab"):
            raise ValueError("\\(b(\\delta)\\) is ill-defined")

        if form.get("F_d"):
            F_d = generate_tensor_function(form["F_d"])
        elif active_tab == "sdp-tab":
            raise ValueError("\\(F_j(\\delta)\\) is ill-defined")

    if form.get("A_da"):
        A_da = generate_tensor_function(form["A_da"])
    if form.get("b_da"):
        b_da = generate_tensor_function(form["b_da"])

    A = generate_matrix(form["A"]) if form.get("A") else np.array([])
    b = generate_matrix(form["b"]) if form.get("b") else np.array([])
    c = generate_matrix(form["c"]) if form.get("c") else np.array([])
    Q = generate_matrix(form["Q"]) if form.get("Q") else np.array([])
    F = generate_tensor(form["F"]) if form.get("F") else {}

    conf = float(form["confidence"]) if form.get("confidence") else 0.0

    rhos = (
        np.array([float(x) for x in form["rho"].split(",")])
        if form.get("rho")
        else np.array([0.0])
    )
    taus = (
        np.array([float(x) for x in form["tau"].split(",")])
        if form.get("tau")
        else np.array([0.0])
    )
    theta_bar = (
        np.array([float(x) for x in form["theta_bar"].split(",")])
        if form.get("theta_bar")
        else 0.0
    )
    p = float(form["p"]) if form.get("p") else 2

    # ---- solve loop (mirrors _solve_inner in app.py) ----
    conf = conf / (len(taus) * len(rhos))

    optimal_x_list, optimal_s_list, optimal_cost_list = [], [], []
    N_list, active_list, constraints_list = [], [], []
    risk_list, e_list, degeneracy_list = [], [], []
    rho_list, tau_list = [], []
    solve_time_list, risk_time_list = [], []

    for tau in taus:
        for rho in rhos:
            solver = "MOSEK"

            optimal_x = np.array([])
            optimal_s = np.array([])
            optimal_cost = None
            N = 0
            complexity = []
            constraints = []
            degeneracy = False
            risk = np.array([])
            e = "None"
            solve_time = 0.0
            risk_time = 0.0

            try:
                t0 = time.perf_counter()
                if active_tab == "lp-tab":
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_lp(
                        scenarios, A_d, b_d, A, b, c, tau, theta_bar, rho, p, solver
                    )
                elif active_tab == "qp-tab":
                    Q_inner = generate_matrix(form["Q"]) if form.get("Q") else np.array([])
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_qp(
                        scenarios, A_d, b_d, A, b, c, Q_inner, tau, theta_bar, rho, p, solver
                    )
                elif active_tab == "sdp-tab":
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_sdp(
                        scenarios, F_d, F, c, Q, tau, theta_bar, rho, p, solver
                    )
                solve_time = time.perf_counter() - t0

                t1 = time.perf_counter()
                risk = np.array(quantify_risk(complexity, N, conf))
                risk_time = time.perf_counter() - t1
            except Exception as error:
                e = str(error)
                if "license" in e.lower() and ("mosek" in e.lower() or "rescode" in e.lower()):
                    e = "MOSEK license cannot be located or is incorrect."

            optimal_x_list.append(_safe_tolist(optimal_x))
            optimal_s_list.append(_safe_tolist(optimal_s))
            optimal_cost_list.append(_safe_tolist(optimal_cost))
            N_list.append(N)
            active_list.append(_safe_tolist(complexity))
            e_list.append(e)
            constraints_list.append(len(constraints))
            risk_list.append(_safe_tolist(risk))
            degeneracy_list.append(degeneracy)
            rho_list.append(_safe_tolist(rho))
            tau_list.append(_safe_tolist(tau))
            solve_time_list.append(round(solve_time, 4))
            risk_time_list.append(round(risk_time, 4))

    result = {
        "form_data": form,
        "optimal_x": optimal_x_list,
        "optimal_s": optimal_s_list,
        "optimal_cost": optimal_cost_list,
        "num_deltas": N_list,
        "tot_con": constraints_list,
        "active_con": active_list,
        "risk": risk_list,
        "conf": 1 - conf,
        "tau_": tau_list,
        "rho_": rho_list,
        "errorcode": e_list,
        "degeneracy": degeneracy_list,
        "solve_time": solve_time_list,
        "risk_time": risk_time_list,
    }

    json.dump(result, real_stdout, default=_safe_tolist)


if __name__ == "__main__":
    real_stdout = sys.stdout
    try:
        main()
    except Exception as exc:
        sys.stdout = real_stdout
        json.dump({"error": str(exc)}, real_stdout)
        sys.exit(0)
