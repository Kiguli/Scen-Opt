"""Solve a scenario program described by the web form's fields.

Used by the Flask app (``app.py``) and by the MOSEK licence subprocess
(``src/mosek_solve.py``), so that both paths compute and report results in
exactly the same way.
"""
import json
import time

import numpy as np

from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Risk import quantify_risk
from src.parsing import (generate_matrix_function, generate_matrix,
                         generate_tensor_function, generate_tensor,
                         generate_numeric_A_b, generate_numeric_F)


def parse_theta_bar(raw):
    """Parse the x̄ (reference point) form field.

    The field is populated by the matrix-grid modal as a JSON-encoded 2-D
    array (e.g. ``[["10"]]`` or ``[[0], [1]]``). Older inputs or an empty
    field may arrive as an empty/comma-separated string. This helper
    accepts any of those and returns a 1-D numpy float array, or 0.0 when
    the field is unset (backward compat with the previous default).
    """
    if raw is None:
        return 0.0
    s = raw.strip()
    if s in ('', 'null', '""', '[]'):
        return 0.0
    # Try JSON first (the standard path for matrix-modal entries).
    try:
        parsed = json.loads(s)
        return np.asarray(parsed, dtype=float).reshape(-1)
    except (ValueError, TypeError):
        pass
    # Fallback: legacy comma-separated scalars (e.g. "0, 0").
    try:
        return np.array([float(x) for x in s.split(',') if x.strip() != ''])
    except ValueError as e:
        raise ValueError(
            f"Could not parse x̄ (theta_bar) value {raw!r}: expected a JSON "
            f"array (e.g. [[10]]) or comma-separated numbers. ({e})"
        )


def compute_base_cost(active_tab, x, c, Q):
    """Compute c'x (LP) or c'x + ½x'Qx (QP/SDP) from the optimal x.

    Regularization (τ·‖x − x̄‖) and relaxation (ρ·Σζ_i) penalty terms are
    numerical tools for informing the solution; they are not part of the
    underlying problem's objective and must not leak into reported cost.
    """
    if x is None:
        return None
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.size == 0:
        return None
    c_arr = np.asarray(c, dtype=float).ravel() if c is not None and getattr(c, 'size', 0) else None
    if c_arr is None or c_arr.size != x_arr.size:
        return None
    base = float(c_arr @ x_arr)
    if active_tab in ('qp-tab', 'sdp-tab') and Q is not None and getattr(Q, 'size', 0):
        Q_arr = np.asarray(Q, dtype=float)
        if Q_arr.shape == (x_arr.size, x_arr.size):
            base += 0.5 * float(x_arr @ Q_arr @ x_arr)
    return base


def parse_confidence(raw):
    """Parse the confidence parameter β, which must lie strictly between 0 and 1."""
    try:
        beta = float(raw)
    except (TypeError, ValueError):
        raise ValueError("Enter the confidence parameter β, a number between 0 and 1 (e.g. 1e-06).")
    if not 0 < beta < 1:
        raise ValueError(f"The confidence parameter β must lie strictly between 0 and 1 (got {raw}).")
    return beta


def parse_norm(raw):
    """Parse the norm order p: a number p ≥ 1, ``inf`` or ``fro``. Empty means p = 2."""
    p_raw = (raw or '').strip().strip("'\"")
    if not p_raw:
        return 2
    if p_raw.lower() in ('inf', 'infinity', 'np.inf'):
        return np.inf
    if p_raw.lower() == 'fro':
        return 'fro'
    try:
        p = float(p_raw)
    except ValueError:
        raise ValueError(
            f"p (norm order) must be a single number, 'inf', or 'fro'. "
            f"You entered {p_raw!r} — comma-separated lists are not "
            f"supported for p (sweep over τ or ρ instead)."
        )
    if p < 1:
        raise ValueError(f"p (norm order) must be at least 1 (got {p_raw}).")
    return p


def solve_form(form, scenarios, solver=None):
    """Solve every (τ, ρ) combination described by the web form.

    Parameters
    ----------
    form : dict
        The web form's fields as strings (``active_tab``, ``mode``, the
        matrices as JSON strings, ``rho``, ``tau``, ``confidence``, ...).
    scenarios : numpy.ndarray or None
        The uploaded scenarios, one row per scenario.
    solver : str or None
        Solver to use. ``None`` uses the form's ``solver`` field
        (default ``CLARABEL``); the MOSEK subprocess passes ``"MOSEK"``.

    Returns
    -------
    dict
        The result lists shown in the results table, one entry per run.
    """
    active_tab = form.get('active_tab')

    mode = form.get('mode', 'symbolic')
    n_x = int(form.get('n_x', 0)) if form.get('n_x') else 0
    rows_A = int(form.get('rows_A', 0)) if form.get('rows_A') else 0
    lmi_size = int(form.get('lmi_size', 0)) if form.get('lmi_size') else 0

    A_d = None
    b_d = None
    F_d = None

    if mode == 'numeric':
        if scenarios is None:
            raise ValueError("Numeric mode requires an uploaded scenario data file.")
        if active_tab in ('lp-tab', 'qp-tab'):
            if n_x <= 0 or rows_A <= 0:
                raise ValueError("Numeric mode requires valid decision variable count (n_x) and soft constraint rows (rows_A).")
            expected_cols = rows_A * n_x + rows_A
            if scenarios.shape[1] != expected_cols:
                raise ValueError(
                    f"Each scenario row must have {expected_cols} elements "
                    f"(rows_A*n_x + rows_A = {rows_A}*{n_x} + {rows_A}), "
                    f"but got {scenarios.shape[1]}.")
            A_d, b_d = generate_numeric_A_b(rows_A, n_x)
        elif active_tab == 'sdp-tab':
            if n_x <= 0 or lmi_size <= 0:
                raise ValueError("Numeric mode requires valid decision variable count (n_x) and LMI matrix size.")
            expected_cols = (n_x + 1) * lmi_size ** 2
            if scenarios.shape[1] != expected_cols:
                raise ValueError(
                    f"Each scenario row must have {expected_cols} elements "
                    f"((n_x+1)*lmi_size² = {n_x + 1}*{lmi_size}² = {expected_cols}), "
                    f"but got {scenarios.shape[1]}.")
            F_d = generate_numeric_F(lmi_size, n_x)
    else:
        if form.get('A_d'):
            A_d = generate_matrix_function(form.get('A_d'))
        elif active_tab in ('lp-tab', 'qp-tab'):
            raise ValueError("\\(A(\\delta)\\) is ill-defined")
        if form.get('b_d'):
            b_d = generate_matrix_function(form.get('b_d'))
        elif active_tab in ('lp-tab', 'qp-tab'):
            raise ValueError("\\(b(\\delta)\\) is ill-defined")
        if form.get('F_d'):
            F_d = generate_tensor_function(form.get('F_d'))
        elif active_tab == 'sdp-tab':
            raise ValueError("\\(F_j(\\delta)\\) is ill-defined")

    # Hard LP/QP constraints use G, h (form fields renamed from legacy A, b).
    G = generate_matrix(form.get('G')) if form.get('G') else np.array([])
    h_vec = generate_matrix(form.get('h')) if form.get('h') else np.array([])
    c = generate_matrix(form.get('c')) if form.get('c') else np.array([])
    Q = generate_matrix(form.get('Q')) if form.get('Q') else np.array([])
    # Hard SDP LMI collection uses E (form field renamed from legacy F).
    E = generate_tensor(form.get('E')) if form.get('E') else {}

    conf = parse_confidence(form.get('confidence'))

    rhos = np.array([float(x) for x in form.get('rho').split(',')]) if form.get('rho') else np.array([0.0])
    taus = np.array([float(x) for x in form.get('tau').split(',')]) if form.get('tau') else np.array([0.0])

    # The selected formulation decides which parameters apply. The fields of
    # the other formulations are hidden in the form but still submitted, so
    # they are ignored here rather than silently applied or swept over.
    selected_option = form.get('option')
    if selected_option in ('robust', 'regularization'):
        rhos = np.array([0.0])
    if selected_option in ('robust', 'relaxation'):
        taus = np.array([0.0])

    theta_bar = parse_theta_bar(form.get('theta_bar'))
    p = parse_norm(form.get('p'))

    # Hard constraints counted in "Total Constraints": the rows of G for
    # LP/QP, or the single hard LMI for SDP.
    if active_tab == 'sdp-tab':
        n_hard = 1 if E else 0
    else:
        n_hard = G.shape[0] if (G.size and h_vec.size) else 0

    # Initialize lists to collect results for each run
    optimal_x_list = []
    optimal_s_list = []
    optimal_cost_list = []
    N_list = []
    active_list = []
    constraints_list = []
    risk_list = []
    e_list = []
    degeneracy_list = []
    rho_list = []
    tau_list = []
    solve_time_list = []
    risk_time_list = []

    # Overall confidence after q sweep attempts is 1 − β·q (union bound).
    # Pass the user-entered β directly to the per-attempt risk computation;
    # only the *reported* confidence folds in q.
    n_attempts = len(taus) * len(rhos)
    conf_reported = conf * n_attempts

    # The relaxation formulation must keep its slack variables ζ_i even when
    # the user has set ρ = 0 (treats that case as genuinely unbounded rather
    # than silently coercing it into the robust LP). When `option` is not in
    # the form payload (older clients / direct API callers), fall through to
    # the legacy "infer slack from ρ" behaviour by leaving include_slack=None
    # so each solver default kicks in.
    if selected_option:
        include_slack = selected_option in ('relaxation', 'regularization-relaxation')
    else:
        include_slack = None

    run_solver = solver or form.get('solver', 'CLARABEL')

    for tau in taus:
        for rho in rhos:
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
                if active_tab == 'lp-tab':
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_lp(
                        scenarios, A_d, b_d, G, h_vec, c, tau, theta_bar, rho, p, run_solver,
                        include_slack=include_slack)
                elif active_tab == 'qp-tab':
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_qp(
                        scenarios, A_d, b_d, G, h_vec, c, Q, tau, theta_bar, rho, p, run_solver,
                        include_slack=include_slack)
                elif active_tab == 'sdp-tab':
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_sdp(
                        scenarios, F_d, E, c, Q, tau, theta_bar, rho, p, run_solver,
                        include_slack=include_slack)
                solve_time = time.perf_counter() - t0

                # Replace full-objective cost (which includes τ·‖x−x̄‖_p and
                # ρ·Σζ_i penalty terms used only as solver guidance) with the
                # base cost: c'x for LP, c'x + ½x'Qx for QP/SDP.
                optimal_cost = compute_base_cost(active_tab, optimal_x, c, Q)

                t1 = time.perf_counter()
                risk = np.array(quantify_risk(complexity, N, conf))
                risk_time = time.perf_counter() - t1
            except Exception as error:
                e = str(error)
                # Shorten verbose MOSEK license errors
                if 'license' in e.lower() and ('mosek' in e.lower() or 'rescode' in e.lower()):
                    e = "MOSEK license cannot be located or is incorrect."
                print(e)

            # Append results for this run
            optimal_x_list.append(optimal_x.tolist() if hasattr(optimal_x, 'tolist') else optimal_x)
            optimal_s_list.append(optimal_s.tolist() if hasattr(optimal_s, 'tolist') else optimal_s)
            optimal_cost_list.append(optimal_cost)
            N_list.append(N)
            active_list.append(complexity)
            e_list.append(e)
            constraints_list.append(len(constraints) + n_hard if len(constraints) else 0)
            risk_list.append(risk.tolist() if hasattr(risk, 'tolist') else risk)
            degeneracy_list.append(degeneracy)
            rho_list.append(float(rho))
            tau_list.append(float(tau))
            solve_time_list.append(round(solve_time, 4))
            risk_time_list.append(round(risk_time, 4))

    # Prepare the result dictionary with lists for each parameter
    return {
        "form_data": dict(form),
        "optimal_x": optimal_x_list,
        "optimal_s": optimal_s_list,
        "optimal_cost": optimal_cost_list,
        "num_deltas": N_list,
        "tot_con": constraints_list,
        "active_con": active_list,
        "risk": risk_list,
        "conf": 1 - conf_reported,
        "tau_": tau_list,
        "rho_": rho_list,
        "errorcode": e_list,
        "degeneracy": degeneracy_list,
        "solve_time": solve_time_list,
        "risk_time": risk_time_list,
        "n_attempts": n_attempts,
    }
