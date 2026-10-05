from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import json as json_module
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Risk import quantify_risk
from src.Miscellaneous import get_solvers
from src.parsing import (generate_matrix_function, generate_matrix,
                         generate_tensor_function, generate_tensor,
                         generate_numeric_A_b, generate_numeric_F)

app = Flask(__name__)


@app.route('/')
def index():
    solvers = get_solvers()
    active_tab = request.args.get('active_tab', 'lp-tab')  # Default to 'lp-tab' if not provided
    return render_template('index.html', solvers=solvers, active_tab=active_tab)


def parse_uploaded_file(file):
    """Parse an uploaded file into a Python object (list/dict/array).
    Supports JSON, CSV, TXT, NPY, NPZ, MAT, Excel, and Parquet."""
    filename = file.filename.lower()
    if filename.endswith('.json'):
        return json_module.load(file)
    elif filename.endswith('.csv') or filename.endswith('.txt') or filename.endswith('.tsv'):
        content = file.read().decode('utf-8')
        sep = '\t' if filename.endswith('.tsv') else ','
        rows = content.strip().split('\n')
        return [row.split(sep) for row in rows]
    elif filename.endswith('.npy'):
        return np.load(file).tolist()
    elif filename.endswith('.npz'):
        data = np.load(file)
        keys = list(data.keys())
        return data[keys[0]].tolist()
    elif filename.endswith('.mat'):
        import scipy.io
        file_bytes = io.BytesIO(file.read())
        mat = scipy.io.loadmat(file_bytes, squeeze_me=False)

        def mat_unwrap(v):
            """Recursively unwrap numpy types to plain Python."""
            if isinstance(v, np.ndarray):
                # Struct array → dict
                if v.dtype.names is not None:
                    item = v.flat[0] if v.size == 1 else v
                    sub = {}
                    for name in v.dtype.names:
                        key = name[1:] if name.startswith('M') and name[1:].isdigit() else name
                        sub[key] = mat_unwrap(item[name])
                    return sub
                # Scalar (1×1) string or char array → plain string
                if v.dtype.kind in ('U', 'S'):
                    return str(v.flat[0]) if v.size == 1 else str(v.flat[0])
                # Scalar (1×1) numeric → float
                if v.dtype.kind in ('f', 'i', 'u') and v.size == 1:
                    return float(v.flat[0])
                # Object (cell) array → recurse into each element
                if v.dtype == object:
                    # 2-D cell array → always preserve as list of lists
                    if v.ndim == 2:
                        return [[mat_unwrap(v[r, c]) for c in range(v.shape[1])] for r in range(v.shape[0])]
                    # 1-D cell array → list
                    if v.ndim == 1:
                        return [mat_unwrap(x) for x in v]
                    # 0-D cell → unwrap
                    return mat_unwrap(v.item())
                # Numeric 2-D+ array → nested lists
                return v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, str):
                return v
            return v

        result = {}
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            result[k] = mat_unwrap(v)
        return result
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        import pandas as pd
        file_bytes = io.BytesIO(file.read())
        df = pd.read_excel(file_bytes, header=None)
        return df.values.tolist()
    elif filename.endswith('.parquet'):
        import pandas as pd
        file_bytes = io.BytesIO(file.read())
        df = pd.read_parquet(file_bytes)
        return df.values.tolist()
    else:
        raise TypeError(f"Unsupported file format: {filename}")


@app.route('/parse-file', methods=['POST'])
def parse_file():
    """Parse an uploaded file and return its content as JSON.
    Used by frontend modals to handle binary file formats."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        result = parse_uploaded_file(file)
        return jsonify({"data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/download-mat', methods=['POST'])
def download_mat():
    """Convert JSON result data to a MATLAB .mat file and return it."""
    import scipy.io
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    mat_dict = {}
    for key, value in data.items():
        try:
            mat_dict[key] = np.array(value)
        except (ValueError, TypeError):
            mat_dict[key] = np.array(value, dtype=object)

    buf = io.BytesIO()
    scipy.io.savemat(buf, mat_dict)
    buf.seek(0)
    return send_file(buf, mimetype='application/x-matlab',
                     as_attachment=True, download_name='results.mat')


@app.route('/solve', methods=['POST'])
def solve():
    # Check if a MOSEK license was uploaded — if so, run the solve in a
    # fresh subprocess so that MOSEKLM_LICENSE_FILE is read before MOSEK
    # is imported.  Each request gets a UUID-based /tmp directory so
    # concurrent users never conflict.
    has_license = (
        'mosek_license' in request.files
        and request.files['mosek_license'].filename != ''
    )

    if not has_license:
        return _solve_inner()

    # ---- MOSEK subprocess path ----
    # Use a platform-neutral temp directory (tempfile.mkdtemp → %TEMP% on
    # Windows, /tmp on Linux/macOS). Hard-coding /tmp broke Windows hosts
    # because the directory may not exist and isn't writable for the user.
    tmp_dir = tempfile.mkdtemp(prefix=f'mosek_{uuid.uuid4().hex}_')
    license_path = os.path.join(tmp_dir, 'mosek.lic')

    try:
        # Save uploaded license to the ephemeral /tmp directory
        request.files['mosek_license'].save(license_path)
        if os.path.getsize(license_path) == 0:
            return jsonify({"error": "Uploaded MOSEK license file is empty."}), 400

        # Parse scenario file (if any) in-process so we can serialise it
        scenarios_list = None
        if 'file' in request.files and request.files['file'].filename != '':
            parsed = parse_uploaded_file(request.files['file'])
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list) and len(v) > 0:
                        scenarios_list = np.array(v, dtype=float).tolist()
                        break
                if scenarios_list is None:
                    return jsonify({"error": "MAT file does not contain a recognizable scenario matrix."}), 400
            else:
                scenarios_list = np.array(parsed, dtype=float).tolist()

        # Build JSON payload for the subprocess
        payload = json_module.dumps({
            "form": request.form.to_dict(),
            "scenarios": scenarios_list,
        })

        # Run the solve in a fresh Python process with the license set
        proc = subprocess.run(
            [sys.executable, '-m', 'src.mosek_solve', license_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return jsonify({"error": f"MOSEK subprocess failed: {stderr}"}), 500

        result = json_module.loads(proc.stdout)
        return jsonify(result)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "MOSEK solve timed out."}), 504
    except json_module.JSONDecodeError:
        return jsonify({"error": "MOSEK subprocess returned invalid output."}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_theta_bar(raw):
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
        parsed = json_module.loads(s)
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


def _solve_inner():
    option = request.form.get('option')
    active_tab = request.form.get('active_tab')

    # Handle scenarios file if present
    scenarios = None
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = file.filename.lower()
        parsed = parse_uploaded_file(file)
        if isinstance(parsed, dict):
            # MAT files return a dict; use the first array-like value
            for k, v in parsed.items():
                if isinstance(v, list) and len(v) > 0:
                    scenarios = np.array(v, dtype=float)
                    break
            if scenarios is None:
                raise TypeError("MAT file does not contain a recognizable scenario matrix.")
        else:
            scenarios = np.array(parsed, dtype=float)

    mode = request.form.get('mode', 'symbolic')
    n_x = int(request.form.get('n_x', 0)) if request.form.get('n_x') else 0
    rows_A = int(request.form.get('rows_A', 0)) if request.form.get('rows_A') else 0
    lmi_size = int(request.form.get('lmi_size', 0)) if request.form.get('lmi_size') else 0

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
        if request.form.get('A_d'):
            A_d = generate_matrix_function(request.form.get('A_d'))
        else:
            if (active_tab == 'lp-tab') or (active_tab == 'qp-tab'):
                raise ValueError("\\(A(\\delta)\\) is ill-defined")
        if request.form.get('b_d'):
            b_d = generate_matrix_function(request.form.get('b_d'))
        else:
            if (active_tab == 'lp-tab') or (active_tab == 'qp-tab'):
                raise ValueError("\\(b(\\delta)\\) is ill-defined")
        if request.form.get('F_d'):
            F_d = generate_tensor_function(request.form.get('F_d'))
        else:
            if (active_tab == 'sdp-tab'):
                raise ValueError("\\(F_j(\\delta)\\) is ill-defined")
    if request.form.get('A_da'):
        A_da = generate_tensor_function(request.form.get('A_da'))
    if request.form.get('b_da'):
        b_da = generate_tensor_function(request.form.get('b_da'))

    # Hard LP/QP constraints use G, h (form fields renamed from legacy A, b).
    G = generate_matrix(request.form.get('G')) if request.form.get('G') else np.array([])
    h_vec = generate_matrix(request.form.get('h')) if request.form.get('h') else np.array([])
    c = generate_matrix(request.form.get('c')) if request.form.get('c') else np.array([])
    Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
    C = generate_matrix(request.form.get('C')) if request.form.get('C') else np.array([])
    # Hard SDP LMI collection uses E (form field renamed from legacy F).
    E = generate_tensor(request.form.get('E')) if request.form.get('E') else {}
    A_a = generate_tensor(request.form.get('A_a')) if request.form.get('A_a') else {}
    b_a = generate_matrix(request.form.get('b_a')) if request.form.get('b_a') else np.array([])

    conf = float(request.form.get('confidence')) if request.form.get('confidence') else 0.0
    # Get values from parameter boxes
    form_data = request.form.to_dict()

    rhos = np.array([float(x) for x in request.form.get('rho', 0).split(',')]) if request.form.get('rho') else np.array(
        [0.0])
    taus = np.array([float(x) for x in request.form.get('tau', 0).split(',')]) if request.form.get('tau') else np.array(
        [0.0])
    theta_bar = _parse_theta_bar(request.form.get('theta_bar'))
    p_raw = request.form.get('p', '').strip().strip("'\"")
    if not p_raw:
        p = 2
    elif p_raw.lower() in ('inf', 'infinity', 'np.inf'):
        p = np.inf
    elif p_raw.lower() == 'fro':
        p = 'fro'
    else:
        try:
            p = float(p_raw)
        except ValueError:
            raise ValueError(
                f"p (norm order) must be a single number, 'inf', or 'fro'. "
                f"You entered {p_raw!r} — comma-separated lists are not "
                f"supported for p (sweep over τ or ρ instead)."
            )

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
    selected_option = form_data.get('option')
    if selected_option:
        include_slack = selected_option in ('relaxation', 'regularization-relaxation')
    else:
        include_slack = None

    for j in range(len(taus)):
        tau = taus[j]
        for i in range(len(rhos)):
            rho = rhos[i]
            solver = form_data.get('solver', 'CLARABEL')

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
                        scenarios, A_d, b_d, G, h_vec, c, tau, theta_bar, rho, p, solver,
                        include_slack=include_slack)
                elif active_tab == 'qp-tab':
                    Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_qp(
                        scenarios, A_d, b_d, G, h_vec, c, Q, tau, theta_bar, rho, p, solver,
                        include_slack=include_slack)
                elif active_tab == 'sdp-tab':
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_sdp(
                        scenarios, F_d, E, c, Q, tau, theta_bar, rho, p, solver,
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
            constraints_list.append(len(constraints))
            risk_list.append(risk.tolist() if hasattr(risk, 'tolist') else risk)
            degeneracy_list.append(degeneracy)
            rho_list.append(rho)
            tau_list.append(tau)
            solve_time_list.append(round(solve_time, 4))
            risk_time_list.append(round(risk_time, 4))

    print(tau_list)
    print(rho_list)

    # Prepare the result dictionary with lists for each parameter
    result = {
        "form_data": form_data,
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

    print(result)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
