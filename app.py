from flask import Flask, render_template, request, jsonify
import numpy as np
import json as json_module
import io
import os
import tempfile
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Risk import quantify_risk
from src.Miscellaneous import get_solvers
import ast

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
        mat = scipy.io.loadmat(file_bytes, squeeze_me=True)

        def mat_to_python(v):
            """Recursively convert MATLAB values to Python types."""
            if isinstance(v, np.ndarray):
                # Struct array → dict (strip 'M' prefix from keys used for numeric MATLAB field names)
                if v.dtype.names is not None:
                    sub = {}
                    for name in v.dtype.names:
                        key = name[1:] if name.startswith('M') and name[1:].isdigit() else name
                        field = v[name].item() if v[name].ndim == 0 else v[name]
                        sub[key] = mat_to_python(field)
                    return sub
                # 1D object arrays that were column vectors → restore to 2D
                if v.ndim == 1 and v.dtype == object:
                    return [[str(x)] for x in v.tolist()]
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
            result[k] = mat_to_python(v)
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


@app.route('/solve', methods=['POST'])
def solve():
    # Handle MOSEK license upload (temporary, not stored)
    mosek_license_path = None
    original_mosek_env = os.environ.get("MOSEKLM_LICENSE_FILE")

    try:
        if 'mosek_license' in request.files and request.files['mosek_license'].filename != '':
            license_file = request.files['mosek_license']
            temp = tempfile.NamedTemporaryFile(delete=False, suffix='.lic')
            temp.write(license_file.stream.read())
            temp.close()
            mosek_license_path = temp.name
            os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path

        return _solve_inner()
    finally:
        if mosek_license_path:
            try:
                os.unlink(mosek_license_path)
            except OSError:
                pass
            if original_mosek_env is not None:
                os.environ["MOSEKLM_LICENSE_FILE"] = original_mosek_env
            elif "MOSEKLM_LICENSE_FILE" in os.environ:
                del os.environ["MOSEKLM_LICENSE_FILE"]


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

    # creates a matrix function for A(delta) and b(delta)
    def generate_matrix_function(expr_matrix_str):
        # Convert string to list of lists
        expr_matrix = ast.literal_eval(expr_matrix_str)

        def matrix_function(delta):
            # Evaluate each expression in the matrix
            return np.array([
                [eval(expr, {"delta": delta, "math": __import__('math')}) for expr in row]
                # TODO: danger using eval on a server! Delete all characters that are not numbers, [,], or "delta"??
                for row in expr_matrix
            ])

        return matrix_function

    # creates a matrix from the values of Q
    def generate_matrix(expr_matrix_str):
        # Convert string to list of lists
        expr_matrix = ast.literal_eval(expr_matrix_str)

        # Evaluate each expression in the matrix and return as a numpy array
        return np.array([
            [eval(expr, {"math": __import__('math')}) for expr in row]
            # TODO: danger using eval on a server!
            for row in expr_matrix
        ])

        # creates a matrix function for A(delta) and b(delta)

    def generate_tensor_function(expr_matrix_str):
        # Parse the string to a dictionary of submatrices
        expr_dict = ast.literal_eval(expr_matrix_str)

        def tensor_function(delta):
            # Evaluate each submatrix for the given delta
            return {
                key: np.array([
                    [eval(expr, {"delta": delta, "math": __import__('math')}) for expr in row]
                    for row in expr_dict[key]
                ])
                for key in expr_dict
            }

        return tensor_function

    def generate_tensor(expr_matrix_str):
        # Parse the string to a dictionary
        expr_dict = ast.literal_eval(expr_matrix_str)
        # Evaluate each submatrix and store as numpy array
        tensor = {
            key: np.array([[float(cell) for cell in row] for row in expr_dict[key]])
            for key in expr_dict
        }
        return tensor

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

    A = generate_matrix(request.form.get('A')) if request.form.get('A') else np.array([])
    b = generate_matrix(request.form.get('b')) if request.form.get('b') else np.array([])
    c = generate_matrix(request.form.get('c')) if request.form.get('c') else np.array([])
    Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
    C = generate_matrix(request.form.get('C')) if request.form.get('C') else np.array([])
    F = generate_tensor(request.form.get('F')) if request.form.get('F') else {}
    A_a = generate_tensor(request.form.get('A_a')) if request.form.get('A_a') else {}
    b_a = generate_matrix(request.form.get('b_a')) if request.form.get('b_a') else np.array([])

    conf = float(request.form.get('confidence')) if request.form.get('confidence') else 0.0
    # Get values from parameter boxes
    form_data = request.form.to_dict()

    rhos = np.array([float(x) for x in request.form.get('rho', 0).split(',')]) if request.form.get('rho') else np.array(
        [0.0])
    taus = np.array([float(x) for x in request.form.get('tau', 0).split(',')]) if request.form.get('tau') else np.array(
        [0.0])
    theta_bar = np.array([float(x) for x in request.form.get('theta_bar', 0).split(',')]) if request.form.get(
        'theta_bar') else 0.0
    p = float(request.form.get('p', 0)) if request.form.get(
        'p') else 2  # TODO: add something to check for 'fro' or 'inf', and any number

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

    # update confidence based on number of tau and rho
    conf = conf / (len(taus) * len(rhos))

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

            try:
                if active_tab == 'lp-tab':
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_lp(
                        scenarios, A_d, b_d, A, b, c, tau, theta_bar, rho, p, solver)
                elif active_tab == 'qp-tab':
                    Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_qp(
                        scenarios, A_d, b_d, A, b, c, Q, tau, theta_bar, rho, p, solver)
                elif active_tab == 'sdp-tab':
                    optimal_x, optimal_s, optimal_cost, N, complexity, constraints, degeneracy = solve_sdp(
                        scenarios, F_d, F, c, Q, tau, theta_bar, rho, p, solver)

                risk = np.array(quantify_risk(complexity, N, conf))
            except Exception as error:
                e = str(error)
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
        "conf": 1 - conf,
        "tau_": tau_list,
        "rho_": rho_list,
        "errorcode": e_list,
        "degeneracy": degeneracy_list
    }

    print(result)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
