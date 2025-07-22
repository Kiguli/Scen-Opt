from flask import Flask, render_template, request, jsonify
import numpy as np
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp1, solve_sdp2
from src.Risk import quantify_risk
from src.Miscellaneous import get_solvers
import ast

app = Flask(__name__)


@app.route('/')
def index():
    solvers = get_solvers()
    active_tab = request.args.get('active_tab', 'lp-tab')  # Default to 'lp-tab' if not provided
    return render_template('index.html', solvers=solvers, active_tab=active_tab)


@app.route('/solve', methods=['POST'])
def solve():
    option = request.form.get('option')
    active_tab = request.form.get('active_tab')

    # Handle scenarios file if present
    scenarios = None
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        # Example: read CSV or TXT as text, JSON as dict/list
        filename = file.filename.lower()
        if filename.endswith('.json'):
            import json
            scenarios = json.load(file)
        elif filename.endswith('.csv') or filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            scenarios = np.array([[float(cell) for cell in row.split(',')] for row in content.strip().split('\n')])
        else:
            raise TypeError("Unsupported file format. Please upload a JSON, CSV, or TXT file.")

    # creates a matrix function for A(delta) and b(delta)
    def generate_matrix_function(expr_matrix_str):
        # Convert string to list of lists
        expr_matrix = ast.literal_eval(expr_matrix_str)

        def matrix_function(delta):
            # Evaluate each expression in the matrix
            return np.array([
                [eval(expr, {"delta": delta, "math": __import__('math')}) for expr in row]
                # TODO: danger using eval on a server!
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
            raise ValueError("A(delta) is ill-defined")
    if request.form.get('b_d'):
        b_d = generate_matrix_function(request.form.get('b_d'))
    else:
        if (active_tab == 'lp-tab') or (active_tab == 'qp-tab'):
            raise ValueError("b(delta) is ill-defined")
    if request.form.get('F_d'):
        F_d = generate_tensor_function(request.form.get('F_d'))
    else:
        if (active_tab == 'sdp-tab'):
            raise ValueError("F_j(delta) is ill-defined")
    if request.form.get('A_da'):
        A_da = generate_tensor_function(request.form.get('A_da'))
    else:
        if (active_tab == 'sdp2-tab'):
            raise ValueError("A_j(delta) is ill-defined")

    A = generate_matrix(request.form.get('A')) if request.form.get('A') else np.array([])
    b = generate_matrix(request.form.get('b')) if request.form.get('b') else np.array([])
    c = generate_matrix(request.form.get('c')) if request.form.get('c') else np.array([])
    Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
    C = generate_matrix(request.form.get('C')) if request.form.get('C') else np.array([])
    F = generate_tensor(request.form.get('F')) if request.form.get('F') else np.array([])
    A_a = generate_tensor(request.form.get('A_a')) if request.form.get('A_a') else np.array([])

    conf = float(request.form.get('confidence')) if request.form.get('confidence') else 0.0
    # Get values from parameter boxes
    form_data = request.form.to_dict()
    print(form_data)

    taus = np.array([float(x) for x in request.form.get('tau', 0).split(',')]) if request.form.get('tau') else np.array(
        [0.0])
    theta_bar = np.array([float(x) for x in request.form.get('theta_bar', 0).split(',')]) if request.form.get(
        'theta_bar') else 0.0
    p = float(request.form.get('p', 0)) if request.form.get(
        'p') else 2  # TODO: add something to check for 'fro' or 'inf', and any number

    rhos = np.array([float(x) for x in request.form.get('rho', 0).split(',')]) if request.form.get('rho') else np.array(
        [0.0])

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

            # try:
            if active_tab == 'lp-tab':
                optimal_x, optimal_s, optimal_cost, N, active, constraints, degeneracy = solve_lp(
                    scenarios, A_d, b_d, A, b, c, tau, theta_bar, rho, p, solver)
            elif active_tab == 'qp-tab':
                Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
                optimal_x, optimal_s, optimal_cost, N, active, constraints, degeneracy = solve_qp(
                    scenarios, A_d, b_d, A, b, c, Q, tau, theta_bar, rho, p, solver)
            elif active_tab == 'sdp-tab':
                print("Solving SDP1")
                optimal_x, optimal_s, optimal_cost, N, active, constraints, degeneracy = solve_sdp1(
                    scenarios, F_d, F, c, Q, tau, theta_bar, rho, p, solver)
            elif active_tab == 'sdp2-tab':
                print("Solving SDP2")
                optimal_x, optimal_s, optimal_cost, N, active, constraints, degeneracy = solve_sdp2(
                    scenarios, C, A_da, A_a, tau, theta_bar, rho, p, solver)

            risk = np.array(quantify_risk(len(active), N, conf))
            e = "None"
            #TODO: put try catch block back
            #TODO: any errors are printed to screen...
        # except Exception as error:
        #    e = str(error)
        #    risk = np.array([])
        #    N, active, constraints, degeneracy = 0, [], [], False

        # Append results for this run
        optimal_x_list.append(optimal_x.tolist() if hasattr(optimal_x, 'tolist') else optimal_x)
        optimal_s_list.append(optimal_s.tolist() if hasattr(optimal_s, 'tolist') else optimal_s)
        optimal_cost_list.append(optimal_cost)
        N_list.append(N)
        active_list.append(len(active))
        constraints_list.append(len(constraints))
        risk_list.append(risk.tolist() if hasattr(risk, 'tolist') else risk)
        e_list.append(e)
        degeneracy_list.append(degeneracy)
        rho_list.append(rho)
        tau_list.append(tau)

    # Prepare the result dictionary with lists for each parameter
    result = {
        "count": len(rhos) * len(taus),
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

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
