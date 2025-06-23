import os
from logging import raiseExceptions

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
from werkzeug.utils import secure_filename
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Risk import quantify_risk, quantify_conf
from src.Miscellaneous import load_file, get_solvers
import ast

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'xlsx', 'json'}
app.secret_key = os.urandom(24)  # Set the secret key to a random 24-byte string


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    solvers = get_solvers()
    active_tab = request.args.get('active_tab', 'lp-tab')  # Default to 'lp-tab' if not provided
    return render_template('index.html', solvers=solvers, active_tab=active_tab)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # TODO: don't save locally
        session['uploaded_file'] = filename
        return redirect(url_for('index'))
    return redirect(request.url)


@app.route('/solve', methods=['POST'])
def solve():
    option = request.form.get('option')
    active_tab = request.form.get('active_tab')

    # Load scenarios from the uploaded file
    filename = session.get('uploaded_file')
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) #TODO: save these in cache rather than in folder
        scenarios = load_file(file_path)
    else:
        return jsonify({'error': 'No file uploaded'}), 400

    # creates a matrix function for A(delta) and b(delta)
    def generate_matrix_function(expr_matrix_str):
        # Convert string to list of lists
        expr_matrix = ast.literal_eval(expr_matrix_str)

        def matrix_function(delta):
            # Evaluate each expression in the matrix
            return np.array([
                [eval(expr, {"delta": delta, "math": __import__('math')}) for expr in row]
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
            for row in expr_matrix
        ])

    if request.form.get('A_d'):
        A_d = generate_matrix_function(request.form.get('A_d'))
    else:
        raise ValueError("A(delta) is ill-defined")
    if request.form.get('b_d'):
        b_d = generate_matrix_function(request.form.get('b_d'))
    else:
        raise ValueError("b(delta) is ill-defined")
    A = generate_matrix(request.form.get('A')) if request.form.get('A') else np.array([])
    b = generate_matrix(request.form.get('b')) if request.form.get('b') else np.array([])
    c = generate_matrix(request.form.get('c')) if request.form.get('c') else np.array([])
    conf = float(request.form.get('confidence'))
    # Get values from parameter boxes
    form_data = request.form.to_dict()
    print(form_data)


    tau = float(request.form.get('tau', 0)) if request.form.get('tau') else 0.0
    theta_bar = np.array([float(x) for x in request.form.get('theta_bar', 0).split(',')]) if request.form.get('theta_bar') else 0.0
    p = float(request.form.get('p', 0)) if request.form.get('p') else 2  #TODO: add something to check for 'fro' or 'inf', and any number
    rho = float(request.form.get('rho', 0)) if request.form.get('rho') else 0.0
    solver = form_data.get('solver', 'SCS')

    # Initialize uninitialized values
    optimal_x = np.array([])
    optimal_s = np.array([])
    optimal_cost = 0.0
    N = 0
    active = []
    constraints = []
    risk = 0.0
    e = "None"

    try:
        # Solve based on the active tab
        if active_tab == 'lp-tab':
            optimal_x, optimal_s, optimal_cost, N, active, constraints = solve_lp(scenarios, A_d, b_d, A, b, c, tau, theta_bar, rho, p, solver)
        elif active_tab == 'qp-tab':
            Q = generate_matrix(request.form.get('Q')) if request.form.get('Q') else np.array([])
            optimal_x, optimal_s, optimal_cost, N, active, constraints = solve_qp(scenarios, A_d, b_d, A, b, c, Q, tau, theta_bar, rho, p, solver)
        elif active_tab == 'sdp-tab':
            optimal_x, optimal_s, optimal_cost = solve_sdp(scenarios, A_d, b_d, A, b, c, tau, theta_bar, rho, p, solver)

        # Calculate risk
        risk = quantify_risk(len(active), N, conf)

    except Exception as error:
        e = str(error)  # Save the error message
        print(e)

    # Prepare the result dictionary
    result = {
        "form_data": form_data,
        "optimal_x": optimal_x.tolist() if optimal_x.size > 0 else [],
        "optimal_s": optimal_s.tolist() if optimal_s.size > 0 else [],
        "optimal_cost": optimal_cost,
        "num_deltas": N,
        "tot_con": len(constraints),
        "active_con": len(active),
        "risk": risk,
        "conf": conf,
        "errorcode": e,  # Include the error message
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
