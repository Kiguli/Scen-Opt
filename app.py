import os
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import cvxpy as cp
from werkzeug.utils import secure_filename
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Miscellaneous import load_file,get_solvers

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'xlsx', 'json'}
app.secret_key = os.urandom(24)  # Set the secret key to a random 24-byte string


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    solvers = get_solvers()
    return render_template('index.html',solvers=solvers)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        scenarios = load_file(file_path)
    else:
        return redirect(url_for('index'))

    def A(deltas: np.ndarray):
        return np.array([[deltas[0], deltas[1]]])

    def b(deltas: np.ndarray):
        return np.array([[deltas[2]]])

    c = np.array([-5, -3])

    # Get values from parameter boxes
    form_data = request.form.to_dict()
    print(form_data)
    tau = float(request.form.get('tau', 0)) if request.form.get('tau') else 0.0
    theta_bar = float(request.form.get('theta_bar', 0)) if request.form.get('theta_bar') else 0.0
    p = float(request.form.get('p', 0)) if request.form.get('p') else 2 # add something to check for 'fro' or 'inf'
    rho = float(request.form.get('rho', 0)) if request.form.get('rho') else 0.0
    solver = form_data.get('solver', 'SCS')

    if active_tab == 'lp-tab':
        optimal_x, optimal_s, optimal_cost = solve_lp(scenarios, A, b, c, tau, theta_bar, rho, p, solver)
    elif active_tab == 'qp-tab':
        optimal_x, optimal_s, optimal_cost = solve_qp(scenarios, A, b, c, tau, theta_bar, rho, p, solver)
    elif active_tab == 'sdp-tab':
        optimal_x, optimal_s, optimal_cost = solve_sdp(scenarios, A, b, c, tau, theta_bar, rho, p, solver)

    return render_template('index.html', result=(form_data, optimal_x, optimal_s, optimal_cost))


if __name__ == '__main__':
    app.run(debug=True)
