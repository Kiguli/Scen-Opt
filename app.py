import os
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import cvxpy as cp
from werkzeug.utils import secure_filename
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp
from src.Miscellaneous import load_file
from tests.LP import norm_type

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'xlsx', 'json'}
app.secret_key = os.urandom(24)  # Set the secret key to a random 24-byte string


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


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
        print(scenarios)
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

    T = 0.0
    P = 0.0
    Theta_Bar = 0.0
    norm_type = 2

    if option == 'robust':
        print(T)
        print(P)
        print(Theta_Bar)
        print(norm_type)
    elif option == 'robust-regularization':
        T = float(form_data.get('tau', 0.0))
        print(T)
        Theta_Bar = float(form_data.get('Theta_Bar', 0.0))
        print(Theta_Bar)
        norm_type = int(form_data.get('p', 2))
        print(norm_type)
        P = 0.0
        print(P)
    elif option == 'robust-relaxation':
        T = 0.0
        Theta_Bar = 0.0
        norm_type = 2
        P = float(form_data.get('rho', 0.0))
    elif option == 'robust-regularization-relaxation':
        P = float(form_data.get('rho', 0.0))
        Theta_Bar = float(form_data.get('theta_bar', 0.0))
        T = float(form_data.get('tau', 0.0))
        norm_type = int(form_data.get('p', 2))

    solver = cp.SCS

    if active_tab == 'lp-tab':
        optimal_x, optimal_s, optimal_cost = solve_lp(scenarios, A, b, c, T, P, norm_type, solver)
    elif active_tab == 'qp-tab':
        optimal_x, optimal_s, optimal_cost = solve_qp(scenarios, A, b, c, T, P, norm_type, solver)
    elif active_tab == 'sdp-tab':
        optimal_x, optimal_s, optimal_cost = solve_sdp(scenarios, A, b, c, T, P, norm_type, solver)

    return render_template('index.html', result=(optimal_x, optimal_s, optimal_cost))


if __name__ == '__main__':
    app.run(debug=True)
