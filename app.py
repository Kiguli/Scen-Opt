from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import cvxpy as cp
from src.LP import solve_lp
from src.QP import solve_qp
from src.SDP import solve_sdp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'xlsx', 'json'}

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
        return redirect(url_for('index'))
    return redirect(request.url)

@app.route('/solve', methods=['POST'])
def solve():
    option = request.form.get('option')
    parameter = request.form.get('parameter')
    active_tab = request.form.get('active_tab')

    # Example parameters
    scenarios = np.array([[2,1,-100],[3,2,-120],[-1,0,0],[0,-1,0]])
    def A(deltas:np.ndarray):
        return np.array([[deltas[0],deltas[1]]])
    def b(deltas:np.ndarray):
        return np.array([[deltas[2]]])
    c = np.array([-5,-3])
    T = 0.0
    P = 0.0
    norm_type = 2
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