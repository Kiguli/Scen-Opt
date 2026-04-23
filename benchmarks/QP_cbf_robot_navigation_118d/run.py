#!/usr/bin/env python3
"""
Solve the CBF Robot Navigation benchmark using the scenario approach QP solver.

Usage:
    python run.py
"""

import sys
import os
import json
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks._loader import load_symbolic_program
from src.QP import solve_qp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


def parse_expression_matrix(filepath):
    """Parse CSV with delta[i] expressions into a function."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    expr_matrix = []
    for line in lines:
        if line.strip():
            row = [cell.strip() for cell in line.split(',')]
            expr_matrix.append(row)

    def matrix_function(delta):
        result = []
        for row in expr_matrix:
            result_row = []
            for expr in row:
                val = eval(expr, {"delta": delta, "math": __import__('math')})
                result_row.append(val)
            result.append(result_row)
        return np.array(result)

    return matrix_function


def load_matrix(filepath):
    """Load a numeric matrix from CSV."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    matrix = []
    for line in lines:
        if line.strip():
            row = [float(x.strip()) for x in line.split(',')]
            matrix.append(row)
    return np.array(matrix)


def load_vector(filepath):
    """Load a numeric vector from CSV."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def main():
    print("=" * 65)
    print("BENCHMARK: QP Robot Navigation")
    print("=" * 65)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    scenarios = load_file(os.path.join(data_dir, 'scenarios.csv'))
    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c, Q, A_d, b_d = prog['c'], prog['Q'], prog['A_d'], prog['b_d']
    G, h = prog.get('G', np.array([])), prog.get('h', np.array([]))
    params = {}
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())

    rho = params.get('rho', 0.0)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.999999)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Scenarios (N): {N}")
    print(f"  Decision variables: {n_vars} (80 states + 38 controls)")
    print(f"  Wall constraints per scenario: {A_d(scenarios[0]).shape[0]}")
    print(f"  Hard constraints: {G.shape[0]}")
    print(f"  rho = {rho}, tau = {tau}")
    print()

    # Solve QP
    print("Solving QP with MOSEK...")
    try:
        x, zeta, cost, N, k, constraints, degeneracy = solve_qp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            Q=Q,
            tau=tau,
            x_ref=np.zeros((n_vars, 1)),
            rho=rho,
            norm_type=2,
            solver='MOSEK'
        )
        status = "SUCCESS"
        print(f"Solved successfully")
    except Exception as e:
        print(f"MOSEK failed: {e}")
        print("Trying CLARABEL...")
        try:
            x, zeta, cost, N, k, constraints, degeneracy = solve_qp(
                deltas=scenarios,
                A_d=A_d,
                b_d=b_d,
                G=G,
                h=h,
                c=c,
                Q=Q,
                tau=tau,
                x_ref=np.zeros((n_vars, 1)),
                rho=rho,
                norm_type=2,
                solver='CLARABEL'
            )
            status = "SUCCESS"
            print(f"Solved successfully with CLARABEL")
        except Exception as e2:
            print(f"Solver failed: {e2}")
            return

    # Calculate risk bounds
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Parse solution into trajectory
    T_HORIZON = 20
    DT = 0.4
    x_flat = x.flatten()

    n_states = 4 * T_HORIZON
    positions = x_flat[:n_states].reshape(T_HORIZON, 4)
    px = positions[:, 0]
    py = positions[:, 1]
    vx = positions[:, 2]
    vy = positions[:, 3]

    n_controls = 2 * (T_HORIZON - 1)
    controls = x_flat[n_states:n_states + n_controls].reshape(T_HORIZON - 1, 2)
    ax_ctrl = controls[:, 0]
    ay_ctrl = controls[:, 1]

    path_length = np.sum(np.sqrt(np.diff(px)**2 + np.diff(py)**2))

    # Wall configuration
    wall = {
        'x_min': 3.0, 'x_max': 7.0,
        'y_min': 0.0, 'y_max': 3.0,
        'face': 'top', 'face_y': 3.0,
        'name': 'Wall',
    }
    R_ROBOT = 0.2
    D_MARGIN = 0.1
    D_SAFE_WALL = R_ROBOT + D_MARGIN

    # Compute wall clearance at each timestep in the wall's x-range
    min_clearance = float('inf')
    for t in range(T_HORIZON):
        if px[t] < wall['x_min'] or px[t] > wall['x_max']:
            continue
        clearance = py[t] - wall['face_y'] - D_SAFE_WALL
        min_clearance = min(min_clearance, clearance)

    if min_clearance == float('inf'):
        min_clearance = float('nan')

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Decision Variables: {n_vars}")
    print(f"Optimal Cost: {cost:.6f}")
    print(f"Complexity (k): {k} support constraints")
    conf_pct = (1 - beta) * 100
    if degeneracy:
        print(f"Risk Bounds ({conf_pct:g}%): [unreliable, {eps_upper:.4f}]")
        print(f"Degeneracy: True (lower bound unreliable)")
    else:
        print(f"Risk Bounds ({conf_pct:g}%): [{eps_lower:.4f}, {eps_upper:.4f}]")
        print(f"Degeneracy: False")
    print("-" * 65)

    print()
    print("TRAJECTORY SUMMARY:")
    print(f"  Start: ({px[0]:.3f}, {py[0]:.3f})")
    print(f"  Goal:  ({px[-1]:.3f}, {py[-1]:.3f})")
    print(f"  Path length: {path_length:.3f} m")
    print(f"  Total time: {T_HORIZON * DT:.1f} s")

    print()
    print("WALL CLEARANCE (negative = violation):")
    status_str = "OK" if min_clearance >= 0 else "VIOLATED"
    print(f"  {wall['name']}: {min_clearance:.3f} m ({status_str})")

    # Save results
    solution_data = {
        'trajectory_x': px.tolist(),
        'trajectory_y': py.tolist(),
        'velocity_x': vx.tolist(),
        'velocity_y': vy.tolist(),
        'control_ax': ax_ctrl.tolist(),
        'control_ay': ay_ctrl.tolist(),
        'path_length': float(path_length),
        'total_time': float(T_HORIZON * DT),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'min_clearance': float(min_clearance),
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    print()
    print("=" * 65)
    print("Benchmark complete!")
    print("Run plot.py to generate the paper figure.")
    print("=" * 65)


if __name__ == '__main__':
    main()
