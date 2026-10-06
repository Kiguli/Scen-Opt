#!/usr/bin/env python3
"""
Solve the LPV Stability 3D (SDP) benchmark using the scenario approach.

Finds a common Lyapunov matrix P that ensures stability of an LPV system
across all sampled parameter values delta in [-0.22, 1].

Usage:
    python run.py [--solver SOLVER]
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks._loader import load_symbolic_program
from src.SDP import solve_sdp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


def parse_expression_matrix(filepath):
    """Parse a CSV file containing expressions with delta[i] terms."""
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
    """Load a matrix from a CSV file."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    matrix = []
    for line in lines:
        if line.strip():
            row = [float(x.strip()) for x in line.split(',')]
            matrix.append(row)
    return np.array(matrix)


def load_vector(filepath):
    """Load a vector from a CSV file (one value per line)."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([float(line.strip()) for line in lines if line.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default='MOSEK',
                        help='CVXPY solver, e.g. MOSEK or CLARABEL (default: MOSEK)')
    args = parser.parse_args()

    print("=" * 60)
    print("BENCHMARK: LPV_stability_3d (SDP)")
    print("Common Lyapunov Function for LPV System")
    print("=" * 60)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("Loading data files...")
    scenarios = load_file(os.path.join(data_dir, 'scenarios.csv'))
    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c = prog['c']
    Q = prog.get('Q', np.array([]))
    F_d = prog['F_d']
    E = prog.get('E', {})
    # Keep scenarios as an (N, 1) array: solve_sdp and the symbolic F_d
    # expressions index each sample as delta[0], so flattening to scalars
    # would break evaluation.
    if scenarios.ndim == 1:
        scenarios = scenarios.reshape(-1, 1)

    N = len(scenarios)
    n_vars = Q.shape[0]
    m = next(iter(F_d(scenarios[0]).values())).shape[0]

    print(f"  Scenarios (parameter samples): {N}")
    print(f"  Decision variables: {n_vars}")
    print(f"  Matrix dimension: {m}x{m}")
    print(f"  Parameter range: [{scenarios.min():.4f}, {scenarios.max():.4f}]")
    print()

    # Solve
    print(f"Solving SDP with {args.solver}...")
    rho_value = 1.0

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
            deltas=scenarios, F_d=F_d, E=E, c=c, Q=Q,
            tau=0.0, x_ref=np.zeros(n_vars), rho=rho_value,
            norm_type=2, solver=args.solver
        )
        print(f"Solved with {args.solver}")
    except Exception as e:
        print(f"{args.solver} failed: {e}")
        print("Attempting with SCS solver...")
        try:
            x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
                deltas=scenarios, F_d=F_d, E=E, c=c, Q=Q,
                tau=0.0, x_ref=np.zeros(n_vars), rho=rho_value,
                norm_type=2, solver='SCS'
            )
            print(f"Solved with SCS")
        except Exception as e2:
            print(f"SCS also failed: {e2}")
            return

    # Risk bounds
    beta = 1e-6
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Results
    print()
    print("-" * 60)
    print(f"Optimal Cost: {cost:.6f}")
    print(f"Complexity (k): {k}")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 60)

    print()
    print("Solution (Lyapunov Parameters):")
    for i in range(n_vars):
        print(f"  x[{i+1}] = {x[i]:.6f}")

    # Stability check
    print()
    print("Stability Analysis:")
    test_deltas = [scenarios.min(), 0, scenarios.max()]
    for delta in test_deltas:
        F_result = F_d([delta])
        F_combined = F_result['0'] + x[0]*F_result['1'] + x[1]*F_result['2'] + x[2]*F_result['3']
        eigs = np.linalg.eigvals(F_combined)
        max_eig = np.max(np.real(eigs))
        print(f"  delta={delta:.4f}: max eigenvalue = {max_eig:.6f} {'(stable)' if max_eig < 0 else '(unstable)'}")

    # Save results
    solution_data = {
        'x': x.tolist(),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'parameter_range': [float(scenarios.min()), float(scenarios.max())]
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
