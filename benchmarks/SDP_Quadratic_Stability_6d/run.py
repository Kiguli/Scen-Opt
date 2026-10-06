#!/usr/bin/env python3
"""
Solve the Quadratic Stability 6D (SDP) benchmark using the scenario approach.

Finds a 3x3 symmetric Lyapunov matrix P certifying stability of a coupled
oscillator LPV system with 2D parameter uncertainty.

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
    print("BENCHMARK: Quadratic_Stability_6d (SDP)")
    print("6D Quadratic Stability with 2D Parameter Uncertainty")
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
    N = len(scenarios)
    n_delta = scenarios.shape[1] if scenarios.ndim > 1 else 1
    n_vars = len(c)
    E_mats = E
    m = next(iter(F_d(scenarios[0]).values())).shape[0]

    print(f"  Scenarios: {N}")
    print(f"  Decision variables: {n_vars}")
    print(f"  Scenario dimensions: {n_delta}")
    print(f"  Matrix dimension: {m}x{m}")
    print(f"  delta[0] range: [{scenarios[:,0].min():.4f}, {scenarios[:,0].max():.4f}]")
    print(f"  delta[1] range: [{scenarios[:,1].min():.4f}, {scenarios[:,1].max():.4f}]")
    print()

    # Solve
    print(f"Solving SDP with {args.solver}...")

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
            deltas=scenarios, F_d=F_d, E=E_mats, c=c, Q=Q,
            tau=0.0, x_ref=np.zeros(n_vars), rho=0.0,
            norm_type=2, solver=args.solver
        )
        print(f"Solved with {args.solver}")
    except Exception as e:
        print(f"{args.solver} failed: {e}")
        print("Attempting with SCS solver...")
        try:
            x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
                deltas=scenarios, F_d=F_d, E=E_mats, c=c, Q=Q,
                tau=0.0, x_ref=np.zeros(n_vars), rho=0.0,
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
        print(f"Risk Bounds ({(1 - beta) * 100:g}%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds ({(1 - beta) * 100:g}%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 60)

    print()
    print("Solution:")
    for i in range(n_vars):
        print(f"  x[{i+1}] = {x[i]:.6f}")

    # Constraint analysis
    print()
    print("Constraint Analysis:")
    grid_size = 20
    d0_range = np.linspace(-1, 1, grid_size)
    d1_range = np.linspace(-1, 1, grid_size)
    max_eig_grid = np.zeros((grid_size, grid_size))

    for i, d0 in enumerate(d0_range):
        for j, d1 in enumerate(d1_range):
            F_dict = F_d([d0, d1])
            F_combined = F_dict['0']
            for idx in range(1, n_vars + 1):
                F_combined = F_combined + x[idx-1] * F_dict[str(idx)]
            eigs = np.linalg.eigvals(F_combined)
            max_eig_grid[j, i] = np.max(np.real(eigs))

    print(f"  Max eigenvalue over grid: {np.max(max_eig_grid):.6f}")
    print(f"  Min eigenvalue over grid: {np.min(max_eig_grid):.6f}")

    # Save results
    solution_data = {
        'x': x.tolist(),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'delta0_range': [float(scenarios[:,0].min()), float(scenarios[:,0].max())],
        'delta1_range': [float(scenarios[:,1].min()), float(scenarios[:,1].max())]
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
