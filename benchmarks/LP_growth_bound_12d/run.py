#!/usr/bin/env python3
"""
Solve the Growth Bound benchmark using the scenario approach LP solver.

Usage:
    python run.py [--mini]
"""

import sys
import os
import json
import argparse
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks._loader import load_symbolic_program
from src.LP import solve_lp
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mini', action='store_true', help='Use mini dataset (100 samples)')
    args = parser.parse_args()

    print("=" * 65)
    print("BENCHMARK: Data-Driven Growth Bound (LP)")
    print("=" * 65)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    if args.mini:
        scenarios = load_file(os.path.join(data_dir, 'growth_bound_mini.csv'))
        print("  Using mini dataset")
    else:
        scenarios = load_file(os.path.join(data_dir, 'growth_bound.csv'))
        print("  Using full dataset")

    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c, A_d, b_d = prog['c'], prog['A_d'], prog['b_d']
    G, h = prog.get('G', np.array([])), prog.get('h', np.array([]))

    # Load parameters
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

    print(f"  Scenarios (state transitions): {N}")
    print(f"  Decision variables: {n_vars} (9 growth matrix + 3 bias)")
    print()

    # Solve LP
    print("Solving LP with MOSEK...")
    try:
        x, zeta, cost, N, k, constraints, degeneracy = solve_lp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            tau=tau,
            x_ref=np.zeros((n_vars, 1)),
            rho=rho,
            norm_type=2,
            solver='MOSEK'
        )
        status = "SUCCESS"
        print(f"Solved successfully")
    except Exception as e:
        print(f"Solver failed: {e}")
        return

    # Calculate risk bounds
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    solution = x.flatten()

    # Extract growth matrix M and bias b
    M = np.array([
        [solution[0], solution[1], solution[2]],
        [solution[3], solution[4], solution[5]],
        [solution[6], solution[7], solution[8]]
    ])
    b_vec = np.array([solution[9], solution[10], solution[11]])

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
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
    print("GROWTH MATRIX M*:")
    for i in range(3):
        row_str = "  ".join(f"{M[i,j]:>8.4f}" for j in range(3))
        print(f"  [{row_str}]")

    print()
    print("BIAS VECTOR b*:")
    print(f"  [{b_vec[0]:.4f}, {b_vec[1]:.4f}, {b_vec[2]:.4f}]")

    # Save results
    solution_data = {
        'solution': solution.tolist(),
        'growth_matrix': M.tolist(),
        'bias_vector': b_vec.tolist(),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy)
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
