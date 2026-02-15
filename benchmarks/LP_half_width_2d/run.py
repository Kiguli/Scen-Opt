#!/usr/bin/env python3
"""
Solve the Smallest Enclosing Interval benchmark using the scenario approach LP solver.

Usage:
    python run.py
"""

import sys
import os
import json
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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


def load_vector(filepath):
    """Load a numeric vector from CSV."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def main():
    print("=" * 65)
    print("BENCHMARK: Smallest Enclosing Interval (LP)")
    print("=" * 65)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    scenarios = load_file(os.path.join(data_dir, 'scenarios.csv'))
    c = load_vector(os.path.join(data_dir, 'c.csv'))

    A_d = parse_expression_matrix(os.path.join(data_dir, 'A_d.csv'))
    b_d = parse_expression_matrix(os.path.join(data_dir, 'b_d.csv'))

    # No hard constraints for this problem
    G = np.array([])
    h = np.array([])

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Data points (N): {N}")
    print(f"  Decision variables: {n_vars} (center, half-width)")
    print()

    # Load parameters
    params = {}
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())

    rho = params.get('rho', 0.0)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.99)

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

    # Extract solution
    center = x[0, 0]
    half_width = x[1, 0]
    interval_min = center - half_width
    interval_max = center + half_width

    # Identify support scenarios
    data_points = scenarios.flatten()
    tol = 1e-4
    at_lower = np.abs(data_points - interval_min) < tol
    at_upper = np.abs(data_points - interval_max) < tol
    support_indices = np.where(at_lower | at_upper)[0]

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Data Points (N): {N}")
    print(f"Optimal Half-width: {half_width:.6f}")
    print(f"Complexity (k): {k} support constraints")
    conf_pct = (1 - beta) * 100
    if degeneracy:
        print(f"Risk Bounds ({conf_pct:g}%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds ({conf_pct:g}%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 65)

    print()
    print("OPTIMAL INTERVAL:")
    print(f"  Center:     {center:.6f}")
    print(f"  Half-width: {half_width:.6f}")
    print(f"  Interval:   [{interval_min:.6f}, {interval_max:.6f}]")

    # Verify
    in_interval = np.all((data_points >= interval_min - 1e-6) &
                         (data_points <= interval_max + 1e-6))
    print(f"  All points contained: {in_interval}")
    print(f"  Data range: [{np.min(data_points):.6f}, {np.max(data_points):.6f}]")
    print(f"  Support scenarios: {len(support_indices)} points at boundaries")

    # Save results
    solution_data = {
        'center': float(center),
        'half_width': float(half_width),
        'interval_min': float(interval_min),
        'interval_max': float(interval_max),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'support_indices': support_indices.tolist()
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
