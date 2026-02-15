#!/usr/bin/env python3
"""
Solve the Iris SVM Classification benchmark using the scenario approach QP solver.

Usage:
    python run.py
"""

import sys
import os
import json
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.QP import solve_qp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


def parse_expression_row(filepath):
    """Parse a single-row CSV with delta[i] expressions into a function."""
    with open(filepath, 'r') as f:
        line = f.read().strip()

    expr_list = [cell.strip() for cell in line.split(',')]

    def row_function(delta):
        result = []
        for expr in expr_list:
            val = eval(expr, {"delta": delta, "math": __import__('math')})
            result.append(val)
        return np.array([result])

    return row_function


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
    print("BENCHMARK: Iris SVM Classification (QP)")
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
    Q = load_matrix(os.path.join(data_dir, 'Q.csv'))

    A_d = parse_expression_row(os.path.join(data_dir, 'A_d.csv'))

    # b_d is constant = 1
    def b_d(delta):
        return np.array([[1.0]])

    # No hard constraints
    G = np.array([])
    h = np.array([])

    # Extract features and labels
    X = scenarios[:, :2]  # petal_length, petal_width
    y = scenarios[:, 2]   # labels: -1 (setosa) or +1 (others)

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

    print(f"  Data points (N): {N}")
    print(f"  Decision variables: {n_vars} (w1, w2, b)")
    print(f"  Class -1 (setosa): {int(np.sum(y == -1))} points")
    print(f"  Class +1 (others): {int(np.sum(y == 1))} points")
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
        print(f"Solver failed: {e}")
        return

    # Calculate risk bounds
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Extract hyperplane parameters
    w1, w2, b = x[0, 0], x[1, 0], x[2, 0]
    w_norm = np.sqrt(w1**2 + w2**2)
    margin = 2.0 / w_norm if w_norm > 0 else float('inf')

    # Classification accuracy
    predictions = np.sign(X @ np.array([w1, w2]) + b)
    accuracy = np.mean(predictions == y)

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Data Points (N): {N}")
    print(f"Optimal Cost: {cost:.6f}")
    print(f"Complexity (k): {k} support constraints")
    conf_pct = (1 - beta) * 100
    if degeneracy:
        print(f"Risk Bounds ({conf_pct:g}%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds ({conf_pct:g}%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 65)

    print()
    print("OPTIMAL HYPERPLANE:")
    print(f"  w1 (petal length): {w1:.6f}")
    print(f"  w2 (petal width):  {w2:.6f}")
    print(f"  b (bias):          {b:.6f}")
    print(f"  ||w||:             {w_norm:.6f}")
    print(f"  Margin (2/||w||):  {margin:.6f}")
    print(f"  Classifier: {b:.2f} = {w1:.2f}*x1 + {w2:.2f}*x2")
    print(f"  Training accuracy: {accuracy*100:.1f}%")

    # Save results
    solution_data = {
        'w1': float(w1),
        'w2': float(w2),
        'b': float(b),
        'w_norm': float(w_norm),
        'margin': float(margin),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'accuracy': float(accuracy),
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
