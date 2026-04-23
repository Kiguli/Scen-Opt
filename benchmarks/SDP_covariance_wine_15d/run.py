#!/usr/bin/env python3
"""
Solve the Robust Covariance Estimation benchmark using the scenario approach SDP solver.

Robust formulation (rho=0): the estimated covariance Sigma must dominate every
subsample covariance in the PSD sense: Sigma >= S_sub for all scenarios.
The scenario approach guarantees that with high probability, Sigma will also
dominate the covariance of a new random subsample.

SDP formulation:
  Decision: x in R^15 (entries of 5x5 symmetric Sigma)
  Scenario LMI (5x5): S_sub(delta) - Sigma << 0  (covariance domination)
  Hard LMI (5x5): -Sigma << 0  (enforces Sigma >> 0)
  Objective: minimize trace(Sigma)

Usage:
    python run.py
"""

import sys
import os
import json
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
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(x) for x in line.split(',')] for line in lines if line.strip()])


def load_vector(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([float(line.strip()) for line in lines if line.strip()])


def load_parameters(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())
    return params


def main():
    print("=" * 60)
    print("BENCHMARK: Robust Covariance Estimation (SDP)")
    print("UCI Wine — Minimum-Trace Covariance Domination")
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
    cov_full = load_matrix(os.path.join(data_dir, 'cov_full.csv'))
    free_entries = load_matrix(os.path.join(data_dir, 'free_entries.csv')).astype(int)

    with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    n_vars = len(c)
    N = len(scenarios)
    p = int(free_entries.max()) + 1  # matrix dimension (5x5)

    params = load_parameters(os.path.join(benchmark_dir, 'parameters.txt'))
    beta = 1.0 - params.get('confidence', 0.999999)

    print(f"  Features: {p} ({', '.join(feature_names)})")
    print(f"  Decision variables: {n_vars} (entries of {p}x{p} symmetric Sigma)")
    print(f"  Scenario LMI dimension: {p}x{p} (covariance domination)")
    print(f"  Hard LMI dimension: {p}x{p} (Sigma >> 0)")
    print(f"  Scenarios: {N}")
    print(f"  Formulation: Robust (rho=0)")
    print()

    # Solve
    print("Solving SDP with MOSEK...")

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
            deltas=scenarios,
            F_d=F_d,
            E=E,
            c=c,
            Q=Q,
            tau=0.0,
            x_ref=np.zeros(n_vars),
            rho=0.0,
            norm_type=2,
            solver='MOSEK'
        )
        status = "SUCCESS"
        print(f"Solved with MOSEK")
    except Exception as e:
        print(f"MOSEK failed: {e}")
        print("Attempting with SCS solver...")
        try:
            x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp(
                deltas=scenarios,
                F_d=F_d,
                E=E,
                c=c,
                Q=Q,
                tau=0.0,
                x_ref=np.zeros(n_vars),
                rho=0.0,
                norm_type=2,
                solver='SCS'
            )
            status = "SUCCESS"
            print(f"Solved with SCS")
        except Exception as e2:
            print(f"SCS also failed: {e2}")
            return

    # Risk bounds
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Reconstruct Sigma from x using free_entries
    Sigma = np.zeros((p, p))
    for idx, (i, j) in enumerate(free_entries):
        if i == j:
            Sigma[i, i] = x[idx]
        else:
            Sigma[i, j] = x[idx]
            Sigma[j, i] = x[idx]

    # Check PSD
    eigvals = np.linalg.eigvalsh(Sigma)
    is_psd = np.all(eigvals >= -1e-8)

    # Compare with full-sample covariance
    frobenius_error = np.linalg.norm(Sigma - cov_full, 'fro')
    entry_errors = np.abs(Sigma - cov_full)
    max_entry_error = np.max(entry_errors)
    mean_entry_error = np.mean(entry_errors)

    # Print results
    print()
    print("-" * 60)
    print(f"{'OPTIMIZATION RESULTS':^60}")
    print("-" * 60)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.4f}")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 60)

    print()
    print("ESTIMATED COVARIANCE MATRIX (Sigma):")
    print("-" * 60)
    print(np.array2string(Sigma, precision=3, suppress_small=True))

    print()
    print("FULL-SAMPLE COVARIANCE (reference):")
    print("-" * 60)
    print(np.array2string(cov_full, precision=3, suppress_small=True))

    print()
    print("COMPARISON:")
    print("-" * 60)
    print(f"  Frobenius error: {frobenius_error:.4f}")
    print(f"  Max entry error: {max_entry_error:.4f}")
    print(f"  Mean entry error: {mean_entry_error:.4f}")
    print(f"  PSD: {is_psd}")
    print(f"  Eigenvalues: {eigvals.round(4)}")

    # Save results
    solution_data = {
        'covariance_estimated': Sigma.tolist(),
        'covariance_full_sample': cov_full.tolist(),
        'frobenius_error': float(frobenius_error),
        'max_entry_error': float(max_entry_error),
        'mean_entry_error': float(mean_entry_error),
        'objective_value': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': 0.0,
        'eigenvalues': eigvals.tolist(),
        'feature_names': feature_names,
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'Sigma_solution.csv'), Sigma, delimiter=',')
    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print("Run plot.py to generate the paper figure.")
    print("=" * 60)


if __name__ == '__main__':
    main()
