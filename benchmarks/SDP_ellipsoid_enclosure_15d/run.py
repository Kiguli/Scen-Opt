#!/usr/bin/env python3
"""
Solve the Minimum Enclosing Ellipsoid benchmark using the scenario approach SDP solver.

Robust formulation (rho=0): every sampled data point must lie inside the ellipsoid.
The scenario approach provides probabilistic guarantees that unseen data points
from the same distribution will also be contained.

SDP formulation:
  Decision: x in R^15 (entries of 5x5 symmetric shape matrix P)
  Scenario LMI (1x1): (x_i - c)' P (x_i - c) - 1 <= 0  (point containment)
  Hard LMI (5x5): -P << 0  (enforces P >> 0)
  Objective: maximize trace(P) = minimize -trace(P)  (tightest ellipsoid)

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
    print("BENCHMARK: Minimum Enclosing Ellipsoid (SDP)")
    print("Breast Cancer Wisconsin — Tightest Containment Ellipsoid")
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
    center = load_vector(os.path.join(data_dir, 'center.csv'))
    data_all = load_matrix(os.path.join(data_dir, 'data_standardized.csv'))
    free_entries = load_matrix(os.path.join(data_dir, 'free_entries.csv')).astype(int)

    with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    n_vars = len(c)
    N = len(scenarios)
    p = int(free_entries.max()) + 1  # matrix dimension (5x5)

    params = load_parameters(os.path.join(benchmark_dir, 'parameters.txt'))
    beta = 1.0 - params.get('confidence', 0.999999)

    print(f"  Features: {p} ({', '.join(feature_names)})")
    print(f"  Decision variables: {n_vars} (entries of {p}x{p} symmetric P)")
    print(f"  Scenario LMI dimension: 1x1 (point containment)")
    print(f"  Hard LMI dimension: {p}x{p} (P >> 0)")
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

    # Reconstruct P from x using free_entries
    P = np.zeros((p, p))
    for idx, (i, j) in enumerate(free_entries):
        if i == j:
            P[i, i] = x[idx]
        else:
            P[i, j] = x[idx]
            P[j, i] = x[idx]

    # Check PSD and properties
    eigvals = np.linalg.eigvalsh(P)
    is_psd = np.all(eigvals >= -1e-8)

    # Ellipsoid volume proportional to 1/sqrt(det(P))
    det_P = np.prod(np.maximum(eigvals, 1e-12))
    trace_P = np.trace(P)

    # Check containment on scenarios and full dataset
    containment_scen = np.zeros(N)
    for i in range(N):
        v = scenarios[i] - center
        containment_scen[i] = v @ P @ v

    n_contained = 0
    n_total = len(data_all)
    containment_all = np.zeros(n_total)
    for i in range(n_total):
        v = data_all[i] - center
        containment_all[i] = v @ P @ v
        if containment_all[i] <= 1.0 + 1e-8:
            n_contained += 1

    containment_rate = n_contained / n_total

    # Print results
    print()
    print("-" * 60)
    print(f"{'OPTIMIZATION RESULTS':^60}")
    print("-" * 60)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.4f} (= -trace(P))")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 60)

    print()
    print("ELLIPSOID PROPERTIES:")
    print("-" * 60)
    print(f"  trace(P): {trace_P:.4f}")
    print(f"  det(P): {det_P:.6f}")
    print(f"  Eigenvalues: {eigvals.round(4)}")
    print(f"  PSD: {is_psd}")
    print(f"  Semi-axis lengths: {(1.0/np.sqrt(np.maximum(eigvals, 1e-12))).round(3)}")

    print()
    print("CONTAINMENT:")
    print("-" * 60)
    print(f"  Sampled points: {N}/{N} (max value: {containment_scen.max():.4f})")
    print(f"  Full dataset: {n_contained}/{n_total} contained ({containment_rate:.1%})")
    print(f"  Max containment value (full): {containment_all.max():.4f}")

    # Save results
    solution_data = {
        'P_matrix': P.tolist(),
        'center': center.tolist(),
        'trace_P': float(trace_P),
        'det_P': float(det_P),
        'eigenvalues': eigvals.tolist(),
        'containment_rate_full': float(containment_rate),
        'objective_value': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': 0.0,
        'feature_names': feature_names,
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'P_solution.csv'), P, delimiter=',')
    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print("Run plot.py to generate the paper figure.")
    print("=" * 60)


if __name__ == '__main__':
    main()
