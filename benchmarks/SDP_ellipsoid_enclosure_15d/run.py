#!/usr/bin/env python3
"""
Solve the Minimum Enclosing Ellipsoid benchmark using the scenario approach SDP1 solver.

Robust formulation (rho=0): every sampled data point must lie inside the ellipsoid.
The scenario approach provides probabilistic guarantees that unseen data points
from the same distribution will also be contained.

SDP1 formulation:
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

from src.SDP import solve_sdp1
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


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
    coeffs = load_matrix(os.path.join(data_dir, 'coeffs.csv'))
    center = load_vector(os.path.join(data_dir, 'center.csv'))
    Q = load_matrix(os.path.join(data_dir, 'Q.csv'))
    c = load_vector(os.path.join(data_dir, 'c.csv'))
    data_all = load_matrix(os.path.join(data_dir, 'data_standardized.csv'))
    basis_matrices = np.load(os.path.join(data_dir, 'basis_matrices.npy'))
    free_entries = load_matrix(os.path.join(data_dir, 'free_entries.csv')).astype(int)

    with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    n_vars = len(c)
    N = len(scenarios)
    p = basis_matrices.shape[1]  # matrix dimension (5x5)

    # Load E matrices (hard constraint: -P << 0)
    E = {}
    for i in range(n_vars + 1):
        E[str(i)] = load_matrix(os.path.join(data_dir, f'E_{i}.csv'))

    params = load_parameters(os.path.join(benchmark_dir, 'parameters.txt'))
    beta = 1.0 - params.get('confidence', 0.99)

    print(f"  Features: {p} ({', '.join(feature_names)})")
    print(f"  Decision variables: {n_vars} (entries of {p}x{p} symmetric P)")
    print(f"  Scenario LMI dimension: 1x1 (point containment)")
    print(f"  Hard LMI dimension: {p}x{p} (P >> 0)")
    print(f"  Scenarios: {N}")
    print(f"  Formulation: Robust (rho=0)")
    print()

    # Build F_d function for scenario LMI
    # Constraint: (x_i - c)' P (x_i - c) - 1 <= 0
    # F_0(delta) = [[-1]], F_k(delta) = [[coeff_k(delta)]]
    # where coeff_k = (delta - center)' B_k (delta - center)
    def F_d(delta):
        v = delta - center
        result = {'0': np.array([[-1.0]])}
        for k in range(n_vars):
            coeff_k = v @ basis_matrices[k] @ v
            result[str(k + 1)] = np.array([[coeff_k]])
        return result

    # Solve
    print("Solving SDP with MOSEK...")

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp1(
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
            x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp1(
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

    # Reconstruct P from x
    P = np.zeros((p, p))
    for idx in range(n_vars):
        P += x[idx] * basis_matrices[idx]

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
