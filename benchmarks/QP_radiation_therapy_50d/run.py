#!/usr/bin/env python3
"""
Solve the Prostate IMRT Radiation Therapy benchmark using the scenario approach QP solver.

Robust formulation (ρ=0) with regularization (τ>0): all dose constraints must be
satisfied for every patient setup scenario. Regularization keeps beamlet intensities
physically realistic.

Clinical Context: Prostate IMRT under patient setup uncertainty (±5mm in 3D).

Usage:
    python run.py
"""

import sys
import os
import json
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.QP import solve_qp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


def compile_expression_matrix(filepath):
    """Parse and compile CSV with delta[i] expressions into efficient functions."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    n_rows = len(lines)
    n_cols = len(lines[0].split(','))

    a0_matrix = np.zeros((n_rows, n_cols))
    a1_matrix = np.zeros((n_rows, n_cols))
    a2_matrix = np.zeros((n_rows, n_cols))
    a3_matrix = np.zeros((n_rows, n_cols))

    for i, line in enumerate(lines):
        if line.strip():
            cells = line.split(',')
            for j, cell in enumerate(cells):
                cell = cell.strip()
                coeffs = {0: 0.0, 1: 0.0, 2: 0.0}

                parts = re.findall(r'([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\*delta\[(\d)\]', cell)
                for coef, idx in parts:
                    coeffs[int(idx)] = float(coef)

                remaining = re.sub(r'[+-]?\d*\.?\d+(?:e[+-]?\d+)?\*delta\[\d\]', '', cell)
                remaining = remaining.replace('+-', '-').replace('--', '+').strip()
                a0 = 0.0
                if remaining:
                    try:
                        a0 = float(remaining)
                    except:
                        try:
                            a0 = eval(cell, {"delta": [0, 0, 0]})
                        except:
                            pass

                a0_matrix[i, j] = a0
                a1_matrix[i, j] = coeffs[0]
                a2_matrix[i, j] = coeffs[1]
                a3_matrix[i, j] = coeffs[2]

    def matrix_function(delta):
        return a0_matrix + a1_matrix * delta[0] + a2_matrix * delta[1] + a3_matrix * delta[2]

    return matrix_function


def compile_expression_vector(filepath):
    """Parse and compile CSV with delta[i] expressions (vector)."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    n = len(lines)
    a0_vec = np.zeros(n)
    a1_vec = np.zeros(n)
    a2_vec = np.zeros(n)
    a3_vec = np.zeros(n)

    for i, cell in enumerate(lines):
        cell = cell.strip()
        coeffs = {0: 0.0, 1: 0.0, 2: 0.0}

        parts = re.findall(r'([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\*delta\[(\d)\]', cell)
        for coef, idx in parts:
            coeffs[int(idx)] = float(coef)

        remaining = re.sub(r'[+-]?\d*\.?\d+(?:e[+-]?\d+)?\*delta\[\d\]', '', cell)
        remaining = remaining.replace('+-', '-').replace('--', '+').strip()
        a0 = 0.0
        if remaining:
            try:
                a0 = float(remaining)
            except:
                pass

        a0_vec[i] = a0
        a1_vec[i] = coeffs[0]
        a2_vec[i] = coeffs[1]
        a3_vec[i] = coeffs[2]

    def vector_function(delta):
        return (a0_vec + a1_vec * delta[0] + a2_vec * delta[1] + a3_vec * delta[2]).reshape(-1, 1)

    return vector_function


def load_matrix(filepath):
    return pd.read_csv(filepath, header=None).values


def load_vector(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def load_parameters(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.split('#')[0].strip()
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params


def load_anatomy(filepath):
    """Load anatomy from key=value format."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params


def main():
    print("=" * 70)
    print("BENCHMARK: Prostate IMRT Radiation Therapy (QP)")
    print("Robust Treatment Under Patient Setup Uncertainty")
    print("=" * 70)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'scenarios.csv'))
    Q = load_matrix(os.path.join(benchmark_dir, 'Q.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))
    G = load_matrix(os.path.join(benchmark_dir, 'G.csv'))
    h = load_vector(os.path.join(benchmark_dir, 'h.csv'))

    print("Compiling constraint expressions...")
    A_d = compile_expression_matrix(os.path.join(benchmark_dir, 'A_d.csv'))
    b_d = compile_expression_vector(os.path.join(benchmark_dir, 'b_d.csv'))

    try:
        D_nominal = load_matrix(os.path.join(benchmark_dir, 'D_nominal.csv'))
    except Exception:
        D_nominal = None

    params = load_parameters(os.path.join(benchmark_dir, 'parameters.txt'))
    anatomy = load_anatomy(os.path.join(benchmark_dir, 'anatomy.txt'))

    n_beamlets = int(anatomy.get('n_beamlets', c.shape[0]))
    n_tumor = int(anatomy.get('n_tumor', 20))
    n_oar1 = int(anatomy.get('n_oar1', 30))
    n_oar2 = int(anatomy.get('n_oar2', 30))
    n_normal = int(anatomy.get('n_normal', 20))
    n_voxels = n_tumor + n_oar1 + n_oar2 + n_normal
    dose_prescribed = anatomy.get('dose_prescribed', 70.0)
    dose_min_tumor = anatomy.get('dose_min_tumor', 66.5)
    dose_max_oar1 = anatomy.get('dose_max_oar1', 50.0)
    dose_max_oar2 = anatomy.get('dose_max_oar2', 60.0)

    tau = params.get('tau', 0.1)
    beta = 1.0 - params.get('confidence', 0.99)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Beamlets: {n_beamlets}")
    print(f"  Voxels: {n_voxels} (tumor:{n_tumor}, rectum:{n_oar1}, bladder:{n_oar2}, normal:{n_normal})")
    print(f"  Scenarios: {N}")
    print(f"  Decision variables: {n_vars}")
    print(f"  Uncertainty: ±5mm setup error (3 directions)")
    print(f"  Formulation: Robust (rho=0) with regularization (tau={tau})")
    print(f"  Prescription: {dose_prescribed} Gy, D_min: {dose_min_tumor} Gy")
    print(f"  OAR limits: rectum ≤ {dose_max_oar1} Gy, bladder ≤ {dose_max_oar2} Gy")
    print()

    # ===== SOLVE USING solve_qp (robust, ρ=0, τ>0) =====
    print("Solving QP with MOSEK...")

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_qp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            Q=Q,
            tau=tau,
            x_ref=np.zeros((n_vars, 1)),
            rho=0.0,
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
    intensities = x.flatten()

    # Compute doses using nominal dose matrix
    if D_nominal is not None:
        doses = D_nominal @ intensities
        tumor_doses = doses[:n_tumor]
        rectum_doses = doses[n_tumor:n_tumor + n_oar1]
        bladder_doses = doses[n_tumor + n_oar1:n_tumor + n_oar1 + n_oar2]
    else:
        doses = tumor_doses = rectum_doses = bladder_doses = None

    # Print results
    print()
    print("-" * 70)
    print(f"{'OPTIMIZATION RESULTS':^70}")
    print("-" * 70)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.4f}")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 70)

    print()
    print("BEAMLET INTENSITIES:")
    print("-" * 70)
    print(f"  Min: {np.min(intensities):.2f}")
    print(f"  Max: {np.max(intensities):.2f}")
    print(f"  Mean: {np.mean(intensities):.2f}")
    print(f"  Std: {np.std(intensities):.2f}")
    print(f"  Active (>0.1): {np.sum(intensities > 0.1)}/{len(intensities)}")

    if tumor_doses is not None:
        d95_tumor = np.percentile(tumor_doses, 5)
        print()
        print("DOSE STATISTICS (Nominal Position):")
        print("-" * 70)
        print(f"  Tumor (PTV):")
        print(f"    min={np.min(tumor_doses):.2f} Gy, mean={np.mean(tumor_doses):.2f} Gy, max={np.max(tumor_doses):.2f} Gy")
        print(f"    D95={d95_tumor:.2f} Gy (target: {dose_min_tumor:.1f} Gy)")
        print(f"  Rectum:")
        print(f"    min={np.min(rectum_doses):.2f} Gy, mean={np.mean(rectum_doses):.2f} Gy, max={np.max(rectum_doses):.2f} Gy")
        print(f"    (limit: {dose_max_oar1:.1f} Gy)")
        print(f"  Bladder:")
        print(f"    min={np.min(bladder_doses):.2f} Gy, mean={np.mean(bladder_doses):.2f} Gy, max={np.max(bladder_doses):.2f} Gy")
        print(f"    (limit: {dose_max_oar2:.1f} Gy)")

    print()
    print("SCENARIO APPROACH GUARANTEES:")
    print("-" * 70)
    print(f"  With {(1-beta)*100:.0f}% confidence:")
    print(f"  Probability of constraint violation: [{eps_lower:.4f}, {eps_upper:.4f}]")

    # Save results
    solution_data = {
        'intensities': intensities.tolist(),
        'objective_value': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': 0.0,
        'tau': float(tau),
        'dose_prescribed': float(dose_prescribed),
        'dose_min_tumor': float(dose_min_tumor),
    }

    if tumor_doses is not None:
        solution_data['tumor_D95'] = float(d95_tumor)
        solution_data['tumor_dose_mean'] = float(np.mean(tumor_doses))
        solution_data['rectum_dose_max'] = float(np.max(rectum_doses))
        solution_data['bladder_dose_max'] = float(np.max(bladder_doses))

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("Run plot.py to generate the paper figure.")
    print("=" * 70)


if __name__ == '__main__':
    main()
