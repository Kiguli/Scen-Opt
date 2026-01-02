#!/usr/bin/env python3
"""
Efficient benchmark runner for Prostate Brachytherapy using MOSEK.
Pre-compiles expressions for faster evaluation.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import re

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

    # Parse coefficients once
    a0_matrix = np.zeros((n_rows, n_cols))
    a1_matrix = np.zeros((n_rows, n_cols))
    a2_matrix = np.zeros((n_rows, n_cols))
    a3_matrix = np.zeros((n_rows, n_cols))

    for i, line in enumerate(lines):
        if line.strip():
            cells = line.split(',')
            for j, cell in enumerate(cells):
                cell = cell.strip()
                # Parse: a0 + a1*delta[0] + a2*delta[1] + a3*delta[2]
                a0 = 0.0
                coeffs = {0: 0.0, 1: 0.0, 2: 0.0}

                # Find all terms
                parts = re.findall(r'([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\*delta\[(\d)\]', cell)
                for coef, idx in parts:
                    coeffs[int(idx)] = float(coef)

                # Find constant term
                remaining = re.sub(r'[+-]?\d*\.?\d+(?:e[+-]?\d+)?\*delta\[\d\]', '', cell)
                remaining = remaining.replace('+-', '-').replace('--', '+').strip()
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
            if ':' in line and not line.startswith('#'):
                key, val = line.split(':', 1)
                try:
                    params[key.strip()] = float(val.strip())
                except ValueError:
                    params[key.strip()] = val.strip()
    return params


def load_anatomy(filepath):
    structures = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':')
                name = parts[0].strip()
                details = parts[1].strip()
                idx_range = details.split('(')[0].strip()
                start, end = idx_range.split('-')
                structures[name] = {
                    'start': int(start),
                    'end': int(end) + 1,
                    'type': 'target' if 'target' in details.lower() else 'OAR'
                }
    return structures


def main():
    print("=" * 70)
    print("BENCHMARK: Prostate Brachytherapy (TROTS Dataset)")
    print("Robust Radiation Therapy Under Implantation Uncertainty")
    print("=" * 70)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

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
    try:
        anatomy = load_anatomy(os.path.join(benchmark_dir, 'anatomy.txt'))
    except:
        anatomy = None

    n_beamlets = int(params.get('n_beamlets', c.shape[0]))
    n_voxels = int(params.get('n_voxels', 100))
    target_dose = params.get('target_dose', 145.0)
    D_min = params.get('D_min', 0.95 * target_dose)

    oar_limits = {}
    for key, val in params.items():
        if '_max' in key:
            oar_name = key.replace('_max', '')
            oar_limits[oar_name] = val

    rho = params.get('rho', 1.0)
    tau = params.get('tau', 0.1)
    beta = 1.0 - params.get('confidence', 0.99)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Beamlets (dwell positions): {n_beamlets}")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 3 (implant positioning)")
    print(f"  Decision variables: {n_vars}")
    print(f"  Target dose: {target_dose} Gy")
    print(f"  Minimum target dose (D95): {D_min} Gy")
    if oar_limits:
        print(f"  OAR limits: {oar_limits}")
    print()

    # ===== SOLVE USING solve_qp =====
    print("Solving QP...")

    # Try MOSEK first, then others
    solvers_to_try = ['MOSEK', 'OSQP', 'SCS', 'ECOS']
    solved = False

    for solver_name in solvers_to_try:
        try:
            print(f"  Trying {solver_name}...")
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
                rho=rho,
                norm_type=2,
                solver=solver_name
            )
            status = f"SUCCESS ({solver_name})"
            print(f"  Solved successfully with {solver_name}")
            solved = True
            break
        except Exception as e:
            print(f"  {solver_name} failed: {str(e)[:80]}...")
            continue

    if not solved:
        print("All solvers failed!")
        return

    # Calculate risk bounds
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    intensities = x.flatten()
    zeta_val = np.max(zeta) if zeta is not None else 0

    # Compute doses
    if D_nominal is not None:
        doses = D_nominal @ x.flatten()
        structure_doses = {}
        if anatomy:
            for name, info in anatomy.items():
                structure_doses[name] = doses[info['start']:info['end']]
    else:
        doses = None
        structure_doses = {}

    # Print results
    print()
    print("-" * 70)
    print(f"{'OPTIMIZATION RESULTS':^70}")
    print("-" * 70)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.4f}")
    print(f"Max Constraint Violation (zeta): {zeta_val:.6f}")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 70)

    print()
    print("BEAMLET INTENSITIES (Dwell Times):")
    print("-" * 70)
    print(f"  Min: {np.min(intensities):.2f}")
    print(f"  Max: {np.max(intensities):.2f}")
    print(f"  Mean: {np.mean(intensities):.2f}")
    print(f"  Std: {np.std(intensities):.2f}")
    print(f"  Active (>0.1): {np.sum(intensities > 0.1)}/{len(intensities)}")

    if structure_doses:
        print()
        print("DOSE STATISTICS (Nominal Position):")
        print("-" * 70)
        for name, dose_vals in structure_doses.items():
            if len(dose_vals) > 0:
                print(f"  {name}:")
                print(f"    min={np.min(dose_vals):.2f} Gy, "
                      f"mean={np.mean(dose_vals):.2f} Gy, "
                      f"max={np.max(dose_vals):.2f} Gy")
                if 'prostate' in name.lower():
                    d95 = np.percentile(dose_vals, 5)
                    print(f"    D95={d95:.2f} Gy (target: {D_min:.1f} Gy)")

    print()
    print("SCENARIO APPROACH GUARANTEES:")
    print("-" * 70)
    print(f"  With {(1-beta)*100:.0f}% confidence:")
    print(f"  Probability of constraint violation ε ∈ [{eps_lower:.4f}, {eps_upper:.4f}]")
    print(f"  This means: in >{(1-eps_upper)*100:.1f}% of implant procedures,")
    print(f"  dose constraints will be satisfied.")

    # Save results
    solution_data = {
        'intensities': intensities.tolist(),
        'objective_value': float(cost),
        'max_zeta': float(zeta_val),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': float(rho),
        'tau': float(tau),
        'target_dose': float(target_dose),
        'D_min': float(D_min),
    }

    for name, dose_vals in structure_doses.items():
        if len(dose_vals) > 0:
            solution_data[f'{name}_dose_min'] = float(np.min(dose_vals))
            solution_data[f'{name}_dose_mean'] = float(np.mean(dose_vals))
            solution_data[f'{name}_dose_max'] = float(np.max(dose_vals))
            if 'prostate' in name.lower():
                solution_data[f'{name}_D95'] = float(np.percentile(dose_vals, 5))

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
