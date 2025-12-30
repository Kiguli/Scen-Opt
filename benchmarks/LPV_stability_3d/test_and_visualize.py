#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and Visualization for LPV_stability_3d (SDP) Benchmark

This script tests the LPV (Linear Parameter-Varying) stability benchmark using
the scenario approach tool and creates visualizations of the results.

The problem finds a common Lyapunov matrix P that ensures stability of an LPV
system across all sampled parameter values δ ∈ [-0.22, 1].

LPV System: dx/dt = A(δ)x where A(δ) = A₀ + δ·A₁
    A₀ = [[0, 1], [-2, -1]]  (stable damped oscillator)
    A₁ = [[0, 0], [0.3, 0.1]]  (parameter-varying perturbation)

Decision Variables: x = [p₁₁, p₁₂, p₂₂] (symmetric Lyapunov matrix P elements)
    P = [[p₁₁, p₁₂], [p₁₂, p₂₂]]

Mathematical formulation:
    minimize    (1/2) x'Qx + c'x
    subject to  A(δ)'P + PA(δ) ≼ 0  (stability, for each scenario δ)
                -P ≼ 0               (positive definiteness)

Where ≼ denotes matrix inequality (negative semidefinite).

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.SDP import solve_sdp1
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


def parse_expression_matrix(filepath):
    """
    Parse a CSV file containing expressions with delta[i] terms.
    Returns a function that evaluates the matrix for a given delta.
    """
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
    print("=" * 60)
    print("BENCHMARK: LPV_stability_3d (SDP)")
    print("Common Lyapunov Function for LPV System")
    print("=" * 60)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load scenario data
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'scheduling_variable_samples.csv'))

    # Flatten if needed (single column)
    if scenarios.ndim == 2 and scenarios.shape[1] == 1:
        scenarios = scenarios.flatten()

    # Load objective function
    Q = load_matrix(os.path.join(benchmark_dir, 'Q.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))

    # Load F matrices (scenario-dependent)
    F_0_func = parse_expression_matrix(os.path.join(benchmark_dir, 'F_0.csv'))
    F_1_func = parse_expression_matrix(os.path.join(benchmark_dir, 'F_1.csv'))
    F_2_func = parse_expression_matrix(os.path.join(benchmark_dir, 'F_2.csv'))
    F_3_func = parse_expression_matrix(os.path.join(benchmark_dir, 'F_3.csv'))

    # Load E matrices (hard constraints)
    E_0 = load_matrix(os.path.join(benchmark_dir, 'E_0.csv'))
    E_1 = load_matrix(os.path.join(benchmark_dir, 'E_1.csv'))
    E_2 = load_matrix(os.path.join(benchmark_dir, 'E_2.csv'))
    E_3 = load_matrix(os.path.join(benchmark_dir, 'E_3.csv'))

    N = len(scenarios)
    n_vars = Q.shape[0]
    m = F_0_func([0]).shape[0]  # Matrix dimension

    print(f"  Scenarios (parameter samples): {N}")
    print(f"  Decision variables: {n_vars}")
    print(f"  Matrix dimension: {m}x{m}")
    print(f"  Parameter range: [{scenarios.min():.4f}, {scenarios.max():.4f}]")
    print()

    # Create F_d function for solve_sdp1
    def F_d(delta):
        # delta may be a scalar or array - ensure it's always an array for indexing
        if np.isscalar(delta):
            delta = np.array([delta])
        elif isinstance(delta, np.ndarray):
            delta = delta.flatten()
            if delta.size == 1:
                delta = np.array([delta.item()])
        return {
            '0': F_0_func(delta),
            '1': F_1_func(delta),
            '2': F_2_func(delta),
            '3': F_3_func(delta)
        }

    # Create E dictionary for hard constraints
    E = {
        '0': E_0,
        '1': E_1,
        '2': E_2,
        '3': E_3
    }

    # Solve the SDP
    print("Solving SDP with MOSEK...")

    # Use rho > 0 to allow slack variables (soft constraints)
    # This finds the minimum constraint violation
    rho_value = 1.0

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp1(
            deltas=scenarios,
            F_d=F_d,
            E=E,
            c=c,
            Q=Q,
            tau=0.0,
            x_ref=np.zeros(n_vars),
            rho=rho_value,
            norm_type=2,
            solver='MOSEK'
        )
        status = "SUCCESS"
        print(f"Solved with MOSEK")
        print(f"Optimal cost: {cost:.6f}")

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
                rho=rho_value,
                norm_type=2,
                solver='SCS'
            )
            status = "SUCCESS (SCS)"
            print(f"Solved with SCS")
            print(f"Optimal cost: {cost:.6f}")
        except Exception as e2:
            print(f"SCS also failed: {e2}")
            return

    # Calculate risk bounds
    beta = 0.01  # 99% confidence
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Print results
    print()
    print("-" * 60)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Decision Variables: {n_vars}")
    print(f"Optimal Cost: {cost:.6f}")
    print(f"Complexity (k): {k}")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
        print(f"Degeneracy: True (lower bound unreliable)")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
        print(f"Degeneracy: False")
    print("-" * 60)

    print()
    print("Solution (Lyapunov Parameters):")
    for i in range(n_vars):
        print(f"  x[{i+1}] = {x[i]:.6f}")

    # Construct and analyze resulting matrices
    print()
    print("Stability Analysis:")

    # Check stability at a few sample points
    test_deltas = [scenarios.min(), 0, scenarios.max()]
    for delta in test_deltas:
        F_result = F_d([delta])
        F_combined = F_result['0'] + x[0]*F_result['1'] + x[1]*F_result['2'] + x[2]*F_result['3']
        eigs = np.linalg.eigvals(F_combined)
        max_eig = np.max(np.real(eigs))
        print(f"  δ={delta:.4f}: max eigenvalue = {max_eig:.6f} {'(stable)' if max_eig < 0 else '(unstable)'}")

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

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(16, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    # ===== Plot 1: Parameter Space (Sampled δ values) =====
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.scatter(scenarios, np.zeros_like(scenarios), c='steelblue', s=30, alpha=0.6)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='δ=0')
    ax1.set_xlim(scenarios.min() - 0.1, scenarios.max() + 0.1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Scheduling Parameter δ', fontsize=11)
    ax1.set_title(f'Sampled Parameters (N={N})', fontsize=12, fontweight='bold')
    ax1.set_yticks([])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')

    # ===== Plot 2: Parameter Histogram =====
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.hist(scenarios, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    ax2.axvline(x=scenarios.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {scenarios.mean():.3f}')
    ax2.set_xlabel('Scheduling Parameter δ', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Parameter Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: Eigenvalue vs Parameter =====
    ax3 = fig.add_subplot(gs[0, 2])

    delta_range = np.linspace(scenarios.min(), scenarios.max(), 100)
    max_eigs = []
    for delta in delta_range:
        F_result = F_d([delta])
        F_combined = F_result['0'] + x[0]*F_result['1'] + x[1]*F_result['2'] + x[2]*F_result['3']
        eigs = np.linalg.eigvals(F_combined)
        max_eigs.append(np.max(np.real(eigs)))

    ax3.plot(delta_range, max_eigs, 'b-', linewidth=2, label='Max eigenvalue of F(δ,x)')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Stability boundary')
    ax3.fill_between(delta_range, max_eigs, 0, where=np.array(max_eigs) < 0,
                     alpha=0.2, color='green', label='Stable region')
    ax3.set_xlabel('Scheduling Parameter δ', fontsize=11)
    ax3.set_ylabel('Maximum Eigenvalue', fontsize=11)
    ax3.set_title('Stability Over Parameter Range', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ===== Plot 4: Decision Variable Values =====
    ax4 = fig.add_subplot(gs[1, 0])

    bars = ax4.bar(['p₁₁', 'p₁₂', 'p₂₂'], x, color=['steelblue', 'coral', 'seagreen'], alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Optimal Decision Variables', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, x):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01*np.sign(val),
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

    # ===== Plot 5: Lyapunov Matrix P Heatmap =====
    ax5 = fig.add_subplot(gs[1, 1])

    # Reconstruct P from solution
    P_opt = np.array([[x[0], x[1]], [x[1], x[2]]])

    im = ax5.imshow(P_opt, cmap='Blues', aspect='equal')
    ax5.set_title('Lyapunov Matrix P', fontsize=12, fontweight='bold')
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # Add value annotations
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f'{P_opt[i,j]:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if P_opt[i,j] > 0.8 else 'black')

    # ===== Plot 6: Risk Bounds Visualization =====
    ax6 = fig.add_subplot(gs[1, 2])

    k_range = np.arange(1, min(N//2, 30))
    eps_lowers = []
    eps_uppers = []
    for k_val in k_range:
        el, eu = quantify_risk(k_val, N, beta)
        eps_lowers.append(el)
        eps_uppers.append(eu)

    ax6.fill_between(k_range, eps_lowers, eps_uppers, alpha=0.3, color='blue', label='Risk interval')
    ax6.plot(k_range, eps_lowers, 'b--', linewidth=1.5, label='Lower bound')
    ax6.plot(k_range, eps_uppers, 'b-', linewidth=1.5, label='Upper bound')
    ax6.axvline(x=k, color='red', linestyle='-', linewidth=2, label=f'Actual k={k}')
    ax6.scatter([k], [eps_lower], c='red', s=100, zorder=5)
    ax6.scatter([k], [eps_upper], c='red', s=100, zorder=5)

    ax6.set_xlabel('Complexity (k)', fontsize=11)
    ax6.set_ylabel('Risk (epsilon)', fontsize=11)
    ax6.set_title(f'Risk Bounds (N={N}, β={beta})', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ===== Plot 7: -P Matrix (should be negative semidefinite) =====
    ax7 = fig.add_subplot(gs[2, 0])

    neg_P = -P_opt
    eigs_negP = np.linalg.eigvals(neg_P)

    im7 = ax7.imshow(neg_P, cmap='RdBu_r', aspect='equal')
    ax7.set_title(f'-P Matrix (hard constraint)\nλ_max={np.max(np.real(eigs_negP)):.4f}', fontsize=12, fontweight='bold')
    ax7.set_xticks([0, 1])
    ax7.set_yticks([0, 1])
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            ax7.text(j, i, f'{neg_P[i,j]:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if abs(neg_P[i,j]) > 0.3 else 'black')

    # ===== Plot 8: Eigenvalue Histogram across all scenarios =====
    ax8 = fig.add_subplot(gs[2, 1])

    all_max_eigs = []
    for delta in scenarios:
        F_result = F_d([delta])
        F_combined = F_result['0'] + x[0]*F_result['1'] + x[1]*F_result['2'] + x[2]*F_result['3']
        eigs = np.linalg.eigvals(F_combined)
        all_max_eigs.append(np.max(np.real(eigs)))

    ax8.hist(all_max_eigs, bins=25, color='steelblue', alpha=0.7, edgecolor='white')
    ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stability boundary')
    ax8.axvline(x=np.max(all_max_eigs), color='orange', linestyle='-', linewidth=2,
                label=f'Max: {np.max(all_max_eigs):.4f}')
    ax8.set_xlabel('Maximum Eigenvalue', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('Eigenvalue of A\'P+PA\nAcross All Scenarios', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # ===== Plot 9: Summary Box =====
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    stability_status = "STABLE" if np.max(all_max_eigs) < 1e-6 else "CHECK REQUIRED"

    # Reconstruct Lyapunov matrix P from solution
    P_matrix = np.array([[x[0], x[1]], [x[1], x[2]]])
    P_eigs = np.linalg.eigvals(P_matrix)
    P_status = "VALID (P > 0)" if all(P_eigs > 0) else "INVALID"

    summary_text = f"""
    LPV STABILITY RESULTS
    ---------------------

    Problem: Common Lyapunov function
    for LPV system dx/dt = A(δ)x

    System Matrices:
      A₀ = [[0,1],[-2,-1]]
      A₁ = [[0,0],[0.3,0.1]]

    Lyapunov Matrix P:
      P = [[{x[0]:.4f}, {x[1]:.4f}]
           [{x[1]:.4f}, {x[2]:.4f}]]
      P eigenvalues: {P_eigs[0]:.4f}, {P_eigs[1]:.4f}
      Status: {P_status}

    Scenario Approach:
      N = {N} sampled parameters
      δ ∈ [{scenarios.min():.3f}, {scenarios.max():.3f}]

    Stability Analysis (A'P+PA):
      Max eigenvalue: {np.max(all_max_eigs):.6f}
      Status: {stability_status}

    Risk Bounds (99% conf.):
      k = {k} support constraints
      ε ∈ [{eps_lower:.4f}, {eps_upper:.4f}]

    Interpretation: With 99% confidence,
    stability holds for at least
    {(1-eps_upper)*100:.1f}% of parameter values.
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='teal'))

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(results_dir, 'visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {fig_path}")

    # Show figure (non-blocking)
    try:
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    except Exception:
        plt.close()

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
