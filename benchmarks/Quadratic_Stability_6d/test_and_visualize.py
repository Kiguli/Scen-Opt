#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and Visualization for Quadratic_Stability_6d (SDP) Benchmark

This script tests the 6-dimensional quadratic stability benchmark using
the scenario approach tool and creates visualizations of the results.

The problem finds a Lyapunov matrix P (parameterized by 6 decision variables)
that ensures stability across a 2D uncertain parameter space.

Decision Variables: x = [x1, x2, x3, x4, x5, x6] (6 variables)
    These parameterize a 2x2 symmetric Lyapunov-like matrix structure.

Scenario Parameters: delta = [delta[0], delta[1]] in [-1, 1] x [-1, 1]
    Two-dimensional uncertainty set (square).

Mathematical formulation:
    minimize    c'x  (minimize x6)
    subject to  F_0 + sum(xi * F_i) <= 0  (for each scenario delta)
                E_0 + sum(xi * E_i) <= 0  (hard constraint)

Where <= denotes matrix inequality (negative semidefinite).

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

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
    print("BENCHMARK: Quadratic_Stability_6d (SDP)")
    print("6D Quadratic Stability with 2D Parameter Uncertainty")
    print("=" * 60)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load scenario data
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'quadratic_stability_data.csv'))

    N = len(scenarios)
    n_delta = scenarios.shape[1] if scenarios.ndim > 1 else 1

    # Load objective function
    Q = load_matrix(os.path.join(benchmark_dir, 'Q.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))
    n_vars = len(c)

    # Load F matrices (scenario-dependent)
    F_funcs = []
    for i in range(n_vars + 1):
        F_funcs.append(parse_expression_matrix(os.path.join(benchmark_dir, f'F_{i}.csv')))

    # Load E matrices (hard constraints)
    E_mats = {}
    for i in range(n_vars + 1):
        E_mats[str(i)] = load_matrix(os.path.join(benchmark_dir, f'E_{i}.csv'))

    m = F_funcs[0]([0, 0]).shape[0]  # Matrix dimension

    print(f"  Scenarios: {N}")
    print(f"  Decision variables: {n_vars}")
    print(f"  Scenario dimensions: {n_delta}")
    print(f"  Matrix dimension: {m}x{m}")
    print(f"  delta[0] range: [{scenarios[:,0].min():.4f}, {scenarios[:,0].max():.4f}]")
    print(f"  delta[1] range: [{scenarios[:,1].min():.4f}, {scenarios[:,1].max():.4f}]")
    print()

    # Create F_d function for solve_sdp1
    def F_d(delta):
        if np.isscalar(delta):
            delta = np.array([delta, 0])
        elif isinstance(delta, np.ndarray):
            delta = delta.flatten()
        return {str(i): F_funcs[i](delta) for i in range(n_vars + 1)}

    # Solve the SDP
    print("Solving SDP with MOSEK...")

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp1(
            deltas=scenarios,
            F_d=F_d,
            E=E_mats,
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
        print(f"Optimal cost: {cost:.6f}")

    except Exception as e:
        print(f"MOSEK failed: {e}")
        print("Attempting with SCS solver...")
        try:
            x, zeta, cost, N_out, k, constraints, degeneracy = solve_sdp1(
                deltas=scenarios,
                F_d=F_d,
                E=E_mats,
                c=c,
                Q=Q,
                tau=0.0,
                x_ref=np.zeros(n_vars),
                rho=0.0,
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
    print("Solution:")
    for i in range(n_vars):
        print(f"  x[{i+1}] = {x[i]:.6f}")

    # Analyze constraint satisfaction across parameter space
    print()
    print("Constraint Analysis:")

    # Evaluate max eigenvalue at grid of points
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
    if np.max(max_eig_grid) > 0:
        print("  WARNING: Positive eigenvalues found!")
    else:
        print("  All constraints satisfied (all eigenvalues <= 0)")

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

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    # ===== Plot 1: Scenario Scatter Plot =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(scenarios[:,0], scenarios[:,1], c='steelblue', s=10, alpha=0.5)
    ax1.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor='red',
                            linewidth=2, linestyle='--', label='Uncertainty set'))
    ax1.set_xlabel('delta[0]', fontsize=11)
    ax1.set_ylabel('delta[1]', fontsize=11)
    ax1.set_title(f'Sampled Scenarios (N={N})', fontsize=12, fontweight='bold')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Scenario Density =====
    ax2 = fig.add_subplot(gs[0, 1])
    h = ax2.hist2d(scenarios[:,0], scenarios[:,1], bins=20, cmap='Blues')
    ax2.set_xlabel('delta[0]', fontsize=11)
    ax2.set_ylabel('delta[1]', fontsize=11)
    ax2.set_title('Scenario Density', fontsize=12, fontweight='bold')
    plt.colorbar(h[3], ax=ax2, label='Count')

    # ===== Plot 3: Max Eigenvalue Heatmap =====
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(max_eig_grid, extent=[-1, 1, -1, 1], origin='lower',
                    cmap='RdBu_r', aspect='equal')
    ax3.contour(d0_range, d1_range, max_eig_grid, levels=[0], colors='black', linewidths=2)
    ax3.set_xlabel('delta[0]', fontsize=11)
    ax3.set_ylabel('delta[1]', fontsize=11)
    ax3.set_title('Max Eigenvalue of F(delta, x*)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Max eigenvalue')

    # ===== Plot 4: Decision Variables =====
    ax4 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, n_vars))
    bars = ax4.bar([f'x{i+1}' for i in range(n_vars)], x, color=colors, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Optimal Decision Variables', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, x):
        offset = 0.02 * (max(x) - min(x)) * np.sign(val) if val != 0 else 0.02
        ax4.text(bar.get_x() + bar.get_width()/2, val + offset,
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    # ===== Plot 5: Risk Bounds =====
    ax5 = fig.add_subplot(gs[1, 1])

    k_range = np.arange(1, min(N//10, 50))
    eps_lowers = []
    eps_uppers = []
    for k_val in k_range:
        el, eu = quantify_risk(k_val, N, beta)
        eps_lowers.append(el)
        eps_uppers.append(eu)

    ax5.fill_between(k_range, eps_lowers, eps_uppers, alpha=0.3, color='blue', label='Risk interval')
    ax5.plot(k_range, eps_lowers, 'b--', linewidth=1.5, label='Lower bound')
    ax5.plot(k_range, eps_uppers, 'b-', linewidth=1.5, label='Upper bound')
    if k > 0:
        ax5.axvline(x=k, color='red', linestyle='-', linewidth=2, label=f'Actual k={k}')
        ax5.scatter([k], [eps_lower], c='red', s=100, zorder=5)
        ax5.scatter([k], [eps_upper], c='red', s=100, zorder=5)

    ax5.set_xlabel('Complexity (k)', fontsize=11)
    ax5.set_ylabel('Risk (epsilon)', fontsize=11)
    ax5.set_title(f'Risk Bounds (N={N}, beta={beta})', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ===== Plot 6: Eigenvalue Distribution =====
    ax6 = fig.add_subplot(gs[1, 2])

    # Compute max eigenvalue at each scenario
    all_max_eigs = []
    for delta in scenarios:
        F_dict = F_d(delta)
        F_combined = F_dict['0']
        for idx in range(1, n_vars + 1):
            F_combined = F_combined + x[idx-1] * F_dict[str(idx)]
        eigs = np.linalg.eigvals(F_combined)
        all_max_eigs.append(np.max(np.real(eigs)))

    ax6.hist(all_max_eigs, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Feasibility boundary')
    ax6.axvline(x=np.max(all_max_eigs), color='orange', linestyle='-', linewidth=2,
                label=f'Max: {np.max(all_max_eigs):.4f}')
    ax6.set_xlabel('Maximum Eigenvalue', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Eigenvalue Distribution\nAcross Scenarios', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    # ===== Plot 7: 3D Surface of Max Eigenvalue =====
    ax7 = fig.add_subplot(gs[2, 0], projection='3d')
    D0, D1 = np.meshgrid(d0_range, d1_range)
    ax7.plot_surface(D0, D1, max_eig_grid, cmap='RdBu_r', alpha=0.8, edgecolor='none')
    ax7.contour(D0, D1, max_eig_grid, levels=[0], colors='black', linewidths=2,
                offset=np.min(max_eig_grid))
    ax7.set_xlabel('delta[0]', fontsize=10)
    ax7.set_ylabel('delta[1]', fontsize=10)
    ax7.set_zlabel('Max Eig', fontsize=10)
    ax7.set_title('Stability Surface', fontsize=12, fontweight='bold')

    # ===== Plot 8: Marginal Distributions =====
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(scenarios[:,0], bins=25, alpha=0.5, label='delta[0]', color='steelblue')
    ax8.hist(scenarios[:,1], bins=25, alpha=0.5, label='delta[1]', color='coral')
    ax8.set_xlabel('Parameter Value', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('Parameter Marginal Distributions', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # ===== Plot 9: Summary Box =====
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    feasible_status = "FEASIBLE" if np.max(all_max_eigs) <= 1e-6 else "CHECK REQUIRED"

    summary_text = f"""
    QUADRATIC STABILITY RESULTS
    ---------------------------

    Problem: 6D Quadratic Stability
    with 2D parameter uncertainty

    Uncertainty Set:
      delta[0] in [-1, 1]
      delta[1] in [-1, 1]

    Scenario Approach:
      N = {N} sampled scenarios
      k = {k} support constraints

    Optimal Solution:
      x1 = {x[0]:.4f}
      x2 = {x[1]:.4f}
      x3 = {x[2]:.4f}
      x4 = {x[3]:.4f}
      x5 = {x[4]:.4f}
      x6 = {x[5]:.4f}

    Constraint Analysis:
      Max eigenvalue: {np.max(all_max_eigs):.6f}
      Status: {feasible_status}

    Risk Bounds (99% conf.):
      epsilon in [{eps_lower:.4f}, {eps_upper:.4f}]

    Interpretation: With 99% confidence,
    stability holds for at least
    {(1-eps_upper)*100:.1f}% of parameters.
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen'))

    plt.tight_layout()

    fig_path = os.path.join(results_dir, 'visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {fig_path}")

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
