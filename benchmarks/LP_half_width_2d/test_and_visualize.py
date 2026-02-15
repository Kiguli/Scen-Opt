#!/usr/bin/env python3
"""
Test and Visualization for Half Width 2D (Smallest Interval) Benchmark

This script tests the smallest interval (half-width) benchmark using the
scenario approach tool and creates visualizations of the results.

The problem finds the smallest interval [center - halfwidth, center + halfwidth]
that contains all the sampled data points. This is a classic robust optimization
problem demonstrating the scenario approach.

Mathematical formulation:
    minimize    halfwidth
    subject to  center - halfwidth <= delta_i <= center + halfwidth,  for all i

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.LP import solve_lp
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


def load_vector(filepath):
    """Load a vector from a CSV file (one value per line)."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def main():
    print("=" * 60)
    print("BENCHMARK: half_width_2d (LP)")
    print("Smallest Interval Containing Data Points")
    print("=" * 60)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'smallest_interval_1d.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))

    # Parse A_d and b_d expressions
    A_d = parse_expression_matrix(os.path.join(benchmark_dir, 'A_d.csv'))
    b_d = parse_expression_matrix(os.path.join(benchmark_dir, 'b_d.csv'))

    # This benchmark has no G/h hard constraints
    G = np.array([])
    h = np.array([])

    n_scenarios = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Scenarios (data points): {n_scenarios}")
    print(f"  Decision variables: {n_vars} (center, half-width)")
    print()

    # Solve the LP using the existing solver
    print("Solving LP...")
    try:
        x, zeta, cost, N, k, constraints, degeneracy = solve_lp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            tau=0.0,
            x_ref=np.zeros((n_vars, 1)),
            rho=0.0,
            norm_type=2,
            solver='MOSEK'
        )
        status = "SUCCESS"
        print(f"Solved with MOSEK")

    except Exception as e:
        print(f"Solver failed: {e}")
        return

    # Calculate risk bounds
    beta = 0.01  # 99% confidence
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Extract solution
    center = x[0, 0]
    half_width = x[1, 0]
    interval_min = center - half_width
    interval_max = center + half_width

    # Print results
    print()
    print("-" * 60)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Decision Variables: {n_vars}")
    print(f"Optimal Cost (Half-width): {cost:.6f}")
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
    print(f"  Center: {center:.6f}")
    print(f"  Half-width: {half_width:.6f}")
    print(f"  Interval: [{interval_min:.6f}, {interval_max:.6f}]")

    # Verify all points are in the interval
    data_points = scenarios.flatten()
    in_interval = np.all((data_points >= interval_min - 1e-6) & (data_points <= interval_max + 1e-6))
    print(f"  All points in interval: {in_interval}")
    print(f"  Data range: [{np.min(data_points):.6f}, {np.max(data_points):.6f}]")

    # Identify support scenarios (those at the boundary)
    tol = 1e-4
    at_lower = np.abs(data_points - interval_min) < tol
    at_upper = np.abs(data_points - interval_max) < tol
    support_indices = np.where(at_lower | at_upper)[0]
    support_points = data_points[support_indices]

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

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(14, 10))

    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)

    # ===== Plot 1: Main Number Line Visualization =====
    ax1 = fig.add_subplot(gs[0, :])

    # Create y-jitter for better visibility of points
    np.random.seed(42)
    y_jitter = np.random.uniform(-0.15, 0.15, len(data_points))

    # Plot all data points
    non_support = ~(at_lower | at_upper)
    ax1.scatter(data_points[non_support], y_jitter[non_support], c='steelblue', s=40, alpha=0.6,
                zorder=3, label=f'Data points (N={N})', edgecolor='white', linewidth=0.5)

    # Plot support points with emphasis
    ax1.scatter(data_points[at_lower | at_upper], y_jitter[at_lower | at_upper],
                c='crimson', s=120, marker='*', zorder=4, label=f'Support scenarios (k={k})',
                edgecolor='darkred', linewidth=0.5)

    # Plot the optimal interval
    ax1.axvline(x=interval_min, color='green', linestyle='-', linewidth=2.5, alpha=0.8)
    ax1.axvline(x=interval_max, color='green', linestyle='-', linewidth=2.5, alpha=0.8)
    ax1.axvline(x=center, color='orange', linestyle='--', linewidth=2, label=f'Center: {center:.4f}')

    # Shade the interval
    ax1.axvspan(interval_min, interval_max, alpha=0.15, color='green',
                label=f'Optimal interval (width={2*half_width:.4f})')

    # Add arrows showing half-width
    arrow_y = 0.35
    ax1.annotate('', xy=(center, arrow_y), xytext=(interval_min, arrow_y),
                 arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2))
    ax1.annotate('', xy=(interval_max, arrow_y), xytext=(center, arrow_y),
                 arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2))
    ax1.text(center, arrow_y + 0.08, f'half-width = {half_width:.4f}', ha='center',
             fontsize=10, color='darkorange', fontweight='bold')

    ax1.set_xlim(min(interval_min, np.min(data_points)) - 0.05,
                  max(interval_max, np.max(data_points)) + 0.05)
    ax1.set_ylim(-0.5, 0.55)
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_title('Smallest Interval Containing Data Points\n(Scenario Approach Optimization)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_yticks([])

    # ===== Plot 2: Histogram with Interval =====
    ax2 = fig.add_subplot(gs[1, 0])

    counts, bins, patches = ax2.hist(data_points, bins=25, color='steelblue', alpha=0.7,
                                       edgecolor='white', linewidth=0.5)
    ax2.axvline(x=interval_min, color='green', linestyle='-', linewidth=2.5, label='Interval bounds')
    ax2.axvline(x=interval_max, color='green', linestyle='-', linewidth=2.5)
    ax2.axvline(x=center, color='orange', linestyle='--', linewidth=2, label='Center')
    ax2.axvline(x=np.mean(data_points), color='purple', linestyle=':', linewidth=2, label='Mean')

    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Data Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Plot 3: CDF with Interval =====
    ax3 = fig.add_subplot(gs[1, 1])

    sorted_data = np.sort(data_points)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    ax3.plot(sorted_data, cdf, 'b-', linewidth=2, label='Empirical CDF')
    ax3.axvline(x=interval_min, color='green', linestyle='-', linewidth=2.5)
    ax3.axvline(x=interval_max, color='green', linestyle='-', linewidth=2.5)
    ax3.axvspan(interval_min, interval_max, alpha=0.15, color='green')

    # Mark where support points are on CDF
    for sp in support_points:
        idx = np.searchsorted(sorted_data, sp)
        if idx < len(cdf):
            ax3.plot(sp, cdf[min(idx, len(cdf)-1)], 'r*', markersize=15)

    ax3.set_xlabel('Value', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Empirical CDF with Optimal Interval', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # ===== Plot 4: Risk Bounds Visualization =====
    ax4 = fig.add_subplot(gs[2, 0])

    # Show the relationship between k, N, and epsilon bounds
    k_range = np.arange(1, min(N//2, 20))
    eps_lowers = []
    eps_uppers = []
    for k_val in k_range:
        el, eu = quantify_risk(k_val, N, beta)
        eps_lowers.append(el)
        eps_uppers.append(eu)

    ax4.fill_between(k_range, eps_lowers, eps_uppers, alpha=0.3, color='blue', label='Risk interval')
    ax4.plot(k_range, eps_lowers, 'b--', linewidth=1.5, label='Lower bound')
    ax4.plot(k_range, eps_uppers, 'b-', linewidth=1.5, label='Upper bound')
    ax4.axvline(x=k, color='red', linestyle='-', linewidth=2, label=f'Actual k={k}')
    ax4.scatter([k], [eps_lower], c='red', s=100, zorder=5)
    ax4.scatter([k], [eps_upper], c='red', s=100, zorder=5)

    ax4.set_xlabel('Complexity (k)', fontsize=11)
    ax4.set_ylabel('Risk (epsilon)', fontsize=11)
    ax4.set_title(f'Risk Bounds vs Complexity (N={N}, beta={beta})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ===== Plot 5: Summary Box =====
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    summary_text = f"""
    SCENARIO APPROACH RESULTS
    -------------------------

    Problem: Find smallest interval
             containing all data points

    Data Statistics:
      N = {N} samples
      Range: [{np.min(data_points):.4f}, {np.max(data_points):.4f}]
      Mean: {np.mean(data_points):.4f}
      Std: {np.std(data_points):.4f}

    Optimal Solution:
      Center: {center:.6f}
      Half-width: {half_width:.6f}
      Interval: [{interval_min:.4f}, {interval_max:.4f}]

    Scenario Approach Theory:
      Complexity k = {k}
      (number of support scenarios)

      Risk bounds (99% confidence):
      epsilon in [{eps_lower:.4f}, {eps_upper:.4f}]

      Interpretation: With 99% confidence,
      at most {eps_upper*100:.1f}% of future samples
      will fall outside this interval.
    """

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(results_dir, 'visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {fig_path}")

    # Show figure (non-blocking for headless environments)
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
