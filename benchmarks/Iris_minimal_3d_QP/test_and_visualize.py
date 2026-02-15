#!/usr/bin/env python3
"""
Test and Visualization for Iris_minimal_3d (SVM Classification) Benchmark

This script tests the SVM classification benchmark using the scenario approach
tool and creates visualizations of the results.

The problem finds an optimal linear classifier (hyperplane) that separates
Iris-setosa from other Iris species using petal length and width.

Mathematical formulation (soft-margin SVM):
    minimize    ||w||^2  (margin maximization)
    subject to  y_i * (w'x_i + b) >= 1,  for all i

Where w = [w1, w2] is the normal vector and b is the bias.

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.QP import solve_qp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk


def parse_expression_row(filepath):
    """
    Parse a single-row CSV file containing expressions with delta[i] terms.
    Returns a function that evaluates the row for a given delta.
    """
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
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def main():
    print("=" * 60)
    print("BENCHMARK: Iris_minimal_3d (QP)")
    print("SVM Classification: Iris-setosa vs Others")
    print("=" * 60)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'iris_minimal.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))
    Q = load_matrix(os.path.join(benchmark_dir, 'Q.csv'))

    # Parse A_d expression (single row)
    A_d = parse_expression_row(os.path.join(benchmark_dir, 'A_d.csv'))

    # b_d is constant = 1
    def b_d(delta):
        return np.array([[1.0]])

    # No hard constraints G/h for this problem
    G = np.array([])
    h = np.array([])

    # Extract features and labels
    X = scenarios[:, :2]  # petal_length, petal_width
    y = scenarios[:, 2]   # labels: -1 (setosa) or +1 (others)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Scenarios (data points): {N}")
    print(f"  Decision variables: {n_vars} (w1, w2, b)")
    print(f"  Class -1 (setosa): {np.sum(y == -1)} points")
    print(f"  Class +1 (others): {np.sum(y == 1)} points")
    print()

    # ===== SOLVE USING solve_qp =====
    print("Solving QP (SVM) with MOSEK...")

    try:
        x, zeta, cost, N, k, constraints, degeneracy = solve_qp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            Q=Q,
            tau=0.0,
            x_ref=np.zeros((n_vars, 1)),
            rho=0.0,
            norm_type=2,
            solver='MOSEK'
        )
        status = "SUCCESS"
        print(f"Solved with MOSEK")
        print(f"Optimal cost: {cost:.6f}")

        # Identify support indices from dual values
        tol = 1e-6
        support_indices = []
        for i, constr in enumerate(constraints[:-1] if not (G.size == 0) else constraints):
            if hasattr(constr, 'dual_value') and constr.dual_value is not None:
                if np.max(np.abs(constr.dual_value)) > tol:
                    support_indices.append(i)

    except Exception as e:
        print(f"Solver failed: {e}")
        return

    # Calculate risk bounds
    beta = 0.01  # 99% confidence
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Extract hyperplane parameters
    w1, w2, b = x[0, 0], x[1, 0], x[2, 0]
    w_norm = np.sqrt(w1**2 + w2**2)
    margin = 2.0 / w_norm if w_norm > 0 else float('inf')

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
    print("Solution (Hyperplane):")
    print(f"  w1 (petal_length coef): {w1:.6f}")
    print(f"  w2 (petal_width coef):  {w2:.6f}")
    print(f"  b (bias):               {b:.6f}")
    print(f"  ||w||:                  {w_norm:.6f}")
    print(f"  Margin (2/||w||):       {margin:.6f}")
    print(f"  Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")

    # Verify classification accuracy
    predictions = np.sign(X @ np.array([w1, w2]) + b)
    accuracy = np.mean(predictions == y)
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
        'degeneracy': bool(degeneracy),
        'support_indices': support_indices
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(15, 10))

    # Create grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # ===== Plot 1: Main Classification Plot with Decision Boundary =====
    ax1 = fig.add_subplot(gs[0, :2])

    # Create meshgrid for decision boundary visualization
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))

    # Decision function values
    Z = w1 * xx1 + w2 * xx2 + b

    # Plot decision regions
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    ax1.contourf(xx1, xx2, np.sign(Z), cmap=cmap_light, alpha=0.3)

    # Plot margin boundaries (where w'x + b = ±1)
    ax1.contour(xx1, xx2, Z, levels=[-1, 0, 1], colors=['blue', 'green', 'red'],
                linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

    # Plot data points
    setosa_mask = y == -1
    others_mask = y == 1

    ax1.scatter(X[setosa_mask, 0], X[setosa_mask, 1], c='blue', s=60, alpha=0.7,
                edgecolor='darkblue', linewidth=0.5, label='Iris-setosa (y=-1)', marker='o')
    ax1.scatter(X[others_mask, 0], X[others_mask, 1], c='red', s=60, alpha=0.7,
                edgecolor='darkred', linewidth=0.5, label='Other Iris (y=+1)', marker='s')

    # Highlight support vectors
    if support_indices:
        ax1.scatter(X[support_indices, 0], X[support_indices, 1],
                    facecolors='none', edgecolors='gold', s=200, linewidth=3,
                    label=f'Support vectors (k={k})')

    ax1.set_xlabel('Petal Length (cm)', fontsize=12)
    ax1.set_ylabel('Petal Width (cm)', fontsize=12)
    ax1.set_title('SVM Classification of Iris Species\n(Scenario Approach)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x1_min, x1_max)
    ax1.set_ylim(x2_min, x2_max)

    # Add decision boundary equation
    eq_text = f'Decision: {w1:.3f}$x_1$ + {w2:.3f}$x_2$ + {b:.3f} = 0'
    ax1.text(0.02, 0.98, eq_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ===== Plot 2: Margin Visualization =====
    ax2 = fig.add_subplot(gs[0, 2])

    # 3D-like visualization of the margin
    angles = np.linspace(0, 2*np.pi, 100)
    circle_r = margin / 2

    # Draw margin as a corridor
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='y = +1 margin')
    ax2.axhline(y=0, color='green', linestyle='-', linewidth=2.5, label='Decision boundary')
    ax2.axhline(y=-1, color='blue', linestyle='--', linewidth=2, label='y = -1 margin')

    # Fill margin region
    ax2.fill_between([-2, 2], [-1, -1], [1, 1], alpha=0.2, color='green', label=f'Margin = {margin:.3f}')

    # Add some conceptual points
    ax2.scatter([0], [1.5], c='red', s=100, marker='s', zorder=5)
    ax2.scatter([0], [-1.5], c='blue', s=100, marker='o', zorder=5)
    ax2.annotate('', xy=(0, 1), xytext=(0, -1),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax2.text(0.15, 0, f'margin\n= {margin:.3f}', fontsize=10, color='orange', fontweight='bold')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('Along hyperplane', fontsize=11)
    ax2.set_ylabel('Distance from hyperplane', fontsize=11)
    ax2.set_title('Margin Concept', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ===== Plot 3: Feature Distribution by Class =====
    ax3 = fig.add_subplot(gs[1, 0])

    positions = [1, 2, 4, 5]
    bp1 = ax3.boxplot([X[setosa_mask, 0], X[others_mask, 0]], positions=[1, 2],
                       patch_artist=True, widths=0.6)
    bp2 = ax3.boxplot([X[setosa_mask, 1], X[others_mask, 1]], positions=[4, 5],
                       patch_artist=True, widths=0.6)

    colors = ['lightblue', 'lightcoral']
    for bp in [bp1, bp2]:
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    ax3.set_xticks([1.5, 4.5])
    ax3.set_xticklabels(['Petal Length', 'Petal Width'])
    ax3.set_ylabel('Value (cm)', fontsize=11)
    ax3.set_title('Feature Distribution by Class', fontsize=12, fontweight='bold')
    ax3.legend([bp1['boxes'][0], bp1['boxes'][1]], ['Setosa', 'Others'],
               loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== Plot 4: Risk Bounds Visualization =====
    ax4 = fig.add_subplot(gs[1, 1])

    k_range = np.arange(1, min(N//2, 25))
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
    ax4.set_title(f'Risk Bounds (N={N}, beta={beta})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ===== Plot 5: Summary Box =====
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    summary_text = f"""
    SVM CLASSIFICATION RESULTS
    --------------------------

    Problem: Binary classification
    Iris-setosa vs Other species
    Features: Petal length & width

    Dataset:
      N = {N} samples
      Setosa: {np.sum(y==-1)}
      Others: {np.sum(y==1)}

    Optimal Hyperplane:
      {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0

    SVM Properties:
      ||w|| = {w_norm:.4f}
      Margin = {margin:.4f}
      Training accuracy: {accuracy*100:.1f}%

    Scenario Approach:
      Support vectors k = {k}
      Risk eps in [{eps_lower:.4f}, {eps_upper:.4f}]

      Interpretation: With 99% confidence,
      at most {eps_upper*100:.1f}% of new samples
      will be misclassified.
    """

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='green'))

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
