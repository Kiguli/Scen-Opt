#!/usr/bin/env python3
"""
Test and Visualization for Growth Bound 12D (Reachability) Benchmark

This script tests the growth bound benchmark using the scenario approach tool
and creates visualizations of the results.

The problem computes growth bounds for a vehicle dynamics system, bounding how
much the state can change from one time step to the next given an initial set
of states.

The vehicle follows bicycle-like dynamics:
    x_dot = v * cos(alpha + theta) / cos(alpha)
    y_dot = v * sin(alpha + theta) / cos(alpha)
    theta_dot = v * tan(steering)

Where alpha = arctan(tan(steering)/2)

The goal is to find bounds [L_i, U_i] such that:
    |x_next_i - x_next_center_i| <= L_i * |x_i - x_center_i| + U_i

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

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


def rk4_step(f, x, u, dt):
    """RK4 integrator for dynamics."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def vehicle_dynamics(x, u):
    """Bicycle model vehicle dynamics."""
    alpha = np.arctan(np.tan(u[1]) / 2.0)
    dx = np.zeros(3)
    dx[0] = u[0] * np.cos(alpha + x[2]) / np.cos(alpha)
    dx[1] = u[0] * np.sin(alpha + x[2]) / np.cos(alpha)
    dx[2] = u[0] * np.tan(u[1])
    return dx


def main():
    print("=" * 60)
    print("BENCHMARK: growth_bound_12d (LP)")
    print("Reachability Analysis for Vehicle Dynamics")
    print("=" * 60)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files - use mini dataset for faster testing
    print("Loading data files...")
    # Try mini dataset first (faster), fall back to full if not available
    mini_path = os.path.join(benchmark_dir, 'growth_bound_mini.csv')
    full_path = os.path.join(benchmark_dir, 'growth_bound.csv')

    if os.path.exists(mini_path):
        scenarios = load_file(mini_path)
        print("  Using mini dataset for faster testing")
    else:
        scenarios = load_file(full_path)
        print("  Using full dataset")

    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))
    G = load_matrix(os.path.join(benchmark_dir, 'G.csv'))
    h = load_vector(os.path.join(benchmark_dir, 'h.csv'))

    # Parse A_d and b_d expressions
    A_d = parse_expression_matrix(os.path.join(benchmark_dir, 'A_d.csv'))
    b_d = parse_expression_matrix(os.path.join(benchmark_dir, 'b_d.csv'))

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Scenarios (state transitions): {N}")
    print(f"  Decision variables: {n_vars}")
    print(f"  Data format: [|dx_0|, |dx_1|, |dx_2|, |dy_0|, |dy_1|, |dy_2|]")
    print()

    # The scenario data contains:
    # columns 0-2: |x_current - x_center| (absolute deviations at current time)
    # columns 3-5: |x_next - x_next_center| (absolute deviations at next time)

    # ===== SOLVE USING solve_lp =====
    print("Solving LP with MOSEK...")

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
        print(f"Optimal cost: {cost:.6f}")

    except Exception as e:
        print(f"Solver failed: {e}")
        return

    # Calculate risk bounds
    beta = 0.01  # 99% confidence
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Extract growth bound parameters
    # The decision variables encode a linear bound:
    # |x_next_i - center_next_i| <= L_{ij} * |x_j - center_j| + U_i
    # With 12 variables: 9 for L matrix entries + 3 for U vector

    # Based on the constraint structure (3 constraints, 12 variables)
    # The solution encodes bounds for each of 3 state dimensions

    solution = x.flatten()

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
    print("Solution (Growth Bound Parameters):")
    for i in range(n_vars):
        print(f"  x[{i}] = {solution[i]:.6f}")

    # Analyze the growth structure
    # The c vector shows which variables we're minimizing
    c_flat = c.flatten()
    growth_vars = np.where(c_flat > 0)[0]
    print()
    print(f"Minimized variables (growth bound terms): {growth_vars}")

    # Save results
    solution_data = {
        'solution': solution.tolist(),
        'optimal_cost': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy)
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
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # ===== Plot 1: 3D Scatter of State Deviations =====
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Current state deviations
    dx_current = scenarios[:, :3]
    dx_next = scenarios[:, 3:]

    # Sample subset for clarity
    sample_idx = np.random.choice(N, min(500, N), replace=False)

    ax1.scatter(dx_current[sample_idx, 0], dx_current[sample_idx, 1], dx_current[sample_idx, 2],
                c='blue', alpha=0.3, s=10, label='Current deviation')
    ax1.set_xlabel('|dx_x|', fontsize=10)
    ax1.set_ylabel('|dx_y|', fontsize=10)
    ax1.set_zlabel('|dx_theta|', fontsize=10)
    ax1.set_title('Current State Deviations\n(from center)', fontsize=11, fontweight='bold')

    # ===== Plot 2: 3D Scatter of Next State Deviations =====
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    ax2.scatter(dx_next[sample_idx, 0], dx_next[sample_idx, 1], dx_next[sample_idx, 2],
                c='red', alpha=0.3, s=10, label='Next deviation')
    ax2.set_xlabel('|dy_x|', fontsize=10)
    ax2.set_ylabel('|dy_y|', fontsize=10)
    ax2.set_zlabel('|dy_theta|', fontsize=10)
    ax2.set_title('Next State Deviations\n(after dynamics)', fontsize=11, fontweight='bold')

    # ===== Plot 3: Growth Ratio Distribution =====
    ax3 = fig.add_subplot(gs[0, 2])

    # Compute growth ratios (next / current) for each dimension
    eps = 1e-10  # avoid division by zero
    growth_ratios = []
    labels = ['x', 'y', 'theta']

    for i in range(3):
        ratio = dx_next[:, i] / (dx_current[:, i] + eps)
        # Filter out extreme values for visualization
        ratio = ratio[ratio < 10]
        growth_ratios.append(ratio)

    bp = ax3.boxplot(growth_ratios, patch_artist=True, labels=labels)
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Growth = 1')
    ax3.set_ylabel('Growth Ratio (next/current)', fontsize=11)
    ax3.set_xlabel('State Dimension', fontsize=11)
    ax3.set_title('State Growth Distribution', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== Plot 4: 2D Projection of Reachable Set =====
    ax4 = fig.add_subplot(gs[1, 0])

    # Parameters from data-collection.py
    center = np.array([0, 1.2, 0])
    width = 1.6
    dt = 0.03
    u = np.array([0.3, 0.3])

    # Compute center trajectory
    x_center_next = rk4_step(vehicle_dynamics, center, u, dt)

    # Show initial set and reached set in x-y plane
    ax4.add_patch(Rectangle((center[0] - width/2, center[1] - width/2),
                             width, width, fill=False, edgecolor='blue',
                             linewidth=2, label='Initial set'))

    # Approximate reached set from data
    reached_x_min = x_center_next[0] - np.max(dx_next[:, 0])
    reached_x_max = x_center_next[0] + np.max(dx_next[:, 0])
    reached_y_min = x_center_next[1] - np.max(dx_next[:, 1])
    reached_y_max = x_center_next[1] + np.max(dx_next[:, 1])

    ax4.add_patch(Rectangle((reached_x_min, reached_y_min),
                             reached_x_max - reached_x_min,
                             reached_y_max - reached_y_min,
                             fill=False, edgecolor='red', linewidth=2,
                             linestyle='--', label='Reached set (approx)'))

    # Plot centers
    ax4.plot(center[0], center[1], 'bo', markersize=10, label='Initial center')
    ax4.plot(x_center_next[0], x_center_next[1], 'r^', markersize=10, label='Next center')

    # Draw arrow showing transition
    ax4.annotate('', xy=(x_center_next[0], x_center_next[1]),
                 xytext=(center[0], center[1]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax4.set_xlabel('x position', fontsize=11)
    ax4.set_ylabel('y position', fontsize=11)
    ax4.set_title('Reachable Set (X-Y Plane)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    # ===== Plot 5: Deviation Correlation =====
    ax5 = fig.add_subplot(gs[1, 1])

    # Plot correlation between current and next deviations
    ax5.scatter(dx_current[sample_idx, 0], dx_next[sample_idx, 0], c='blue', alpha=0.3, s=20, label='x')
    ax5.scatter(dx_current[sample_idx, 1], dx_next[sample_idx, 1], c='green', alpha=0.3, s=20, label='y')
    ax5.scatter(dx_current[sample_idx, 2], dx_next[sample_idx, 2], c='orange', alpha=0.3, s=20, label='theta')

    # Add reference line (growth = 1)
    max_val = max(np.max(dx_current[sample_idx]), np.max(dx_next[sample_idx]))
    ax5.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='y=x (no growth)')

    ax5.set_xlabel('Current deviation |x_i - center|', fontsize=11)
    ax5.set_ylabel('Next deviation |x_next_i - center_next|', fontsize=11)
    ax5.set_title('Deviation Correlation', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ===== Plot 6: Risk Bounds Visualization =====
    ax6 = fig.add_subplot(gs[1, 2])

    k_range = np.arange(1, min(N//10, 30))
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
    ax6.set_title(f'Risk Bounds (N={N}, beta={beta})', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ===== Plot 7: Vehicle Trajectory Concept =====
    ax7 = fig.add_subplot(gs[2, 0])

    # Draw a simple vehicle trajectory concept
    t_steps = 20
    trajectory_x = [center[0]]
    trajectory_y = [center[1]]
    x_state = center.copy()

    for _ in range(t_steps):
        x_state = rk4_step(vehicle_dynamics, x_state, u, dt)
        trajectory_x.append(x_state[0])
        trajectory_y.append(x_state[1])

    ax7.plot(trajectory_x, trajectory_y, 'g-', linewidth=2, label='Nominal trajectory')
    ax7.plot(trajectory_x[0], trajectory_y[0], 'bo', markersize=12, label='Start')
    ax7.plot(trajectory_x[-1], trajectory_y[-1], 'r^', markersize=12, label='End')

    # Add uncertainty tube
    for i in range(0, len(trajectory_x), 3):
        circle = plt.Circle((trajectory_x[i], trajectory_y[i]),
                            width/4 * (1 + i/len(trajectory_x)),
                            fill=False, edgecolor='gray', alpha=0.5, linestyle='--')
        ax7.add_patch(circle)

    ax7.set_xlabel('x position', fontsize=11)
    ax7.set_ylabel('y position', fontsize=11)
    ax7.set_title('Vehicle Trajectory with\nGrowing Uncertainty', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')

    # ===== Plot 8: Histogram of Max Deviations =====
    ax8 = fig.add_subplot(gs[2, 1])

    max_devs = np.max(dx_next, axis=1)
    ax8.hist(max_devs, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax8.axvline(x=np.mean(max_devs), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(max_devs):.4f}')
    ax8.axvline(x=np.max(max_devs), color='orange', linestyle='--', linewidth=2,
                label=f'Max: {np.max(max_devs):.4f}')

    ax8.set_xlabel('Maximum deviation (any dimension)', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('Distribution of Maximum\nNext-State Deviations', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # ===== Plot 9: Summary Box =====
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = f"""
    REACHABILITY ANALYSIS RESULTS
    -----------------------------

    Problem: Bound state evolution
    for vehicle dynamics model

    Vehicle Model:
      Bicycle-like dynamics
      Control: v={u[0]}, steer={u[1]}
      Time step: dt={dt}

    Initial Set:
      Center: ({center[0]}, {center[1]}, {center[2]})
      Width: {width}

    Simulation:
      N = {N} sample transitions
      State dims: 3 (x, y, theta)

    Growth Bound Solution:
      Optimal cost: {cost:.4f}
      Complexity k = {k}

    Scenario Approach:
      Risk eps in [{eps_lower:.4f}, {eps_upper:.4f}]

      Interpretation: With 99% confidence,
      the computed growth bound holds
      for at least {(1-eps_upper)*100:.1f}% of
      possible state transitions.
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

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
