#!/usr/bin/env python3
"""
Test and Visualization for Fresh Produce Distribution Benchmark

This script tests the inventory benchmark using the scenario approach tool
and creates visualizations of the results.

Problem: A regional produce distributor must decide daily order quantities
for 5 perishable products under uncertainty in:
- Yield (spoilage during transport/storage)
- Demand (varies by day, weather, season)
- Space efficiency (packing variability)

The goal is to minimize ordering costs while maintaining service levels,
subject to budget, warehouse capacity, and supplier constraints.

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.LP import solve_lp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk

# Product configuration (must match generate.py)
PRODUCT_NAMES = ['Strawberries', 'Tomatoes', 'Lettuce', 'Avocados', 'Bell Peppers']
PRODUCT_COLORS = ['#E74C3C', '#E67E22', '#27AE60', '#2ECC71', '#F1C40F']
N_PRODUCTS = 5


def parse_expression_matrix(filepath):
    """Parse CSV with delta[i] expressions into a function."""
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


def parse_expression_vector(filepath):
    """Parse CSV with delta[i] expressions into a function (vector)."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    expressions = [line.strip() for line in lines if line.strip()]

    def vector_function(delta):
        result = []
        for expr in expressions:
            val = eval(expr, {"delta": delta, "math": __import__('math')})
            result.append([val])
        return np.array(result)

    return vector_function


def load_matrix(filepath):
    """Load a numeric matrix from CSV."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    matrix = []
    for line in lines:
        if line.strip():
            row = [float(x.strip()) for x in line.split(',')]
            matrix.append(row)
    return np.array(matrix)


def load_vector(filepath):
    """Load a numeric vector from CSV."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def main():
    print("=" * 65)
    print("BENCHMARK: Fresh Produce Distribution (LP)")
    print("Regional Distributor Ordering Under Uncertainty")
    print("=" * 65)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'scenarios.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))
    G = load_matrix(os.path.join(benchmark_dir, 'G.csv'))
    h = load_vector(os.path.join(benchmark_dir, 'h.csv'))

    # Parse A_d and b_d expressions
    A_d = parse_expression_matrix(os.path.join(benchmark_dir, 'A_d.csv'))
    b_d = parse_expression_vector(os.path.join(benchmark_dir, 'b_d.csv'))

    # Load parameters
    params = {}
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())

    rho = params.get('rho', 25.0)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.99)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Products: {N_PRODUCTS} ({', '.join(PRODUCT_NAMES)})")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 15")
    print(f"  rho = {rho}, tau = {tau}")
    print()

    # Extract scenario components
    yields = scenarios[:, 0:5]
    demands = scenarios[:, 5:10]
    space_factors = scenarios[:, 10:15]

    # ===== SOLVE USING solve_lp =====
    print("Solving LP with MOSEK...")

    try:
        x, zeta, cost, N_out, k, constraints, degeneracy = solve_lp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            tau=tau,
            x_ref=np.zeros((n_vars, 1)),
            rho=rho,
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
    q = x.flatten()
    zeta_vals = zeta.flatten()
    zeta_max = np.max(zeta_vals)
    costs = c.flatten()

    # Calculate derived quantities
    ordering_cost = np.sum(costs * q)
    slack_cost = rho * np.sum(zeta_vals)
    usable = yields * q  # yield * order for each scenario
    shortfall = demands - usable  # positive = unmet demand

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Total Cost: ${cost:.2f}")
    print(f"  - Ordering: ${ordering_cost:.2f}")
    print(f"  - Service penalty: ${slack_cost:.2f}")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 65)

    print()
    print("OPTIMAL ORDER QUANTITIES:")
    print("-" * 65)
    print(f"{'Product':<15} {'Order':>8} {'Cost/Case':>10} {'Total Cost':>12} {'Mean Demand':>12}")
    print("-" * 65)
    for i, name in enumerate(PRODUCT_NAMES):
        total = q[i] * costs[i]
        mean_dem = np.mean(demands[:, i])
        print(f"{name:<15} {q[i]:>8.1f} ${costs[i]:>9.2f} ${total:>11.2f} {mean_dem:>12.1f}")
    print("-" * 65)
    print(f"{'TOTAL':<15} {np.sum(q):>8.1f} {'':<10} ${ordering_cost:>11.2f}")

    print()
    print("SERVICE LEVEL ANALYSIS:")
    print("-" * 65)
    for i, name in enumerate(PRODUCT_NAMES):
        service_pct = np.mean(usable[:, i] >= demands[:, i]) * 100
        avg_short = np.mean(np.maximum(0, shortfall[:, i]))
        worst_short = np.max(np.maximum(0, shortfall[:, i]))
        print(f"{name:<15}: {service_pct:>5.1f}% satisfied, "
              f"avg shortfall={avg_short:>5.1f}, worst={worst_short:>5.1f}")

    # Save results
    solution_data = {
        'order_quantities': {name: float(q[i]) for i, name in enumerate(PRODUCT_NAMES)},
        'optimal_cost': float(cost),
        'ordering_cost': float(ordering_cost),
        'slack_cost': float(slack_cost),
        'zeta_values': zeta_vals.tolist(),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': float(rho)
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # ===== Plot 1: Order Quantities vs Mean Demand =====
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(N_PRODUCTS)
    width = 0.35
    mean_demands = [np.mean(demands[:, i]) for i in range(N_PRODUCTS)]

    bars1 = ax1.bar(x_pos - width/2, mean_demands, width, label='Mean Demand',
                    color='lightgray', edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, q, width, label='Optimal Order',
                    color=PRODUCT_COLORS, edgecolor='black')

    ax1.set_ylabel('Cases', fontsize=11)
    ax1.set_title('Order Quantities vs Mean Demand', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([n[:4] for n in PRODUCT_NAMES], fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== Plot 2: Yield Distributions =====
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([yields[:, i] for i in range(N_PRODUCTS)],
                     patch_artist=True, labels=[n[:4] for n in PRODUCT_NAMES])
    for patch, color in zip(bp['boxes'], PRODUCT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Yield Factor', fontsize=11)
    ax2.set_title('Product Yield Distributions\n(due to spoilage)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect yield')

    # ===== Plot 3: Service Level by Product =====
    ax3 = fig.add_subplot(gs[0, 2])
    service_levels = [np.mean(usable[:, i] >= demands[:, i]) * 100 for i in range(N_PRODUCTS)]
    bars = ax3.bar(x_pos, service_levels, color=PRODUCT_COLORS, edgecolor='black')
    ax3.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% target')
    ax3.set_ylabel('Service Level (%)', fontsize=11)
    ax3.set_title('Service Level by Product', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([n[:4] for n in PRODUCT_NAMES], fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, pct in zip(bars, service_levels):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.0f}%', ha='center', fontsize=9)

    # ===== Plot 4: Cost Breakdown =====
    ax4 = fig.add_subplot(gs[0, 3])
    product_costs = [q[i] * costs[i] for i in range(N_PRODUCTS)]
    wedges, texts, autotexts = ax4.pie(product_costs, labels=[n[:4] for n in PRODUCT_NAMES],
                                        colors=PRODUCT_COLORS, autopct='$%.0f',
                                        explode=[0.02]*N_PRODUCTS, startangle=90)
    ax4.set_title('Ordering Cost by Product', fontsize=12, fontweight='bold')

    # ===== Plot 5: Yield vs Demand Scatter =====
    ax5 = fig.add_subplot(gs[1, 0])
    for i in range(N_PRODUCTS):
        ax5.scatter(yields[:, i], demands[:, i], c=PRODUCT_COLORS[i],
                   alpha=0.3, s=15, label=PRODUCT_NAMES[i][:4])
    ax5.set_xlabel('Yield Factor', fontsize=11)
    ax5.set_ylabel('Demand (cases)', fontsize=11)
    ax5.set_title('Yield vs Demand Scenarios', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)

    # ===== Plot 6: Demand Correlation Heatmap =====
    ax6 = fig.add_subplot(gs[1, 1])
    demand_corr = np.corrcoef(demands.T)
    im = ax6.imshow(demand_corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax6.set_xticks(range(N_PRODUCTS))
    ax6.set_yticks(range(N_PRODUCTS))
    ax6.set_xticklabels([n[:4] for n in PRODUCT_NAMES], fontsize=9)
    ax6.set_yticklabels([n[:4] for n in PRODUCT_NAMES], fontsize=9)
    ax6.set_title('Demand Correlation', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6, shrink=0.8)

    # ===== Plot 7: Risk Bounds =====
    ax7 = fig.add_subplot(gs[1, 2])
    k_range = np.arange(1, min(N//10, 30))
    eps_lowers = []
    eps_uppers = []
    for k_val in k_range:
        el, eu = quantify_risk(k_val, N, beta)
        eps_lowers.append(el)
        eps_uppers.append(eu)

    ax7.fill_between(k_range, eps_lowers, eps_uppers, alpha=0.3, color='blue')
    ax7.plot(k_range, eps_lowers, 'b--', linewidth=1.5, label='Lower bound')
    ax7.plot(k_range, eps_uppers, 'b-', linewidth=1.5, label='Upper bound')
    ax7.axvline(x=k, color='red', linestyle='-', linewidth=2, label=f'k={k}')
    ax7.scatter([k], [eps_lower], c='red', s=100, zorder=5)
    ax7.scatter([k], [eps_upper], c='red', s=100, zorder=5)
    ax7.set_xlabel('Complexity (k)', fontsize=11)
    ax7.set_ylabel('Risk (epsilon)', fontsize=11)
    ax7.set_title(f'Campi-Garatti Risk Bounds\n(N={N}, 99% confidence)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # ===== Plot 8: Warehouse Capacity Usage =====
    ax8 = fig.add_subplot(gs[1, 3])
    space_per_case = [1.2, 1.5, 2.0, 0.8, 1.3]  # From generate.py
    used_space = np.sum(space_factors * q * space_per_case, axis=1)
    ax8.hist(used_space, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax8.axvline(x=800, color='red', linestyle='--', linewidth=2, label='Capacity (800 cu.ft)')
    ax8.axvline(x=np.mean(used_space), color='green', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(used_space):.0f}')
    ax8.set_xlabel('Warehouse Space Used (cu.ft)', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('Warehouse Capacity Usage', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # ===== Plot 9: Shortfall Distribution by Product =====
    ax9 = fig.add_subplot(gs[2, 0])
    shortfall_data = [np.maximum(0, shortfall[:, i]) for i in range(N_PRODUCTS)]
    bp2 = ax9.boxplot(shortfall_data, patch_artist=True,
                      labels=[n[:4] for n in PRODUCT_NAMES])
    for patch, color in zip(bp2['boxes'], PRODUCT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax9.set_ylabel('Shortfall (cases)', fontsize=11)
    ax9.set_title('Shortfall Distribution by Product', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')

    # ===== Plot 10: Budget Utilization =====
    ax10 = fig.add_subplot(gs[2, 1])
    budget_used = ordering_cost
    budget_remaining = 12000 - budget_used
    ax10.barh(['Used', 'Remaining'], [budget_used, budget_remaining],
              color=['steelblue', 'lightgray'], edgecolor='black')
    ax10.set_xlabel('Dollars', fontsize=11)
    ax10.set_title(f'Daily Budget Utilization\n(Budget: $12,000)', fontsize=12, fontweight='bold')
    ax10.axvline(x=12000, color='red', linestyle='--', linewidth=2)
    for i, v in enumerate([budget_used, budget_remaining]):
        ax10.text(v + 100, i, f'${v:.0f}', va='center', fontsize=10)

    # ===== Plot 11: Usable Supply vs Demand =====
    ax11 = fig.add_subplot(gs[2, 2])
    sample_idx = np.random.choice(N, min(200, N), replace=False)
    for i in range(N_PRODUCTS):
        ax11.scatter(demands[sample_idx, i], usable[sample_idx, i],
                    c=PRODUCT_COLORS[i], alpha=0.4, s=20, label=PRODUCT_NAMES[i][:4])
    max_val = max(np.max(demands), np.max(usable)) * 1.1
    ax11.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Supply=Demand')
    ax11.set_xlabel('Demand (cases)', fontsize=11)
    ax11.set_ylabel('Usable Supply (cases)', fontsize=11)
    ax11.set_title('Usable Supply vs Demand', fontsize=12, fontweight='bold')
    ax11.legend(fontsize=8, ncol=2)
    ax11.grid(True, alpha=0.3)

    # ===== Plot 12: Summary Box =====
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    summary_text = f"""
    FRESH PRODUCE DISTRIBUTION
    ===========================

    Problem:
      5 perishable products
      15-dimensional uncertainty
      Budget: $12,000/day
      Warehouse: 800 cu.ft

    Optimal Orders:
      Strawberries: {q[0]:>6.1f} cases
      Tomatoes:     {q[1]:>6.1f} cases
      Lettuce:      {q[2]:>6.1f} cases
      Avocados:     {q[3]:>6.1f} cases
      Bell Peppers: {q[4]:>6.1f} cases

    Costs:
      Ordering:  ${ordering_cost:>7.2f}
      Penalty:   ${slack_cost:>7.2f}
      Total:     ${cost:>7.2f}

    Scenario Approach:
      N = {N} scenarios
      k = {k} support constraints
      Risk: [{eps_lower:.4f}, {eps_upper:.4f}]

      With 99% confidence,
      solution is feasible for
      >{(1-eps_upper)*100:.1f}% of realizations.
    """

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    plt.suptitle('Fresh Produce Distribution - Scenario Approach Optimization',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
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
    print("=" * 65)
    print("Benchmark complete!")
    print("=" * 65)


if __name__ == '__main__':
    main()
