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

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.LP import solve_lp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk

# Product configuration (must match generate.py)
PRODUCT_NAMES = ['Strawberries', 'Tomatoes', 'Lettuce', 'Avocados', 'Bell Peppers']
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
    data_dir = os.path.join(benchmark_dir, 'data')
    print("Loading data files...")
    scenarios = load_file(os.path.join(data_dir, 'scenarios.csv'))
    c = load_vector(os.path.join(data_dir, 'c.csv'))
    G = load_matrix(os.path.join(data_dir, 'G.csv'))
    h = load_vector(os.path.join(data_dir, 'h.csv'))

    # Parse A_d and b_d expressions
    A_d = parse_expression_matrix(os.path.join(data_dir, 'A_d.csv'))
    b_d = parse_expression_vector(os.path.join(data_dir, 'b_d.csv'))

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

    # ===== REFORMULATE: embed shared slack into decision variable =====
    # Equivalent to old shared-ζ relaxation: min c'x + ρ·1'ζ  s.t. A(δ_i)x + b(δ_i) ≤ ζ, ζ ≥ 0
    # Augment x̃ = [x; ζ] and call solver with ρ=0.
    A_d_orig = A_d
    b_d_orig = b_d
    m_scenario = A_d_orig(scenarios[0]).shape[0]

    c_aug = np.vstack([c, rho * np.ones((m_scenario, 1))])

    def A_d_aug(delta):
        return np.hstack([A_d_orig(delta), -np.eye(m_scenario)])

    def b_d_aug(delta):
        return b_d_orig(delta)

    if not (G.size == 0 or h.size == 0):
        G_aug = np.block([
            [G, np.zeros((G.shape[0], m_scenario))],
            [np.zeros((m_scenario, n_vars)), -np.eye(m_scenario)]
        ])
        h_aug = np.vstack([h, np.zeros((m_scenario, 1))])
    else:
        G_aug = np.hstack([np.zeros((m_scenario, n_vars)), -np.eye(m_scenario)])
        h_aug = np.zeros((m_scenario, 1))

    print("Solving LP with MOSEK (shared-slack reformulation)...")

    try:
        x_full, _, cost, N_out, k, constraints, degeneracy = solve_lp(
            deltas=scenarios,
            A_d=A_d_aug,
            b_d=b_d_aug,
            G=G_aug,
            h=h_aug,
            c=c_aug,
            tau=0.0,
            x_ref=np.zeros((n_vars + m_scenario, 1)),
            rho=0.0,
            norm_type=2,
            solver='MOSEK'
        )
        x = x_full[:n_vars]
        zeta = x_full[n_vars:]
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

    print()
    print("=" * 65)
    print("Benchmark complete!")
    print("Run plot.py to generate the paper figure.")
    print("=" * 65)


if __name__ == '__main__':
    main()
