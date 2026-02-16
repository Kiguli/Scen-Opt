#!/usr/bin/env python3
"""
Fresh Produce Distribution Benchmark Data Generator

Based on real-world fresh produce logistics challenges faced by regional
distributors. The scenario models a distributor serving grocery stores
with perishable products where:
- Yield varies due to spoilage during transport and storage
- Demand is uncertain and varies by day of week and season
- Warehouse capacity and refrigeration are limited

Products modeled after USDA wholesale market data for common produce items.

Data sources:
- USDA Agricultural Marketing Service Terminal Market Reports
- FDA Food Loss and Waste estimates (30-40% for fresh produce)
- Industry benchmarks for cold chain logistics

Usage:
    python generate.py [--n_scenarios N] [--seed SEED]
"""

import os
import numpy as np
import argparse

# ============================================================
# PRODUCT CONFIGURATION
# Based on real wholesale produce market data
# ============================================================

PRODUCTS = {
    'Strawberries': {
        'cost_per_case': 18.50,      # $/case (USDA avg wholesale)
        'mean_demand': 85,            # cases/day
        'demand_std': 25,             # high variability
        'yield_mean': 0.75,           # 25% spoilage rate (highly perishable)
        'yield_std': 0.08,
        'space_per_case': 1.2,        # cubic feet (refrigerated)
        'max_order': 150,             # supplier constraint
    },
    'Tomatoes': {
        'cost_per_case': 14.25,
        'mean_demand': 120,
        'demand_std': 30,
        'yield_mean': 0.88,           # 12% loss
        'yield_std': 0.05,
        'space_per_case': 1.5,
        'max_order': 200,
    },
    'Lettuce': {
        'cost_per_case': 12.00,
        'mean_demand': 95,
        'demand_std': 20,
        'yield_mean': 0.82,           # 18% loss
        'yield_std': 0.06,
        'space_per_case': 2.0,        # bulky
        'max_order': 180,
    },
    'Avocados': {
        'cost_per_case': 32.00,       # premium product
        'mean_demand': 60,
        'demand_std': 18,
        'yield_mean': 0.90,           # relatively stable
        'yield_std': 0.04,
        'space_per_case': 0.8,        # compact
        'max_order': 120,
    },
    'Bell Peppers': {
        'cost_per_case': 22.50,
        'mean_demand': 75,
        'demand_std': 22,
        'yield_mean': 0.85,
        'yield_std': 0.05,
        'space_per_case': 1.3,
        'max_order': 140,
    },
}

N_PRODUCTS = len(PRODUCTS)
PRODUCT_NAMES = list(PRODUCTS.keys())

# Warehouse constraints
WAREHOUSE_CAPACITY = 800  # cubic feet of refrigerated space
BUDGET_LIMIT = 12000      # daily ordering budget

# Correlation structure (products from same suppliers/regions correlate)
# Strawberries-Lettuce (CA), Tomatoes-Peppers (Mexico), Avocados independent
DEMAND_CORRELATION = np.array([
    [1.0,  0.2,  0.4,  0.1,  0.2],   # Strawberries
    [0.2,  1.0,  0.2,  0.3,  0.6],   # Tomatoes
    [0.4,  0.2,  1.0,  0.1,  0.2],   # Lettuce
    [0.1,  0.3,  0.1,  1.0,  0.3],   # Avocados
    [0.2,  0.6,  0.2,  0.3,  1.0],   # Bell Peppers
])

YIELD_CORRELATION = np.array([
    [1.0,  0.3,  0.5,  0.1,  0.3],   # Similar cold chain
    [0.3,  1.0,  0.3,  0.2,  0.7],   # Same origin
    [0.5,  0.3,  1.0,  0.1,  0.3],
    [0.1,  0.2,  0.1,  1.0,  0.2],   # Independent supply
    [0.3,  0.7,  0.3,  0.2,  1.0],
])


def ensure_positive_definite(corr_matrix, min_eigenvalue=0.01):
    """Ensure correlation matrix is positive definite."""
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Normalize to correlation matrix
    d = np.sqrt(np.diag(result))
    return result / np.outer(d, d)


def generate_correlated_samples(n_samples, means, stds, corr_matrix, seed_offset=0):
    """Generate correlated samples using Cholesky decomposition."""
    n_vars = len(means)
    corr_pd = ensure_positive_definite(corr_matrix)
    L = np.linalg.cholesky(corr_pd)

    # Generate uncorrelated standard normal samples
    Z = np.random.randn(n_samples, n_vars)

    # Apply correlation structure
    Z_corr = Z @ L.T

    # Transform to target distribution
    samples = means + stds * Z_corr

    return samples


def generate_scenarios(n_scenarios, seed=42):
    """
    Generate scenarios with 15 uncertain parameters:
    - delta[0:5]  = yields for 5 products (correlated)
    - delta[5:10] = demands for 5 products (correlated)
    - delta[10:15] = space efficiency factors (slightly variable)

    Total: 15-dimensional uncertainty
    """
    np.random.seed(seed)

    scenarios = np.zeros((n_scenarios, 15))

    # Extract product parameters
    yield_means = np.array([PRODUCTS[p]['yield_mean'] for p in PRODUCT_NAMES])
    yield_stds = np.array([PRODUCTS[p]['yield_std'] for p in PRODUCT_NAMES])
    demand_means = np.array([PRODUCTS[p]['mean_demand'] for p in PRODUCT_NAMES])
    demand_stds = np.array([PRODUCTS[p]['demand_std'] for p in PRODUCT_NAMES])

    # Generate correlated yields (truncated to [0.5, 1.0])
    yields = generate_correlated_samples(n_scenarios, yield_means, yield_stds,
                                         YIELD_CORRELATION)
    yields = np.clip(yields, 0.5, 1.0)
    scenarios[:, 0:5] = yields

    # Generate correlated demands (truncated to positive)
    demands = generate_correlated_samples(n_scenarios, demand_means, demand_stds,
                                          DEMAND_CORRELATION, seed_offset=1000)
    demands = np.maximum(demands, 10)  # Minimum demand
    scenarios[:, 5:10] = demands

    # Generate space efficiency factors (slight variability around 1.0)
    # Represents packing efficiency variations
    space_factors = np.random.uniform(0.95, 1.10, (n_scenarios, 5))
    scenarios[:, 10:15] = space_factors

    return scenarios


def load_parameters():
    """Load rho and other parameters from parameters.txt."""
    params = {}
    params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.txt')
    with open(params_path, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())
    return params


def save_constraint_files():
    """Generate and save constraint CSV files with shared-slack augmentation.

    The data files encode the augmented formulation directly:
        Original:  min c'q + rho * 1'zeta   s.t. A(d)q + b(d) <= zeta, Gq+h <= 0, zeta >= 0
        Augmented: min c_aug' x_aug          s.t. A_aug(d) x_aug + b(d) <= 0, G_aug x_aug + h_aug <= 0

    where x_aug = [q; zeta] in R^(n + m), with:
        c_aug = [c; rho * 1_m]
        A_aug = [A, -I_m]        (slack absorbs scenario violations)
        G_aug = [[G, 0]; [0, -I_m]]   (original hard constraints + zeta >= 0)
        h_aug = [h; 0]
    """
    params = load_parameters()
    rho = params.get('rho', 100.0)

    # ---- Build original A_d rows (6 x 5 expression strings) ----
    # Rows 0-4: Service constraints (-yield[j] * q[j] + demand[j] <= zeta_j)
    # Row 5: Capacity constraint (space[j] * q[j] - capacity <= zeta_5)
    a_d_rows = []
    for j in range(N_PRODUCTS):
        row = ['0'] * N_PRODUCTS
        row[j] = f'-delta[{j}]'
        a_d_rows.append(','.join(row))

    space_row = []
    for j, name in enumerate(PRODUCT_NAMES):
        space = PRODUCTS[name]['space_per_case']
        space_row.append(f'{space}*delta[{10+j}]')
    a_d_rows.append(','.join(space_row))

    m_scenario = len(a_d_rows)  # 6

    # ---- Augment A_d: append -I_m slack columns ----
    a_d_aug_rows = []
    for i, row_str in enumerate(a_d_rows):
        slack_cols = ['0'] * m_scenario
        slack_cols[i] = '-1'
        a_d_aug_rows.append(row_str + ',' + ','.join(slack_cols))

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'A_d.csv'), 'w') as f:
        f.write('\n'.join(a_d_aug_rows) + '\n')

    # ---- b_d.csv: unchanged (6 rows) ----
    b_d_rows = []
    for j in range(N_PRODUCTS):
        b_d_rows.append(f'delta[{5+j}]')
    b_d_rows.append(f'-{WAREHOUSE_CAPACITY}')

    with open(os.path.join(data_dir, 'b_d.csv'), 'w') as f:
        f.write('\n'.join(b_d_rows) + '\n')

    # ---- c.csv: augmented cost vector [c; rho * 1_m] ----
    costs = [PRODUCTS[p]['cost_per_case'] for p in PRODUCT_NAMES]
    costs_aug = costs + [rho] * m_scenario
    with open(os.path.join(data_dir, 'c.csv'), 'w') as f:
        for val in costs_aug:
            f.write(f'{val}\n')

    # ---- G.csv and h.csv: augmented hard constraints ----
    # Original hard constraints on q (padded with zero slack columns):
    #   Non-negativity: -q[j] <= 0
    #   Upper bounds:    q[j] <= max_order[j]
    #   Budget:          sum(cost[j] * q[j]) <= BUDGET_LIMIT
    # Plus zeta >= 0 block: -zeta_i <= 0

    n_vars = N_PRODUCTS
    n_aug = n_vars + m_scenario
    G_rows = []
    h_vals = []

    # Non-negativity on q
    for j in range(N_PRODUCTS):
        row = [0] * n_aug
        row[j] = -1
        G_rows.append(row)
        h_vals.append(0)

    # Upper bounds on q
    for j, name in enumerate(PRODUCT_NAMES):
        row = [0] * n_aug
        row[j] = 1
        G_rows.append(row)
        h_vals.append(-PRODUCTS[name]['max_order'])

    # Budget constraint on q
    budget_row = [0] * n_aug
    for j, p in enumerate(PRODUCT_NAMES):
        budget_row[j] = PRODUCTS[p]['cost_per_case']
    G_rows.append(budget_row)
    h_vals.append(-BUDGET_LIMIT)

    # Non-negativity on zeta: -zeta_i <= 0
    for i in range(m_scenario):
        row = [0] * n_aug
        row[n_vars + i] = -1
        G_rows.append(row)
        h_vals.append(0)

    with open(os.path.join(data_dir, 'G.csv'), 'w') as f:
        for row in G_rows:
            f.write(','.join(str(x) for x in row) + '\n')

    with open(os.path.join(data_dir, 'h.csv'), 'w') as f:
        for h in h_vals:
            f.write(f'{h}\n')

    print(f"  Augmented formulation: x_aug = [q; zeta] in R^{n_aug}")
    print(f"  A_d: {m_scenario} x {n_aug}, c: {n_aug}, G: {len(G_rows)} x {n_aug}, h: {len(h_vals)}")


def main():
    parser = argparse.ArgumentParser(description='Generate Fresh Produce Distribution benchmark')
    parser.add_argument('--n_scenarios', type=int, default=500,
                        help='Number of scenarios (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("Fresh Produce Distribution Benchmark Generator")
    print("=" * 60)
    print()
    print(f"Products: {N_PRODUCTS}")
    print(f"  " + ", ".join(PRODUCT_NAMES))
    print(f"Scenarios: {args.n_scenarios}")
    print(f"Uncertainty dimensions: 15")
    print(f"  - 5 yield factors (correlated)")
    print(f"  - 5 demand values (correlated)")
    print(f"  - 5 space efficiency factors")
    print()

    # Generate scenarios
    print("Generating correlated scenarios...")
    scenarios = generate_scenarios(args.n_scenarios, args.seed)

    # Save scenarios
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(os.path.join(data_dir, 'scenarios.csv'), scenarios, delimiter=',', fmt='%.10f')
    print(f"Saved data/scenarios.csv: {args.n_scenarios} x 15")

    # Save constraint files
    print("Generating constraint files...")
    save_constraint_files()
    print("Saved: A_d.csv, b_d.csv, c.csv, G.csv, h.csv")

    # Print summary statistics
    print()
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nYields (delta[0:5]):")
    for j, name in enumerate(PRODUCT_NAMES):
        y = scenarios[:, j]
        print(f"  {name:12s}: mean={np.mean(y):.3f}, std={np.std(y):.3f}, "
              f"range=[{np.min(y):.3f}, {np.max(y):.3f}]")

    print("\nDemands (delta[5:10]):")
    for j, name in enumerate(PRODUCT_NAMES):
        d = scenarios[:, 5+j]
        print(f"  {name:12s}: mean={np.mean(d):.1f}, std={np.std(d):.1f}, "
              f"range=[{np.min(d):.1f}, {np.max(d):.1f}]")

    print("\nSpace Factors (delta[10:15]):")
    sf = scenarios[:, 10:15]
    print(f"  All products: mean={np.mean(sf):.3f}, std={np.std(sf):.3f}")

    print("\nConstraints:")
    print(f"  Warehouse capacity: {WAREHOUSE_CAPACITY} cubic feet")
    print(f"  Daily budget: ${BUDGET_LIMIT}")
    for name in PRODUCT_NAMES:
        print(f"  {name:12s} max order: {PRODUCTS[name]['max_order']} cases")


if __name__ == '__main__':
    main()
