#!/usr/bin/env python3
"""
Solve the Economic Dispatch benchmark using the scenario approach LP solver.

Usage:
    python run.py
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

# System configuration (must match generate.py)
T_HORIZON = 24
GENERATOR_NAMES = ['Gas1', 'Gas2', 'Coal']
N_GENERATORS = len(GENERATOR_NAMES)
GENERATOR_COSTS = [40, 50, 30]  # $/MWh
GENERATOR_CAPACITIES = [400, 300, 500]  # MW


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
    print("BENCHMARK: Economic Dispatch with Renewable Uncertainty (LP)")
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

    # Load demand profile
    demand_data = np.loadtxt(os.path.join(data_dir, 'demand.csv'), delimiter=',')
    demand = demand_data[:, 1]

    # Load parameters
    params = {}
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())

    rho = params.get('rho', 100.0)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.999999)

    N = len(scenarios)
    n_vars = N_GENERATORS * T_HORIZON       # original: generator dispatch variables
    n_aug = c.shape[0]                     # augmented: x_aug = [P; zeta]

    print(f"  Generators: {N_GENERATORS} ({', '.join(GENERATOR_NAMES)})")
    print(f"  Time horizon: {T_HORIZON} hours")
    print(f"  Decision variables: {n_aug} ({n_vars} dispatch + {n_aug - n_vars} slack)")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 48 (24 wind + 24 solar)")
    print(f"  rho = {rho}, tau = {tau}")
    print()

    # Data files contain the augmented formulation x_aug = [P; zeta] with
    # rho embedded in c, A_d augmented with -I slack columns, and G/h
    # extended with zeta >= 0 constraints. See generate.py for details.
    print("Solving LP with MOSEK...")

    try:
        x_full, _, cost, N_out, k, constraints, degeneracy = solve_lp(
            deltas=scenarios,
            A_d=A_d,
            b_d=b_d,
            G=G,
            h=h,
            c=c,
            tau=0.0,
            x_ref=np.zeros((n_aug, 1)),
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

    # Extract solution as generation schedule
    P = x.flatten().reshape(N_GENERATORS, T_HORIZON)
    zeta_vals = zeta.flatten()

    # Calculate costs
    generation_cost = sum(GENERATOR_COSTS[m] * np.sum(P[m, :]) for m in range(N_GENERATORS))
    slack_cost = rho * np.sum(zeta_vals)

    # Mean renewable generation
    wind_mean = np.mean(scenarios[:, :24], axis=0)
    solar_mean = np.mean(scenarios[:, 24:], axis=0)
    total_thermal = np.sum(P, axis=0)
    total_supply = total_thermal + wind_mean + solar_mean

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Total Cost: ${cost:.2f}")
    print(f"  - Generation: ${generation_cost:.2f}")
    print(f"  - Imbalance penalty: ${slack_cost:.2f}")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 65)

    print()
    print("GENERATOR DISPATCH SUMMARY (MW):")
    print("-" * 65)
    print(f"{'Generator':<10} {'Min':>8} {'Mean':>8} {'Max':>8} {'Total MWh':>12} {'Cost ($)':>12}")
    print("-" * 65)
    for m, name in enumerate(GENERATOR_NAMES):
        total_mwh = np.sum(P[m, :])
        gen_cost = GENERATOR_COSTS[m] * total_mwh
        print(f"{name:<10} {np.min(P[m,:]):>8.1f} {np.mean(P[m,:]):>8.1f} "
              f"{np.max(P[m,:]):>8.1f} {total_mwh:>12.1f} ${gen_cost:>11.2f}")
    print("-" * 65)
    print(f"{'TOTAL':<10} {np.min(total_thermal):>8.1f} {np.mean(total_thermal):>8.1f} "
          f"{np.max(total_thermal):>8.1f} {np.sum(P):>12.1f} ${generation_cost:>11.2f}")

    print()
    print("SUPPLY vs DEMAND (hourly average):")
    print("-" * 65)
    print(f"  Mean thermal:    {np.mean(total_thermal):>8.1f} MW")
    print(f"  Mean wind:       {np.mean(wind_mean):>8.1f} MW")
    print(f"  Mean solar:      {np.mean(solar_mean):>8.1f} MW")
    print(f"  Mean total:      {np.mean(total_supply):>8.1f} MW")
    print(f"  Mean demand:     {np.mean(demand):>8.1f} MW")

    # Save results
    solution_data = {
        'dispatch': {name: P[m, :].tolist() for m, name in enumerate(GENERATOR_NAMES)},
        'optimal_cost': float(cost),
        'generation_cost': float(generation_cost),
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
