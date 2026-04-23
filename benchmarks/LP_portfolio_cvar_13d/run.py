#!/usr/bin/env python3
"""
Solve the CVaR Portfolio benchmark using the scenario approach LP solver.

Usage:
    python run.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks._loader import load_symbolic_program
from src.LP import solve_lp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk

# Asset configuration (must match generate.py)
TICKERS = ['SPY', 'AGG', 'VNQ', 'GLD', 'EFA', 'TLT', 'VWO', 'LQD']
N_ASSETS = 8

ASSET_CLASSES = {
    'SPY': 'US Equity', 'AGG': 'Fixed Income', 'VNQ': 'Real Assets',
    'GLD': 'Commodities', 'EFA': 'Intl Equity', 'TLT': 'Long Bonds',
    'VWO': 'Intl Equity', 'LQD': 'Credit'
}


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
    print("BENCHMARK: Pension Fund CVaR Portfolio Optimization (LP)")
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
    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c, A_d, b_d = prog['c'], prog['A_d'], prog['b_d']
    G, h = prog.get('G', np.array([])), prog.get('h', np.array([]))
    params = {}
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())

    rho = params.get('rho', 0.04)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.999999)

    N = len(scenarios)
    n_vars = N_ASSETS + 1                  # original: 8 weights + 1 VaR threshold
    n_aug = c.shape[0]                     # augmented: x_aug = [x; zeta]

    print(f"  Assets: {N_ASSETS} ({', '.join(TICKERS)})")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 12")
    print(f"  Decision variables: {n_aug} ({n_vars} original + {n_aug - n_vars} slack)")
    print(f"  rho = {rho:.6f}, tau = {tau}")
    print()

    # Extract scenario components
    returns = scenarios[:, 0:8]

    # Data files contain the augmented formulation x_aug = [x; zeta] with
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

    # Extract solution
    weights = x[:N_ASSETS].flatten()
    alpha = x[N_ASSETS, 0]  # VaR threshold
    zeta_vals = zeta.flatten()

    # Calculate portfolio returns
    portfolio_returns = returns @ weights

    # Calculate CVaR (average of worst 5% scenarios)
    sorted_returns = np.sort(portfolio_returns)
    cvar_threshold = int(0.05 * N)
    cvar_actual = -np.mean(sorted_returns[:max(1, cvar_threshold)])

    # Asset class allocations
    equity_weight = weights[0] + weights[4] + weights[6]
    fixed_income = weights[1] + weights[5] + weights[7]
    alternatives = weights[2] + weights[3]

    # Print results
    print()
    print("-" * 65)
    print(f"{'OPTIMIZATION RESULTS':^65}")
    print("-" * 65)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.6f}")
    print(f"VaR Threshold (alpha): {alpha*100:.4f}%")
    print(f"CVaR (95%): {cvar_actual*100:.4f}%")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 65)

    print()
    print("OPTIMAL PORTFOLIO WEIGHTS:")
    print("-" * 65)
    print(f"{'Ticker':<8} {'Class':<16} {'Weight':>8}")
    print("-" * 65)
    for i, ticker in enumerate(TICKERS):
        print(f"{ticker:<8} {ASSET_CLASSES[ticker]:<16} {weights[i]*100:>7.2f}%")
    print("-" * 65)
    print(f"{'TOTAL':<8} {'':<16} {np.sum(weights)*100:>7.2f}%")

    print()
    print("ALLOCATION SUMMARY:")
    print(f"  Equities (SPY+EFA+VWO):       {equity_weight*100:>6.2f}% (limit: 30-60%)")
    print(f"  Fixed Income (AGG+TLT+LQD):   {fixed_income*100:>6.2f}% (min: 25%)")
    print(f"  Alternatives (VNQ+GLD):       {alternatives*100:>6.2f}%")

    print()
    print("PORTFOLIO RISK METRICS:")
    print(f"  Mean Daily Return:   {np.mean(portfolio_returns)*100:>8.4f}%")
    print(f"  Std Daily Return:    {np.std(portfolio_returns)*100:>8.4f}%")
    print(f"  VaR (95%):           {-np.percentile(portfolio_returns, 5)*100:>8.4f}%")
    print(f"  CVaR (95%):          {cvar_actual*100:>8.4f}%")

    # Save results
    solution_data = {
        'portfolio_weights': {ticker: float(weights[i]) for i, ticker in enumerate(TICKERS)},
        'alpha_var_threshold': float(alpha),
        'cvar_95': float(cvar_actual),
        'equity_allocation': float(equity_weight),
        'fixed_income_allocation': float(fixed_income),
        'alternatives_allocation': float(alternatives),
        'mean_return': float(np.mean(portfolio_returns)),
        'std_return': float(np.std(portfolio_returns)),
        'objective_value': float(cost),
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
