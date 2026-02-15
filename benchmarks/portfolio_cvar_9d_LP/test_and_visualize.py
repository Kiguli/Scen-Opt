#!/usr/bin/env python3
"""
Test and Visualization for Golden State Teachers' Pension Fund CVaR Benchmark

This script tests the CVaR portfolio optimization benchmark using the
scenario approach tool and creates visualizations of the results.

Problem: A pension fund must minimize CVaR (Conditional Value at Risk)
while meeting allocation constraints:
- Minimum 30% in equities, maximum 60%
- Minimum 25% in fixed income
- 5-30% position limits per asset
- Fully invested (sum to 100%)

The CVaR formulation finds portfolio weights w and VaR threshold alpha that
minimize the expected loss in the worst (1-β)% of scenarios.

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.LP import solve_lp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk

# Asset configuration (must match generate.py)
TICKERS = ['SPY', 'AGG', 'VNQ', 'GLD', 'EFA', 'TLT', 'VWO', 'LQD']
N_ASSETS = 8

# Asset class colors
CLASS_COLORS = {
    'US Equity': '#3498DB',
    'Fixed Income': '#27AE60',
    'Real Assets': '#E74C3C',
    'Commodities': '#F39C12',
    'Intl Equity': '#9B59B6',
    'Long Bonds': '#1ABC9C',
    'Credit': '#34495E',
}

ASSET_CLASSES = {
    'SPY': 'US Equity', 'AGG': 'Fixed Income', 'VNQ': 'Real Assets',
    'GLD': 'Commodities', 'EFA': 'Intl Equity', 'TLT': 'Long Bonds',
    'VWO': 'Intl Equity', 'LQD': 'Credit'
}

ASSET_COLORS = [CLASS_COLORS[ASSET_CLASSES[t]] for t in TICKERS]


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
    print("=" * 70)
    print("BENCHMARK: Golden State Teachers' Pension Fund CVaR Portfolio (LP)")
    print("CVaR Optimization Under Multi-Source Uncertainty")
    print("=" * 70)
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

    # Load historical prices for visualization
    try:
        prices = pd.read_csv(os.path.join(benchmark_dir, 'historical_prices.csv'),
                            index_col=0, parse_dates=True)
    except Exception:
        prices = None

    # Load asset info
    try:
        assets_df = pd.read_csv(os.path.join(benchmark_dir, 'assets.csv'))
    except Exception:
        assets_df = None

    # Load parameters
    params = {}
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())

    rho = params.get('rho', 0.04)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.99)

    N = len(scenarios)
    n_vars = c.shape[0]  # 8 weights + 1 alpha

    print(f"  Assets: {N_ASSETS} ({', '.join(TICKERS)})")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 12")
    print(f"  Decision variables: {n_vars} (8 weights + 1 VaR threshold)")
    print(f"  rho = {rho:.6f}, tau = {tau}")
    print()

    # Extract scenario components
    returns = scenarios[:, 0:8]
    stress = scenarios[:, 8]
    credit_shock = scenarios[:, 9]
    rate_shock = scenarios[:, 10]
    vol_scale = scenarios[:, 11]

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
    equity_weight = weights[0] + weights[4] + weights[6]  # SPY + EFA + VWO
    fixed_income = weights[1] + weights[5] + weights[7]    # AGG + TLT + LQD
    alternatives = weights[2] + weights[3]                  # VNQ + GLD

    # Print results
    print()
    print("-" * 70)
    print(f"{'OPTIMIZATION RESULTS':^70}")
    print("-" * 70)
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
    print("-" * 70)

    print()
    print("OPTIMAL PORTFOLIO WEIGHTS:")
    print("-" * 70)
    print(f"{'Ticker':<8} {'Name':<24} {'Weight':>8} {'Class':<16}")
    print("-" * 70)
    for i, ticker in enumerate(TICKERS):
        if assets_df is not None:
            name = assets_df[assets_df['ticker'] == ticker]['name'].values[0]
            asset_class = assets_df[assets_df['ticker'] == ticker]['class'].values[0]
        else:
            name = ticker
            asset_class = ASSET_CLASSES[ticker]
        print(f"{ticker:<8} {name[:24]:<24} {weights[i]*100:>7.2f}% {asset_class:<16}")
    print("-" * 70)
    print(f"{'TOTAL':<8} {'':<24} {np.sum(weights)*100:>7.2f}%")

    print()
    print("ALLOCATION SUMMARY:")
    print("-" * 70)
    print(f"  Equities (SPY+EFA+VWO):       {equity_weight*100:>6.2f}% (limit: 30-60%)")
    print(f"  Fixed Income (AGG+TLT+LQD):   {fixed_income*100:>6.2f}% (min: 25%)")
    print(f"  Alternatives (VNQ+GLD):       {alternatives*100:>6.2f}%")

    print()
    print("PORTFOLIO RISK METRICS:")
    print("-" * 70)
    print(f"  Mean Daily Return:   {np.mean(portfolio_returns)*100:>8.4f}%")
    print(f"  Std Daily Return:    {np.std(portfolio_returns)*100:>8.4f}%")
    print(f"  Worst Scenario:      {np.min(portfolio_returns)*100:>8.4f}%")
    print(f"  Best Scenario:       {np.max(portfolio_returns)*100:>8.4f}%")
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
        'worst_scenario': float(np.min(portfolio_returns)),
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

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # ===== Plot 1: Portfolio Weights =====
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(N_ASSETS)
    bars = ax1.bar(x_pos, weights * 100, color=ASSET_COLORS, edgecolor='black')
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.7, label='Min 5%')
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Max 30%')
    ax1.set_ylabel('Weight (%)', fontsize=11)
    ax1.set_title('Optimal CVaR Portfolio Weights', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(TICKERS, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 35)

    # ===== Plot 2: Historical Prices (Normalized) =====
    ax2 = fig.add_subplot(gs[0, 1])
    if prices is not None:
        norm_prices = prices / prices.iloc[0] * 100
        for ticker in TICKERS:
            if ticker in norm_prices.columns:
                color = CLASS_COLORS[ASSET_CLASSES[ticker]]
                ax2.plot(norm_prices.index, norm_prices[ticker], label=ticker,
                        color=color, alpha=0.7, linewidth=1)
        ax2.set_ylabel('Normalized Price (start=100)', fontsize=11)
        ax2.set_title('5-Year Price History', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=7, ncol=2, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=30)
    else:
        ax2.text(0.5, 0.5, 'Historical prices not available', ha='center', va='center')
        ax2.set_title('5-Year Price History', fontsize=12, fontweight='bold')

    # ===== Plot 3: Portfolio Return Distribution =====
    ax3 = fig.add_subplot(gs[0, 2])
    n_bins = 40
    ax3.hist(portfolio_returns * 100, bins=n_bins, color='steelblue', alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=np.mean(portfolio_returns) * 100, color='green', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(portfolio_returns)*100:.3f}%')
    ax3.axvline(x=np.percentile(portfolio_returns, 5) * 100, color='orange', linestyle='--', linewidth=2,
                label=f'VaR(95%)={-np.percentile(portfolio_returns, 5)*100:.3f}%')
    ax3.axvline(x=-cvar_actual * 100, color='red', linestyle='-', linewidth=2,
                label=f'CVaR(95%)={cvar_actual*100:.3f}%')
    ax3.set_xlabel('Daily Portfolio Return (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Portfolio Return Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== Plot 4: Asset Class Allocation Pie =====
    ax4 = fig.add_subplot(gs[0, 3])
    class_weights = {}
    for i, ticker in enumerate(TICKERS):
        ac = ASSET_CLASSES[ticker]
        class_weights[ac] = class_weights.get(ac, 0) + weights[i]
    labels = list(class_weights.keys())
    values = [class_weights[l] for l in labels]
    colors = [CLASS_COLORS[l] for l in labels]
    wedges, texts, autotexts = ax4.pie(values, labels=labels, colors=colors,
                                        autopct='%.1f%%', explode=[0.02]*len(labels),
                                        startangle=90)
    ax4.set_title('Asset Class Allocation', fontsize=12, fontweight='bold')
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(8)

    # ===== Plot 5: Return Correlation Heatmap =====
    ax5 = fig.add_subplot(gs[1, 0])
    return_corr = np.corrcoef(returns.T)
    im = ax5.imshow(return_corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(N_ASSETS))
    ax5.set_yticks(range(N_ASSETS))
    ax5.set_xticklabels(TICKERS, fontsize=8)
    ax5.set_yticklabels(TICKERS, fontsize=8)
    ax5.set_title('Return Correlation', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # ===== Plot 6: Stress Indicator Distribution =====
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(stress, bins=30, color='coral', alpha=0.7, edgecolor='white')
    ax6.axvline(x=np.mean(stress), color='red', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(stress):.2f}')
    ax6.set_xlabel('Market Stress Indicator (0=calm, 1=crisis)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Market Stress Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    # ===== Plot 7: Risk Bounds =====
    ax7 = fig.add_subplot(gs[1, 2])
    k_range = np.arange(1, min(N//10, 40))
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

    # ===== Plot 8: Credit vs Rate Shocks =====
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(credit_shock * 100, rate_shock * 100, alpha=0.3, c=stress, cmap='coolwarm', s=20)
    ax8.set_xlabel('Credit Spread Shock (%)', fontsize=11)
    ax8.set_ylabel('Interest Rate Shock (%)', fontsize=11)
    ax8.set_title('Credit vs Rate Shocks\n(color = market stress)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # ===== Plot 9: CVaR Tail Analysis =====
    ax9 = fig.add_subplot(gs[2, 0])
    sorted_rets = np.sort(portfolio_returns) * 100
    cvar_idx = int(0.05 * N)
    ax9.fill_between(range(cvar_idx), sorted_rets[:cvar_idx], color='red', alpha=0.5, label='CVaR tail (worst 5%)')
    ax9.fill_between(range(cvar_idx, N), sorted_rets[cvar_idx:], color='steelblue', alpha=0.5, label='Rest (95%)')
    ax9.axhline(y=-cvar_actual * 100, color='darkred', linestyle='--', linewidth=2, label=f'CVaR={cvar_actual*100:.3f}%')
    ax9.set_xlabel('Scenario (sorted by return)', fontsize=11)
    ax9.set_ylabel('Daily Return (%)', fontsize=11)
    ax9.set_title('CVaR Tail Analysis', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # ===== Plot 10: Volatility Scaling Impact =====
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.scatter(vol_scale, portfolio_returns * 100, alpha=0.3, c='steelblue', s=20)
    ax10.set_xlabel('Volatility Scaling Factor', fontsize=11)
    ax10.set_ylabel('Portfolio Return (%)', fontsize=11)
    ax10.set_title('Return vs Volatility Regime', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)

    # ===== Plot 11: Asset Returns Boxplot =====
    ax11 = fig.add_subplot(gs[2, 2])
    bp = ax11.boxplot([returns[:, i] * 100 for i in range(N_ASSETS)],
                      patch_artist=True, tick_labels=TICKERS)
    for patch, color in zip(bp['boxes'], ASSET_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax11.set_ylabel('Daily Return (%)', fontsize=11)
    ax11.set_title('Asset Return Distributions', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3, axis='y')

    # ===== Plot 12: Summary Box =====
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    summary_text = f"""
    GOLDEN STATE PENSION FUND
    =========================

    Problem:
      8 ETF assets
      12-dimensional uncertainty
      CVaR optimization (95%)
      Min equity: 30%
      Max equity: 60%
      Min fixed income: 25%

    Portfolio Allocation:
      Equities:      {equity_weight*100:>6.1f}%
      Fixed Income:  {fixed_income*100:>6.1f}%
      Alternatives:  {alternatives*100:>6.1f}%

    Risk Metrics:
      VaR (95%):  {-np.percentile(portfolio_returns, 5)*100:>6.4f}%
      CVaR (95%): {cvar_actual*100:>6.4f}%
      Alpha:      {alpha*100:>6.4f}%

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

    plt.suptitle("Golden State Teachers' Pension Fund - CVaR Portfolio Optimization",
                 fontsize=14, fontweight='bold', y=0.98)
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        pass  # Ignore tight_layout warnings

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
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
