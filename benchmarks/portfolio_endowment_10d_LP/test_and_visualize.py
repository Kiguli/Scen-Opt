#!/usr/bin/env python3
"""
Test and Visualization for Sierra Vista University Endowment Portfolio Benchmark

This script tests the portfolio benchmark using the scenario approach tool
and creates visualizations of the results.

Problem: A university endowment must allocate assets across 10 stocks under
uncertainty in:
- Individual asset returns (bootstrapped from historical data)
- Volatility regimes
- Correlation stress during market turbulence
- Systematic factor exposures (market, size, value)

The goal is to maximize risk-adjusted returns while satisfying constraints
on position sizes, sector exposure, and diversification.

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
# import seaborn as sns  # Optional

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.LP import solve_lp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk

# Asset configuration (must match generate.py)
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'PG', 'JPM', 'JNJ', 'XOM', 'CAT', 'MCD']
N_ASSETS = 10

# Sector colors
SECTOR_COLORS = {
    'Technology': '#3498DB',
    'Consumer Discretionary': '#9B59B6',
    'Consumer Staples': '#27AE60',
    'Financials': '#E74C3C',
    'Healthcare': '#1ABC9C',
    'Energy': '#F39C12',
    'Industrials': '#34495E',
}

ASSET_SECTORS = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer Discretionary', 'PG': 'Consumer Staples',
    'JPM': 'Financials', 'JNJ': 'Healthcare', 'XOM': 'Energy',
    'CAT': 'Industrials', 'MCD': 'Consumer Discretionary'
}

ASSET_COLORS = [SECTOR_COLORS[ASSET_SECTORS[t]] for t in TICKERS]


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
    print("BENCHMARK: Sierra Vista University Endowment Portfolio (LP)")
    print("Asset Allocation Under Multi-Source Uncertainty")
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

    rho = params.get('rho', 50.0)
    tau = params.get('tau', 0.0)
    beta = 1.0 - params.get('confidence', 0.99)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Assets: {N_ASSETS} ({', '.join(TICKERS)})")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 15")
    print(f"  rho = {rho}, tau = {tau}")
    print()

    # Extract scenario components
    returns = scenarios[:, 0:10]
    vol_regime = scenarios[:, 10]
    corr_stress = scenarios[:, 11]
    factor_returns = scenarios[:, 12:15]

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
    w = x.flatten()
    zeta_vals = zeta.flatten()
    expected_returns = -c.flatten()  # c was negative expected returns

    # Calculate derived quantities
    portfolio_return = np.sum(expected_returns * w) * 100  # As percentage
    portfolio_vol = np.std(returns @ w) * np.sqrt(252) * 100  # Annualized
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

    # Sector allocations
    sector_weights = {}
    for ticker, weight in zip(TICKERS, w):
        sector = ASSET_SECTORS[ticker]
        sector_weights[sector] = sector_weights.get(sector, 0) + weight

    tech_weight = sum(w[i] for i, t in enumerate(TICKERS) if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

    # Print results
    print()
    print("-" * 70)
    print(f"{'OPTIMIZATION RESULTS':^70}")
    print("-" * 70)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.6f}")
    print(f"Expected Return: {portfolio_return:.2f}% (annualized)")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 70)

    print()
    print("OPTIMAL PORTFOLIO WEIGHTS:")
    print("-" * 70)
    print(f"{'Ticker':<8} {'Company':<22} {'Weight':>8} {'Exp.Ret':>10} {'Sector':<18}")
    print("-" * 70)
    for i, ticker in enumerate(TICKERS):
        if assets_df is not None:
            name = assets_df[assets_df['ticker'] == ticker]['name'].values[0]
        else:
            name = ticker
        sector = ASSET_SECTORS[ticker]
        ret = expected_returns[i] * 100
        print(f"{ticker:<8} {name[:22]:<22} {w[i]*100:>7.2f}% {ret:>9.2f}% {sector:<18}")
    print("-" * 70)
    print(f"{'TOTAL':<8} {'':<22} {np.sum(w)*100:>7.2f}%")

    print()
    print("SECTOR ALLOCATION:")
    print("-" * 70)
    for sector, weight in sorted(sector_weights.items(), key=lambda x: -x[1]):
        print(f"  {sector:<25}: {weight*100:>6.2f}%")
    print("-" * 70)
    print(f"  Tech Sector (AAPL+MSFT+GOOGL+AMZN): {tech_weight*100:.2f}% (limit: 40%)")

    # Save results
    solution_data = {
        'portfolio_weights': {ticker: float(w[i]) for i, ticker in enumerate(TICKERS)},
        'sector_weights': {k: float(v) for k, v in sector_weights.items()},
        'expected_return': float(portfolio_return),
        'portfolio_volatility': float(portfolio_vol),
        'sharpe_ratio': float(sharpe),
        'objective_value': float(cost),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': float(rho),
        'tech_weight': float(tech_weight)
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
    bars = ax1.bar(x_pos, w * 100, color=ASSET_COLORS, edgecolor='black')
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.7, label='Min 5%')
    ax1.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Max 25%')
    ax1.set_ylabel('Weight (%)', fontsize=11)
    ax1.set_title('Optimal Portfolio Weights', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(TICKERS, fontsize=9, rotation=45)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 30)

    # ===== Plot 2: Historical Prices (Normalized) =====
    ax2 = fig.add_subplot(gs[0, 1])
    if prices is not None:
        norm_prices = prices / prices.iloc[0] * 100
        for ticker in TICKERS:
            if ticker in norm_prices.columns:
                color = SECTOR_COLORS[ASSET_SECTORS[ticker]]
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

    # ===== Plot 3: Return Distribution =====
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([returns[:, i] * 100 for i in range(N_ASSETS)],
                     patch_artist=True, labels=TICKERS)
    for patch, color in zip(bp['boxes'], ASSET_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Daily Return (%)', fontsize=11)
    ax3.set_title('Scenario Return Distributions', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)

    # ===== Plot 4: Sector Allocation Pie =====
    ax4 = fig.add_subplot(gs[0, 3])
    sector_labels = list(sector_weights.keys())
    sector_values = [sector_weights[s] for s in sector_labels]
    sector_colors = [SECTOR_COLORS[s] for s in sector_labels]
    wedges, texts, autotexts = ax4.pie(sector_values, labels=sector_labels,
                                        colors=sector_colors, autopct='%.1f%%',
                                        explode=[0.02]*len(sector_labels), startangle=90)
    ax4.set_title('Sector Allocation', fontsize=12, fontweight='bold')
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)

    # ===== Plot 5: Return Correlation Heatmap =====
    ax5 = fig.add_subplot(gs[1, 0])
    return_corr = np.corrcoef(returns.T)
    im = ax5.imshow(return_corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(N_ASSETS))
    ax5.set_yticks(range(N_ASSETS))
    ax5.set_xticklabels(TICKERS, fontsize=8, rotation=45)
    ax5.set_yticklabels(TICKERS, fontsize=8)
    ax5.set_title('Return Correlation', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # ===== Plot 6: Volatility Regime Distribution =====
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(vol_regime, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax6.axvline(x=np.mean(vol_regime), color='red', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(vol_regime):.2f}')
    ax6.set_xlabel('Volatility Regime (0=low, 1=high)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Volatility Regime Distribution', fontsize=12, fontweight='bold')
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

    # ===== Plot 8: Factor Exposures =====
    ax8 = fig.add_subplot(gs[1, 3])
    if assets_df is not None:
        betas = assets_df['beta'].values
        sizes = assets_df['size'].values
        values = assets_df['value'].values

        port_beta = np.sum(betas * w)
        port_size = np.sum(sizes * w)
        port_value = np.sum(values * w)

        factors = ['Market Beta', 'Size', 'Value']
        exposures = [port_beta, port_size, port_value]
        colors = ['steelblue', 'orange', 'green']
        bars = ax8.bar(factors, exposures, color=colors, edgecolor='black')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax8.set_ylabel('Portfolio Exposure', fontsize=11)
        ax8.set_title('Factor Exposures', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        for bar, exp in zip(bars, exposures):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{exp:.2f}', ha='center', fontsize=10)
    else:
        ax8.text(0.5, 0.5, 'Factor data not available', ha='center', va='center')
        ax8.set_title('Factor Exposures', fontsize=12, fontweight='bold')

    # ===== Plot 9: Portfolio Return Distribution =====
    ax9 = fig.add_subplot(gs[2, 0])
    portfolio_returns = returns @ w * 100  # Daily returns in %
    ax9.hist(portfolio_returns, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax9.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax9.axvline(x=np.mean(portfolio_returns), color='green', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(portfolio_returns):.3f}%')
    ax9.axvline(x=np.percentile(portfolio_returns, 5), color='red', linestyle='--', linewidth=2,
                label=f'5th pctl={np.percentile(portfolio_returns, 5):.3f}%')
    ax9.set_xlabel('Daily Portfolio Return (%)', fontsize=11)
    ax9.set_ylabel('Frequency', fontsize=11)
    ax9.set_title('Portfolio Return Distribution', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')

    # ===== Plot 10: Correlation Stress Impact =====
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.scatter(corr_stress, vol_regime, alpha=0.3, c='steelblue', s=20)
    ax10.set_xlabel('Correlation Stress', fontsize=11)
    ax10.set_ylabel('Volatility Regime', fontsize=11)
    ax10.set_title('Correlation vs Volatility Regimes', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)

    # ===== Plot 11: Expected Return vs Weight =====
    ax11 = fig.add_subplot(gs[2, 2])
    for i, ticker in enumerate(TICKERS):
        ax11.scatter(expected_returns[i] * 100, w[i] * 100,
                    c=ASSET_COLORS[i], s=100, label=ticker, edgecolor='black')
    ax11.set_xlabel('Expected Return (%)', fontsize=11)
    ax11.set_ylabel('Portfolio Weight (%)', fontsize=11)
    ax11.set_title('Weight vs Expected Return', fontsize=12, fontweight='bold')
    ax11.legend(fontsize=7, ncol=2)
    ax11.grid(True, alpha=0.3)

    # ===== Plot 12: Summary Box =====
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    summary_text = f"""
    SIERRA VISTA ENDOWMENT
    =======================

    Problem:
      10 S&P 500 stocks
      15-dimensional uncertainty
      Min position: 5%
      Max position: 25%
      Tech limit: 40%

    Portfolio Allocation:
      Technology:    {sector_weights.get('Technology', 0)*100:>6.1f}%
      Financials:    {sector_weights.get('Financials', 0)*100:>6.1f}%
      Healthcare:    {sector_weights.get('Healthcare', 0)*100:>6.1f}%
      Cons.Discr:    {sector_weights.get('Consumer Discretionary', 0)*100:>6.1f}%
      Cons.Staples:  {sector_weights.get('Consumer Staples', 0)*100:>6.1f}%
      Energy:        {sector_weights.get('Energy', 0)*100:>6.1f}%
      Industrials:   {sector_weights.get('Industrials', 0)*100:>6.1f}%

    Performance:
      Exp. Return: {portfolio_return:>6.2f}%
      Est. Vol:    {portfolio_vol:>6.2f}%
      Sharpe:      {sharpe:>6.2f}

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

    plt.suptitle('Sierra Vista University Endowment - Scenario Approach Portfolio Optimization',
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
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
