#!/usr/bin/env python3
"""
Golden State Teachers' Pension Fund CVaR Portfolio Benchmark

This benchmark models the investment decisions faced by Sarah Mitchell,
Chief Risk Officer of the Golden State Teachers' Pension Fund - a $45 billion
defined benefit pension serving 300,000 active and retired California teachers.

The fund must maintain strict risk controls while generating returns sufficient
to meet long-term pension obligations. California state law requires that the
fund's CVaR (Conditional Value at Risk) at the 95% confidence level must not
exceed 2.5% in any single day, while targeting 7% annual returns.

Real ETF data is downloaded from Yahoo Finance via yfinance.

Data Sources:
- Yahoo Finance: Historical ETF prices (5 years daily)
- Reference: Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional
  Value-at-Risk." Journal of Risk, 2, 21-42.

Usage:
    python generate.py [--n_scenarios N] [--seed SEED] [--years Y]
"""

import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance not installed. Run: pip install yfinance")
    exit(1)


# ============================================================
# PORTFOLIO CONFIGURATION
# Diversified pension fund with 8 assets across asset classes
# ============================================================

TICKERS = ['SPY', 'AGG', 'VNQ', 'GLD', 'EFA', 'TLT', 'VWO', 'LQD']

ASSET_INFO = {
    'SPY': {'name': 'S&P 500 ETF', 'class': 'US Equity', 'role': 'Core Growth'},
    'AGG': {'name': 'US Aggregate Bond', 'class': 'Fixed Income', 'role': 'Stability'},
    'VNQ': {'name': 'Real Estate (REIT)', 'class': 'Real Assets', 'role': 'Inflation Hedge'},
    'GLD': {'name': 'Gold', 'class': 'Commodities', 'role': 'Crisis Hedge'},
    'EFA': {'name': 'Intl Developed', 'class': 'Intl Equity', 'role': 'Diversification'},
    'TLT': {'name': '20+ Year Treasury', 'class': 'Long Bonds', 'role': 'Duration'},
    'VWO': {'name': 'Emerging Markets', 'class': 'Intl Equity', 'role': 'Growth'},
    'LQD': {'name': 'Investment Grade Corp', 'class': 'Credit', 'role': 'Yield'},
}

N_ASSETS = len(TICKERS)

# CVaR parameters
CVAR_ALPHA = 0.95  # 95% CVaR (worst 5% of scenarios)

# Position limits (pension fund style)
MIN_EQUITY = 0.30  # At least 30% in equities (SPY + EFA + VWO)
MAX_EQUITY = 0.60  # At most 60% in equities
MIN_FIXED_INCOME = 0.25  # At least 25% in bonds (AGG + TLT + LQD)
MIN_POSITION = 0.05  # 5% minimum per asset
MAX_POSITION = 0.30  # 30% maximum per asset


def download_stock_data(tickers, years=5):
    """Download historical stock data from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)

    print(f"Downloading {years} years of data for {len(tickers)} tickers...")

    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    return prices, returns


def compute_statistics(returns):
    """Compute mean returns, volatility, and correlation matrix."""
    mean_returns = returns.mean() * 252  # Annualized
    volatility = returns.std() * np.sqrt(252)
    correlation = returns.corr()

    return mean_returns, volatility, correlation


def generate_scenarios(returns, n_scenarios, seed=42):
    """
    Generate scenarios with 12 uncertain parameters:
    - delta[0:8]  = asset returns (bootstrapped from historical data)
    - delta[8]    = market stress indicator [0, 1]
    - delta[9]    = credit spread shock [-0.02, 0.02]
    - delta[10]   = interest rate shock [-0.01, 0.01]
    - delta[11]   = volatility scaling factor [0.8, 1.5]

    Total: 12-dimensional uncertainty
    """
    np.random.seed(seed)

    n_days = len(returns)
    scenarios = np.zeros((n_scenarios, 12))

    # Compute statistics
    mean_returns, volatility, correlation = compute_statistics(returns)

    # Compute rolling volatility for stress detection
    rolling_vol = returns.rolling(window=20).std().mean(axis=1).dropna()
    vol_percentiles = rolling_vol.rank(pct=True).values

    for i in range(n_scenarios):
        # Bootstrap: sample a random day's returns
        idx = np.random.randint(0, n_days)
        daily_returns = returns.iloc[idx].values

        # Volatility scaling based on regime
        vol_scale = np.random.uniform(0.8, 1.5)

        # Scale returns by volatility factor
        scaled_returns = daily_returns * vol_scale
        scenarios[i, 0:8] = scaled_returns

        # Market stress indicator
        if len(vol_percentiles) > 0 and idx < len(vol_percentiles):
            stress = vol_percentiles[min(idx, len(vol_percentiles)-1)]
        else:
            stress = np.random.uniform(0, 1)
        scenarios[i, 8] = stress

        # Credit spread shock (wider in stress)
        credit_shock = np.random.uniform(-0.01, 0.01) - stress * 0.01
        scenarios[i, 9] = credit_shock

        # Interest rate shock
        rate_shock = np.random.uniform(-0.005, 0.005)
        scenarios[i, 10] = rate_shock

        # Volatility scaling factor
        scenarios[i, 11] = vol_scale

    return scenarios, mean_returns, volatility, correlation


def save_constraint_files(mean_returns, n_scenarios):
    """Generate and save all constraint CSV files for CVaR formulation.

    CVaR Formulation:
    Decision variables: x = [w_1, ..., w_8, alpha]
    where w_i are portfolio weights and alpha is the VaR threshold.

    CVaR at level β is computed as:
        CVaR_β = alpha + (1/(1-β)) * E[max(0, -r^T w - alpha)]

    The scenario approach constraint is:
        -r^T w - alpha <= zeta  for each scenario

    This ensures the portfolio loss doesn't exceed alpha + zeta.
    With rho = 1/((1-β)*N), minimizing alpha + rho*sum(zeta) approximates CVaR.
    """

    n_vars = N_ASSETS + 1  # 8 weights + 1 alpha (VaR threshold)

    # A_d.csv: Scenario-dependent constraint coefficients
    # For CVaR: -return * w - alpha <= zeta
    # Rows:
    #   Row 0-7: Base loss constraint with stress amplification
    #   Row 8: Credit-sensitive assets (AGG, TLT, LQD) with credit shock
    #   Row 9: Rate-sensitive assets with rate shock

    a_d_rows = []

    # Main CVaR constraint: -(1 + stress * delta[8]) * delta[i] * w[i] - alpha <= zeta
    # This creates a single row that sums losses across all assets
    main_row = []
    for j in range(N_ASSETS):
        # Amplify losses during stress periods
        main_row.append(f'-(1+0.3*delta[8])*delta[{j}]')
    main_row.append('-1')  # Coefficient for alpha
    a_d_rows.append(','.join(main_row))

    # Credit spread impact on bond assets (AGG=1, TLT=5, LQD=7)
    credit_row = ['0'] * n_vars
    credit_row[1] = '-delta[9]*0.5'   # AGG: moderate credit sensitivity
    credit_row[5] = '-delta[9]*0.3'   # TLT: some credit sensitivity
    credit_row[7] = '-delta[9]*1.0'   # LQD: high credit sensitivity
    credit_row[8] = '0'  # alpha coefficient
    a_d_rows.append(','.join(credit_row))

    # Interest rate impact on duration assets
    rate_row = ['0'] * n_vars
    rate_row[1] = '-delta[10]*5'    # AGG: moderate duration (~5 years)
    rate_row[5] = '-delta[10]*18'   # TLT: high duration (~18 years)
    rate_row[7] = '-delta[10]*8'    # LQD: medium duration (~8 years)
    rate_row[8] = '0'  # alpha coefficient
    a_d_rows.append(','.join(rate_row))

    # Equity correlation stress (equities move together in crisis)
    equity_row = ['0'] * n_vars
    equity_idx = [0, 4, 6]  # SPY, EFA, VWO
    for idx in equity_idx:
        equity_row[idx] = f'-(delta[8]*0.5)*delta[{idx}]'
    equity_row[8] = '0'
    a_d_rows.append(','.join(equity_row))

    with open('A_d.csv', 'w') as f:
        f.write('\n'.join(a_d_rows) + '\n')

    # b_d.csv: All zeros (constraints are <= zeta)
    b_d_rows = ['0'] * 4

    with open('b_d.csv', 'w') as f:
        f.write('\n'.join(b_d_rows) + '\n')

    # c.csv: Objective coefficients
    # Minimize: alpha + rho * sum(zeta)
    # c = [0, 0, ..., 0, 1] - weights have 0 cost, alpha has cost 1
    c_values = [0] * N_ASSETS + [1]
    with open('c.csv', 'w') as f:
        for val in c_values:
            f.write(f'{val}\n')

    # G.csv and h.csv: Hard constraints
    # Format: G @ x + h <= 0 => G @ x <= -h
    G_rows = []
    h_vals = []

    # 1. Minimum position constraints: w[j] >= MIN_POSITION => -w[j] + MIN_POSITION <= 0
    for j in range(N_ASSETS):
        row = [0] * n_vars
        row[j] = -1
        G_rows.append(row)
        h_vals.append(MIN_POSITION)

    # 2. Maximum position constraints: w[j] <= MAX_POSITION => w[j] - MAX_POSITION <= 0
    for j in range(N_ASSETS):
        row = [0] * n_vars
        row[j] = 1
        G_rows.append(row)
        h_vals.append(-MAX_POSITION)

    # 3. Budget constraint: sum(w) = 1
    # sum(w) <= 1
    sum_row = [1] * N_ASSETS + [0]
    G_rows.append(sum_row)
    h_vals.append(-1.0)

    # sum(w) >= 1 => -sum(w) + 1 <= 0
    neg_sum_row = [-1] * N_ASSETS + [0]
    G_rows.append(neg_sum_row)
    h_vals.append(1.0)

    # 4. Minimum equity constraint: w[SPY] + w[EFA] + w[VWO] >= 0.30
    # => -(w[0] + w[4] + w[6]) + 0.30 <= 0
    min_equity_row = [0] * n_vars
    min_equity_row[0] = -1  # SPY
    min_equity_row[4] = -1  # EFA
    min_equity_row[6] = -1  # VWO
    G_rows.append(min_equity_row)
    h_vals.append(MIN_EQUITY)

    # 5. Maximum equity constraint: w[SPY] + w[EFA] + w[VWO] <= 0.60
    max_equity_row = [0] * n_vars
    max_equity_row[0] = 1  # SPY
    max_equity_row[4] = 1  # EFA
    max_equity_row[6] = 1  # VWO
    G_rows.append(max_equity_row)
    h_vals.append(-MAX_EQUITY)

    # 6. Minimum fixed income: w[AGG] + w[TLT] + w[LQD] >= 0.25
    min_fi_row = [0] * n_vars
    min_fi_row[1] = -1  # AGG
    min_fi_row[5] = -1  # TLT
    min_fi_row[7] = -1  # LQD
    G_rows.append(min_fi_row)
    h_vals.append(MIN_FIXED_INCOME)

    # 7. Alpha (VaR threshold) bounds: -0.05 <= alpha <= 0.05
    # These bounds allow reasonable VaR values for daily returns
    # Lower bound: alpha >= -0.05 => -alpha <= 0.05
    alpha_lower = [0] * n_vars
    alpha_lower[8] = -1
    G_rows.append(alpha_lower)
    h_vals.append(0.05)

    # Upper bound: alpha <= 0.05
    alpha_upper = [0] * n_vars
    alpha_upper[8] = 1
    G_rows.append(alpha_upper)
    h_vals.append(-0.05)

    with open('G.csv', 'w') as f:
        for row in G_rows:
            f.write(','.join(str(x) for x in row) + '\n')

    with open('h.csv', 'w') as f:
        for h in h_vals:
            f.write(f'{h}\n')

    # Save asset info for visualization
    with open('assets.csv', 'w') as f:
        f.write('ticker,name,class,role\n')
        for ticker in TICKERS:
            info = ASSET_INFO[ticker]
            f.write(f"{ticker},{info['name']},{info['class']},{info['role']}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate Pension Fund CVaR benchmark')
    parser.add_argument('--n_scenarios', type=int, default=500,
                        help='Number of scenarios (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data (default: 5)')
    args = parser.parse_args()

    print("=" * 65)
    print("Golden State Teachers' Pension Fund CVaR Benchmark")
    print("=" * 65)
    print()
    print(f"Assets: {N_ASSETS}")
    for ticker in TICKERS:
        info = ASSET_INFO[ticker]
        print(f"  {ticker:5s} - {info['name']:22s} ({info['class']}, {info['role']})")
    print()
    print(f"Scenarios: {args.n_scenarios}")
    print(f"Uncertainty dimensions: 12")
    print(f"  - 8 asset returns (bootstrapped from {args.years} years of data)")
    print(f"  - 1 market stress indicator")
    print(f"  - 1 credit spread shock")
    print(f"  - 1 interest rate shock")
    print(f"  - 1 volatility scaling factor")
    print()

    # Download real data
    print("Downloading data from Yahoo Finance...")
    prices, returns = download_stock_data(TICKERS, args.years)
    print(f"Downloaded {len(returns)} trading days of data")
    print()

    # Generate scenarios
    print("Generating scenarios...")
    scenarios, mean_returns, volatility, correlation = generate_scenarios(
        returns, args.n_scenarios, args.seed
    )

    # Save scenarios
    np.savetxt('scenarios.csv', scenarios, delimiter=',', fmt='%.10f')
    print(f"Saved scenarios.csv: {args.n_scenarios} x 12")

    # Save prices for visualization
    prices.to_csv('historical_prices.csv')
    print("Saved historical_prices.csv")

    # Save returns for reference
    returns.to_csv('historical_returns.csv')
    print("Saved historical_returns.csv")

    # Save constraint files
    print("Generating constraint files...")
    save_constraint_files(mean_returns, args.n_scenarios)
    print("Saved: A_d.csv, b_d.csv, c.csv, G.csv, h.csv, assets.csv")

    # Save parameters
    rho = 1.0 / ((1 - CVAR_ALPHA) * args.n_scenarios)
    with open('parameters.txt', 'w') as f:
        f.write("# Golden State Teachers' Pension Fund CVaR Benchmark\n")
        f.write("# 8 ETF assets, 12D uncertainty\n")
        f.write("#\n")
        f.write("# CVaR formulation at 95% confidence level\n")
        f.write("# rho = 1/((1-alpha)*N) where alpha=0.95, N=500\n")
        f.write("#\n")
        f.write("# Uncertainty:\n")
        f.write("#   delta[0:8]  = asset returns\n")
        f.write("#   delta[8]    = market stress indicator\n")
        f.write("#   delta[9]    = credit spread shock\n")
        f.write("#   delta[10]   = interest rate shock\n")
        f.write("#   delta[11]   = volatility scaling\n")
        f.write("#\n")
        f.write("# Decision: portfolio weights w[0:8] + VaR threshold alpha\n")
        f.write("\n")
        f.write(f"rho = {rho:.6f}\n")
        f.write("tau = 0\n")
        f.write("confidence = 0.99\n")
    print("Saved parameters.txt")

    # Print summary statistics
    print()
    print("=" * 65)
    print("Summary Statistics")
    print("=" * 65)

    print("\nAnnualized Returns:")
    for ticker in TICKERS:
        ret = mean_returns[ticker]
        vol = volatility[ticker]
        sharpe = ret / vol if vol > 0 else 0
        print(f"  {ticker:5s}: return={ret*100:6.2f}%, vol={vol*100:5.2f}%, Sharpe={sharpe:.2f}")

    print("\nScenario Statistics:")
    print(f"  Asset returns (delta[0:8]):")
    for j, ticker in enumerate(TICKERS):
        r = scenarios[:, j]
        print(f"    {ticker:5s}: mean={np.mean(r)*100:6.3f}%, std={np.std(r)*100:5.3f}%")

    print(f"\n  Market stress (delta[8]):")
    print(f"    mean={np.mean(scenarios[:, 8]):.3f}, range=[{np.min(scenarios[:, 8]):.3f}, {np.max(scenarios[:, 8]):.3f}]")

    print(f"\n  Credit shock (delta[9]):")
    print(f"    mean={np.mean(scenarios[:, 9])*100:.3f}%, range=[{np.min(scenarios[:, 9])*100:.3f}%, {np.max(scenarios[:, 9])*100:.3f}%]")

    print(f"\n  Rate shock (delta[10]):")
    print(f"    mean={np.mean(scenarios[:, 10])*100:.4f}%, range=[{np.min(scenarios[:, 10])*100:.4f}%, {np.max(scenarios[:, 10])*100:.4f}%]")

    print("\nConstraints:")
    print(f"  Min position: {MIN_POSITION*100:.0f}%")
    print(f"  Max position: {MAX_POSITION*100:.0f}%")
    print(f"  Equity range: {MIN_EQUITY*100:.0f}% - {MAX_EQUITY*100:.0f}%")
    print(f"  Min fixed income: {MIN_FIXED_INCOME*100:.0f}%")
    print(f"  CVaR confidence: {CVAR_ALPHA*100:.0f}%")


if __name__ == '__main__':
    main()
