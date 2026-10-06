#!/usr/bin/env python3
# Requires yfinance (not in requirements.txt): pip install yfinance
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
    python generate.py [--seed SEED] [--years Y]
"""

import os
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


def generate_scenarios(returns, seed=42):
    """
    Generate scenarios from ALL trading days (no bootstrap).

    Each trading day becomes one scenario with 12 uncertain parameters:
    - delta[0:8]  = actual daily asset returns
    - delta[8]    = market stress indicator [0, 1] (rolling vol percentile)
    - delta[9]    = credit spread proxy (LQD return minus AGG return)
    - delta[10]   = interest rate proxy (negative TLT return)
    - delta[11]   = volatility scaling (realized/long-term vol ratio)

    Total: 12-dimensional uncertainty, N = number of trading days
    """
    np.random.seed(seed)

    n_days = len(returns)
    scenarios = np.zeros((n_days, 12))

    # Compute statistics
    mean_returns, volatility, correlation = compute_statistics(returns)

    # Compute rolling volatility for stress detection
    rolling_vol = returns.rolling(window=20).std().mean(axis=1)
    long_term_vol = returns.std().mean()
    vol_percentiles = rolling_vol.rank(pct=True)

    for i in range(n_days):
        # Actual daily returns (no bootstrap, no scaling)
        scenarios[i, 0:8] = returns.iloc[i].values

        # Market stress indicator from rolling volatility percentile
        if not np.isnan(vol_percentiles.iloc[i]):
            scenarios[i, 8] = vol_percentiles.iloc[i]
        else:
            scenarios[i, 8] = 0.5  # neutral for initial window

        # Credit spread proxy: LQD (credit) minus AGG (treasury) return
        lqd_ret = returns.iloc[i][TICKERS[7]]  # LQD
        agg_ret = returns.iloc[i][TICKERS[1]]  # AGG
        scenarios[i, 9] = lqd_ret - agg_ret

        # Interest rate proxy: negative TLT return (rates up = bonds down)
        scenarios[i, 10] = -returns.iloc[i][TICKERS[5]]  # -TLT

        # Volatility scaling: realized/long-term ratio
        if not np.isnan(rolling_vol.iloc[i]) and long_term_vol > 0:
            scenarios[i, 11] = rolling_vol.iloc[i] / long_term_vol
        else:
            scenarios[i, 11] = 1.0

    return scenarios, mean_returns, volatility, correlation


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


def save_constraint_files(data_dir, mean_returns, n_scenarios):
    """Generate and save constraint CSV files with shared-slack augmentation.

    The data files encode the augmented formulation directly:
        Original:  min c'x + rho * 1'zeta   s.t. A(d)x + b(d) <= zeta, Gx+h <= 0, zeta >= 0
        Augmented: min c_aug' x_aug          s.t. A_aug(d) x_aug + b(d) <= 0, G_aug x_aug + h_aug <= 0

    where x_aug = [x; zeta] in R^(n + m), with:
        c_aug = [c; rho * 1_m]
        A_aug = [A, -I_m]        (slack absorbs scenario violations)
        G_aug = [[G, 0]; [0, -I_m]]   (original hard constraints + zeta >= 0)
        h_aug = [h; 0]

    CVaR Formulation:
    Decision variables: x = [w_1, ..., w_8, alpha]
    where w_i are portfolio weights and alpha is the VaR threshold.

    CVaR at level beta is computed as:
        CVaR_beta = alpha + (1/(1-beta)) * E[max(0, -r^T w - alpha)]

    The scenario approach constraint is:
        -r^T w - alpha <= zeta  for each scenario

    This ensures the portfolio loss doesn't exceed alpha + zeta.
    With rho = 1/((1-beta)*N), minimizing alpha + rho*sum(zeta) approximates CVaR.
    """
    params = load_parameters()
    rho = params.get('rho', 0.015936)

    n_vars = N_ASSETS + 1  # 8 weights + 1 alpha (VaR threshold)

    # ---- Build original A_d rows (4 x 9 expression strings) ----

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

    m_scenario = len(a_d_rows)  # 4

    # ---- Augment A_d: append -I_m slack columns ----
    a_d_aug_rows = []
    for i, row_str in enumerate(a_d_rows):
        slack_cols = ['0'] * m_scenario
        slack_cols[i] = '-1'
        a_d_aug_rows.append(row_str + ',' + ','.join(slack_cols))

    with open(os.path.join(data_dir, 'A_d.csv'), 'w') as f:
        f.write('\n'.join(a_d_aug_rows) + '\n')

    # ---- b_d.csv: unchanged (4 rows) ----
    b_d_rows = ['0'] * m_scenario

    with open(os.path.join(data_dir, 'b_d.csv'), 'w') as f:
        f.write('\n'.join(b_d_rows) + '\n')

    # ---- c.csv: augmented cost vector [c; rho * 1_m] ----
    # Original: c = [0, ..., 0, 1] (weights have 0 cost, alpha has cost 1)
    c_values = [0] * N_ASSETS + [1] + [rho] * m_scenario
    with open(os.path.join(data_dir, 'c.csv'), 'w') as f:
        for val in c_values:
            f.write(f'{val}\n')

    # ---- G.csv and h.csv: augmented hard constraints ----
    # Original hard constraints on x (padded with zero slack columns):
    # Format: G @ x + h <= 0 => G @ x <= -h
    # Plus zeta >= 0 block: -zeta_i <= 0

    n_aug = n_vars + m_scenario
    G_rows = []
    h_vals = []

    # 1. Minimum position constraints: w[j] >= MIN_POSITION => -w[j] + MIN_POSITION <= 0
    for j in range(N_ASSETS):
        row = [0] * n_aug
        row[j] = -1
        G_rows.append(row)
        h_vals.append(MIN_POSITION)

    # 2. Maximum position constraints: w[j] <= MAX_POSITION => w[j] - MAX_POSITION <= 0
    for j in range(N_ASSETS):
        row = [0] * n_aug
        row[j] = 1
        G_rows.append(row)
        h_vals.append(-MAX_POSITION)

    # 3. Budget constraint: sum(w) = 1
    # sum(w) <= 1
    sum_row = [1] * N_ASSETS + [0] + [0] * m_scenario
    G_rows.append(sum_row)
    h_vals.append(-1.0)

    # sum(w) >= 1 => -sum(w) + 1 <= 0
    neg_sum_row = [-1] * N_ASSETS + [0] + [0] * m_scenario
    G_rows.append(neg_sum_row)
    h_vals.append(1.0)

    # 4. Minimum equity constraint: w[SPY] + w[EFA] + w[VWO] >= 0.30
    # => -(w[0] + w[4] + w[6]) + 0.30 <= 0
    min_equity_row = [0] * n_aug
    min_equity_row[0] = -1  # SPY
    min_equity_row[4] = -1  # EFA
    min_equity_row[6] = -1  # VWO
    G_rows.append(min_equity_row)
    h_vals.append(MIN_EQUITY)

    # 5. Maximum equity constraint: w[SPY] + w[EFA] + w[VWO] <= 0.60
    max_equity_row = [0] * n_aug
    max_equity_row[0] = 1  # SPY
    max_equity_row[4] = 1  # EFA
    max_equity_row[6] = 1  # VWO
    G_rows.append(max_equity_row)
    h_vals.append(-MAX_EQUITY)

    # 6. Minimum fixed income: w[AGG] + w[TLT] + w[LQD] >= 0.25
    min_fi_row = [0] * n_aug
    min_fi_row[1] = -1  # AGG
    min_fi_row[5] = -1  # TLT
    min_fi_row[7] = -1  # LQD
    G_rows.append(min_fi_row)
    h_vals.append(MIN_FIXED_INCOME)

    # 7. Alpha (VaR threshold) bounds: -0.05 <= alpha <= 0.05
    # These bounds allow reasonable VaR values for daily returns
    # Lower bound: alpha >= -0.05 => -alpha <= 0.05
    alpha_lower = [0] * n_aug
    alpha_lower[8] = -1
    G_rows.append(alpha_lower)
    h_vals.append(0.05)

    # Upper bound: alpha <= 0.05
    alpha_upper = [0] * n_aug
    alpha_upper[8] = 1
    G_rows.append(alpha_upper)
    h_vals.append(-0.05)

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

    print(f"  Augmented formulation: x_aug = [x; zeta] in R^{n_aug}")
    print(f"  A_d: {m_scenario} x {n_aug}, c: {n_aug}, G: {len(G_rows)} x {n_aug}, h: {len(h_vals)}")

    # Save asset info for visualization
    with open(os.path.join(data_dir, 'assets.csv'), 'w') as f:
        f.write('ticker,name,class,role\n')
        for ticker in TICKERS:
            info = ASSET_INFO[ticker]
            f.write(f"{ticker},{info['name']},{info['class']},{info['role']}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate Pension Fund CVaR benchmark')
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
    print(f"Uncertainty dimensions: 12")
    print(f"  - 8 asset returns (all trading days from {args.years} years)")
    print(f"  - 1 market stress indicator (rolling vol percentile)")
    print(f"  - 1 credit spread proxy (LQD - AGG return)")
    print(f"  - 1 interest rate proxy (negative TLT return)")
    print(f"  - 1 volatility scaling (realized/long-term vol ratio)")
    print()

    # Set up directories
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Download real data
    print("Downloading data from Yahoo Finance...")
    prices, returns = download_stock_data(TICKERS, args.years)
    n_scenarios = len(returns)
    print(f"Downloaded {n_scenarios} trading days of data")
    print(f"Using ALL {n_scenarios} trading days as scenarios (no bootstrap)")
    print()

    # Update rho in parameters.txt based on actual N
    rho = 1.0 / ((1 - CVAR_ALPHA) * n_scenarios)
    params_path = os.path.join(benchmark_dir, 'parameters.txt')
    with open(params_path, 'w') as f:
        f.write("# Pension Fund CVaR Portfolio Optimization (LP)\n")
        f.write("# 8 ETFs: SPY, AGG, VNQ, GLD, EFA, TLT, VWO, LQD\n")
        f.write("#\n")
        f.write(f"# Augmented formulation: x_aug = [w; alpha; zeta] in R^13\n")
        f.write("#   w[0:8]    = portfolio weights\n")
        f.write("#   alpha     = VaR threshold\n")
        f.write("#   zeta[0:4] = shared slack variables (CVaR constraint violations)\n")
        f.write("#\n")
        f.write("# Uncertainty (12D):\n")
        f.write("#   delta[0:8]  = asset returns (all trading days from Yahoo Finance)\n")
        f.write("#   delta[8]    = market stress indicator\n")
        f.write("#   delta[9]    = credit spread proxy\n")
        f.write("#   delta[10]   = interest rate proxy\n")
        f.write("#   delta[11]   = volatility scaling factor\n")
        f.write("#\n")
        f.write(f"# Scenario constraints (4): CVaR loss, credit spread, rate duration, equity correlation\n")
        f.write(f"# Hard constraints (27): position limits 5-30%, sum(w)=1, equity 30-60%,\n")
        f.write(f"#                         fixed income >= 25%, alpha bounds, zeta >= 0\n")
        f.write(f"# Objective: min alpha + rho * sum(zeta)  (approximates CVaR at 95% level)\n")
        f.write(f"# rho = 1/((1-0.95)*N) = {rho:.6f} for N={n_scenarios}\n")
        f.write(f"\n")
        f.write(f"rho = {rho:.6f}\n")
        f.write("tau = 0\n")
        f.write("confidence = 0.999999\n")
    print(f"Updated parameters.txt: rho = {rho:.6f} (for N={n_scenarios})")

    # Generate scenarios
    print("Generating scenarios...")
    scenarios, mean_returns, volatility, correlation = generate_scenarios(
        returns, args.seed
    )

    # Save scenarios
    np.savetxt(os.path.join(data_dir, 'scenarios.csv'), scenarios, delimiter=',', fmt='%.10f')
    print(f"Saved data/scenarios.csv: {n_scenarios} x 12")

    # Save prices for visualization
    prices.to_csv(os.path.join(data_dir, 'historical_prices.csv'))
    print("Saved data/historical_prices.csv")

    # Save returns for reference
    returns.to_csv(os.path.join(data_dir, 'historical_returns.csv'))
    print("Saved data/historical_returns.csv")

    # Save constraint files
    print("Generating constraint files...")
    save_constraint_files(data_dir, mean_returns, n_scenarios)
    print("Saved: data/A_d.csv, b_d.csv, c.csv, G.csv, h.csv, assets.csv")

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
