#!/usr/bin/env python3
"""
Sierra Vista University Endowment Portfolio Benchmark Data Generator

This benchmark models the asset allocation decision faced by Dr. Elizabeth Chen,
Chief Investment Officer of Sierra Vista University's $850M endowment fund.

The portfolio must balance growth objectives against the university's need for
stable annual distributions while managing multiple sources of uncertainty:
- Individual asset returns
- Volatility regime shifts
- Correlation breakdowns during market stress
- Systematic factor exposures (market, size, value)

Real stock data is downloaded from Yahoo Finance via yfinance.

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
# 10 diversified assets across sectors
# ============================================================

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'PG', 'JPM', 'JNJ', 'XOM', 'CAT', 'MCD']

ASSET_INFO = {
    'AAPL': {'name': 'Apple', 'sector': 'Technology', 'role': 'Growth'},
    'MSFT': {'name': 'Microsoft', 'sector': 'Technology', 'role': 'Growth'},
    'GOOGL': {'name': 'Alphabet', 'sector': 'Technology', 'role': 'Growth'},
    'AMZN': {'name': 'Amazon', 'sector': 'Consumer Discretionary', 'role': 'Growth'},
    'PG': {'name': 'Procter & Gamble', 'sector': 'Consumer Staples', 'role': 'Defensive'},
    'JPM': {'name': 'JPMorgan Chase', 'sector': 'Financials', 'role': 'Value'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'role': 'Defensive'},
    'XOM': {'name': 'ExxonMobil', 'sector': 'Energy', 'role': 'Value/Cyclical'},
    'CAT': {'name': 'Caterpillar', 'sector': 'Industrials', 'role': 'Cyclical'},
    'MCD': {'name': "McDonald's", 'sector': 'Consumer Discretionary', 'role': 'Defensive'},
}

N_ASSETS = len(TICKERS)

# Fama-French factor loadings (approximate)
# Format: [market_beta, size_loading, value_loading]
FACTOR_LOADINGS = {
    'AAPL': [1.20, -0.30, -0.40],  # Large growth
    'MSFT': [1.10, -0.25, -0.30],
    'GOOGL': [1.15, -0.20, -0.35],
    'AMZN': [1.30, -0.35, -0.50],  # High beta growth
    'PG': [0.60, 0.10, 0.30],      # Defensive value
    'JPM': [1.10, 0.15, 0.40],     # Financial value
    'JNJ': [0.70, 0.05, 0.25],     # Defensive
    'XOM': [0.90, 0.20, 0.50],     # Energy value
    'CAT': [1.20, 0.25, 0.35],     # Cyclical
    'MCD': [0.80, 0.10, 0.20],     # Defensive
}

# Position limits
MIN_POSITION = 0.05  # 5% minimum per asset
MAX_POSITION = 0.25  # 25% maximum per asset
TECH_SECTOR_LIMIT = 0.40  # 40% max in tech


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


def compute_rolling_volatility(returns, window=20):
    """Compute rolling volatility for regime detection."""
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol


def generate_scenarios(returns, n_scenarios, seed=42):
    """
    Generate scenarios with 15 uncertain parameters:
    - delta[0:10]  = asset returns (bootstrapped from historical + noise)
    - delta[10]    = volatility regime indicator [0, 1]
    - delta[11]    = correlation stress factor [0, 0.5]
    - delta[12:15] = factor returns (market, size, value)

    Total: 15-dimensional uncertainty
    """
    np.random.seed(seed)

    n_days = len(returns)
    scenarios = np.zeros((n_scenarios, 15))

    # Compute statistics
    mean_returns, volatility, correlation = compute_statistics(returns)

    # Compute rolling volatility for regime detection
    rolling_vol = compute_rolling_volatility(returns)
    vol_percentiles = rolling_vol.mean(axis=1).dropna().rank(pct=True)
    vol_pct_values = vol_percentiles.values

    for i in range(n_scenarios):
        # Bootstrap: sample a random day's returns
        idx = np.random.randint(0, n_days)
        daily_returns = returns.iloc[idx].values

        # Add some noise to create variation
        noise = np.random.randn(N_ASSETS) * volatility.values * 0.1
        scenarios[i, 0:10] = daily_returns + noise

        # Volatility regime: sample from historical percentiles
        if len(vol_pct_values) > 0:
            vol_idx = np.random.randint(0, len(vol_pct_values))
            vol_regime = vol_pct_values[vol_idx]
        else:
            vol_regime = np.random.uniform(0, 1)
        scenarios[i, 10] = vol_regime

        # Correlation stress: higher in high-vol regimes
        base_corr_stress = np.random.uniform(0, 0.3)
        corr_stress = base_corr_stress + vol_regime * 0.2  # More stress in high vol
        scenarios[i, 11] = min(corr_stress, 0.5)

        # Factor returns (market, size, value)
        # Market factor: correlated with average return
        market_factor = np.mean(daily_returns) + np.random.randn() * 0.01
        scenarios[i, 12] = market_factor

        # Size factor: small minus big (random)
        size_factor = np.random.randn() * 0.008
        scenarios[i, 13] = size_factor

        # Value factor: high minus low (random)
        value_factor = np.random.randn() * 0.008
        scenarios[i, 14] = value_factor

    return scenarios, mean_returns, volatility, correlation


def save_constraint_files(mean_returns, volatility):
    """Generate and save all constraint CSV files."""

    # A_d.csv: 14 rows x 10 cols
    # Rows 0-9: Volatility-adjusted individual asset loss
    # Row 10: Correlation stress on tech sector
    # Rows 11-13: Factor exposure constraints

    a_d_rows = []

    # Rows 0-9: -(1 + 0.5*delta[10]) * delta[i] * w[i]
    # This creates diagonal entries for each asset
    for j in range(N_ASSETS):
        row = ['0'] * N_ASSETS
        row[j] = f'-(1+0.5*delta[10])*delta[{j}]'
        a_d_rows.append(','.join(row))

    # Row 10: Correlation stress on tech sector (AAPL, MSFT, GOOGL, AMZN)
    # -(1 + delta[11]) * sum(delta[i] * w[i]) for tech stocks
    tech_row = []
    for j, ticker in enumerate(TICKERS):
        if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            tech_row.append(f'-(1+delta[11])*delta[{j}]')
        else:
            tech_row.append('0')
    a_d_rows.append(','.join(tech_row))

    # Rows 11-13: Factor exposures
    # Market factor: -delta[12] * beta[i]
    market_row = []
    for ticker in TICKERS:
        beta = FACTOR_LOADINGS[ticker][0]
        market_row.append(f'-delta[12]*{beta}')
    a_d_rows.append(','.join(market_row))

    # Size factor: -delta[13] * size_loading[i]
    size_row = []
    for ticker in TICKERS:
        size = FACTOR_LOADINGS[ticker][1]
        size_row.append(f'-delta[13]*{size}')
    a_d_rows.append(','.join(size_row))

    # Value factor: -delta[14] * value_loading[i]
    value_row = []
    for ticker in TICKERS:
        value = FACTOR_LOADINGS[ticker][2]
        value_row.append(f'-delta[14]*{value}')
    a_d_rows.append(','.join(value_row))

    with open('A_d.csv', 'w') as f:
        f.write('\n'.join(a_d_rows) + '\n')

    # b_d.csv: 14 rows, all zeros
    # (constraints are: A_d @ w <= zeta, i.e., loss <= zeta)
    b_d_rows = ['0'] * 14

    with open('b_d.csv', 'w') as f:
        f.write('\n'.join(b_d_rows) + '\n')

    # c.csv: negative expected returns (to maximize return while minimizing cost)
    # We use negative because solve_lp minimizes c'x
    with open('c.csv', 'w') as f:
        for ticker in TICKERS:
            # Annualized mean return, converted to negative for minimization
            ret = -float(mean_returns[ticker])
            f.write(f'{ret}\n')

    # G.csv and h.csv: hard constraints
    # 1. Non-negativity: -w[j] <= 0 (not needed with min position, skip)
    # 2. Min position: -w[j] <= -MIN_POSITION
    # 3. Max position: w[j] <= MAX_POSITION
    # 4. Tech sector: w[AAPL]+w[MSFT]+w[GOOGL]+w[AMZN] <= TECH_SECTOR_LIMIT
    # 5. Budget: sum(w) = 1 (as two inequalities: sum >= 1 and sum <= 1)

    G_rows = []
    h_vals = []

    # Min position: w[j] >= MIN_POSITION  =>  -w[j] + MIN_POSITION <= 0
    for j in range(N_ASSETS):
        row = [0] * N_ASSETS
        row[j] = -1
        G_rows.append(row)
        h_vals.append(MIN_POSITION)  # Positive because: -w + MIN_POS <= 0 => w >= MIN_POS

    # Max position: w[j] <= MAX_POSITION  =>  w[j] - MAX_POSITION <= 0
    for j in range(N_ASSETS):
        row = [0] * N_ASSETS
        row[j] = 1
        G_rows.append(row)
        h_vals.append(-MAX_POSITION)  # Note: negative because G @ x + h <= 0

    # Tech sector limit: w[0]+w[1]+w[2]+w[3] <= TECH_SECTOR_LIMIT
    tech_row = [0] * N_ASSETS
    for j, ticker in enumerate(TICKERS):
        if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            tech_row[j] = 1
    G_rows.append(tech_row)
    h_vals.append(-TECH_SECTOR_LIMIT)  # Note: negative

    # Budget: sum(w) <= 1  =>  sum(w) - 1 <= 0
    sum_row = [1] * N_ASSETS
    G_rows.append(sum_row)
    h_vals.append(-1.0)  # Note: negative

    # Budget: sum(w) >= 1  =>  -sum(w) + 1 <= 0
    neg_sum_row = [-1] * N_ASSETS
    G_rows.append(neg_sum_row)
    h_vals.append(1.0)  # Positive because: -sum(w) + 1 <= 0 => sum(w) >= 1

    with open('G.csv', 'w') as f:
        for row in G_rows:
            f.write(','.join(str(x) for x in row) + '\n')

    with open('h.csv', 'w') as f:
        for h in h_vals:
            f.write(f'{h}\n')

    # Save asset info for visualization
    with open('assets.csv', 'w') as f:
        f.write('ticker,name,sector,role,beta,size,value\n')
        for ticker in TICKERS:
            info = ASSET_INFO[ticker]
            factors = FACTOR_LOADINGS[ticker]
            f.write(f"{ticker},{info['name']},{info['sector']},{info['role']},"
                    f"{factors[0]},{factors[1]},{factors[2]}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate Portfolio Endowment benchmark')
    parser.add_argument('--n_scenarios', type=int, default=500,
                        help='Number of scenarios (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data (default: 5)')
    args = parser.parse_args()

    print("=" * 60)
    print("Sierra Vista University Endowment Portfolio Benchmark")
    print("=" * 60)
    print()
    print(f"Assets: {N_ASSETS}")
    for ticker in TICKERS:
        info = ASSET_INFO[ticker]
        print(f"  {ticker:5s} - {info['name']:20s} ({info['sector']}, {info['role']})")
    print()
    print(f"Scenarios: {args.n_scenarios}")
    print(f"Uncertainty dimensions: 15")
    print(f"  - 10 asset returns (bootstrapped from {args.years} years of data)")
    print(f"  - 1 volatility regime indicator")
    print(f"  - 1 correlation stress factor")
    print(f"  - 3 factor returns (market, size, value)")
    print()

    # Download real stock data
    print("Downloading stock data from Yahoo Finance...")
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
    print(f"Saved scenarios.csv: {args.n_scenarios} x 15")

    # Save prices for visualization
    prices.to_csv('historical_prices.csv')
    print("Saved historical_prices.csv")

    # Save returns for reference
    returns.to_csv('historical_returns.csv')
    print("Saved historical_returns.csv")

    # Save constraint files
    print("Generating constraint files...")
    save_constraint_files(mean_returns, volatility)
    print("Saved: A_d.csv, b_d.csv, c.csv, G.csv, h.csv, assets.csv")

    # Print summary statistics
    print()
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nAnnualized Returns:")
    for ticker in TICKERS:
        ret = mean_returns[ticker]
        vol = volatility[ticker]
        sharpe = ret / vol if vol > 0 else 0
        print(f"  {ticker:5s}: return={ret*100:6.2f}%, vol={vol*100:5.2f}%, Sharpe={sharpe:.2f}")

    print("\nScenario Statistics:")
    print(f"  Asset returns (delta[0:10]):")
    for j, ticker in enumerate(TICKERS):
        r = scenarios[:, j]
        print(f"    {ticker:5s}: mean={np.mean(r)*100:6.3f}%, std={np.std(r)*100:5.3f}%")

    print(f"\n  Volatility regime (delta[10]):")
    print(f"    mean={np.mean(scenarios[:, 10]):.3f}, range=[{np.min(scenarios[:, 10]):.3f}, {np.max(scenarios[:, 10]):.3f}]")

    print(f"\n  Correlation stress (delta[11]):")
    print(f"    mean={np.mean(scenarios[:, 11]):.3f}, range=[{np.min(scenarios[:, 11]):.3f}, {np.max(scenarios[:, 11]):.3f}]")

    print(f"\n  Factor returns (delta[12:15]):")
    print(f"    Market: mean={np.mean(scenarios[:, 12])*100:.3f}%")
    print(f"    Size:   mean={np.mean(scenarios[:, 13])*100:.3f}%")
    print(f"    Value:  mean={np.mean(scenarios[:, 14])*100:.3f}%")

    print("\nConstraints:")
    print(f"  Min position: {MIN_POSITION*100:.0f}%")
    print(f"  Max position: {MAX_POSITION*100:.0f}%")
    print(f"  Tech sector limit: {TECH_SECTOR_LIMIT*100:.0f}%")
    print(f"  Budget: 100% (fully invested)")


if __name__ == '__main__':
    main()
