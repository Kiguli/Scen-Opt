# Sierra Vista University Endowment Portfolio Benchmark

## The Story

**Sierra Vista University** is a mid-sized research university in the Pacific Northwest with an $850 million endowment. Chief Investment Officer Dr. Elizabeth Chen manages the portfolio with a 20-year investment horizon, balancing growth objectives against the university's need for stable annual distributions to fund scholarships, research, and operations.

The endowment must generate 4.5% annually ($38.25M) while preserving purchasing power against inflation. Dr. Chen faces multiple sources of uncertainty: market returns, volatility regimes, correlation breakdowns during crises, and factor exposures. The Board of Trustees requires a 95% confidence level that the portfolio won't experience catastrophic drawdowns.

If she tilts too aggressively toward growth, a market correction could force selling at the worst time. If she's too conservative, the endowment fails to meet distribution requirements, leading to program cuts. The challenge: find the allocation that balances growth and protection across hundreds of possible market scenarios.

**This benchmark captures Dr. Chen's asset allocation problem using the scenario approach.**

## Problem Description

Determine optimal portfolio weights for 10 assets under uncertainty in returns, volatility regimes, correlation structure, and systematic factor exposures.

| Ticker | Company | Sector | Role | Beta |
|--------|---------|--------|------|------|
| AAPL | Apple | Technology | Growth | 1.20 |
| MSFT | Microsoft | Technology | Growth | 1.10 |
| GOOGL | Alphabet | Technology | Growth | 1.15 |
| AMZN | Amazon | Consumer Discretionary | Growth | 1.30 |
| PG | Procter & Gamble | Consumer Staples | Defensive | 0.60 |
| JPM | JPMorgan Chase | Financials | Value | 1.10 |
| JNJ | Johnson & Johnson | Healthcare | Defensive | 0.70 |
| XOM | ExxonMobil | Energy | Value/Cyclical | 0.90 |
| CAT | Caterpillar | Industrials | Cyclical | 1.20 |
| MCD | McDonald's | Consumer Discretionary | Defensive | 0.80 |

### Constraints
- **Budget**: Weights sum to 1 (fully invested)
- **Position Limits**: 5% minimum, 25% maximum per asset
- **Sector Limits**: Technology ≤ 40%

## Data Sources

### Stock Price Data
- **Source**: Yahoo Finance via yfinance
- **Period**: 5 years of daily adjusted close prices
- **Tickers**: Real S&P 500 components selected for sector diversity

### Factor Loadings
- **Source**: Fama-French 3-Factor Model estimates
- **Reference**: Fama, E.F. & French, K.R. (1993). "Common risk factors in the returns on stocks and bonds." Journal of Financial Economics.
- **Factors**: Market (beta), Size (SMB), Value (HML)

## Mathematical Formulation

```
minimize    -expected_return'w + rho * zeta

subject to  (for each scenario):
    # Volatility-adjusted loss constraints (per asset)
    -(1 + 0.5*delta[10]) * delta[i] * w[i] <= zeta    for i=0..9

    # Correlation stress on tech sector
    -(1 + delta[11]) * sum(delta[i] * w[i]) for tech stocks <= zeta

    # Factor exposure constraints
    -delta[12] * sum(beta[i] * w[i]) <= zeta      (market)
    -delta[13] * sum(size[i] * w[i]) <= zeta      (size)
    -delta[14] * sum(value[i] * w[i]) <= zeta     (value)

hard constraints:
    sum(w[i]) = 1                    (budget)
    w[i] >= 0.05                     (minimum position)
    w[i] <= 0.25                     (maximum position)
    w[AAPL]+w[MSFT]+w[GOOGL]+w[AMZN] <= 0.40    (tech sector limit)
```

Where:
- `w[i]` = portfolio weight for asset i (decision variable)
- `delta[0:10]` = asset returns (uncertain, bootstrapped from historical data)
- `delta[10]` = volatility regime indicator [0=low, 1=high]
- `delta[11]` = correlation stress factor [0, 0.5]
- `delta[12:15]` = factor returns (market, size, value)
- `zeta` = worst-case loss slack variable
- `rho` = penalty for constraint violation

## Uncertainty Characterization

**15-dimensional uncertainty** (delta vector):

| Index | Parameter | Distribution | Notes |
|-------|-----------|--------------|-------|
| delta[0:10] | Asset returns | Bootstrap + noise | Sampled from 5 years of daily returns |
| delta[10] | Volatility regime | Empirical [0, 1] | Based on rolling 20-day realized volatility |
| delta[11] | Correlation stress | Uniform(0, 0.5) | Higher during market turbulence |
| delta[12] | Market factor | Empirical + Normal(0, 0.01) | Correlated with average return |
| delta[13] | Size factor | Normal(0, 0.008) | Small minus Big |
| delta[14] | Value factor | Normal(0, 0.008) | High minus Low |

### Key Features

1. **Delta-dependent A_d matrix**: Constraint coefficients depend on uncertain parameters (volatility regime, correlation stress, factor returns), not just the RHS.

2. **Volatility regime adjustment**: Loss constraints are amplified during high-volatility regimes: `-(1 + 0.5*delta[10])` scales losses by up to 50% more.

3. **Correlation stress**: During market crises, correlations increase. The tech sector constraint models this: `-(1 + delta[11])` amplifies correlated losses.

4. **Factor exposures**: Systematic risk from market, size, and value factors captured through empirically-estimated loadings.

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 10 (portfolio weights) |
| Uncertainty dimensions | 15 |
| Scenarios (N) | 500 |
| Scenario constraints | 14 per scenario |
| Hard constraints | 23 (position limits, sector limit, budget) |

## Expected Results

Running `test_and_visualize.py` produces:

- **Complexity (k)**: 8-15 support constraints
- **Risk bounds**: ε ∈ [0.01, 0.05] at 99% confidence
- **Tech allocation**: Near 40% limit (growth seeking)
- **Defensive allocation**: 20-30% (PG, JNJ, MCD)
- **Budget utilization**: 100% (fully invested)

### Interpretation

The scenario approach finds the return-maximizing portfolio weights that satisfy all 500 observed scenarios (minus k support constraints). The Campi-Garatti theory guarantees that with 99% confidence, this solution will satisfy at least (1-ε) fraction of all possible future scenarios.

For Dr. Chen, this means: "If I allocate assets according to this portfolio, I can expect to meet my risk constraints at least 95% of the time, with 99% confidence in that estimate."

## Files

| File | Description |
|------|-------------|
| `scenarios.csv` | 500 x 15 uncertainty scenarios |
| `historical_prices.csv` | 5 years of daily prices for visualization |
| `historical_returns.csv` | Daily returns computed from prices |
| `assets.csv` | Asset metadata (ticker, name, sector, factor loadings) |
| `A_d.csv` | Scenario-dependent constraint matrix (14 x 10) |
| `b_d.csv` | Scenario-dependent RHS (14 x 1, all zeros) |
| `c.csv` | Negative expected returns (10 x 1) |
| `G.csv` | Hard constraint matrix (23 x 10) |
| `h.csv` | Hard constraint RHS |
| `parameters.txt` | rho=50, tau=0, confidence=0.99 |
| `generate.py` | Download data and regenerate scenarios |
| `test_and_visualize.py` | Solve and create 12-panel visualization |

## Usage

```bash
# Generate fresh scenarios (downloads from Yahoo Finance)
python generate.py --n_scenarios 500 --seed 42

# Solve and visualize
python test_and_visualize.py
```

Results are saved to `results/`:
- `metrics.json`: Optimal weights, sector allocations, risk bounds
- `solution.csv`: Raw solution vector
- `visualization.png`: 12-panel analysis figure

## Scenario Approach Insights

This benchmark demonstrates several key features of the scenario approach:

1. **Multiple uncertainty sources**: Returns, volatility regime, correlation stress, and factor returns create a rich 15-dimensional uncertainty space.

2. **Regime-dependent constraints**: The A_d matrix changes with volatility regime (delta[10]) and correlation stress (delta[11]), capturing the reality that risk constraints bind differently in different market conditions.

3. **Real-world data**: Using actual stock returns via yfinance grounds the scenarios in empirical market behavior, including the 2020 COVID crash and subsequent recovery.

4. **Practical constraints**: Position limits and sector constraints reflect real investment policy guidelines that endowments must follow.

5. **Risk certification**: The [eps_lower, eps_upper] bounds give Dr. Chen a concrete probability guarantee she can present to the Board of Trustees.

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
======================================================================
BENCHMARK: University Endowment Portfolio Optimization
======================================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 500
Decision Variables: 10 (portfolio weights)
Expected Return: 13.61%
Complexity (k): 12 support constraints
Risk Bounds (99%): [0.0074, 0.0573]
----------------------------------------------------------------------
```

### Interpretation

**Expected Return = 13.61%**:
- The optimized portfolio targets an annualized return of 13.61%
- This exceeds the 4.5% distribution requirement by a healthy margin
- The optimization maximizes return while respecting risk constraints

**Complexity (k = 12)**:
- 12 scenario constraints are binding at the optimal solution
- This moderate complexity indicates the portfolio is shaped by multiple extreme scenarios
- The solution balances growth objectives against various stress conditions

**Risk Bounds [0.0074, 0.0573]**:
- With 99% confidence, the probability of violating constraints is between 0.74% and 5.73%
- This means: at least 94.3% of market scenarios will satisfy all risk constraints
- The endowment can report strong risk management to the Board of Trustees

**Portfolio Allocation Insights**:
- Technology sector likely near the 40% limit (growth-seeking behavior)
- Defensive stocks (PG, JNJ, MCD) provide downside protection
- The higher complexity (k=12) compared to CVaR (k=3) reflects the multi-factor risk model

**Practical Meaning for Dr. Chen**:
- The portfolio is well-positioned to meet the 4.5% annual distribution
- Risk constraints are satisfied with high probability across market regimes
- Board reporting can cite: "With 99% confidence, our portfolio satisfies risk limits in at least 94% of market conditions"
