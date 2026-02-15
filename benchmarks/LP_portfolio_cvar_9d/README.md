# CVaR Portfolio Optimization

## Problem Description

A pension fund must allocate capital across 8 ETF asset classes to minimize tail risk (Conditional Value-at-Risk) while satisfying regulatory allocation constraints. CVaR at the 95% level measures the average loss in the worst 5% of market scenarios — a coherent risk measure that captures the severity of extreme losses, not just their probability.

| Ticker | Asset | Class | Role |
|--------|-------|-------|------|
| SPY | S&P 500 ETF | US Equity | Core Growth |
| AGG | US Aggregate Bond | Fixed Income | Stability |
| VNQ | Real Estate (REIT) | Real Assets | Inflation Hedge |
| GLD | Gold | Commodities | Crisis Hedge |
| EFA | Intl Developed | Intl Equity | Diversification |
| TLT | 20+ Year Treasury | Long Bonds | Duration |
| VWO | Emerging Markets | Intl Equity | Growth |
| LQD | Investment Grade Corp | Credit | Yield |

Position limits are 5-30% per asset. Equities (SPY+EFA+VWO) must be 30-60%, fixed income (AGG+TLT+LQD) at least 25%. Scenario data is bootstrapped from 5 years of real daily ETF returns downloaded via yfinance.

## Formulation

This is a Linear Program using the Rockafellar-Uryasev CVaR formulation. The decision variable is x = [w, alpha] in R^9, where w in R^8 are portfolio weights and alpha is the VaR threshold. The objective c = [0,...,0,1] minimizes alpha, with slack penalty rho = 1/((1-0.95)*500) = 0.04 so that alpha + rho*sum(zeta) approximates CVaR at the 95% level. No regularisation (tau = 0).

Each scenario delta_i in R^12 encodes 8 asset returns (delta[0:8]), a market stress indicator (delta[8]), credit spread shock (delta[9]), interest rate shock (delta[10]), and volatility scaling (delta[11]). The 4 scenario-dependent constraints capture: (1) portfolio loss with stress amplification, (2) credit spread impact on bonds, (3) interest rate duration exposure, and (4) equity correlation breakdown during crises. Hard constraints encode position limits, budget (weights sum to 1), equity allocation range, and fixed income minimum (23 rows total).

## Results

![CVaR Portfolio Results](results/portfolio_cvar.png)

The LP solves with N = 500 scenarios. The optimizer tilts heavily toward alternatives: GLD (24.1%) and VNQ (20.9%) dominate, while equities sit at the 30% regulatory minimum and fixed income at 25%. This defensive posture minimizes tail risk — the CVaR(95%) is only 1.52% daily loss. The LP had complexity k = 3, no degeneracy, and risk bounds [0.0000, 0.0278].

## Files

```
LP_portfolio_cvar_9d/
├── README.md           This file
├── parameters.txt      Solver parameters (rho, tau, confidence)
├── generate.py         Download ETF data and generate scenarios (requires yfinance)
├── run.py              Solve the LP and print results
├── plot.py             Generate the paper figure
├── data/
│   ├── A_d.csv         Scenario-dependent constraint matrix (4 x 9)
│   ├── b_d.csv         Scenario-dependent RHS vector (4 x 1, all zeros)
│   ├── c.csv           Objective vector (9 x 1)
│   ├── G.csv           Hard constraint matrix (23 x 9)
│   ├── h.csv           Hard constraint RHS (23 x 1)
│   ├── scenarios.csv   500 x 12 uncertainty scenarios
│   ├── assets.csv      Asset metadata
│   ├── historical_prices.csv   5-year ETF price history
│   └── historical_returns.csv  Daily returns
└── results/
    ├── metrics.json    Solver output (weights, CVaR, risk bounds)
    ├── solution.csv    Raw solution vector
    ├── portfolio_cvar.png   Paper figure (300 dpi)
    └── portfolio_cvar.pdf   Paper figure (vector)
```

## Usage

```bash
# Regenerate scenarios (requires yfinance and internet)
python generate.py --n_scenarios 500 --seed 42

# Solve the LP
python run.py

# Generate paper figure
python plot.py
```
