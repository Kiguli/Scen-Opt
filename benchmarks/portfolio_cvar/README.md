# Golden State Teachers' Pension Fund: CVaR Portfolio Optimization

## The Story

**Golden State Teachers' Pension Fund (GSTPF)** manages $45 billion in retirement assets for over 300,000 current and former public school teachers across California. Chief Investment Officer Maria Santos faces a challenging mandate: generate stable returns to meet pension obligations while strictly limiting downside risk.

The fund's board requires that in 95% of market scenarios, the portfolio's worst-case losses remain within acceptable bounds. After the 2008 financial crisis devastated similar funds, GSTPF adopted **Conditional Value-at-Risk (CVaR)** as their primary risk metric, focusing on the average loss in the worst 5% of scenarios rather than just point estimates.

Maria's team must allocate across 8 asset classes while navigating multiple sources of uncertainty: market returns, credit spreads, interest rate movements, and correlation breakdowns during stress periods. The scenario approach provides data-driven risk bounds that satisfy the board's regulatory requirements.

**This benchmark demonstrates CVaR optimization using the scenario approach framework.**

---

## Background: What is CVaR?

**Conditional Value-at-Risk (CVaR)**, also known as Expected Shortfall, measures the average loss in the worst (1-β)% of scenarios. Unlike Value-at-Risk (VaR) which only identifies a threshold, CVaR captures the severity of tail losses.

For β = 95%:
- **VaR(95%)**: "We're 95% confident losses won't exceed X"
- **CVaR(95%)**: "When losses do exceed VaR, they average Y"

CVaR is preferred by regulators because:
1. It's a **coherent risk measure** (satisfies subadditivity)
2. It captures **tail risk** severity, not just frequency
3. It can be formulated as a **linear program**

### The Rockafellar-Uryasev Formulation

CVaR minimization can be expressed as:

```
minimize    α + (1/(1-β)N) * Σ max(0, loss_i - α)

subject to  portfolio constraints
```

Where α is the VaR threshold (optimized jointly) and N is the number of scenarios.

---

## Problem Description

Determine optimal portfolio weights for 8 ETF asset classes to minimize CVaR at the 95% confidence level, subject to pension fund regulatory constraints.

### Asset Classes (Real Data via yfinance)

| Ticker | Asset Class | Role | Characteristics |
|--------|-------------|------|-----------------|
| SPY | US Large Cap Equity | Growth | High return, high volatility |
| AGG | US Aggregate Bonds | Stability | Low return, low volatility |
| VNQ | Real Estate (REITs) | Income | Moderate return, high correlation to equity |
| GLD | Gold | Hedge | Low correlation, inflation protection |
| EFA | Developed Intl Equity | Diversification | Moderate return, currency exposure |
| TLT | Long-Term Treasuries | Safety | Interest rate sensitive, flight-to-quality |
| VWO | Emerging Markets | Growth | High return, high volatility |
| LQD | Investment Grade Corp Bonds | Income | Credit spread sensitive |

### Regulatory Constraints

GSTPF operates under California Public Employees' Pension Reform Act guidelines:

- **Equity Exposure**: Between 30% and 60% (SPY + EFA + VWO)
- **Fixed Income Minimum**: At least 25% in bonds (AGG + TLT + LQD)
- **Position Limits**: Each position between 5% and 30%
- **Full Investment**: All capital must be deployed (weights sum to 1)

---

## Mathematical Formulation

### Decision Variables

```
x = [w_SPY, w_AGG, w_VNQ, w_GLD, w_EFA, w_TLT, w_VWO, w_LQD, α]
```

Where w_i are portfolio weights and α is the VaR threshold.

### CVaR Objective

```
minimize    α + ρ * ζ

where ρ = 1/((1-0.95) * 500) = 0.04
```

The auxiliary variable ζ captures max(0, loss - α) across scenarios.

### Scenario Constraints (A_d matrix)

For each scenario with uncertainty vector δ:

**Row 1 - Main CVaR Loss Constraint:**
```
-(1 + 0.3*δ[8]) * Σ δ[i]*w[i] - α ≤ ζ
```
The (1 + 0.3*δ[8]) term amplifies losses during high-volatility regimes.

**Row 2 - Credit Spread Stress:**
```
-δ[9] * (0.5*w_AGG + 0.3*w_TLT + 1.0*w_LQD) ≤ ζ
```
Corporate bonds (LQD) are most sensitive to credit spreads.

**Row 3 - Interest Rate Shock:**
```
-δ[10] * (0.3*w_AGG + 1.0*w_TLT + 0.5*w_LQD) ≤ ζ
```
Long-term treasuries (TLT) have highest duration exposure.

**Row 4 - Equity Correlation Stress:**
```
-(1 + δ[11]) * (δ[0]*w_SPY + δ[4]*w_EFA + δ[6]*w_VWO) ≤ ζ
```
During crises, equity correlations approach 1.0.

### Hard Constraints (G, h matrices)

```
Position limits:           0.05 ≤ w[i] ≤ 0.30  for all i
Equity allocation:         0.30 ≤ w_SPY + w_EFA + w_VWO ≤ 0.60
Fixed income minimum:      w_AGG + w_TLT + w_LQD ≥ 0.25
Budget constraint:         Σ w[i] = 1
```

---

## Uncertainty Characterization

**12-dimensional uncertainty vector** δ:

| Index | Parameter | Distribution | Source |
|-------|-----------|--------------|--------|
| δ[0:8] | Asset returns | Historical bootstrap | yfinance 5-year daily returns |
| δ[8] | Stress indicator | Uniform(0, 1) | Volatility regime proxy |
| δ[9] | Credit shock | Normal(0, 0.02) | Credit spread movement |
| δ[10] | Rate shock | Normal(0, 0.01) | Interest rate change |
| δ[11] | Correlation stress | Uniform(0, 0.5) | Crisis correlation increase |

### Scenario Generation Process

1. Download 5 years of daily returns for all 8 ETFs via yfinance
2. Bootstrap sample returns from historical distribution
3. Add Gaussian perturbations for regime changes
4. Generate stress indicators from volatility percentiles
5. Sample credit and rate shocks from calibrated distributions
6. Create 500 scenarios for robust optimization

---

## File Structure

| File | Description | Dimensions |
|------|-------------|------------|
| `generate.py` | Downloads data, generates scenarios and constraint matrices | - |
| `scenarios.csv` | 500 scenarios × 12 uncertainty parameters | 500 × 12 |
| `A_d.csv` | Scenario-dependent constraint matrix with delta expressions | 4 × 9 |
| `b_d.csv` | RHS of scenario constraints | 4 × 1 |
| `c.csv` | Objective coefficients (CVaR: [0,...,0,1]) | 9 × 1 |
| `G.csv` | Hard constraint matrix | 19 × 9 |
| `h.csv` | Hard constraint RHS | 19 × 1 |
| `parameters.txt` | rho=0.04, confidence=0.95 | - |
| `test_and_visualize.py` | Solves problem, generates 12-panel visualization | - |

---

## Running the Benchmark

### Generate Data and Constraints

```bash
cd benchmarks/portfolio_cvar
python generate.py
```

This downloads real market data and creates all constraint files.

### Solve and Visualize

```bash
python test_and_visualize.py
```

Produces:
- Optimal portfolio weights
- CVaR and VaR estimates
- Risk bounds from scenario approach
- 12-panel visualization saved to `cvar_portfolio_analysis.png`

---

## Expected Results

| Metric | Expected Range |
|--------|----------------|
| Complexity (k) | 4-8 support constraints |
| CVaR (95%) | 1.5% - 3.0% |
| VaR (95%) | 1.0% - 2.5% |
| Risk bound (ε) | 0.01 - 0.05 at 95% confidence |
| Equity allocation | Near 45-55% |
| Fixed income | Near 30-40% |

The scenario approach provides **finite-sample guarantees** on out-of-sample constraint satisfaction, critical for pension fund regulatory compliance.

---

## Visualization Guide

The 12-panel visualization includes:

1. **Historical Prices**: 5-year normalized price history
2. **Optimal Weights**: Bar chart of portfolio allocation
3. **Return Distributions**: Box plots by asset class
4. **Correlation Heatmap**: Asset return correlations
5. **Stress Regime Distribution**: Histogram of δ[8]
6. **CVaR Tail Analysis**: Worst 5% of portfolio returns
7. **Portfolio vs Benchmark**: Cumulative return comparison
8. **Drawdown Analysis**: Maximum drawdown over time
9. **Asset Class Allocation**: Pie chart by category
10. **Credit vs Rate Shocks**: Scatter plot of shock scenarios
11. **Risk Bounds**: Scenario approach confidence intervals
12. **Summary Statistics**: Key metrics table

---

## Mathematical Details

### Why ρ = 0.04?

In the CVaR formulation:
```
CVaR_β = α + (1/(1-β)) * E[max(0, Loss - α)]
```

With N=500 scenarios and β=0.95:
```
ρ = 1/((1-0.95) * 500) = 1/(0.05 * 500) = 0.04
```

This ensures the objective correctly computes the CVaR.

### Scenario Approach Risk Bounds

Given:
- N = 500 scenarios
- d = 9 decision variables
- k = complexity (active constraints)
- β = 0.95 confidence level

The scenario approach guarantees:
```
P(ε ≤ ε_upper) ≥ β
```

Where ε is the true probability of constraint violation.

---

## References

1. Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." Journal of Risk.

2. Campi, M.C. & Garatti, S. (2008). "The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs." SIAM J. Optimization.

3. Garatti, S. & Campi, M.C. (2024). "Risk and Complexity in Scenario Optimization." Mathematical Programming.

4. California Public Employees' Pension Reform Act (PEPRA) Guidelines.

---

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
======================================================================
BENCHMARK: CVaR Portfolio Optimization (Pension Fund)
======================================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 500
Decision Variables: 9 (8 weights + VaR threshold)
CVaR (95%): 1.52%
Complexity (k): 3 support constraints
Risk Bounds (99%): [0.0000, 0.0278]
----------------------------------------------------------------------
```

### Interpretation

**CVaR = 1.52%**:
- The expected loss in the worst 5% of market scenarios is 1.52%
- For a $45 billion portfolio, this translates to a worst-case average loss of ~$684 million
- This is well within regulatory tolerances for pension funds

**Complexity (k = 3)**:
- Only 3 scenario constraints are binding at the optimal solution
- This low complexity indicates the CVaR-minimizing portfolio is determined by just 3 extreme scenarios
- Most scenarios (497 out of 500) are satisfied with slack

**Risk Bounds [0.0000, 0.0278]**:
- With 99% confidence, the probability of violating the CVaR constraint is at most 2.78%
- The lower bound of 0 indicates the solution may be fully robust within the sampled distribution
- This provides strong regulatory assurance for the pension fund's risk management

**Practical Meaning for Maria**:
- The optimized portfolio satisfies regulatory CVaR requirements with high confidence
- The low complexity suggests a stable solution that won't change dramatically with new scenarios
- Board reporting can cite: "With 99% confidence, our portfolio limits tail risk in at least 97.2% of market conditions"

---

## Data Sources

- **Market Data**: Yahoo Finance via yfinance (5 years daily)
- **ETF Information**: iShares, Vanguard, SPDR prospectuses
- **Factor Models**: Fama-French Research Data Library

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust optimization with finite-sample risk guarantees.*
