# Fresh Produce Distribution Benchmark

## The Story

**Valley Fresh Distributors** is a regional produce company serving 47 grocery stores
across Northern California. Every morning at 4 AM, purchasing manager Maria Rodriguez
must decide how many cases of each product to order from suppliers at the Oakland
Terminal Market for delivery that day.

The challenge: fresh produce is highly perishable. Strawberries ordered today might
have 25% spoilage by the time they reach store shelves. Demand fluctuates based on
weather, local events, and seasonal patterns. And Maria must work within a fixed daily
budget and limited refrigerated warehouse space.

If she orders too little, stores face empty shelves and lost sales. Order too much,
and spoiled produce goes straight to compost. The cost of getting it wrong compounds
daily across 260 business days per year.

**This benchmark captures Maria's daily decision problem using the scenario approach.**

## Problem Description

Determine optimal daily order quantities for 5 perishable produce items under
uncertainty in yield (spoilage), demand, and warehouse space efficiency.

| Product | Cost/Case | Mean Demand | Yield (Spoilage) | Space |
|---------|-----------|-------------|------------------|-------|
| Strawberries | $18.50 | 85 cases | 75% (25% loss) | 1.2 cu.ft |
| Tomatoes | $14.25 | 120 cases | 88% (12% loss) | 1.5 cu.ft |
| Lettuce | $12.00 | 95 cases | 82% (18% loss) | 2.0 cu.ft |
| Avocados | $32.00 | 60 cases | 90% (10% loss) | 0.8 cu.ft |
| Bell Peppers | $22.50 | 75 cases | 85% (15% loss) | 1.3 cu.ft |

### Constraints
- **Daily Budget**: $12,000
- **Warehouse Capacity**: 800 cubic feet (refrigerated)
- **Supplier Limits**: Product-specific maximum orders (150-200 cases)

## Data Sources

### Yield (Spoilage) Data
- **Source**: FDA/USDA Food Loss and Waste estimates
- **Reference**: Buzby, J.C., et al. (2014). "The Estimated Amount, Value, and Calories
  of Postharvest Food Losses at the Retail and Consumer Levels in the United States."
  USDA Economic Research Service, Economic Information Bulletin 121.
- **Key Finding**: Fresh produce experiences 30-40% total loss from farm to consumer,
  with 8-12% at distribution/retail level. Highly perishable items (berries, leafy greens)
  see higher losses.

### Demand Patterns
- **Source**: USDA Agricultural Marketing Service Terminal Market Reports
- **Reference**: USDA AMS (2024). "National Fruit and Vegetable Retail Report."
  Weekly data from major terminal markets.
- **URL**: https://www.ams.usda.gov/market-news/fruit-and-vegetable
- **Correlation Source**: Products from same growing regions (California leafy greens,
  Mexican tomatoes/peppers) show correlated demand due to shared supply chain disruptions.

### Wholesale Pricing
- **Source**: USDA Agricultural Marketing Service
- **Reference**: Oakland Terminal Market daily price reports
- **Note**: Prices reflect Q4 2024 wholesale case prices for #1 grade produce

## Mathematical Formulation

```
minimize    c'q + rho * sum(zeta)

subject to  -yield[j] * q[j] + demand[j] <= zeta[j]    for each product j, scenario
            space[j] * q[j] <= capacity + zeta[capacity]
            sum(cost[j] * q[j]) <= budget
            0 <= q[j] <= max_order[j]
```

Where:
- `q[j]` = order quantity for product j (decision variable)
- `yield[j]` = fraction of ordered product that's usable (uncertain)
- `demand[j]` = customer demand in cases (uncertain)
- `space[j]` = warehouse space per case, with packing variability (uncertain)
- `zeta[j]` = service shortfall slack variable
- `rho` = penalty for unmet demand ($100/case shortfall cost)

## Uncertainty Characterization

**15-dimensional uncertainty** (delta vector):

| Index | Parameter | Distribution | Notes |
|-------|-----------|--------------|-------|
| delta[0:5] | Yields | Correlated Normal | Mean 0.75-0.90, products sharing cold chain correlate |
| delta[5:10] | Demands | Correlated Normal | Mean 60-120 cases, regional demand patterns correlate |
| delta[10:15] | Space factors | Uniform(0.95, 1.10) | Packing efficiency variability |

### Correlation Structure
- **Strawberries-Lettuce**: 0.4-0.5 correlation (both California sourced)
- **Tomatoes-Bell Peppers**: 0.6-0.7 correlation (both Mexico sourced)
- **Avocados**: Low correlation with others (independent supply chain)

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 5 (order quantities) |
| Uncertainty dimensions | 15 |
| Scenarios (N) | 500 |
| Hard constraints | 11 (non-negativity, upper bounds, budget) |
| Scenario constraints | 6 per scenario (5 service + 1 capacity) |

## Expected Results

Running `test_and_visualize.py` produces:

- **Complexity (k)**: 5-8 support constraints
- **Risk bounds**: ε ∈ [0.002, 0.05] at 99% confidence
- **Budget utilization**: Typically 90-100% of $12,000
- **Service levels**: Vary by product based on cost/yield tradeoff

### Interpretation

The scenario approach finds the cost-minimizing order quantities that satisfy all
500 observed scenarios (minus k support constraints). The Campi-Garatti theory
guarantees that with 99% confidence, this solution will satisfy at least (1-ε)
fraction of all possible future scenarios.

For Maria, this means: "If I order these quantities every day, I can expect to
meet demand at least 95% of the time, with 99% confidence in that estimate."

## Files

| File | Description |
|------|-------------|
| `scenarios.csv` | 500 x 15 uncertainty scenarios |
| `A_d.csv` | Scenario-dependent constraint matrix (yield, space) |
| `b_d.csv` | Scenario-dependent RHS (demand, capacity) |
| `c.csv` | Ordering cost per case |
| `G.csv` | Hard constraint matrix (bounds, budget) |
| `h.csv` | Hard constraint RHS |
| `parameters.txt` | rho=100, tau=0, confidence=0.99 |
| `generate.py` | Regenerate scenarios with different N or seed |
| `test_and_visualize.py` | Solve and create 12-panel visualization |

## Usage

```bash
# Generate fresh scenarios
python generate.py --n_scenarios 500 --seed 42

# Solve and visualize
python test_and_visualize.py
```

Results are saved to `results/`:
- `metrics.json`: Optimal quantities, costs, risk bounds
- `solution.csv`: Raw solution vector
- `visualization.png`: 12-panel analysis figure

## Scenario Approach Insights

This benchmark demonstrates several key features of the scenario approach:

1. **Delta-dependent A matrix**: The constraint coefficient matrix depends on
   uncertain yields, not just the RHS. This captures real supply chain variability.

2. **Correlated uncertainty**: Demands and yields are correlated across products,
   reflecting real-world patterns where a cold snap affects all California produce.

3. **Multiple binding constraints**: The optimal solution typically has k=5-8
   support constraints, showing that multiple scenarios "matter" for the solution.

4. **Practical risk certification**: The [eps_lower, eps_upper] bounds give
   Maria a concrete probability guarantee she can communicate to store managers.

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
======================================================================
BENCHMARK: Fresh Produce Distribution (Inventory)
======================================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 500
Decision Variables: 5
Optimal Cost: $45,099.67
Max Constraint Violation (zeta): 0.000000
Complexity (k): 7 support constraints
Risk Bounds (99%): [0.0022, 0.0418]
----------------------------------------------------------------------
```

### Interpretation

**Complexity (k = 7)**:
- Out of 500 scenarios, exactly 7 constraints are active (binding) at the optimal solution
- These 7 "support constraints" fully determine the optimal order quantities
- The remaining 493 scenarios are satisfied with slack

**Risk Bounds [0.0022, 0.0418]**:
- With 99% confidence, the probability that a randomly drawn scenario violates constraints is between 0.22% and 4.18%
- This means: with 99% confidence, **at least 95.8% of future days** will have demand satisfied by these order quantities

**Practical Meaning for Maria**:
- The optimal order quantities provide near-certain service levels
- Only about 1 in 25 days (at worst) might experience a stockout
- The low complexity (k=7) indicates the solution is robust—few extreme scenarios drive the decision

**Cost Breakdown**:
- Total daily ordering cost: approximately $45,100
- Budget utilization: ~$12,000 (near full utilization)
- The solution balances ordering costs against stockout penalties (rho=100 per case shortfall)

---

## References

1. Buzby, J.C., et al. (2014). "The Estimated Amount, Value, and Calories of
   Postharvest Food Losses." USDA ERS Economic Information Bulletin 121.

2. USDA Agricultural Marketing Service. Terminal Market Reports.
   https://www.ams.usda.gov/market-news

3. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized
   solutions of uncertain convex programs." SIAM J. Optimization, 19(3), 1211-1230.

4. Bertsimas, D. & Thiele, A. (2006). "A Robust Optimization Approach to
   Inventory Theory." Operations Research, 54(1), 150-168.
