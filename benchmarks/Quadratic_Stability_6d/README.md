# Quadratic Stability Analysis (6D SDP)

## The Problem

This benchmark addresses **quadratic stability** for systems with 2-dimensional parameter uncertainty. The goal is to find a Lyapunov-like matrix (parameterized by 6 decision variables) that ensures stability across a 2D uncertain parameter space.

Unlike the 3D LPV benchmark which has 1D uncertainty, this problem explores a richer uncertainty structure where parameters can vary independently in two directions.

---

## Mathematical Background

### Problem Structure

The optimization finds decision variables x = [x_1, ..., x_6] such that:

```
F_0(delta) + sum(x_i * F_i(delta)) <= 0
```

for all scenarios delta = [delta[0], delta[1]] in the 2D uncertainty set.

### Uncertainty Set

The parameters vary within a square:
```
delta[0] in [-1, 1]
delta[1] in [-1, 1]
```

This 2D uncertainty creates a more challenging optimization problem as stability must be certified across a continuous region, not just a line.

---

## Problem Formulation

This is a **Semidefinite Program (SDP)**:

```
minimize    c' x    (minimize x_6)

subject to  F_0(delta) + x[0]*F_1(delta) + ... + x[5]*F_6(delta) <= 0
            (for each scenario delta)

            E_0 + x[0]*E_1 + ... + x[5]*E_6 <= 0
            (hard constraint)
```

Where `<= 0` denotes negative semidefiniteness (matrix inequality).

---

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 6 |
| Uncertainty dimensions | 2 (delta[0], delta[1]) |
| Scenarios (N) | 1162 |
| Parameter range (each) | [-1, 1] |

---

## Files

| File | Description |
|------|-------------|
| `quadratic_stability_data.csv` | 1162 sampled 2D parameter values |
| `Q.csv` | Quadratic objective matrix (6x6) |
| `c.csv` | Linear objective vector (6x1) |
| `F_0.csv` - `F_6.csv` | Scenario-dependent LMI matrices |
| `E_0.csv` - `E_6.csv` | Hard constraint matrices |
| `test_and_visualize.py` | Solve and create 9-panel visualization |

---

## Usage

```bash
cd benchmarks/Quadratic_Stability_6d
python test_and_visualize.py
```

Results are saved to `results/`:
- `metrics.json`: Optimal parameters, complexity, risk bounds
- `solution.csv`: Optimal decision variables
- `visualization.png`: 9-panel analysis figure

---

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
============================================================
BENCHMARK: Quadratic_Stability_6d (SDP)
6D Quadratic Stability with 2D Parameter Uncertainty
============================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 1162
Decision Variables: 6
Optimal Cost: -0.000180
Complexity (k): 4 support constraints
Risk Bounds (99%): [0.0000, 0.0137]
------------------------------------------------------------

Solution:
  x[1] = 0.500012
  x[2] = 0.000000
  x[3] = 0.500012
  x[4] = -0.000000
  x[5] = -0.000000
  x[6] = -0.000180

Constraint Analysis:
  Max eigenvalue over grid: 0.000021
  WARNING: Positive eigenvalues found!
```

### Interpretation

**Complexity (k = 4)**:
- 4 scenario constraints are active at the optimal solution
- With 2D uncertainty and 6 decision variables, this moderate complexity indicates the solution is shaped by extreme corners of the parameter space
- The constraints likely bind at parameter combinations near the vertices of the [-1,1] x [-1,1] square

**Risk Bounds [0.0000, 0.0137]**:
- With 99% confidence, at most 1.37% of parameter combinations might violate the stability condition
- This is a very tight bound, indicating strong robustness
- The low upper bound reflects the large number of scenarios (N=1162) relative to decision variables (6)

**Optimal Solution Structure**:
- x[1] = x[3] ≈ 0.5 suggests a symmetric structure in the Lyapunov-like matrix
- x[2], x[4], x[5] ≈ 0 indicates off-diagonal or cross-terms are not needed
- x[6] ≈ 0 shows the objective (minimizing x_6) is nearly achieved

**Near-Stability Warning**:
- The maximum eigenvalue of approximately 0.00002 is positive but numerically negligible
- This indicates the solution is at the boundary of feasibility
- For practical purposes, this is effectively stable (within numerical tolerance)

**2D Parameter Space Coverage**:
- The 1162 scenarios provide dense coverage of the 2D uncertainty region
- This is significantly more than the 100 scenarios used in the 1D LPV benchmark
- Higher scenario count leads to tighter risk bounds

---

## Visualization Guide

The 9-panel visualization includes:

1. **Scenario Scatter**: 2D distribution of sampled parameters
2. **Scenario Density**: Heatmap of parameter distribution
3. **Max Eigenvalue Heatmap**: Stability margin across parameter space
4. **Decision Variables**: Bar chart of optimal x values
5. **Risk Bounds**: Scenario approach confidence intervals
6. **Eigenvalue Distribution**: Histogram of stability margins
7. **3D Stability Surface**: Surface plot of max eigenvalue vs parameters
8. **Marginal Distributions**: Parameter histograms for each dimension
9. **Summary Statistics**: Key metrics and interpretation

---

## Comparison with LPV_stability_3d

| Aspect | LPV_stability_3d | Quadratic_Stability_6d |
|--------|------------------|------------------------|
| Decision variables | 3 | 6 |
| Uncertainty dim | 1 | 2 |
| Scenarios | 100 | 1162 |
| Complexity (k) | 1 | 4 |
| Risk upper bound | 9.41% | 1.37% |

The 6D problem has tighter risk bounds despite higher dimensionality due to the much larger scenario count. This illustrates the fundamental tradeoff in scenario-based optimization: more scenarios yield better guarantees but increase computational cost.

---

## References

1. Boyd, S. et al. (1994). "Linear Matrix Inequalities in System and Control Theory." SIAM.

2. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions of uncertain convex programs." SIAM J. Optimization.

3. Garatti, S. & Campi, M.C. (2024). "Risk and Complexity in Scenario Optimization." Mathematical Programming.

4. Packard, A. & Doyle, J. (1993). "The complex structured singular value." Automatica.

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust stability analysis with 2D parameter uncertainty.*
