# Coupled Oscillator Stability Analysis (6D SDP)

## The Problem

This benchmark addresses **robust stability** of a coupled oscillator Linear Parameter-Varying (LPV) system with 2-dimensional uncertainty. The goal is to find a common Lyapunov matrix P that certifies stability for all parameter combinations in the uncertainty set.

The system represents two coupled oscillator modes with uncertain damping and coupling parameters - a realistic model that arises in mechanical vibration analysis, structural dynamics, and multi-body systems.

---

## Physical System Model

### Coupled Oscillator Dynamics

The system is described by:
```
dx/dt = A(delta) * x
```

where x is the 3-dimensional state vector and the system matrix depends on uncertain parameters:
```
A(delta) = A_0 + delta[0]*A_1 + delta[1]*A_2
```

- **A_0**: Nominal dynamics (damped coupled oscillations)
- **A_1**: Perturbation to the first oscillator mode
- **A_2**: Perturbation to the coupling strength

### Uncertainty Set

The parameters vary within a square:
```
delta[0] in [-1, 1]   (first mode uncertainty)
delta[1] in [-1, 1]   (coupling uncertainty)
```

This 2D uncertainty creates a challenging optimization problem as stability must be certified across the entire continuous region, not just isolated points.

---

## Mathematical Background

### Lyapunov Stability Condition

For the system dx/dt = A(delta)x to be stable for all delta in the uncertainty set, we seek a common Lyapunov matrix P such that:
```
A(delta)' P + P A(delta) < 0   for all delta in [-1,1]^2
P > 0                          (positive definite)
```

### Decision Variables

The 3x3 symmetric Lyapunov matrix P has 6 independent elements:
```
P = [[p11, p12, p13],
     [p12, p22, p23],
     [p13, p23, p33]]

x = [p11, p12, p13, p22, p23, p33]
```

---

## Problem Formulation

This is a **Semidefinite Program (SDP)**:

```
minimize    trace(P) = p11 + p22 + p33

subject to  A(delta)' P + P A(delta) <= 0   (for each scenario delta)
            P >= 0.1 * I                     (positive definiteness)
```

Where `<= 0` denotes negative semidefiniteness (matrix inequality).

---

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 6 (symmetric 3x3 matrix) |
| Uncertainty dimensions | 2 (delta[0], delta[1]) |
| Scenarios (N) | 500 |
| Parameter range (each) | [-1, 1] |

---

## Files

| File | Description |
|------|-------------|
| `generate.py` | Generate all benchmark data files |
| `quadratic_stability_data.csv` | 500 sampled 2D parameter values |
| `Q.csv` | Quadratic regularization matrix (6x6) |
| `c.csv` | Linear objective vector (minimize trace(P)) |
| `F_0.csv` - `F_6.csv` | Lyapunov constraint matrices (delta-dependent) |
| `E_0.csv` - `E_6.csv` | Positive definiteness constraint matrices |
| `test_and_visualize.py` | Solve and create 9-panel visualization |

---

## Usage

```bash
cd benchmarks/Quadratic_Stability_6d
python generate.py            # (Optional) Regenerate benchmark data
python test_and_visualize.py  # Solve and visualize
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
Scenarios (N): 500
Decision Variables: 6
Optimal Cost: 0.300015 (trace(P))
Complexity (k): 0 support constraints
Risk Bounds (99%): [0.0000, 0.0144]
------------------------------------------------------------

Lyapunov Matrix P:
  [  0.100   0.000   0.000]
  [  0.000   0.100   0.000]
  [  0.000   0.000   0.100]

Constraint Analysis:
  Max eigenvalue over grid: -0.041505
  Min eigenvalue over grid: -0.072697
  All constraints satisfied (all eigenvalues <= 0)
```

### Interpretation

**Complexity (k = 0)**:
- No scenario constraints are active at the optimal solution
- The hard constraint P >= 0.1*I is the binding constraint
- This indicates the system has a large stability margin - even the smallest allowable Lyapunov matrix certifies robust stability

**Risk Bounds [0.0000, 0.0144]**:
- With 99% confidence, at most 1.44% of parameter combinations might violate the stability condition
- This is a very tight bound, indicating strong robustness
- The low upper bound reflects the conservative nature of the solution

**Optimal Solution: P = 0.1*I**:
- The identity-scaled Lyapunov matrix is sufficient for stability
- This means the system's eigenvector structure doesn't require a non-trivial P
- The coupled oscillator dynamics are inherently well-damped

**Stability Margin**:
- Maximum eigenvalue of A(delta)'P + PA(delta) is **-0.041505** (strictly negative)
- All eigenvalues are bounded away from zero by at least 0.04
- No numerical issues - the solution is robustly feasible

**Physical Interpretation**:
- The system is robustly stable for all combinations of mode and coupling uncertainty
- The identity Lyapunov function V(x) = x'Px = 0.1||x||² works universally
- This suggests the nominal design has excellent stability margins

---

## Visualization Guide

The 9-panel visualization includes:

1. **Scenario Scatter**: 2D distribution of sampled parameters
2. **Scenario Density**: Heatmap of parameter distribution
3. **Max Eigenvalue Heatmap**: Stability margin across parameter space
4. **Decision Variables**: Bar chart of Lyapunov matrix elements
5. **Risk Bounds**: Scenario approach confidence intervals
6. **Eigenvalue Distribution**: Histogram of stability margins
7. **3D Stability Surface**: Surface plot of max eigenvalue vs parameters
8. **Marginal Distributions**: Parameter histograms for each dimension
9. **Summary Statistics**: Lyapunov matrix and key metrics

---

## Comparison with LPV_stability_3d

| Aspect | LPV_stability_3d | Quadratic_Stability_6d |
|--------|------------------|------------------------|
| System dimension | 2x2 | 3x3 |
| Decision variables | 3 | 6 |
| Uncertainty dim | 1 | 2 |
| Scenarios | 100 | 500 |
| Complexity (k) | 1 | 0 |
| Max eigenvalue | ~0 | -0.04 |
| Physical model | Generic LPV | Coupled oscillators |

The 6D problem demonstrates a well-designed system with substantial stability margins, while the 3D problem operates closer to the feasibility boundary.

---

## References

1. Boyd, S. et al. (1994). "Linear Matrix Inequalities in System and Control Theory." SIAM.

2. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions of uncertain convex programs." SIAM J. Optimization.

3. Garatti, S. & Campi, M.C. (2024). "Risk and Complexity in Scenario Optimization." Mathematical Programming.

4. Packard, A. & Doyle, J. (1993). "The complex structured singular value." Automatica.

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust stability analysis for coupled oscillator systems with 2D parameter uncertainty.*
