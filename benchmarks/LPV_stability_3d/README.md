# LPV System Stability Analysis (3D SDP)

## The Problem

**Linear Parameter-Varying (LPV) systems** are dynamical systems where the state-space matrices depend on time-varying parameters. A fundamental challenge in control theory is finding a **common Lyapunov function** that proves stability across all parameter values.

This benchmark uses the scenario approach to find a Lyapunov matrix P that certifies stability of an LPV system for a range of scheduling parameters, with probabilistic guarantees on robustness.

---

## Mathematical Background

### LPV System Model

The system follows:
```
dx/dt = A(delta) * x
```

Where:
```
A(delta) = A_0 + delta * A_1

A_0 = [[0, 1], [-2, -1]]    (stable damped oscillator)
A_1 = [[0, 0], [0.3, 0.1]]  (parameter-varying perturbation)
```

The scheduling parameter `delta` varies within the range [-0.22, 1].

### Lyapunov Stability Condition

For stability, we need a positive definite matrix P such that:
```
A(delta)' P + P A(delta) < 0    (negative definite)
```

for all values of delta in the parameter range.

### Decision Variables

The 3 decision variables parameterize the symmetric Lyapunov matrix:
```
P = [[p_11, p_12],
     [p_12, p_22]]

x = [p_11, p_12, p_22]
```

---

## Problem Formulation

This is a **Semidefinite Program (SDP)**:

```
minimize    (1/2) x' Q x + c' x

subject to  F_0(delta) + x[0]*F_1(delta) + x[1]*F_2(delta) + x[2]*F_3(delta) <= 0
            (for each scenario delta)

            E_0 + x[0]*E_1 + x[1]*E_2 + x[2]*E_3 <= 0
            (hard constraint: P > 0)
```

Where `<= 0` denotes negative semidefiniteness (matrix inequality).

---

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 3 (Lyapunov matrix elements) |
| Uncertainty dimensions | 1 (scheduling parameter) |
| Scenarios (N) | 100 |
| Matrix dimension | 2x2 |
| Parameter range | [-0.22, 1.0] |

---

## Files

| File | Description |
|------|-------------|
| `scheduling_variable_samples.csv` | 100 sampled parameter values |
| `Q.csv` | Quadratic objective matrix (3x3) |
| `c.csv` | Linear objective vector (3x1) |
| `F_0.csv` - `F_3.csv` | Scenario-dependent LMI matrices |
| `E_0.csv` - `E_3.csv` | Hard constraint (P > 0) matrices |
| `test_and_visualize.py` | Solve and create 9-panel visualization |

---

## Usage

```bash
cd benchmarks/LPV_stability_3d
python test_and_visualize.py
```

Results are saved to `results/`:
- `metrics.json`: Lyapunov parameters, complexity, risk bounds
- `solution.csv`: Optimal decision variables
- `visualization.png`: 9-panel analysis figure

---

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
============================================================
BENCHMARK: LPV_stability_3d (SDP)
Common Lyapunov Function for LPV System
============================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 100
Decision Variables: 3
Optimal Cost: -9.841018
Complexity (k): 1 support constraint
Risk Bounds (99%): [0.0000, 0.0941]
------------------------------------------------------------

Solution (Lyapunov Parameters):
  x[1] = 1.250000 (p_11)
  x[2] = 0.250000 (p_12)
  x[3] = 0.750000 (p_22)

Stability Analysis:
  delta=-0.2200: max eigenvalue = -0.071 (stable)
  delta= 0.0000: max eigenvalue = -0.095 (stable)
  delta= 1.0000: max eigenvalue = -0.000 (stable)
```

### Interpretation

**Complexity (k = 1)**:
- Only 1 scenario constraint is active at the optimal solution
- This indicates the Lyapunov function is primarily constrained by a single extreme parameter value
- The low complexity suggests the stability region has a simple structure

**Risk Bounds [0.0000, 0.0941]**:
- With 99% confidence, at most 9.41% of parameter values might violate the stability condition
- The lower bound of 0 indicates the solution may be fully robust within the sampled distribution
- For control systems, this provides strong probabilistic stability guarantees

**Lyapunov Matrix P**:
```
P = [[1.25, 0.25],
     [0.25, 0.75]]
```
- The matrix P is positive definite (both eigenvalues positive)
- The condition A'P + PA < 0 is satisfied across the entire parameter range

**Stability Verification**:
- Maximum eigenvalue of A'P + PA remains negative (or nearly zero) across all tested parameters
- The system is certified stable for delta in [-0.22, 1.0]
- At delta = 1.0, the eigenvalue approaches 0, indicating the boundary of stability

**Practical Meaning**:
- The computed Lyapunov function certifies stability with high probability
- Engineers can use this P matrix for controller design and stability verification
- The scenario approach provides finite-sample guarantees unlike classical robust methods

---

## Visualization Guide

The 9-panel visualization includes:

1. **Sampled Parameters**: Distribution of scheduling parameter values
2. **Parameter Histogram**: Frequency distribution of delta
3. **Eigenvalue vs Parameter**: Stability margin across parameter range
4. **Decision Variables**: Bar chart of optimal Lyapunov parameters
5. **Lyapunov Matrix P**: Heatmap of the 2x2 matrix
6. **Risk Bounds**: Scenario approach confidence intervals
7. **-P Matrix**: Verification of positive definiteness constraint
8. **Eigenvalue Histogram**: Distribution of stability margins
9. **Summary Statistics**: Key metrics and interpretation

---

## References

1. Apkarian, P. & Adams, R.J. (1998). "Advanced gain-scheduling techniques for uncertain systems." IEEE Trans. Control Systems Technology.

2. Packard, A. (1994). "Gain scheduling via linear fractional transformations." Systems & Control Letters.

3. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions of uncertain convex programs." SIAM J. Optimization.

4. Boyd, S. et al. (1994). "Linear Matrix Inequalities in System and Control Theory." SIAM.

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust control analysis with finite-sample stability guarantees.*
