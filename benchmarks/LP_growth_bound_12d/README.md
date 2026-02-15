# Data-Driven Growth Bounds for Vehicle Dynamics (12D LP)

## The Problem

This benchmark computes **data-driven growth bounds** for a vehicle dynamics system, providing probabilistic guarantees on reachable set over-approximations. The approach is based on the methodology from:

> **Kazemi, M., Salamati, M., Wooding, B., Soudjani, S., & Majumdar, R.** "Data-Driven Reachability Analysis for Stochastic Systems"

The goal is to bound how much the system state can evolve from one time step to the next, enabling efficient reachability analysis without requiring explicit knowledge of the system dynamics.

---

## Background: Reachability Analysis

**Reachability analysis** answers the question: "Given an initial set of states, what states can the system reach after one time step?"

Traditional approaches require:
- Explicit system dynamics model
- Conservative approximations (often too loose)
- Computationally expensive set propagation

**Data-driven approach**:
- Sample state transitions from simulations or experiments
- Learn growth bounds from data using scenario optimization
- Obtain probabilistic guarantees on over-approximation quality

---

## Vehicle Dynamics Model

The system follows **bicycle-like dynamics**:

```
x_dot = v * cos(alpha + theta) / cos(alpha)
y_dot = v * sin(alpha + theta) / cos(alpha)
theta_dot = v * tan(steering)
```

Where:
- `(x, y)` is the position
- `theta` is the heading angle
- `v` is the velocity (control input)
- `steering` is the steering angle (control input)
- `alpha = arctan(tan(steering)/2)` is the slip angle

**State**: x = [x, y, theta]
**Control**: u = [v, steering]

---

## Growth Bound Formulation

For a set centered at `x_center` with deviations `|x - x_center|`, we seek bounds:

```
|x_next - x_next_center| <= L * |x - x_center| + U
```

Where:
- L is a linear growth factor matrix (or vector)
- U is an additive offset term
- The bound holds with high probability across sampled transitions

### Decision Variables

The 12 decision variables encode the growth bound parameters:
- 9 variables for the L matrix (3x3 for linear growth)
- 3 variables for the U vector (additive terms)

---

## Problem Formulation

This is a **Linear Program (LP)**:

```
minimize    c' x    (minimize growth bound terms)

subject to  A_d(delta) @ x + b_d(delta) <= 0    for each scenario

            G @ x + h <= 0    (hard constraints)
```

Where each scenario `delta` contains:
- `delta[0:3]`: |x_current - x_center| (current deviations)
- `delta[3:6]`: |x_next - x_next_center| (next-step deviations)

---

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 12 (growth bound parameters) |
| State dimensions | 3 (x, y, theta) |
| Scenarios (N) | 100 (mini dataset) |
| Initial set width | 1.6 (square around center) |
| Time step | 0.03 seconds |

---

## Files

| File | Description |
|------|-------------|
| `growth_bound.csv` | Full dataset of state transitions |
| `growth_bound_mini.csv` | Mini dataset (100 samples) for faster testing |
| `c.csv` | Linear objective vector (12x1) |
| `G.csv` | Hard constraint matrix |
| `h.csv` | Hard constraint RHS |
| `A_d.csv` | Scenario-dependent constraint matrix |
| `b_d.csv` | Scenario-dependent constraint RHS |
| `data-collection.py` | Script to generate transition data |
| `test_and_visualize.py` | Solve LP and create 9-panel visualization |

---

## Usage

```bash
cd benchmarks/growth_bound_12d
python test_and_visualize.py
```

Results are saved to `results/`:
- `metrics.json`: Growth parameters, complexity, risk bounds
- `solution.csv`: Optimal decision variables
- `visualization.png`: 9-panel reachability analysis

---

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
============================================================
BENCHMARK: growth_bound_12d (LP)
Reachability Analysis for Vehicle Dynamics
============================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 100
Decision Variables: 12
Optimal Cost: -2.986429
Complexity (k): 5 support constraints
Risk Bounds (99%): [unreliable, 0.1663]
Degeneracy: True (lower bound unreliable)
------------------------------------------------------------

Solution (Growth Bound Parameters):
  x[0] = 0.000000
  x[1] = 0.000000
  x[2] = 0.000000
  ...
  (parameters define growth bound L, U)

Minimized variables (growth bound terms): [indices of active terms]
```

### Interpretation

**Complexity (k = 5)**:
- 5 scenario constraints are active at the optimal solution
- These correspond to the most "extreme" state transitions that shape the growth bound
- The growth bound is determined by transitions near the boundary of the reachable set

**Risk Bounds [unreliable, 0.1663]**:
- With 99% confidence, at most 16.63% of state transitions might exceed the computed bound
- **Degeneracy detected**: The lower bound is unreliable
- Degeneracy occurs when multiple growth bounds achieve the same optimal objective

**Degeneracy Explanation**:
- In reachability analysis, multiple parameter combinations can yield equivalent bounds
- This is common when the dynamics have symmetry or the initial set is symmetric
- The upper bound (16.63%) remains valid for conservative analysis

**Practical Meaning**:
- The computed growth bound is valid for at least 83.4% of state transitions
- For safety-critical applications, this provides a probabilistic over-approximation
- Additional scenarios would tighten the bounds

**Vehicle Dynamics Insights**:
- The growth bound captures how position and heading errors propagate
- Larger initial deviations lead to larger next-step deviations (growth)
- The L matrix encodes the linearized growth rate; U captures nonlinear effects

---

## Visualization Guide

The 9-panel visualization includes:

1. **3D Current Deviations**: Scatter plot of initial state deviations
2. **3D Next Deviations**: Scatter plot of next-step deviations
3. **Growth Ratio Distribution**: Box plot of deviation growth per dimension
4. **Reachable Set (X-Y Plane)**: 2D projection with initial and reached sets
5. **Deviation Correlation**: Current vs next deviation scatter
6. **Risk Bounds**: Scenario approach confidence intervals
7. **Vehicle Trajectory**: Nominal trajectory with uncertainty tube
8. **Max Deviation Histogram**: Distribution of maximum deviations
9. **Summary Statistics**: Key metrics and interpretation

---

## Connection to Formal Methods

This benchmark bridges **data-driven methods** with **formal verification**:

| Classical Reachability | Data-Driven Reachability |
|------------------------|--------------------------|
| Requires explicit dynamics | Works with simulation data |
| Conservative interval arithmetic | Tight probabilistic bounds |
| Exponential complexity in time | Linear in number of scenarios |
| Guaranteed (but loose) | Probabilistic (but tight) |

The scenario approach provides a principled way to obtain safety guarantees from data, complementing model-based formal methods.
