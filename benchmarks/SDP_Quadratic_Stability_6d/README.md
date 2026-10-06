# Quadratic Stability Benchmark

## Problem Description

A 3x3 coupled oscillator LPV system dx/dt = A(delta)x has two uncertain parameters delta = (delta_1, delta_2) in [-1, 1]^2. The system matrices are:

    A(delta) = A_0 + delta_1 * A_1 + delta_2 * A_2

    A_0 = [[-0.3,  1.0,  0.0],        (nominal coupled oscillator)
           [-2.0, -1.0,  0.2],
           [ 0.2,  0.0, -1.2]]

    A_1 = [[0.0,  0.0,  0.0],         (stiffness/coupling uncertainty)
           [0.5,  0.15, 0.0],
           [0.0,  0.0,  0.3]]

    A_2 = [[0.0,  0.0,   0.0],        (damping uncertainty)
           [0.0,  0.3,   0.15],
           [0.15, 0.0,   0.45]]

The goal is to find a symmetric positive definite Lyapunov matrix P in R^{3x3} such that A(delta)'P + PA(delta) < 0 for all delta in [-1,1]^2, certifying quadratic stability over the entire 2D parameter space.

## Formulation

This is a Semidefinite Program (SDP) in the scenario approach inequality form. The decision variable is x = [p11, p12, p13, p22, p23, p33] (6 independent elements of the symmetric 3x3 Lyapunov matrix P). Each scenario delta_i in R^2 imposes a 3x3 LMI:

    F_0(delta_i) + sum_{j=1}^{6} x_j F_j(delta_i) <= 0

where F_j encode the Lyapunov equation structure. A hard constraint E_0 + sum x_j E_j <= 0 enforces P >= 0.01*I. The objective minimises trace(P) with a small quadratic regularisation (Q = 0.001*I). There is no slack penalty (rho = 0) and no regularisation (tau = 0).

## Results

![Quadratic Stability Results](results/quadratic_stability.png)

The SDP solves with N = 500 scenarios. The optimal Lyapunov matrix is:

    P = [[0.0101, -0.0005,  0.0001],
         [-0.0005, 0.0127, -0.0003],
         [ 0.0001, -0.0003, 0.0100]]

with trace(P) = 0.033. The solver found complexity k = 1 with no degeneracy, meaning one scenario constraint is active at the optimal solution. The risk bounds at 99.9999% confidence (beta = 10^{-6}) are [0.000, 0.040], meaning stability is certified for at least 96.0% of the parameter space.

## Files

```
SDP_Quadratic_Stability_6d/
├── README.md           This file
├── parameters.txt      Solver parameters (rho, tau, confidence)
├── generate.py         Generate scenarios and constraint matrices
├── run.py              Solve the SDP and print results
├── plot.py             Generate the paper figure
├── data/
│   ├── program_symbolic.json  Upload-Program definition (symbolic mode)
│   ├── program_numeric.json   Upload-Program definition (numeric mode)
│   ├── scenarios.csv          500 x 2 parameter samples
│   └── scenarios_numeric.csv  Per-row-flattened matrices for numeric mode
└── results/
    ├── metrics.json    Solver output (cost, risk bounds, etc.)
    ├── solution.csv    Raw solution vector
    ├── quadratic_stability.png   Paper figure (300 dpi)
    └── quadratic_stability.pdf   Paper figure (vector)
```

## Usage

```bash
# Generate scenarios and constraint matrices (optional)
python generate.py

# Solve the SDP (MOSEK by default; use --solver to pick another, e.g. CLARABEL)
python run.py
python run.py --solver CLARABEL

# Generate paper figure
python plot.py
```

## Web Interface Usage

### One-Shot Method (Recommended)

1. Start the web server: `python3 app.py`
2. Click **"Upload Program"** button (next to LP/QP/SDP tabs)
3. Upload `data/program_symbolic.json` (or `data/program_numeric.json` for numeric mode)
4. Upload `data/scenarios.csv` (or `data/scenarios_numeric.csv` with `program_numeric.json`) in the Scenarios box
5. Set solver to **MOSEK** and press **Solve**

### Manual Method

1. Select the **SDP** tab, formulation: **Robust**
2. Enter the matrices with their **Edit** buttons, using the matching fields of `data/program_symbolic.json`:
   - **F(delta)**: `F_d` (matrices F_0 ... F_6)
   - **E**: `E` (matrices E_0 ... E_6)
3. Enter:
   - **c**: `c`
   - **Q**: `Q`
4. Set parameters: rho = 0.0, tau = 0, confidence (beta) = 1e-06
5. Upload `data/scenarios.csv` in the Scenarios box
6. Press **Solve**

### Expected Results

- Optimal cost: 0.03285895857852664
- Complexity k: 1
- Risk bounds: [0.0, 0.04027646044222639]
