# Smallest Enclosing Interval

## Problem Description

We sample 100 points from the 1-dimensional space [0, 1] and seek the smallest interval that encloses all samples, following the setup in Campi & Garatti (2018). The solution consists of the interval half-width h and center o, i.e., x = [o, h]. Each data point imposes a constraint that it must lie within the interval.

## Formulation

This is a Linear Program with 2 decision variables x = [center, h] and no hard constraints. The objective c = [0, 1] minimizes the half-width. Each scenario δᵢ (a single data point) generates two soft constraints via:

A(δᵢ) = [[-1, -1], [1, -1]], b(δᵢ) = [δᵢ, -δᵢ]

No regularisation (τ = 0) or slack penalty (ρ = 0).

## Results

![Smallest Enclosing Interval](results/half_width.png)

The LP solves with N = 100 data points. The optimal interval is [0.002, 0.999] with center 0.501 and half-width 0.498. The complexity is k = 2 (the two extreme data points define the interval). With β = 10⁻⁶, the risk bounds are ε̲ = 0 and ε̄ = 0.209.

## Files

```
LP_half_width_2d/
├── README.md           This file
├── parameters.txt      Solver parameters (rho, tau, confidence)
├── run.py              Solve the LP and print results
├── plot.py             Generate the paper figure
├── data/
│   ├── program_symbolic.json  Upload-Program definition (symbolic mode)
│   ├── program_numeric.json   Upload-Program definition (numeric mode)
│   ├── scenarios.csv          100 data points sampled from [0, 1]
│   └── scenarios_numeric.csv  Per-row-flattened matrices for numeric mode
└── results/
    ├── metrics.json    Solver output (center, half-width, risk bounds)
    ├── solution.csv    Raw solution vector
    ├── half_width.png  Paper figure (300 dpi)
    └── half_width.pdf  Paper figure (vector)
```

## Usage

```bash
# Solve the LP (MOSEK by default; use --solver to pick another, e.g. CLARABEL)
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

1. Select the **LP** tab, formulation: **Robust**
2. Enter each matrix with its **Edit** button, using the matching field of `data/program_symbolic.json`:
   - **A(delta)**: `A_d`
   - **b(delta)**: `b_d`
   - **c**: `c`
3. Set parameters: rho = 0, tau = 0, confidence (beta) = 1e-06
4. Upload `data/scenarios.csv` in the Scenarios box
5. Press **Solve**

### Expected Results

- Optimal cost: 0.498309968223239
- Complexity k: 2
- Risk bounds: [0.0, 0.20852406037505716]
