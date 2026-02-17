# Minimum Enclosing Ellipsoid -- Breast Cancer Wisconsin

## Problem Description

Given a collection of data points, we seek the tightest ellipsoid (in the trace sense) that contains all sampled observations. This is a fundamental problem in robust statistics and anomaly detection: the scenario approach provides probabilistic guarantees that unseen data points from the same distribution will also be contained.

The Breast Cancer Wisconsin dataset contains 569 samples with 30 features. We select the first 5 features (mean radius, mean texture, mean perimeter, mean area, mean smoothness) and standardize each to zero mean and unit variance, with data center x_bar = mean(data).

The decision variable is x = [p_11, p_12, ..., p_55] in R^15, the entries of the 5 x 5 symmetric shape matrix P = sum_{k=1}^{15} x_k Pi_k, defining the ellipsoid E = {z : (z - x_bar)' P (z - x_bar) <= 1}.

Each scenario delta_i in R^5 is a data point drawn uniformly at random from the dataset (N = 200 scenarios).

## Formulation

This is a Semidefinite Program (SDP) in the scenario approach inequality form. The scenario-dependent constraint matrices encode point containment as a 1 x 1 (scalar) LMI:

    F_0(delta_i) = -1,   F_j(delta_i) = (delta_i - x_bar)' Pi_j (delta_i - x_bar),   j = 1,...,15

so that (delta_i - x_bar)' P (delta_i - x_bar) - 1 <= 0. The hard constraint enforces P > 0 via E_0 = 0 and E_j = -Pi_j. The objective uses c = [-1, 0, 0, -1, ..., -1] and Q = 0, maximizing trace(P) (i.e., minimizing -trace(P)) to find the tightest ellipsoid. There is no slack penalty (rho = 0) and no regularization (tau = 0).

## Results

![Minimum Enclosing Ellipsoid](results/minimum_enclosing_ellipsoid.png)

The optimal P has trace 53.77 and eigenvalues {~0, ~0, ~0, 0.61, 53.16}, revealing that the standardized data effectively lies in a 2-dimensional subspace (the first 5 breast cancer features are highly correlated). Of the full 569-point dataset, 559 (98.2%) lie inside the ellipsoid; the 10 points outside were not among the 200 sampled scenarios. The complexity is k = 6 (six boundary data points define the ellipsoid), no degeneracy, and risk bounds [0.0029, 0.0942] at 99% confidence, guaranteeing that a new random data point will fall inside the ellipsoid with probability at least 90.6%.

## Files

```
SDP_ellipsoid_enclosure_15d/
├── README.md           This file
├── parameters.txt      Solver parameters (rho, tau, confidence)
├── generate.py         Generate data point scenarios from Breast Cancer dataset
├── run.py              Solve the SDP and print results
├── plot.py             Generate the paper figure
├── data/
│   ├── benchmark.json             One-shot program definition (JSON)
│   ├── benchmark.mat              One-shot program definition (MATLAB)
│   ├── scenarios.csv              200 x 5 data point scenarios
│   ├── basis_matrices.npy         15 x 5 x 5 symmetric basis matrices Pi_k
│   ├── free_entries.csv           Mapping of free entries in symmetric matrix
│   ├── coeffs.csv                 Precomputed containment coefficients
│   ├── center.csv                 Data center x_bar
│   ├── c.csv                      Linear objective vector (-trace)
│   ├── Q.csv                      Quadratic objective matrix (zero)
│   ├── E_0.csv ... E_15.csv       Hard constraint matrices (P > 0)
│   ├── data_standardized.csv      Full standardized dataset (569 x 5)
│   └── feature_names.txt          Feature labels
└── results/
    ├── metrics.json                     Solver output (cost, risk bounds, etc.)
    ├── solution.csv                     Raw solution vector
    ├── P_solution.csv                   Reconstructed shape matrix
    ├── minimum_enclosing_ellipsoid.png  Paper figure (300 dpi)
    └── minimum_enclosing_ellipsoid.pdf  Paper figure (vector)
```

## Usage

```bash
# Generate scenarios from Breast Cancer dataset (optional)
python generate.py

# Solve the SDP
python run.py

# Generate paper figure
python plot.py
```

## Web Interface Usage

### One-Shot Method (Recommended)

1. Start the web server: `python3 app.py`
2. Click **"Detect Program"** button (next to LP/QP/SDP tabs)
3. Upload `data/benchmark.json` or `data/benchmark.mat`
4. Upload `data/scenarios.csv` in the Scenarios box
5. Set solver to **MOSEK** and press **Solve**

### Manual Method

1. Select the **SDP** tab, formulation: **Robust**
2. Upload matrices using one of these approaches:

   **Option A — One-shot LMI upload:**
   - Click **Edit F(delta)** -> upload `data/benchmark.json` (the F_d section) as a JSON file, or enter the F_d dict
   - Click **Edit E** -> upload the E matrices similarly

   **Option B — Individual matrix entry:**
   - Click **Edit F(delta)** -> set n = 15 -> click each F_i button and upload `data/F_0.csv` through `data/F_15.csv`
   - Click **Edit E** -> set n = 15 -> click each E_i button and upload `data/E_0.csv` through `data/E_15.csv`

3. Upload or enter:
   - **c**: `data/c.csv`
   - **Q**: `data/Q.csv`
4. Set parameters: rho = 0.0, tau = 0.0, confidence (beta) = 0.01
5. Upload `data/scenarios.csv` in the Scenarios box
6. Press **Solve**

### Expected Results

- Optimal cost: -53.77272568355289
- Complexity k: 6
- Risk bounds: [0.0029051032103598116, 0.0942128561489517]
