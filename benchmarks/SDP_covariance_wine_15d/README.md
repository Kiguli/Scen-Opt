# Robust Covariance Estimation -- UCI Wine

## Problem Description

We seek the minimum-trace positive semidefinite covariance matrix Sigma that dominates the outer product of every data point from the UCI Wine dataset. This robust covariance estimation has applications in anomaly detection, portfolio risk, and Mahalanobis distance classification.

The Wine dataset contains 178 samples with 13 chemical measurements per wine. We select the first 5 features (alcohol, malic acid, ash, alcalinity of ash, magnesium), standardize each to zero mean and unit variance, and compute the full-sample covariance Sigma_full in R^{5 x 5}.

The decision variable is x = [sigma_11, sigma_12, ..., sigma_55] in R^15, the 15 entries of the 5 x 5 symmetric matrix Sigma = sum_{k=1}^{15} x_k Pi_k, where Pi_k are the standard basis matrices for the symmetric 5 x 5 space.

Each scenario delta_i in R^5 is one of the 178 standardized wine data points (N = 178 scenarios, one per sample).

## Formulation

This is a Semidefinite Program (SDP) in the scenario approach inequality form. The scenario-dependent constraint matrices encode outer-product domination:

    F_0(delta_i) = (delta_i - mean)(delta_i - mean)',   F_j(delta_i) = -Pi_j,   j = 1,...,15

so that (delta_i - mean)(delta_i - mean)' - Sigma <= 0 (PSD sense), i.e., Sigma >= (delta_i - mean)(delta_i - mean)' for each data point. The hard constraint enforces Sigma > 0 via E_0 = 0 and E_j = -Pi_j. The objective uses c = [1, 0, 0, 1, ..., 1] and Q = 0, minimizing trace(Sigma). There is no slack penalty (rho = 0) and no regularization (tau = 0).

## Results

![Robust Covariance Wine](results/robust_covariance_wine.png)

The optimal Sigma_hat has trace 75.71 and eigenvalues {6.20, 9.29, 12.59, 21.05, 26.58}, confirming positive definiteness. The Frobenius error ||Sigma_hat - Sigma_full||_F = 35.61 reflects the conservatism of the domination requirement: every individual point's outer product must be dominated, inflating the diagonal entries substantially. The complexity is k = 7, no degeneracy, and risk bounds [0.000, 0.175] at 99.9999% confidence.

## Files

```
SDP_covariance_wine_15d/
├── README.md           This file
├── parameters.txt      Solver parameters (rho, tau, confidence)
├── generate.py         Generate data point scenarios from Wine dataset
├── run.py              Solve the SDP and print results
├── plot.py             Generate the paper figure
├── data/
│   ├── program_symbolic.json         One-shot program definition (JSON)
│   ├── program_symbolic.json          One-shot program definition (MATLAB)
│   ├── scenarios.csv          178 x 5 data point scenarios
│   ├── basis_matrices.npy     15 x 5 x 5 symmetric basis matrices Pi_k
│   ├── free_entries.csv       Mapping of free entries in symmetric matrix
│   ├── c.csv                  Linear objective vector (trace)
│   ├── Q.csv                  Quadratic objective matrix (zero)
│   ├── E_0.csv ... E_15.csv   Hard constraint matrices (Sigma > 0)
│   ├── data_standardized.csv  Standardized wine data
│   ├── feature_names.txt      Feature labels
│   └── cov_full.csv           Full-sample covariance for comparison
└── results/
    ├── metrics.json                Solver output (cost, risk bounds, etc.)
    ├── solution.csv                Raw solution vector
    ├── Sigma_solution.csv          Reconstructed covariance matrix
    ├── robust_covariance_wine.png  Paper figure (300 dpi)
    └── robust_covariance_wine.pdf  Paper figure (vector)
```

## Usage

```bash
# Generate scenarios from Wine dataset (optional)
python generate.py

# Solve the SDP
python run.py

# Generate paper figure
python plot.py
```

## Web Interface Usage

### One-Shot Method (Recommended)

1. Start the web server: `python3 app.py`
2. Click **"Upload Program"** button (next to LP/QP/SDP tabs)
3. Upload `data/program_symbolic.json` or `data/program_symbolic.json`
4. Upload `data/scenarios.csv` in the Scenarios box
5. Set solver to **MOSEK** and press **Solve**

### Manual Method

1. Select the **SDP** tab, formulation: **Robust**
2. Upload matrices using one of these approaches:

   **Option A — One-shot LMI upload:**
   - Click **Edit F(delta)** -> upload `data/program_symbolic.json` (the F_d section) as a JSON file, or enter the F_d dict
   - Click **Edit E** -> upload the E matrices similarly

   **Option B — Individual matrix entry:**
   - Click **Edit F(delta)** -> set n = 15 -> click each F_i button and upload `data/F_0.csv` through `data/F_15.csv`
   - Click **Edit E** -> set n = 15 -> click each E_i button and upload `data/E_0.csv` through `data/E_15.csv`

3. Upload or enter:
   - **c**: `data/c.csv`
   - **Q**: `data/Q.csv`
4. Set parameters: rho = 0.0, tau = 0.0, confidence (beta) = 1e-06
5. Upload `data/scenarios.csv` in the Scenarios box
6. Press **Solve**

### Expected Results

- Optimal cost: 75.71438155818954
- Complexity k: 7
- Risk bounds: [0.0, 0.17538841818544354]
