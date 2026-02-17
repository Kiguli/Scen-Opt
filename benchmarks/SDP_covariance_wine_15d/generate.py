#!/usr/bin/env python3
"""
Generate robust covariance estimation data from UCI Wine dataset for SDP formulation.

Each scenario draws a random subsample of wines and computes its sample covariance.
The SDP finds a PSD covariance matrix Sigma that dominates all subsample covariances:
  Sigma >= S_sub  (in PSD sense) for all subsamples

SDP formulation (robust, rho=0):
  15 scalar decision variables x (entries of 5x5 symmetric Sigma)
  Scenario LMI (5x5): S_sub(delta) - Sigma << 0  (covariance domination)
  Hard LMI (5x5): -Sigma << 0  (enforces Sigma >> 0)
  Objective: minimize trace(Sigma)
"""

import os
import numpy as np
from sklearn.datasets import load_wine

np.random.seed(42)

# ===== Load and preprocess data =====
wine = load_wine()
data_full = wine.data[:, :5]  # First 5 features
feature_names = list(wine.feature_names[:5])

# Standardize
means = data_full.mean(axis=0)
stds = data_full.std(axis=0)
data = (data_full - means) / stds

n_samples, p = data.shape
print(f"Wine dataset: {n_samples} samples, {p} features")
print(f"Features: {feature_names}")

# Full-sample covariance
cov_full = np.cov(data, rowvar=False)
print(f"\nFull-sample covariance eigenvalues: {np.linalg.eigvalsh(cov_full).round(3)}")

# ===== Enumerate variables =====
# x = upper triangle of Sigma (p x p symmetric)
free_entries = []
for i in range(p):
    for j in range(i, p):
        free_entries.append((i, j))

n_vars = len(free_entries)  # 15
print(f"\nDecision variables: {n_vars}")

# Basis matrices B_k: Sigma = sum(x_k * B_k)
basis_matrices = []
for i, j in free_entries:
    B = np.zeros((p, p))
    if i == j:
        B[i, i] = 1.0
    else:
        B[i, j] = 1.0
        B[j, i] = 1.0
    basis_matrices.append(B)

# Cost vector: minimize trace(Sigma)
c = np.zeros(n_vars)
for k, (i, j) in enumerate(free_entries):
    if i == j:
        c[k] = 1.0

Q = np.zeros((n_vars, n_vars))

# ===== Generate scenarios =====
N = 200
subsample_size = 50

# Each scenario: random subsample -> full sample covariance (p x p)
# Store delta as flattened upper triangle of S_sub
scenario_deltas = np.zeros((N, n_vars))

for s in range(N):
    idx = np.random.choice(n_samples, size=subsample_size, replace=False)
    subsample = data[idx]
    cov_sub = np.cov(subsample, rowvar=False)
    # Flatten upper triangle
    for k, (i, j) in enumerate(free_entries):
        scenario_deltas[s, k] = cov_sub[i, j]

print(f"\nScenarios: {N}")
print(f"Subsample size: {subsample_size}")
print(f"Delta dimension: {n_vars} (flattened upper triangle of S_sub)")

# ===== Save everything =====
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(benchmark_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

np.savetxt(os.path.join(data_dir, 'scenarios.csv'), scenario_deltas, delimiter=',')
np.savetxt(os.path.join(data_dir, 'Q.csv'), Q, delimiter=',')
np.savetxt(os.path.join(data_dir, 'c.csv'), c, delimiter=',')
np.savetxt(os.path.join(data_dir, 'cov_full.csv'), cov_full, delimiter=',')
np.savetxt(os.path.join(data_dir, 'data_standardized.csv'), data, delimiter=',')

# E matrices for hard constraint: -Sigma << 0 (Sigma >> 0)
np.savetxt(os.path.join(data_dir, 'E_0.csv'), np.zeros((p, p)), delimiter=',')
for k in range(n_vars):
    np.savetxt(os.path.join(data_dir, f'E_{k+1}.csv'), -basis_matrices[k], delimiter=',')

# Save basis info
np.save(os.path.join(data_dir, 'basis_matrices.npy'), np.array(basis_matrices))
np.savetxt(os.path.join(data_dir, 'free_entries.csv'),
           np.array(free_entries), delimiter=',', fmt='%d')

with open(os.path.join(data_dir, 'feature_names.txt'), 'w') as f:
    for name in feature_names:
        f.write(name + '\n')

# F_d expression CSVs for scenario LMI: S_sub(delta) - Sigma << 0
# F_0: S_sub(delta) — 5x5 symmetric with delta[k] at each entry
entry_map = {}
for k, (i, j) in enumerate(free_entries):
    entry_map[(i, j)] = k
    entry_map[(j, i)] = k  # symmetric

with open(os.path.join(data_dir, 'F_0.csv'), 'w') as f:
    for i in range(p):
        row = []
        for j in range(p):
            row.append(f'delta[{entry_map[(i, j)]}]')
        f.write(','.join(row) + '\n')

# F_1 through F_15: -B_k (constant numeric matrices)
for k in range(n_vars):
    np.savetxt(os.path.join(data_dir, f'F_{k+1}.csv'), -basis_matrices[k], delimiter=',')

print(f"\nSaved data files to: {data_dir}")
print(f"  F_0.csv: 5x5 expression matrix (delta[k] entries)")
print(f"  F_1.csv-F_{n_vars}.csv: 5x5 numeric matrices (-B_k)")
