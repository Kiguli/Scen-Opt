#!/usr/bin/env python3
"""
Generate robust covariance estimation data from UCI Wine dataset for SDP formulation.

Each scenario is a wine data point. The SDP finds a PSD covariance matrix Sigma that
dominates all individual outer products:
  Sigma >= (x_i - mu)(x_i - mu)^T  (in PSD sense) for all data points

This guarantees that Sigma will also dominate the outer product of a new random wine.

SDP formulation (robust, rho=0):
  15 scalar decision variables x (entries of 5x5 symmetric Sigma)
  Scenario LMI (5x5): (delta - mu)(delta - mu)^T - Sigma << 0  (outer product domination)
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
center = data.mean(axis=0)
print(f"Wine dataset: {n_samples} samples, {p} features")
print(f"Features: {feature_names}")
print(f"Center (mean): {center.round(3)}")

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
# Use ALL wine data points as direct scenarios (no subsampling)
N = n_samples
scenario_deltas = data  # Each row is a 5D data point

print(f"\nScenarios: {N} (all wine data points)")
print(f"Delta dimension: {p} (raw feature values)")

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

# F_d expression CSVs for scenario LMI: (delta - center)(delta - center)^T - Sigma << 0
# F_0: outer product (delta - center)(delta - center)^T — 5x5 symmetric
def float_str(v):
    """Format float for expression string, wrapping negatives in parens."""
    s = repr(float(v))
    return f'({s})' if s.startswith('-') else s

with open(os.path.join(data_dir, 'F_0.csv'), 'w') as f:
    for i in range(p):
        row = []
        for j in range(p):
            c_i = float_str(center[i])
            c_j = float_str(center[j])
            if i == j:
                row.append(f'(delta[{i}] - {c_i})**2')
            else:
                row.append(f'(delta[{i}] - {c_i})*(delta[{j}] - {c_j})')
        f.write(','.join(row) + '\n')

# F_1 through F_15: -B_k (constant numeric matrices)
for k in range(n_vars):
    np.savetxt(os.path.join(data_dir, f'F_{k+1}.csv'), -basis_matrices[k], delimiter=',')

print(f"\nSaved data files to: {data_dir}")
print(f"  F_0.csv: 5x5 expression matrix (outer product entries)")
print(f"  F_1.csv-F_{n_vars}.csv: 5x5 numeric matrices (-B_k)")
