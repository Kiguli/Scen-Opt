#!/usr/bin/env python3
"""
Generate minimum enclosing ellipsoid data from Breast Cancer Wisconsin dataset for SDP.

Each scenario is a data point from the full dataset. The SDP finds the tightest PSD shape
matrix P such that all points lie inside the ellipsoid {x : (x-c)'P(x-c) <= 1}.
The scenario approach guarantees containment of future random data points.

SDP formulation (robust, rho=0):
  15 scalar decision variables x (entries of 5x5 symmetric P)
  Scenario LMI (1x1): (x_i - c)' P (x_i - c) - 1 <= 0  (point containment)
  Hard LMI (5x5): -P << 0  (enforces P >> 0)
  Objective: maximize trace(P) = minimize -trace(P)
"""

import os
import numpy as np
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

# ===== Load and preprocess data =====
bc = load_breast_cancer()
data_full = bc.data[:, :5]  # First 5 features
feature_names = list(bc.feature_names[:5])

# Standardize
means = data_full.mean(axis=0)
stds = data_full.std(axis=0)
data = (data_full - means) / stds

n_samples, p = data.shape
center = data.mean(axis=0)

print(f"Breast Cancer dataset: {n_samples} samples, {p} features")
print(f"Features: {feature_names}")
print(f"Center (mean): {center.round(3)}")

# ===== Enumerate variables =====
# x = upper triangle of P (p x p symmetric)
free_entries = []
for i in range(p):
    for j in range(i, p):
        free_entries.append((i, j))

n_vars = len(free_entries)  # 15
print(f"\nDecision variables: {n_vars}")

# Basis matrices B_k: P = sum(x_k * B_k)
basis_matrices = []
for i, j in free_entries:
    B = np.zeros((p, p))
    if i == j:
        B[i, i] = 1.0
    else:
        B[i, j] = 1.0
        B[j, i] = 1.0
    basis_matrices.append(B)

# Cost vector: maximize trace(P) = minimize -trace(P)
c = np.zeros(n_vars)
for k, (i, j) in enumerate(free_entries):
    if i == j:
        c[k] = -1.0

Q = np.zeros((n_vars, n_vars))

# ===== Generate scenarios =====
# Use ALL data points as scenarios (no subsampling)
N = n_samples
scenario_points = data

print(f"\nScenarios: {N} (all data points)")
print(f"Data shape: {scenario_points.shape}")

# Precompute coefficients for each scenario
# coeff[i, k] = (x_i - center)' B_k (x_i - center)
coeffs = np.zeros((N, n_vars))
for i in range(N):
    v = scenario_points[i] - center
    for k in range(n_vars):
        coeffs[i, k] = v @ basis_matrices[k] @ v

print(f"Coefficient matrix shape: {coeffs.shape}")
print(f"Max coeff: {coeffs.max():.3f}, Min coeff: {coeffs.min():.3f}")

# ===== Save everything =====
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(benchmark_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

np.savetxt(os.path.join(data_dir, 'scenarios.csv'), scenario_points, delimiter=',')
np.savetxt(os.path.join(data_dir, 'coeffs.csv'), coeffs, delimiter=',')
np.savetxt(os.path.join(data_dir, 'center.csv'), center, delimiter=',')
np.savetxt(os.path.join(data_dir, 'Q.csv'), Q, delimiter=',')
np.savetxt(os.path.join(data_dir, 'c.csv'), c, delimiter=',')
np.savetxt(os.path.join(data_dir, 'data_standardized.csv'), data, delimiter=',')

# E matrices for hard constraint: -P << 0 (P >> 0)
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

# Save full dataset covariance for reference
cov_full = np.cov(data, rowvar=False)
np.savetxt(os.path.join(data_dir, 'cov_full.csv'), cov_full, delimiter=',')

# F_d expression CSVs for scenario LMI: (x_i - c)' P (x_i - c) - 1 <= 0
# F_0: constant [[-1.0]]
with open(os.path.join(data_dir, 'F_0.csv'), 'w') as f:
    f.write('-1.0\n')

# F_1 through F_15: quadratic form (delta - center)' B_k (delta - center)
def float_str(v):
    """Format float for expression string, wrapping negatives in parens."""
    s = repr(float(v))
    return f'({s})' if s.startswith('-') else s

for k, (a, b) in enumerate(free_entries):
    if a == b:
        c_a = float_str(center[a])
        expr = f'(delta[{a}] - {c_a})**2'
    else:
        c_a = float_str(center[a])
        c_b = float_str(center[b])
        expr = f'2*(delta[{a}] - {c_a})*(delta[{b}] - {c_b})'
    with open(os.path.join(data_dir, f'F_{k+1}.csv'), 'w') as f:
        f.write(expr + '\n')

print(f"\nSaved data files to: {data_dir}")
print(f"  F_0.csv: 1x1 constant matrix [[-1.0]]")
print(f"  F_1.csv-F_{n_vars}.csv: 1x1 quadratic form expressions")
