#!/usr/bin/env python3
"""Generate a paper-ready figure showing only the Iris data (no solution overlay)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Style
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Load data
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(benchmark_dir, 'data')
scenarios = np.loadtxt(os.path.join(data_dir, 'scenarios.csv'), delimiter=',')
X = scenarios[:, :2]
y = scenarios[:, 2]

# Colours
C_SETOSA = '#2166ac'
C_OTHERS = '#b2182b'

fig, ax = plt.subplots(figsize=(5, 4))

setosa = y == -1
others = y == 1
ax.scatter(X[setosa, 0], X[setosa, 1], c=C_SETOSA, s=40, alpha=0.7,
           edgecolor='white', linewidth=0.3, label='Setosa ($y=-1$)', marker='o')
ax.scatter(X[others, 0], X[others, 1], c=C_OTHERS, s=40, alpha=0.7,
           edgecolor='white', linewidth=0.3, label='Others ($y=+1$)', marker='s')

ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.set_title('Iris dataset (petal features)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='lower right')
ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
ax.set_ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'iris_data_only.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
