#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the Iris SVM classification benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.Risk import quantify_risk

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

# Load results
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(benchmark_dir, 'data')

with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

scenarios = np.loadtxt(os.path.join(data_dir, 'scenarios.csv'), delimiter=',')
X = scenarios[:, :2]
y = scenarios[:, 2]

N = metrics['N']
k = metrics['k']
w1 = metrics['w1']
w2 = metrics['w2']
b = metrics['b']
margin = metrics['margin']

# Load beta from parameters.txt
params = {}
with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            params[key.strip()] = float(val.split('#')[0].strip())
beta = 1.0 - params.get('confidence', 0.999999)

# Identify support vectors (points closest to margin)
distances = np.abs(X @ np.array([w1, w2]) + b) / np.sqrt(w1**2 + w2**2)
tol = 1e-3
support_mask = np.abs(y * (X @ np.array([w1, w2]) + b) - 1.0) < tol

# Colours
C_SETOSA = '#2166ac'
C_OTHERS = '#b2182b'
C_BOUNDARY = '#2d6a4f'
C_MARGIN = '#e66101'
C_SUPPORT = '#ffd700'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Classification with decision boundary ──
ax = axes[0]

# Decision regions
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                       np.linspace(x2_min, x2_max, 200))
Z = w1 * xx1 + w2 * xx2 + b
cmap_light = ListedColormap(['#d1e5f0', '#fddbc7'])
ax.contourf(xx1, xx2, np.sign(Z), cmap=cmap_light, alpha=0.4)

# Margin boundaries
ax.contour(xx1, xx2, Z, levels=[-1, 0, 1],
           colors=[C_SETOSA, C_BOUNDARY, C_OTHERS],
           linestyles=['--', '-', '--'], linewidths=[1.2, 2.0, 1.2])

# Data points
setosa = y == -1
others = y == 1
ax.scatter(X[setosa, 0], X[setosa, 1], c=C_SETOSA, s=40, alpha=0.7,
           edgecolor='white', linewidth=0.3, label='Setosa ($y=-1$)', marker='o')
ax.scatter(X[others, 0], X[others, 1], c=C_OTHERS, s=40, alpha=0.7,
           edgecolor='white', linewidth=0.3, label='Others ($y=+1$)', marker='s')

# Support vectors
if np.any(support_mask):
    ax.scatter(X[support_mask, 0], X[support_mask, 1],
               facecolors='none', edgecolors=C_SUPPORT, s=150, linewidth=2.5,
               label=f'Support vectors ($k={k}$)', zorder=5)

ax.set_xlabel('Petal length (cm)')
ax.set_ylabel('Petal width (cm)')
ax.set_title('(a) SVM classification')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='lower right')
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

# ── Panel (b): Campi–Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 5, 25) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx, kv in enumerate(k_range):
    eps_lo[idx], eps_hi[idx] = quantify_risk(kv, N, beta)

ax.fill_between(k_range, eps_lo, eps_hi, alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_range, eps_lo, '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_range, eps_hi, '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_OTHERS, linestyle='-', linewidth=1.5, alpha=0.8)
k_idx = min(k, len(eps_lo) - 1)
ax.scatter([k], [eps_lo[k_idx]], c=C_OTHERS, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi[k_idx]], c=C_OTHERS, s=50, zorder=5, marker='o')
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k_idx]), xytext=(k + 2, eps_hi[k_idx] + 0.06),
            fontsize=9, color=C_OTHERS,
            arrowprops=dict(arrowstyle='->', color=C_OTHERS, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta=10^{{-6}}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Feature distribution by class ──
ax = axes[2]

positions_setosa = [0.8, 2.8]
positions_others = [1.2, 3.2]

bp1 = ax.boxplot([X[setosa, 0], X[setosa, 1]], positions=positions_setosa,
                  patch_artist=True, widths=0.35)
bp2 = ax.boxplot([X[others, 0], X[others, 1]], positions=positions_others,
                  patch_artist=True, widths=0.35)

for patch in bp1['boxes']:
    patch.set_facecolor(C_SETOSA)
    patch.set_alpha(0.6)
for patch in bp2['boxes']:
    patch.set_facecolor(C_OTHERS)
    patch.set_alpha(0.6)
for median in bp1['medians'] + bp2['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

ax.set_xticks([1.0, 3.0])
ax.set_xticklabels(['Petal length', 'Petal width'])
ax.set_ylabel('Value (cm)')
ax.set_title('(c) Feature distribution by class')
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Setosa', 'Others'],
          frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper right')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'iris_svm.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
