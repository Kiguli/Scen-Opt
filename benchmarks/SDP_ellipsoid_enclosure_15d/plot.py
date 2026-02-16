#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the minimum enclosing ellipsoid benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

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
results_dir = os.path.join(benchmark_dir, 'results')
data_dir = os.path.join(benchmark_dir, 'data')

with open(os.path.join(results_dir, 'metrics.json')) as f:
    metrics = json.load(f)

P = np.array(metrics['P_matrix'])
center = np.array(metrics['center'])
feature_names = metrics['feature_names']
eigvals = np.array(metrics['eigenvalues'])

N = metrics['N']
k = metrics['k']
degeneracy = metrics['degeneracy']

# Load beta from parameters.txt
params = {}
with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            params[key.strip()] = float(val.split('#')[0].strip())
beta = 1.0 - params.get('confidence', 0.99)

# Load data
data_all = np.loadtxt(os.path.join(data_dir, 'data_standardized.csv'), delimiter=',')
scenarios = np.loadtxt(os.path.join(data_dir, 'scenarios.csv'), delimiter=',')

# Compute containment for all data
containment_all = np.array([((x - center) @ P @ (x - center)) for x in data_all])
containment_scen = np.array([((x - center) @ P @ (x - center)) for x in scenarios])

# Colours
C_INSIDE = '#2166ac'
C_OUTSIDE = '#b2182b'
C_SCENARIO = '#4393c3'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_MARKER = '#b2182b'

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ── Panel (a): 2D projection with containment ──
ax = axes[0]
fi, fj = 0, 1  # first two features

inside = containment_all <= 1.0 + 1e-8
ax.scatter(data_all[inside, fi], data_all[inside, fj], c=C_INSIDE, s=12, alpha=0.4,
           label=f'Inside ({inside.sum()})', zorder=2, edgecolors='none')
ax.scatter(data_all[~inside, fi], data_all[~inside, fj], c=C_OUTSIDE, s=25, alpha=0.8,
           label=f'Outside ({(~inside).sum()})', zorder=3, marker='x', linewidths=1.5)

# Draw projected ellipse (Schur complement onto features fi, fj)
idx = [fi, fj]
idx_rest = [i for i in range(P.shape[0]) if i not in idx]
P11 = P[np.ix_(idx, idx)]
P12 = P[np.ix_(idx, idx_rest)]
P22 = P[np.ix_(idx_rest, idx_rest)]

try:
    P22_inv = np.linalg.pinv(P22)
    P_proj = P11 - P12 @ P22_inv @ P12.T
    proj_eigvals, proj_eigvecs = np.linalg.eigh(P_proj)
    if np.all(proj_eigvals > 1e-10):
        angle = np.degrees(np.arctan2(proj_eigvecs[1, 0], proj_eigvecs[0, 0]))
        width = 2.0 / np.sqrt(proj_eigvals[0])
        height = 2.0 / np.sqrt(proj_eigvals[1])
        ellipse = Ellipse(xy=(center[fi], center[fj]), width=width, height=height,
                          angle=angle, fill=False, edgecolor=C_MARKER, linewidth=2,
                          linestyle='-', zorder=4, label='Projected ellipse')
        ax.add_patch(ellipse)
except np.linalg.LinAlgError:
    pass

short_names = [n.replace('mean ', '').capitalize() for n in feature_names]
ax.set_xlabel(f'{short_names[fi]} (std.)')
ax.set_ylabel(f'{short_names[fj]} (std.)')
ax.set_title('(a) Ellipsoid containment (2D projection)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper right')

# ── Panel (b): Campi-Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N, 60) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx_k, kv in enumerate(k_range):
    eps_lo[idx_k], eps_hi[idx_k] = quantify_risk(kv, N, beta)

ax.fill_between(k_range, eps_lo, eps_hi, alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_range, eps_lo, '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_range, eps_hi, '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
k_plot = min(k, k_range[-1])
ax.axvline(x=k_plot, color=C_MARKER, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k_plot], [eps_lo[k_plot]], c=C_MARKER, s=50, zorder=5, marker='o')
ax.scatter([k_plot], [eps_hi[k_plot]], c=C_MARKER, s=50, zorder=5, marker='o')

if k <= k_range[-1]:
    ax.annotate(f'$k={k}$', xy=(k, eps_hi[k_plot]),
                xytext=(k + 3, eps_hi[k_plot] + 0.02),
                fontsize=9, color=C_MARKER,
                arrowprops=dict(arrowstyle='->', color=C_MARKER, lw=1.0))

ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
beta_str = f'{beta:.2g}'
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta={beta_str}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Sorted containment values ──
ax = axes[2]
sorted_vals = np.sort(containment_all)
idx_arr = np.arange(1, len(sorted_vals) + 1)

ax.plot(idx_arr, sorted_vals, '-', color=C_INSIDE, linewidth=1.5, label='All data points')
ax.axhline(y=1.0, color=C_MARKER, linestyle='--', linewidth=1.5, alpha=0.8, label='Boundary ($v^T P v = 1$)')
n_outside = np.sum(containment_all > 1.0 + 1e-8)
if n_outside > 0:
    ax.fill_between(idx_arr, sorted_vals, 1.0, where=sorted_vals > 1.0,
                    alpha=0.15, color=C_OUTSIDE, label=f'Outside ({n_outside} pts)')

ax.set_xlabel('Data point (sorted)')
ax.set_ylabel(r"Containment value $(\mathbf{x}-\mathbf{c})' P (\mathbf{x}-\mathbf{c})$")
ax.set_title(f'(c) Containment profile ($k={k}$, $N={N}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left')

plt.tight_layout()
out_path = os.path.join(results_dir, 'minimum_enclosing_ellipsoid.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
