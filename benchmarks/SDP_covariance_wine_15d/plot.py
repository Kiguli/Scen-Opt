#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the wine covariance estimation benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

with open(os.path.join(results_dir, 'metrics.json')) as f:
    metrics = json.load(f)

X_est = np.array(metrics['covariance_estimated'])
cov_full = np.array(metrics['covariance_full_sample'])
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

# Short feature labels
short_names = ['Alc', 'Mal', 'Ash', 'Alk', 'Mg']

# Colours
C_EST = '#b2182b'
C_FULL = '#2166ac'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_MARKER = '#b2182b'

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ── Panel (a): Covariance matrices side by side ──
ax = axes[0]
p = X_est.shape[0]
vmax = max(np.max(np.abs(X_est)), np.max(np.abs(cov_full)))
im = ax.imshow(X_est - cov_full, cmap='RdBu_r', interpolation='nearest',
               vmin=-vmax * 0.5, vmax=vmax * 0.5)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Difference', fontsize=9)

# Annotate entries
for i in range(p):
    for j in range(p):
        diff = X_est[i, j] - cov_full[i, j]
        color = 'white' if abs(diff) > vmax * 0.3 else 'black'
        ax.text(j, i, f'{diff:+.2f}', ha='center', va='center', fontsize=7, color=color)

ax.set_xticks(range(p))
ax.set_xticklabels(short_names, fontsize=8)
ax.set_yticks(range(p))
ax.set_yticklabels(short_names, fontsize=8)
ax.set_title(r'(a) $\hat{\Sigma} - \Sigma_{\mathrm{full}}$ (entry errors)')

# ── Panel (b): Campi-Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N, 60) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx, kv in enumerate(k_range):
    eps_lo[idx], eps_hi[idx] = quantify_risk(kv, N, beta)

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
else:
    ax.annotate(f'$k={k}$ (off scale)', xy=(k_range[-1], eps_hi[-1]),
                xytext=(k_range[-1] - 15, eps_hi[-1] - 0.05),
                fontsize=9, color=C_MARKER,
                arrowprops=dict(arrowstyle='->', color=C_MARKER, lw=1.0))

ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
beta_str = f'{beta:.2g}'
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta={beta_str}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Eigenvalue comparison ──
ax = axes[2]
eigvals_full = np.linalg.eigvalsh(cov_full)

x_pos = np.arange(p)
width = 0.35
bars1 = ax.bar(x_pos - width / 2, sorted(eigvals_full), width, color=C_FULL,
               edgecolor='black', linewidth=0.5, label='Full-sample', alpha=0.8)
bars2 = ax.bar(x_pos + width / 2, sorted(eigvals), width, color=C_EST,
               edgecolor='black', linewidth=0.5, label='Estimated', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'$\\lambda_{i + 1}$' for i in range(p)])
ax.set_ylabel('Eigenvalue')
ax.set_title(r'(c) Eigenvalue spectrum of $\hat{\Sigma}$')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left')

plt.tight_layout()
out_path = os.path.join(results_dir, 'robust_covariance_wine.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
