#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the smallest enclosing interval benchmark."""

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
data_dir = os.path.join(benchmark_dir, 'data')

with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

data_points = np.loadtxt(os.path.join(data_dir, 'scenarios.csv'), delimiter=',').flatten()
N = metrics['N']
k = metrics['k']
# Load beta from parameters.txt
params = {}
with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            params[key.strip()] = float(val.split('#')[0].strip())
beta = 1.0 - params.get('confidence', 0.999999)
center = metrics['center']
half_width = metrics['half_width']
interval_min = metrics['interval_min']
interval_max = metrics['interval_max']
support_indices = metrics['support_indices']

# Identify support points
tol = 1e-4
at_lower = np.abs(data_points - interval_min) < tol
at_upper = np.abs(data_points - interval_max) < tol
is_support = at_lower | at_upper

# Colours
C_DATA = '#4393c3'
C_SUPPORT = '#b2182b'
C_INTERVAL = '#2166ac'
C_CENTER = '#e66101'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_HIST = '#4393c3'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Data points with optimal interval ──
ax = axes[0]
np.random.seed(42)
y_jitter = np.random.uniform(-0.3, 0.3, len(data_points))

# Non-support points
ax.scatter(data_points[~is_support], y_jitter[~is_support], c=C_DATA, s=30, alpha=0.6,
           edgecolor='white', linewidth=0.3, label=f'Data points ($N={N}$)', zorder=3)

# Support points
ax.scatter(data_points[is_support], y_jitter[is_support], c=C_SUPPORT, s=100, marker='*',
           edgecolor='darkred', linewidth=0.5, label=f'Support scenarios ($k={k}$)', zorder=4)

# Interval shading
ax.axvspan(interval_min, interval_max, alpha=0.12, color=C_INTERVAL)
ax.axvline(x=interval_min, color=C_INTERVAL, linewidth=2.0, alpha=0.8)
ax.axvline(x=interval_max, color=C_INTERVAL, linewidth=2.0, alpha=0.8)
ax.axvline(x=center, color=C_CENTER, linestyle='--', linewidth=1.5,
           label=f'Center = {center:.4f}')

# Half-width annotation
arrow_y = 0.45
ax.annotate('', xy=(center, arrow_y), xytext=(interval_min, arrow_y),
            arrowprops=dict(arrowstyle='<->', color=C_CENTER, lw=1.5))
ax.annotate('', xy=(interval_max, arrow_y), xytext=(center, arrow_y),
            arrowprops=dict(arrowstyle='<->', color=C_CENTER, lw=1.5))
ax.text(center, arrow_y + 0.1, f'$h = {half_width:.4f}$', ha='center',
        fontsize=9, color=C_CENTER)

ax.set_xlim(interval_min - 0.05, interval_max + 0.05)
ax.set_ylim(-0.55, 0.7)
ax.set_xlabel('Value')
ax.set_yticks([])
ax.set_title('(a) Smallest enclosing interval')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='lower right')

# ── Panel (b): Campi–Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 5, 20) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx, kv in enumerate(k_range):
    eps_lo[idx], eps_hi[idx] = quantify_risk(kv, N, beta)

ax.fill_between(k_range, eps_lo, eps_hi, alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_range, eps_lo, '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_range, eps_hi, '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_SUPPORT, linestyle='-', linewidth=1.5, alpha=0.8)
k_idx = min(k, len(eps_lo) - 1)
ax.scatter([k], [eps_lo[k_idx]], c=C_SUPPORT, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi[k_idx]], c=C_SUPPORT, s=50, zorder=5, marker='o')
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k_idx]), xytext=(k + 2, eps_hi[k_idx] + 0.06),
            fontsize=9, color=C_SUPPORT,
            arrowprops=dict(arrowstyle='->', color=C_SUPPORT, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
conf_pct = (1 - beta) * 100
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta=10^{{-6}}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Histogram with interval bounds ──
ax = axes[2]
ax.hist(data_points, bins=20, color=C_HIST, alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(x=interval_min, color=C_INTERVAL, linewidth=2.0,
           label=f'Interval bounds')
ax.axvline(x=interval_max, color=C_INTERVAL, linewidth=2.0)
ax.axvline(x=center, color=C_CENTER, linestyle='--', linewidth=1.5,
           label=f'Center')
ax.axvline(x=np.mean(data_points), color='#2d6a4f', linestyle=':', linewidth=1.5,
           label=f'Mean = {np.mean(data_points):.4f}')

ax.set_xlabel('Value')
ax.set_ylabel('Count')
ax.set_title('(c) Data distribution')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper right')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'half_width.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
