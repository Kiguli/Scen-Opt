#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the inventory benchmark."""

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
with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

scenarios = np.loadtxt(os.path.join(benchmark_dir, 'data', 'scenarios.csv'), delimiter=',')
yields = scenarios[:, 0:5]
demands = scenarios[:, 5:10]

NAMES = ['Strawberries', 'Tomatoes', 'Lettuce', 'Avocados', 'Bell Peppers']
SHORT = ['Straw.', 'Tom.', 'Lett.', 'Avoc.', 'Bell P.']
q = np.array([metrics['order_quantities'][n] for n in NAMES])
N = metrics['N']
k = metrics['k']
beta = 0.01

# Colours: muted palette suitable for print
C_ORDER = '#2166ac'
C_DEMAND = '#b2182b'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_SUPPLY = '#4393c3'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Order quantities vs mean demand ──
ax = axes[0]
x_pos = np.arange(len(NAMES))
width = 0.35
mean_dem = np.mean(demands, axis=0)

ax.bar(x_pos - width/2, mean_dem, width, label='Mean demand',
       color=C_DEMAND, alpha=0.7, edgecolor='white', linewidth=0.5)
ax.bar(x_pos + width/2, q, width, label='Optimal order',
       color=C_ORDER, alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_ylabel('Cases')
ax.set_xticks(x_pos)
ax.set_xticklabels(SHORT, rotation=0)
ax.legend(frameon=True, framealpha=0.9, edgecolor='none')
ax.set_title('(a) Order quantities vs. mean demand')
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

# ── Panel (b): Campi–Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 10, 30) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx, kv in enumerate(k_range):
    eps_lo[idx], eps_hi[idx] = quantify_risk(kv, N, beta)

ax.fill_between(k_range, eps_lo, eps_hi, alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_range, eps_lo, '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_range, eps_hi, '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_DEMAND, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k], [eps_lo[k]], c=C_DEMAND, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi[k]], c=C_DEMAND, s=50, zorder=5, marker='o')
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k]), xytext=(k + 4, eps_hi[k] + 0.025),
            fontsize=9, color=C_DEMAND,
            arrowprops=dict(arrowstyle='->', color=C_DEMAND, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, 99\\% confidence)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Usable supply vs demand ──
ax = axes[2]
usable = yields * q  # (N, 5)
product_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#ffff33']
for i in range(len(NAMES)):
    ax.scatter(demands[:, i], usable[:, i], c=product_colors[i],
               alpha=0.35, s=12, label=SHORT[i], edgecolors='none')

max_val = max(np.max(demands), np.max(usable)) * 1.05
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.0, alpha=0.6, label='Supply = Demand')
ax.set_xlabel('Demand (cases)')
ax.set_ylabel('Usable supply (cases)')
ax.set_title('(c) Usable supply vs. demand')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8,
          ncol=2, loc='upper left', markerscale=1.5)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'inventory.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
