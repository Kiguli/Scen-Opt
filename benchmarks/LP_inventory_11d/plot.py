#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the inventory benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline

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
beta = 1e-6

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

# ── Panel (b): Scenario approach risk bounds ──
ax = axes[1]
k_max = min(N // 10, 30)
k_int = np.arange(0, k_max + 1)
eps_lo_int = np.zeros_like(k_int, dtype=float)
eps_hi_int = np.zeros_like(k_int, dtype=float)
for idx, kv in enumerate(k_int):
    eps_lo_int[idx], eps_hi_int[idx] = quantify_risk(int(kv), N, beta)
eps_lo_k, eps_hi_k = quantify_risk(k, N, beta)
k_smooth = np.linspace(0, k_max, 500)
spl_lo = UnivariateSpline(k_int, eps_lo_int, s=1e-4)
spl_hi = UnivariateSpline(k_int, eps_hi_int, s=1e-4)

ax.fill_between(k_smooth, spl_lo(k_smooth), spl_hi(k_smooth), alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_smooth, spl_lo(k_smooth), '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_smooth, spl_hi(k_smooth), '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_DEMAND, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k], [eps_lo_k], c=C_DEMAND, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi_k], c=C_DEMAND, s=50, zorder=5, marker='o')
# Place the k-label without crossing either blue curve.
eu = float(eps_hi_k)                       # eps_up at the operating k
el = float(eps_lo_k)                        # eps_lo at the operating k
eu_max = float(np.max(spl_hi(k_smooth)))   # top of the plotted eps_up range
if eu < 0.6 * eu_max:
    # Ample headroom: place the label ABOVE eps_up in the white space, shifted to
    # the RIGHT of the red marker line so the line does not run through the text.
    # The connector arrow still points to the operating dot at (k, eps_up(k)).
    dx = 0.035 * (k_smooth[-1] - k_smooth[0])
    x_txt = k + dx
    # eps_up rises to the right, so shifting the text right moves it toward the
    # curve. Evaluate eps_up a few k-units past the label start and keep the whole
    # text box above it with a margin, so neither the box nor the arrow touch it.
    eu_right = float(spl_hi(min(x_txt + 3.5, k_smooth[-1])))
    y_txt = max(eu + 0.35 * (eu_max - eu), eu + 0.05 * eu_max,
                eu_right + 0.06 * eu_max)
    ax.annotate(f'$k={k}$', xy=(k, eu), xytext=(x_txt, y_txt),
                fontsize=9, color=C_DEMAND, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color=C_DEMAND, lw=1.0))
else:
    # Little headroom: place the label INSIDE the shaded feasible region and
    # drop the arrow (the red vertical line already marks k).
    y_txt = 0.5 * (el + eu)
    ax.annotate(f'$k={k}$', xy=(k + 0.6, y_txt), xytext=(k + 0.6, y_txt),
                fontsize=9, color=C_DEMAND, ha='left', va='center')
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta=10^{{-6}}$)')
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
