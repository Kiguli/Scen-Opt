#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the wine covariance estimation benchmark."""

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
beta = 1.0 - params.get('confidence', 0.999999)

# Short feature labels
short_names = ['Alc', 'Mal', 'Ash', 'Alk', 'Mg']

# Colours
C_EST = '#b2182b'
C_FULL = '#2166ac'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_MARKER = '#b2182b'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

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

# ── Panel (b): Scenario approach risk bounds ──
ax = axes[1]
k_max = min(N, 60)
k_int = np.arange(0, k_max + 1)
eps_lo_int = np.zeros_like(k_int, dtype=float)
eps_hi_int = np.zeros_like(k_int, dtype=float)
for idx, kv in enumerate(k_int):
    eps_lo_int[idx], eps_hi_int[idx] = quantify_risk(int(kv), N, beta)
eps_lo_k, eps_hi_k = quantify_risk(min(k, k_max), N, beta)
k_smooth = np.linspace(0, k_max, 500)
spl_lo = UnivariateSpline(k_int, eps_lo_int, s=1e-2)
spl_hi = UnivariateSpline(k_int, eps_hi_int, s=1e-2)

ax.fill_between(k_smooth, spl_lo(k_smooth), spl_hi(k_smooth), alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_smooth, spl_lo(k_smooth), '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_smooth, spl_hi(k_smooth), '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
k_plot = min(k, k_max)
ax.axvline(x=k_plot, color=C_MARKER, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k_plot], [eps_lo_k], c=C_MARKER, s=50, zorder=5, marker='o')
ax.scatter([k_plot], [eps_hi_k], c=C_MARKER, s=50, zorder=5, marker='o')

if k <= k_max:
    # Place the k-label so neither the text nor its connector crosses a blue curve.
    eu = float(eps_hi_k)                       # eps_up at the operating k
    el = float(eps_lo_k)                       # eps_lo at the operating k
    eu_max = float(np.max(spl_hi(k_smooth)))   # top of the plotted eps_up range
    if eu < 0.6 * eu_max:
        # Ample headroom: put the label just to the RIGHT of the red marker line, in the
        # white space above eps_up, so the vertical line no longer runs through the text.
        dx = 0.035 * (k_smooth[-1] - k_smooth[0])   # ~3.5% of the plotted k-range
        x_txt = k_plot + dx
        # eps_up rises to the right, so shifting the text right moves it toward the curve.
        # Keep the whole text box clear of eps_up: raise y to the larger of the nominal
        # headroom target and eps_up a few k-units right of the label, plus a margin.
        eu_right = float(spl_hi(min(x_txt + 0.06 * k_max, k_max)))
        y_txt = max(eu + 0.35 * (eu_max - eu), eu_right + 0.05 * eu_max)
        ax.annotate(f'$k={k}$', xy=(k_plot, eu),
                    xytext=(x_txt, y_txt),
                    ha='left', va='bottom',
                    fontsize=9, color=C_MARKER,
                    arrowprops=dict(arrowstyle='->', color=C_MARKER, lw=1.0))
    else:
        # Little headroom above, but the feasible band is tall: drop the label inside
        # the shaded region and drop the arrow (the red line already marks k).
        dx = 0.015 * k_max
        y_txt = 0.5 * (el + eu)
        ax.text(k_plot + dx, y_txt, f'$k={k}$',
                ha='left', va='center', fontsize=9, color=C_MARKER)
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

plt.tight_layout()
out_path = os.path.join(results_dir, 'robust_covariance_wine.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
