#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the CVaR portfolio benchmark."""

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

scenarios = np.loadtxt(os.path.join(data_dir, 'scenarios.csv'), delimiter=',')
returns = scenarios[:, 0:8]

TICKERS = ['SPY', 'AGG', 'VNQ', 'GLD', 'EFA', 'TLT', 'VWO', 'LQD']
N = metrics['N']
k = metrics['k']
beta = 0.01
weights = np.array([metrics['portfolio_weights'][t] for t in TICKERS])
alpha = metrics['alpha_var_threshold']

# Portfolio returns
portfolio_returns = returns @ weights

# Colours
C_BAR = '#2166ac'
C_LIMIT = '#b2182b'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_HIST = '#4393c3'
C_VAR = '#e66101'
C_CVAR = '#b2182b'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Optimal portfolio weights ──
ax = axes[0]
x_pos = np.arange(len(TICKERS))
bars = ax.bar(x_pos, weights * 100, color=C_BAR, alpha=0.85, edgecolor='white', linewidth=0.5)

# Min/max constraint lines
ax.axhline(y=5, color=C_LIMIT, linestyle='--', linewidth=1.0, alpha=0.7, label='Position limits (5–30%)')
ax.axhline(y=30, color=C_LIMIT, linestyle='--', linewidth=1.0, alpha=0.7)

ax.set_ylabel('Weight (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(TICKERS, rotation=0)
ax.set_title('(a) Optimal portfolio allocation')
ax.set_ylim(0, 35)
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper right', fontsize=8)

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
ax.axvline(x=k, color=C_CVAR, linestyle='-', linewidth=1.5, alpha=0.8)
k_idx = min(k, len(eps_lo) - 1)
ax.scatter([k], [eps_lo[k_idx]], c=C_CVAR, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi[k_idx]], c=C_CVAR, s=50, zorder=5, marker='o')
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k_idx]), xytext=(k + 4, eps_hi[k_idx] + 0.035),
            fontsize=9, color=C_CVAR,
            arrowprops=dict(arrowstyle='->', color=C_CVAR, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, 99\\% confidence)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Portfolio return distribution with VaR/CVaR ──
ax = axes[2]
ax.hist(portfolio_returns * 100, bins=40, color=C_HIST, alpha=0.7, edgecolor='white', linewidth=0.5)

# VaR line (95th percentile loss)
var_95 = -np.percentile(portfolio_returns, 5)
ax.axvline(x=-var_95 * 100, color=C_VAR, linewidth=2.0, linestyle='-',
           label=f'VaR(95%) = {var_95*100:.2f}%')

# CVaR line (average of worst 5%)
sorted_ret = np.sort(portfolio_returns)
n_tail = max(1, int(0.05 * N))
cvar_95 = -np.mean(sorted_ret[:n_tail])
ax.axvline(x=-cvar_95 * 100, color=C_CVAR, linewidth=2.0, linestyle='-',
           label=f'CVaR(95%) = {cvar_95*100:.2f}%')

# Shade the tail
tail_returns = sorted_ret[:n_tail] * 100
ax.axvspan(ax.get_xlim()[0], -var_95 * 100, alpha=0.1, color=C_CVAR)

# Mean return
mean_ret = np.mean(portfolio_returns)
ax.axvline(x=mean_ret * 100, color='#2d6a4f', linewidth=1.5, linestyle='--',
           label=f'Mean = {mean_ret*100:.3f}%')

ax.set_xlabel('Daily return (%)')
ax.set_ylabel('Count')
ax.set_title('(c) Portfolio return distribution')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'portfolio_cvar.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
