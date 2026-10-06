#!/usr/bin/env python3
# Requires matplotlib (not in requirements.txt): pip install matplotlib
"""Generate a clean, paper-ready figure for the power dispatch benchmark."""

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
data_dir = os.path.join(benchmark_dir, 'data')

with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

scenarios = np.loadtxt(os.path.join(data_dir, 'scenarios.csv'), delimiter=',')
demand_data = np.loadtxt(os.path.join(data_dir, 'demand.csv'), delimiter=',')
demand = demand_data[:, 1]

wind_scenarios = scenarios[:, :24]
solar_scenarios = scenarios[:, 24:]

NAMES = ['Gas1', 'Gas2', 'Coal']
N = metrics['N']
k = metrics['k']
beta = 1e-6
hours = np.arange(24)

# Extract dispatch
P = np.array([metrics['dispatch'][name] for name in NAMES])  # (3, 24)
wind_mean = np.mean(wind_scenarios, axis=0)
solar_mean = np.mean(solar_scenarios, axis=0)

# Colours
C_GAS1 = '#e66101'
C_GAS2 = '#fdb863'
C_COAL = '#5e3c99'
C_WIND = '#2166ac'
C_SOLAR = '#f4a582'
C_DEMAND = '#b2182b'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Dispatch schedule ──
ax = axes[0]

# Stack: Coal (bottom), Gas1 (middle), Gas2 (top), then renewables
coal = P[2, :]
gas1 = P[0, :]
gas2 = P[1, :]

ax.fill_between(hours, 0, coal, color=C_COAL, alpha=0.8, label='Coal')
ax.fill_between(hours, coal, coal + gas1, color=C_GAS1, alpha=0.8, label='Gas 1')
ax.fill_between(hours, coal + gas1, coal + gas1 + gas2, color=C_GAS2, alpha=0.8, label='Gas 2')
thermal_top = coal + gas1 + gas2
ax.fill_between(hours, thermal_top, thermal_top + wind_mean, color=C_WIND, alpha=0.5, label='Wind (mean)')
ax.fill_between(hours, thermal_top + wind_mean, thermal_top + wind_mean + solar_mean,
                color=C_SOLAR, alpha=0.5, label='Solar (mean)')

ax.plot(hours, demand, 'k-', linewidth=2.0, label='Demand', zorder=5)

ax.set_xlabel('Hour of day')
ax.set_ylabel('Power (MW)')
ax.set_title('(a) Generation dispatch schedule')
ax.set_xlim(0, 23)
ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left', ncol=2)
ax.set_xticks([0, 4, 8, 12, 16, 20])

# ── Panel (b): Scenario approach risk bounds ──
ax = axes[1]
k_max = min(N // 3, 60)
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
# Place the "$k=N$" label so neither the text nor its connector crosses a blue
# curve: either fully ABOVE eps_up (white space) or INSIDE the feasible region.
eu = float(eps_hi_k)                       # eps_up at the operating complexity k
el = float(eps_lo_k)                       # eps_lo at the operating complexity k
eu_max = float(np.max(spl_hi(k_smooth)))   # top of the plotted eps_up range
if eu < 0.6 * eu_max:
    # Ample headroom above the operating point: label in the white space, with a
    # vertical/left-leaning arrow (x <= k) that meets the rising curve only at the dot.
    label_y = eu + max(0.35 * (eu_max - eu), 0.05)
    ax.annotate(f'$k={k}$', xy=(k, eu), xytext=(k, label_y),
                fontsize=9, color=C_DEMAND, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color=C_DEMAND, lw=1.0))
else:
    # Operating point near the top: drop the label into the tall feasible region.
    # The vertical red line already marks k, so no arrow is needed.
    label_y = 0.5 * (el + eu)
    ax.annotate(f'$k={k}$', xy=(k + 1.0, label_y),
                fontsize=9, color=C_DEMAND, ha='left', va='center')
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta=10^{{-6}}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Renewable scenario variability ──
ax = axes[2]

# Wind: percentile bands
wind_p10 = np.percentile(wind_scenarios, 10, axis=0)
wind_p25 = np.percentile(wind_scenarios, 25, axis=0)
wind_p75 = np.percentile(wind_scenarios, 75, axis=0)
wind_p90 = np.percentile(wind_scenarios, 90, axis=0)

ax.fill_between(hours, wind_p10, wind_p90, color=C_WIND, alpha=0.15)
ax.fill_between(hours, wind_p25, wind_p75, color=C_WIND, alpha=0.3)
ax.plot(hours, wind_mean, color=C_WIND, linewidth=1.8, label='Wind')

# Solar: percentile bands
solar_p10 = np.percentile(solar_scenarios, 10, axis=0)
solar_p25 = np.percentile(solar_scenarios, 25, axis=0)
solar_p75 = np.percentile(solar_scenarios, 75, axis=0)
solar_p90 = np.percentile(solar_scenarios, 90, axis=0)

ax.fill_between(hours, solar_p10, solar_p90, color=C_SOLAR, alpha=0.2)
ax.fill_between(hours, solar_p25, solar_p75, color=C_SOLAR, alpha=0.4)
ax.plot(hours, solar_mean, color='#d6604d', linewidth=1.8, label='Solar')

ax.set_xlabel('Hour of day')
ax.set_ylabel('Generation (MW)')
ax.set_title('(c) Renewable generation scenarios')
ax.set_xlim(0, 23)
ax.set_ylim(0)
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper right')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'power_dispatch.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
