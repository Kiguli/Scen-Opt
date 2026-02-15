#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the growth bound benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

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

# Load scenarios (use full dataset for visualization)
full_path = os.path.join(data_dir, 'growth_bound.csv')
scenarios = np.loadtxt(full_path, delimiter=',')

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

dx_current = scenarios[:, :3]
dx_next = scenarios[:, 3:]

# Vehicle dynamics for trajectory
def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def vehicle_dynamics(x, u):
    alpha = np.arctan(np.tan(u[1]) / 2.0)
    dx = np.zeros(3)
    dx[0] = u[0] * np.cos(alpha + x[2]) / np.cos(alpha)
    dx[1] = u[0] * np.sin(alpha + x[2]) / np.cos(alpha)
    dx[2] = u[0] * np.tan(u[1])
    return dx

center = np.array([0, 1.2, 0])
width = 1.6
dt = 0.03
u = np.array([0.3, 0.3])
x_center_next = rk4_step(vehicle_dynamics, center, u, dt)

# Colours
C_INITIAL = '#2166ac'
C_REACHED = '#b2182b'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_MARKER = '#b2182b'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Reachable set (X-Y plane) ──
ax = axes[0]

# Initial set
ax.add_patch(Rectangle((center[0] - width/2, center[1] - width/2),
                         width, width, fill=True, facecolor=C_INITIAL, alpha=0.15,
                         edgecolor=C_INITIAL, linewidth=2, label='Initial set'))

# Reached set (approximate from data)
reached_x_min = x_center_next[0] - np.max(dx_next[:, 0])
reached_x_max = x_center_next[0] + np.max(dx_next[:, 0])
reached_y_min = x_center_next[1] - np.max(dx_next[:, 1])
reached_y_max = x_center_next[1] + np.max(dx_next[:, 1])

ax.add_patch(Rectangle((reached_x_min, reached_y_min),
                         reached_x_max - reached_x_min,
                         reached_y_max - reached_y_min,
                         fill=True, facecolor=C_REACHED, alpha=0.15,
                         edgecolor=C_REACHED, linewidth=2, linestyle='--',
                         label='Reached set'))

# Centers and transition arrow
ax.plot(center[0], center[1], 'o', color=C_INITIAL, markersize=8, zorder=5)
ax.plot(x_center_next[0], x_center_next[1], '^', color=C_REACHED, markersize=8, zorder=5)
ax.annotate('', xy=(x_center_next[0], x_center_next[1]),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle='->', color='#2d6a4f', lw=2))

ax.set_xlabel('$x$ position')
ax.set_ylabel('$y$ position')
ax.set_title('(a) Reachable set (X–Y plane)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left')
ax.axis('equal')

# ── Panel (b): Campi–Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 50, 40) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx, kv in enumerate(k_range):
    eps_lo[idx], eps_hi[idx] = quantify_risk(kv, N, beta)

ax.fill_between(k_range, eps_lo, eps_hi, alpha=0.25, color=C_RISK_FILL, label='Feasible region')
ax.plot(k_range, eps_lo, '--', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_range, eps_hi, '-', color=C_RISK_LINE, linewidth=1.2, label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_MARKER, linestyle='-', linewidth=1.5, alpha=0.8)
k_idx = min(k, len(eps_lo) - 1)
ax.scatter([k], [eps_lo[k_idx]], c=C_MARKER, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi[k_idx]], c=C_MARKER, s=50, zorder=5, marker='o')
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k_idx]), xytext=(k + 3, eps_hi[k_idx] + 0.003),
            fontsize=9, color=C_MARKER,
            arrowprops=dict(arrowstyle='->', color=C_MARKER, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta=10^{{-6}}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Growth ratio distribution ──
ax = axes[2]
eps_denom = 1e-10
growth_ratios = []
labels = ['$x$', '$y$', r'$\theta$']
for i in range(3):
    ratio = dx_next[:, i] / (dx_current[:, i] + eps_denom)
    ratio = ratio[ratio < 5]  # filter extreme outliers
    growth_ratios.append(ratio)

bp = ax.boxplot(growth_ratios, patch_artist=True, labels=labels, widths=0.5)
box_colors = ['#4393c3', '#74add1', '#abd9e9']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('#b2182b')
    median.set_linewidth(1.5)

ax.axhline(y=1.0, color=C_MARKER, linestyle='--', linewidth=1.0, alpha=0.7, label='Growth = 1')
ax.set_ylabel('Growth ratio (next / current)')
ax.set_xlabel('State dimension')
ax.set_title('(c) State growth distribution')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper right')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'growth_bound.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
