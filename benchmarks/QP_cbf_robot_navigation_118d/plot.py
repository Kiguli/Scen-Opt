#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the robot navigation benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
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

N = metrics['N']
k = metrics['k']
degeneracy = metrics['degeneracy']
px = np.array(metrics['trajectory_x'])
py = np.array(metrics['trajectory_y'])
vx = np.array(metrics['velocity_x'])
vy = np.array(metrics['velocity_y'])
min_clearance = metrics['min_clearance']

# Load beta from parameters.txt
params = {}
with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            params[key.strip()] = float(val.split('#')[0].strip())
beta = 1.0 - params.get('confidence', 0.999999)

# Wall configuration
wall = {
    'x_min': 3.0, 'x_max': 7.0,
    'y_min': 0.0, 'y_max': 3.0,
    'face_y': 3.0,
    'name': 'Wall',
}
R_ROBOT = 0.2
D_MARGIN = 0.1
D_SAFE = R_ROBOT + D_MARGIN
OBSTACLE_SIGMA = 0.05
T_HORIZON = 20
DT = 0.4
WORKSPACE_Y = 8.0

# Colours
C_TRAJ = '#2166ac'
C_WALL = '#b2182b'
C_SAFE = '#2d6a4f'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_MARKER = '#b2182b'

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ── Panel (a): Robot trajectory with wall obstacle ──
ax = axes[0]

w = wall['x_max'] - wall['x_min']
h_wall = wall['y_max'] - wall['y_min']

# Safety margin region (above wall face)
safety_rect = Rectangle(
    (wall['x_min'], wall['y_max']),
    w, D_SAFE,
    facecolor=C_WALL, alpha=0.12,
    edgecolor=C_WALL, linestyle='--', linewidth=0.8)
ax.add_patch(safety_rect)

# Wall body
rect = Rectangle(
    (wall['x_min'], wall['y_min']),
    w, h_wall,
    facecolor=C_WALL, alpha=0.45,
    edgecolor='black', linewidth=1.5)
ax.add_patch(rect)

# Label
cx = (wall['x_min'] + wall['x_max']) / 2
cy = (wall['y_min'] + wall['y_max']) / 2
ax.annotate(wall['name'], (cx, cy), ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

# Trajectory
ax.plot(px, py, '-', color=C_TRAJ, linewidth=2.5, zorder=5, label='Optimal path')
ax.scatter(px, py, c=C_TRAJ, s=25, zorder=6, edgecolors='white', linewidths=0.5)

# Velocity vectors
scale = 0.2
for i in range(len(px)):
    ax.arrow(px[i], py[i], vx[i] * scale, vy[i] * scale,
             head_width=0.06, head_length=0.03, fc='cyan', ec='darkblue',
             alpha=0.6, zorder=4, linewidth=0.5)

# Start and goal
ax.scatter([0.5], [2.5], c=C_SAFE, s=120, marker='o', zorder=7,
           edgecolors='black', linewidths=1.5, label='Start')
ax.scatter([9.5], [2.5], c=C_MARKER, s=150, marker='*', zorder=7,
           edgecolors='black', linewidths=0.8, label='Goal')

ax.set_xlabel('$x$ position (m)')
ax.set_ylabel('$y$ position (m)')
ax.set_title('(a) QP robot navigation')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=7.5, loc='upper left')
ax.set_xlim(-0.3, 10.5)
ax.set_ylim(-0.3, 4.5)
ax.set_aspect('equal')

# ── Panel (b): Scenario approach risk bounds ──
ax = axes[1]
k_max = min(N // 10, 40)
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
ax.axvline(x=k, color=C_MARKER, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k], [eps_lo_k], c=C_MARKER, s=50, zorder=5, marker='o')
ax.scatter([k], [eps_hi_k], c=C_MARKER, s=50, zorder=5, marker='o')
ax.annotate(f'$k={k}$', xy=(k, eps_hi_k), xytext=(k + 3, eps_hi_k + 0.01),
            fontsize=9, color=C_MARKER,
            arrowprops=dict(arrowstyle='->', color=C_MARKER, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
beta_str = f'{beta:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta={beta_str}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Wall clearance over time ──
ax = axes[2]

time_full = np.arange(T_HORIZON) * DT
clearance_over_time = []
for t in range(T_HORIZON):
    if px[t] < wall['x_min'] or px[t] > wall['x_max']:
        clearance_over_time.append(np.nan)
    else:
        clearance_over_time.append(py[t] - wall['face_y'] - D_SAFE)

clearance_arr = np.array(clearance_over_time)
# Plot non-NaN portions
valid = ~np.isnan(clearance_arr)
ax.plot(time_full[valid], clearance_arr[valid], '-o', color=C_SAFE, linewidth=2, markersize=5)
# Show NaN timesteps as grey dots at y=0
nan_mask = np.isnan(clearance_arr)
if np.any(nan_mask):
    ax.scatter(time_full[nan_mask], np.zeros(nan_mask.sum()), c='grey', s=15, alpha=0.4, zorder=3)
ax.axhline(y=0, color=C_MARKER, linewidth=1.5, linestyle='--', label='Safety boundary')
if np.any(valid):
    valid_clearance = clearance_arr.copy()
    valid_clearance[nan_mask] = 0
    ax.fill_between(time_full, valid_clearance, 0, where=(valid_clearance >= 0) & valid,
                    color=C_SAFE, alpha=0.15, label='Safe')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Wall clearance (m)')
ax.set_title('(c) Wall clearance profile')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper right')
ax.set_xlim([0, (T_HORIZON - 1) * DT])

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'cbf_navigation.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
