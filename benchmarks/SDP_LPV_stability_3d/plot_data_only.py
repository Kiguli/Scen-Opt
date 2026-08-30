#!/usr/bin/env python3
"""Generate a paper-ready figure showing only the LPV trajectories (no Lyapunov ellipse)."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp

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

benchmark_dir = os.path.dirname(os.path.abspath(__file__))

# System matrices
A0 = np.array([[0, 1], [-2, -1]])
A1 = np.array([[0, 0], [0.3, 0.1]])

def A(delta):
    return A0 + delta * A1

# Colours
cmap = plt.cm.coolwarm
delta_vals = [-0.2, 0.0, 0.3, 0.6, 1.0]
colors = [cmap(i / (len(delta_vals) - 1)) for i in range(len(delta_vals))]

fig, ax = plt.subplots(figsize=(5, 4.5))

t_span = (0, 8)
t_eval = np.linspace(0, 8, 200)
n_ic = 4  # initial conditions per delta

for i, dv in enumerate(delta_vals):
    Ai = A(dv)
    for j in range(n_ic):
        theta = 2 * np.pi * j / n_ic
        x0 = 1.5 * np.array([np.cos(theta), np.sin(theta)])
        sol = solve_ivp(lambda t, y: Ai @ y, t_span, x0, t_eval=t_eval,
                        method='RK45')
        alpha = 0.6 if j == 0 else 0.3
        label = f'$\\delta={dv}$' if j == 0 else None
        ax.plot(sol.y[0], sol.y[1], color=colors[i], alpha=alpha,
                linewidth=0.8, label=label)

ax.set_xlabel(r'$\xi_1$')
ax.set_ylabel(r'$\xi_2$')
ax.set_title('LPV system trajectories')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=7,
          ncol=2, loc='upper right')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect('equal')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'lpv_trajectories_only.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
