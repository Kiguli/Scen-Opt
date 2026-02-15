#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the LPV Stability 3D benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp

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

benchmark_dir = os.path.dirname(os.path.abspath(__file__))

# Load results
with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

scenarios = np.loadtxt(os.path.join(benchmark_dir, 'data', 'scenarios.csv'))
x_sol = np.array(metrics['x'])
N = metrics['N']
k = metrics['k']
beta = 0.01

# Reconstruct P
P = np.array([[x_sol[0], x_sol[1]],
              [x_sol[1], x_sol[2]]])

# System matrices
A0 = np.array([[0, 1], [-2, -1]])
A1 = np.array([[0, 0], [0.3, 0.1]])

def A(delta):
    return A0 + delta * A1

# Colours
C_STABLE = '#2166ac'
C_UNSTABLE = '#b2182b'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_SAMPLES = '#2166ac'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

# ── Panel (a): Stabilising trajectories under Lyapunov certificate ──
ax = axes[0]

# Simulate trajectories for several delta values and initial conditions
delta_vals = [-0.2, 0.0, 0.3, 0.6, 1.0]
cmap = plt.cm.coolwarm
colors = [cmap(i / (len(delta_vals) - 1)) for i in range(len(delta_vals))]
t_span = (0, 8)
t_eval = np.linspace(0, 8, 200)

np.random.seed(42)
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

# Draw Lyapunov ellipse V(x) = x'Px = c for a level set
eigvals, eigvecs = np.linalg.eigh(P)
theta_ellipse = np.linspace(0, 2 * np.pi, 200)
# Level set at V = 1
ellipse = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ np.array(
    [np.cos(theta_ellipse), np.sin(theta_ellipse)])
ax.plot(ellipse[0], ellipse[1], 'k--', linewidth=1.5, alpha=0.6,
        label='$V(x)=1$')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('(a) Stabilising trajectories')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=7,
          ncol=2, loc='upper right')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect('equal')

# ── Panel (b): Risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 5, 20) + 1)
eps_lo = np.zeros_like(k_range, dtype=float)
eps_hi = np.zeros_like(k_range, dtype=float)
for idx, kv in enumerate(k_range):
    eps_lo[idx], eps_hi[idx] = quantify_risk(kv, N, beta)

ax.fill_between(k_range, eps_lo, eps_hi, alpha=0.25, color=C_RISK_FILL,
                label='Feasible region')
ax.plot(k_range, eps_lo, '--', color=C_RISK_LINE, linewidth=1.2,
        label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_range, eps_hi, '-', color=C_RISK_LINE, linewidth=1.2,
        label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_UNSTABLE, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k], [eps_lo[k]], c=C_UNSTABLE, s=50, zorder=5)
ax.scatter([k], [eps_hi[k]], c=C_UNSTABLE, s=50, zorder=5)
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k]),
            xytext=(k + 3, eps_hi[k] + 0.08),
            fontsize=9, color=C_UNSTABLE,
            arrowprops=dict(arrowstyle='->', color=C_UNSTABLE, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, 99\\% confidence)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'lpv_stability.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
