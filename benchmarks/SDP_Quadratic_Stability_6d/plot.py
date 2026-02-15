#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the Quadratic Stability 6D benchmark."""

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

scenarios = np.loadtxt(os.path.join(benchmark_dir, 'data', 'scenarios.csv'),
                       delimiter=',')
x_sol = np.array(metrics['x'])
N = metrics['N']
k = metrics['k']
beta = 0.01

# Reconstruct P (3x3 symmetric)
P = np.array([[x_sol[0], x_sol[1], x_sol[2]],
              [x_sol[1], x_sol[3], x_sol[4]],
              [x_sol[2], x_sol[4], x_sol[5]]])

# System matrices
A0 = np.array([[-0.3, 1.0, 0.0],
               [-2.0, -1.0, 0.2],
               [0.2, 0.0, -1.2]])
A1 = np.array([[0.0, 0.0, 0.0],
               [0.5, 0.15, 0.0],
               [0.0, 0.0, 0.3]])
A2 = np.array([[0.0, 0.0, 0.0],
               [0.0, 0.3, 0.15],
               [0.15, 0.0, 0.45]])

def A(d0, d1):
    return A0 + d0 * A1 + d1 * A2

# Colours
C_TRAJ = '#2166ac'
C_UNSTABLE = '#b2182b'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# ── Panel (a): Stabilising trajectories ──
ax = axes[0]

# Pick several parameter combos at corners and center
delta_combos = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
combo_labels = ['(-1,-1)', '(-1,+1)', '(+1,-1)', '(+1,+1)', '(0,0)']
cmap = plt.cm.coolwarm
colors = [cmap(i / (len(delta_combos) - 1)) for i in range(len(delta_combos))]

t_span = (0, 6)
t_eval = np.linspace(0, 6, 300)

np.random.seed(42)
for i, (d0, d1) in enumerate(delta_combos):
    Ai = A(d0, d1)
    # Two initial conditions per combo
    for j, x0 in enumerate([np.array([1.0, 0.5, -0.5]),
                             np.array([-0.5, 1.0, 0.3])]):
        sol = solve_ivp(lambda t, y: Ai @ y, t_span, x0, t_eval=t_eval,
                        method='RK45')
        # Plot norm of state vs time
        norms = np.linalg.norm(sol.y, axis=0)
        alpha = 0.8 if j == 0 else 0.4
        label = f'$\\delta={combo_labels[i]}$' if j == 0 else None
        ax.plot(sol.t, norms, color=colors[i], alpha=alpha,
                linewidth=0.9, label=label)

ax.set_xlabel('Time $t$')
ax.set_ylabel('$\\|x(t)\\|$')
ax.set_title('(a) Stabilising trajectories')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=7,
          loc='upper right')
ax.set_ylim(bottom=0)

# ── Panel (b): Risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 10, 30) + 1)
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

if k > 0:
    ax.axvline(x=k, color=C_UNSTABLE, linestyle='-', linewidth=1.5, alpha=0.8)
    ax.scatter([k], [eps_lo[k]], c=C_UNSTABLE, s=50, zorder=5)
    ax.scatter([k], [eps_hi[k]], c=C_UNSTABLE, s=50, zorder=5)
    ax.annotate(f'$k={k}$', xy=(k, eps_hi[k]),
                xytext=(k + 3, eps_hi[k] + 0.06),
                fontsize=9, color=C_UNSTABLE,
                arrowprops=dict(arrowstyle='->', color=C_UNSTABLE, lw=1.0))
else:
    # k=0: mark on y-axis
    ax.scatter([0], [eps_lo[0]], c=C_UNSTABLE, s=50, zorder=5)
    ax.scatter([0], [eps_hi[0]], c=C_UNSTABLE, s=50, zorder=5)
    ax.annotate(f'$k=0$', xy=(0, eps_hi[0]),
                xytext=(3, eps_hi[0] + 0.005),
                fontsize=9, color=C_UNSTABLE,
                arrowprops=dict(arrowstyle='->', color=C_UNSTABLE, lw=1.0))

ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, 99\\% confidence)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Scenario samples coloured by max eigenvalue ──
ax = axes[2]

# Compute max eigenvalue at each scenario sample
max_eigs = np.zeros(len(scenarios))
for i in range(len(scenarios)):
    d0, d1 = scenarios[i, 0], scenarios[i, 1]
    Ai = A(d0, d1)
    lyap = Ai.T @ P + P @ Ai
    max_eigs[i] = np.max(np.linalg.eigvalsh(lyap))

sc = ax.scatter(scenarios[:, 0], scenarios[:, 1], c=max_eigs, s=15,
                cmap='RdYlBu_r', edgecolors='none', alpha=0.7)
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label(r'$\lambda_{\max}(A^\top P + PA)$', fontsize=9)

# Mark the boundary of parameter space
rect = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='k',
                      linewidth=0.8, linestyle='--', alpha=0.5)
ax.add_patch(rect)

ax.set_xlabel(r'$\delta_1$')
ax.set_ylabel(r'$\delta_2$')
ax.set_title('(c) Stability margin over parameter space')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')

plt.tight_layout()
results_dir = os.path.join(benchmark_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, 'quadratic_stability.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
