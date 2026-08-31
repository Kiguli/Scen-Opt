#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the LPV Stability 3D benchmark."""

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
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

benchmark_dir = os.path.dirname(os.path.abspath(__file__))

# Load results
with open(os.path.join(benchmark_dir, 'results', 'metrics.json')) as f:
    metrics = json.load(f)

scenarios = np.loadtxt(os.path.join(benchmark_dir, 'data', 'scenarios.csv'))
x_sol = np.array(metrics['x'])
N = metrics['N']
k = metrics['k']
beta = 1e-6

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
k_max = min(N // 5, 20)
k_int = np.arange(0, k_max + 1)
eps_lo_int = np.zeros_like(k_int, dtype=float)
eps_hi_int = np.zeros_like(k_int, dtype=float)
for idx, kv in enumerate(k_int):
    eps_lo_int[idx], eps_hi_int[idx] = quantify_risk(int(kv), N, beta)
eps_lo_k, eps_hi_k = quantify_risk(k, N, beta)
k_smooth = np.linspace(0, k_max, 500)
spl_lo = UnivariateSpline(k_int, eps_lo_int, s=1e-4)
spl_hi = UnivariateSpline(k_int, eps_hi_int, s=1e-4)

ax.fill_between(k_smooth, spl_lo(k_smooth), spl_hi(k_smooth), alpha=0.25, color=C_RISK_FILL,
                label='Feasible region')
ax.plot(k_smooth, spl_lo(k_smooth), '--', color=C_RISK_LINE, linewidth=1.2,
        label=r'$\epsilon_{\mathrm{lo}}(k)$')
ax.plot(k_smooth, spl_hi(k_smooth), '-', color=C_RISK_LINE, linewidth=1.2,
        label=r'$\epsilon_{\mathrm{up}}(k)$')
ax.axvline(x=k, color=C_UNSTABLE, linestyle='-', linewidth=1.5, alpha=0.8)
ax.scatter([k], [eps_lo_k], c=C_UNSTABLE, s=50, zorder=5)
ax.scatter([k], [eps_hi_k], c=C_UNSTABLE, s=50, zorder=5)

# Place the k-annotation so that neither the text box nor its connector
# crosses either blue curve. eps_up rises to the right, so we anchor the
# label at x=k (never to the right) and lift it straight up into the white
# space when there is headroom; otherwise we drop it into the feasible band.
eu = float(eps_hi_k)                       # eps_up at operating k
el = float(eps_lo_k)                       # eps_lo at operating k
eu_max = float(np.max(spl_hi(k_smooth)))   # top of plotted eps_up range
headroom = eu_max - eu
if eu < 0.6 * eu_max:
    # Ample headroom: park the label ABOVE the eps_up curve and slightly to
    # the RIGHT of the red marker line so the line no longer runs through
    # the text. The arrow still points back down to the operating dot.
    k_width = float(k_smooth[-1] - k_smooth[0])
    dx = 0.035 * k_width                     # small horizontal offset
    label_x = k + dx
    # eps_up rises to the right, so evaluate the curve out past the right
    # edge of the text box and lift the label clear of it with a margin.
    eu_right = float(spl_hi(min(label_x + 0.08 * k_width, k_smooth[-1])))
    label_y = max(eu + max(0.35 * headroom, 0.08 * eu_max),
                  eu_right + 0.06 * eu_max)
    ax.annotate(f'$k={k}$', xy=(k, eu),
                xytext=(label_x, label_y),
                fontsize=9, color=C_UNSTABLE, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color=C_UNSTABLE, lw=1.0))
else:
    # Operating point near the top: put the label INSIDE the shaded band
    # (between eps_lo and eps_up) and drop the arrow, since the red line
    # already marks k.
    label_y = 0.5 * (el + eu)
    ax.annotate(f'$k={k}$', xy=(k + 0.4, label_y),
                fontsize=9, color=C_UNSTABLE, ha='left', va='center')
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta=10^{{-6}}$)')
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
