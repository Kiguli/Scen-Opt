#!/usr/bin/env python3
"""Generate a clean, paper-ready figure for the prostate brachytherapy benchmark."""

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
results_dir = os.path.join(benchmark_dir, 'results')

with open(os.path.join(results_dir, 'metrics.json')) as f:
    metrics = json.load(f)

# Load anatomy
anatomy = {}
with open(os.path.join(data_dir, 'anatomy.txt'), 'r') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            try:
                anatomy[key] = float(val)
            except ValueError:
                anatomy[key] = val

# Load beta from parameters.txt
params = {}
with open(os.path.join(benchmark_dir, 'parameters.txt'), 'r') as f:
    for line in f:
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            params[key.strip()] = float(val.split('#')[0].strip())
beta = 1.0 - params.get('confidence', 0.99)

# Load dose matrix and compute doses
import pandas as pd
D_nominal = pd.read_csv(os.path.join(data_dir, 'D_nominal.csv'), header=None).values
intensities = np.array(metrics['intensities'])
doses = D_nominal @ intensities

n_tumor = int(anatomy.get('n_tumor', 20))
n_oar1 = int(anatomy.get('n_oar1', 30))
n_oar2 = int(anatomy.get('n_oar2', 30))
n_normal = int(anatomy.get('n_normal', 20))

tumor_doses = doses[:n_tumor]
rectum_doses = doses[n_tumor:n_tumor + n_oar1]
bladder_doses = doses[n_tumor + n_oar1:n_tumor + n_oar1 + n_oar2]

N = metrics['N']
k = metrics['k']
degeneracy = metrics['degeneracy']
dose_prescribed = metrics['dose_prescribed']
dose_min_tumor = metrics['dose_min_tumor']

# Colours
C_TUMOR = '#b2182b'
C_RECTUM = '#2166ac'
C_BLADDER = '#1b7837'
C_RISK_FILL = '#4393c3'
C_RISK_LINE = '#2166ac'
C_MARKER = '#b2182b'
C_BEAM = '#2166ac'

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ── Panel (a): Dose-Volume Histogram (DVH) ──
ax = axes[0]

def compute_dvh(doses_arr, n_bins=200):
    """Compute cumulative DVH: fraction of volume receiving >= dose."""
    d_max = max(np.max(doses_arr) * 1.1, dose_prescribed * 1.3)
    dose_bins = np.linspace(0, d_max, n_bins)
    volume_frac = np.array([np.mean(doses_arr >= d) for d in dose_bins])
    return dose_bins, volume_frac

d_bins_t, v_frac_t = compute_dvh(tumor_doses)
d_bins_r, v_frac_r = compute_dvh(rectum_doses)
d_bins_b, v_frac_b = compute_dvh(bladder_doses)

ax.plot(d_bins_t, v_frac_t * 100, '-', color=C_TUMOR, linewidth=2.5, label='PTV (tumor)')
ax.plot(d_bins_r, v_frac_r * 100, '-', color=C_RECTUM, linewidth=2.5, label='Rectum')
ax.plot(d_bins_b, v_frac_b * 100, '-', color=C_BLADDER, linewidth=2.5, label='Bladder')

# Prescription and constraint lines
ax.axvline(x=dose_prescribed, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)
ax.annotate(f'{dose_prescribed:.0f} Gy\n(Rx)', xy=(dose_prescribed, 50),
            fontsize=7.5, ha='left', va='center', color='grey')
ax.axvline(x=dose_min_tumor, color=C_TUMOR, linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=float(anatomy.get('dose_max_oar1', 50.0)), color=C_RECTUM,
           linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=float(anatomy.get('dose_max_oar2', 60.0)), color=C_BLADDER,
           linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Dose (Gy)')
ax.set_ylabel('Volume (%)')
ax.set_title('(a) Dose-volume histogram')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper right')
# Cap x-axis at 3x prescription for clinically meaningful DVH (brachytherapy has
# extreme hot spots near dwell positions that would otherwise compress the plot)
ax.set_xlim([0, dose_prescribed * 3])
ax.set_ylim([0, 105])

# ── Panel (b): Campi-Garatti risk bounds ──
ax = axes[1]
k_range = np.arange(0, min(N // 4, 60) + 1)
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
ax.annotate(f'$k={k}$', xy=(k, eps_hi[k_idx]), xytext=(k + 5, eps_hi[k_idx] + 0.02),
            fontsize=9, color=C_MARKER,
            arrowprops=dict(arrowstyle='->', color=C_MARKER, lw=1.0))
ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'Risk $\varepsilon$')
beta_str = f'{beta:.2g}'
ax.set_title(f'(b) Risk bounds ($N={N}$, $\\beta={beta_str}$)')
ax.legend(frameon=True, framealpha=0.9, edgecolor='none', loc='upper left')

# ── Panel (c): Dwell-position intensities ──
ax = axes[2]
n_beamlets = int(anatomy.get('n_beamlets', 50))
intensities_clipped = np.maximum(intensities, 0)

# Colour bars by intensity: high = red, low = blue
colors = plt.cm.YlOrRd(intensities_clipped / max(np.max(intensities_clipped), 1e-8))
ax.bar(range(n_beamlets), intensities_clipped, color=colors, width=0.8, edgecolor='none')
ax.set_xlabel('Dwell position')
ax.set_ylabel('Intensity (Gy$\cdot$s)')
ax.set_title('(c) Dwell-position intensities')
ax.set_xlim([-0.5, n_beamlets - 0.5])
ax.set_ylim([0, np.max(intensities_clipped) * 1.1])

# Add a colourbar matching the bar colours
sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                            norm=mpl.colors.Normalize(0, np.max(intensities_clipped)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Intensity', fontsize=9)

plt.tight_layout()
out_path = os.path.join(results_dir, 'prostate_brachytherapy_robust_planning.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".png", ".pdf")}')
plt.close()
