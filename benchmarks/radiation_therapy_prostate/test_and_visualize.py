#!/usr/bin/env python3
"""
Test and Visualization for Prostate Brachytherapy Benchmark (TROTS Dataset)

This script tests the robust prostate brachytherapy optimization using the
scenario approach tool and creates visualizations of the results.

Data Source: TROTS (Radiotherapy Optimisation Test Set)
- Breedveld et al., "The TROTS dataset: Open radiotherapy optimization data"
- Real clinical dose-influence matrices from patient CT scans

Clinical Context:
Prostate brachytherapy delivers high-dose radiation via radioactive seeds
implanted directly into the prostate. The proximity of the urethra (inside
the prostate) and rectum/bladder (adjacent) makes precise dose optimization
critical. The scenario approach provides robust guarantees against
positioning uncertainty during seed implantation.

Usage:
    python test_and_visualize.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.QP import solve_qp
from src.Miscellaneous import load_file
from src.Risk import quantify_risk

# Colors for visualization
COLORS = {
    'prostate': '#E74C3C',
    'rectum': '#3498DB',
    'bladder': '#F39C12',
    'urethra': '#9B59B6',
    'normal': '#95A5A6'
}


def parse_expression_matrix(filepath):
    """Parse CSV with delta[i] expressions into a function."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    expr_matrix = []
    for line in lines:
        if line.strip():
            row = [cell.strip() for cell in line.split(',')]
            expr_matrix.append(row)

    def matrix_function(delta):
        result = []
        for row in expr_matrix:
            result_row = []
            for expr in row:
                try:
                    val = eval(expr, {"delta": delta, "math": __import__('math')})
                except:
                    val = float(expr)
                result_row.append(val)
            result.append(result_row)
        return np.array(result)

    return matrix_function


def parse_expression_vector(filepath):
    """Parse CSV with delta[i] expressions into a function (vector)."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    expressions = [line.strip() for line in lines if line.strip()]

    def vector_function(delta):
        result = []
        for expr in expressions:
            try:
                val = eval(expr, {"delta": delta, "math": __import__('math')})
            except:
                val = float(expr)
            result.append([val])
        return np.array(result)

    return vector_function


def load_matrix(filepath):
    """Load a numeric matrix from CSV."""
    return pd.read_csv(filepath, header=None).values


def load_vector(filepath):
    """Load a numeric vector from CSV."""
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    return np.array([[float(line.strip())] for line in lines if line.strip()])


def load_parameters(filepath):
    """Load parameters from text file."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, val = line.split(':', 1)
                try:
                    params[key.strip()] = float(val.strip())
                except ValueError:
                    params[key.strip()] = val.strip()
    return params


def load_anatomy(filepath):
    """Load anatomy structure definitions."""
    structures = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':')
                name = parts[0].strip()
                details = parts[1].strip()
                # Parse "0-99 (target, 100 voxels)"
                idx_range = details.split('(')[0].strip()
                start, end = idx_range.split('-')
                structures[name] = {
                    'start': int(start),
                    'end': int(end) + 1,  # Make end exclusive
                    'type': 'target' if 'target' in details.lower() else 'OAR'
                }
    return structures


def compute_doses(D, x):
    """Compute dose to each voxel given beamlet intensities."""
    return D @ x.flatten()


def main():
    print("=" * 70)
    print("BENCHMARK: Prostate Brachytherapy (TROTS Dataset)")
    print("Robust Radiation Therapy Under Implantation Uncertainty")
    print("=" * 70)
    print()

    # Get benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(benchmark_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data files
    print("Loading data files...")
    scenarios = load_file(os.path.join(benchmark_dir, 'scenarios.csv'))
    Q = load_matrix(os.path.join(benchmark_dir, 'Q.csv'))
    c = load_vector(os.path.join(benchmark_dir, 'c.csv'))
    G = load_matrix(os.path.join(benchmark_dir, 'G.csv'))
    h = load_vector(os.path.join(benchmark_dir, 'h.csv'))

    # Parse A_d and b_d expressions
    A_d = parse_expression_matrix(os.path.join(benchmark_dir, 'A_d.csv'))
    b_d = parse_expression_vector(os.path.join(benchmark_dir, 'b_d.csv'))

    # Load nominal dose matrix for visualization
    try:
        D_nominal = load_matrix(os.path.join(benchmark_dir, 'D_nominal.csv'))
    except Exception:
        D_nominal = None

    # Load parameters and anatomy
    params = load_parameters(os.path.join(benchmark_dir, 'parameters.txt'))
    try:
        anatomy = load_anatomy(os.path.join(benchmark_dir, 'anatomy.txt'))
    except:
        anatomy = None

    # Extract parameters
    n_beamlets = int(params.get('n_beamlets', c.shape[0]))
    n_voxels = int(params.get('n_voxels', 100))
    target_dose = params.get('target_dose', 145.0)
    D_min = params.get('D_min', 0.95 * target_dose)

    # OAR limits
    oar_limits = {}
    for key, val in params.items():
        if '_max' in key:
            oar_name = key.replace('_max', '')
            oar_limits[oar_name] = val

    rho = params.get('rho', 1.0)
    tau = params.get('tau', 0.1)
    beta = 1.0 - params.get('confidence', 0.99)

    N = len(scenarios)
    n_vars = c.shape[0]

    print(f"  Beamlets (dwell positions): {n_beamlets}")
    print(f"  Scenarios: {N}")
    print(f"  Uncertainty dimensions: 3 (implant positioning)")
    print(f"  Decision variables: {n_vars}")
    print(f"  Target dose: {target_dose} Gy")
    print(f"  Minimum target dose (D95): {D_min} Gy")
    if oar_limits:
        print(f"  OAR limits: {oar_limits}")
    print()

    # ===== SOLVE USING solve_qp =====
    print("Solving QP...")

    # Try multiple solvers in order of preference
    solvers_to_try = ['OSQP', 'SCS', 'MOSEK', 'ECOS']
    solved = False

    for solver_name in solvers_to_try:
        try:
            print(f"  Trying {solver_name}...")
            x, zeta, cost, N_out, k, constraints, degeneracy = solve_qp(
                deltas=scenarios,
                A_d=A_d,
                b_d=b_d,
                G=G,
                h=h,
                c=c,
                Q=Q,
                tau=tau,
                x_ref=np.zeros((n_vars, 1)),
                rho=rho,
                norm_type=2,
                solver=solver_name
            )
            status = f"SUCCESS ({solver_name})"
            print(f"  Solved successfully with {solver_name}")
            solved = True
            break
        except Exception as e:
            print(f"  {solver_name} failed: {str(e)[:80]}...")
            continue

    if not solved:
        print("All solvers failed!")
        return

    # Calculate risk bounds
    eps_lower, eps_upper = quantify_risk(k, N, beta)

    # Extract solution
    intensities = x.flatten()
    zeta_val = np.max(zeta) if zeta is not None else 0

    # Compute doses using nominal dose matrix
    if D_nominal is not None:
        doses = compute_doses(D_nominal, x)

        # Parse doses by structure
        structure_doses = {}
        if anatomy:
            for name, info in anatomy.items():
                structure_doses[name] = doses[info['start']:info['end']]
        else:
            # Default split (estimate from typical benchmark)
            n_target = int(n_voxels * 0.4)
            structure_doses['Prostate'] = doses[:n_target]
            structure_doses['OARs'] = doses[n_target:]
    else:
        doses = None
        structure_doses = {}

    # Print results
    print()
    print("-" * 70)
    print(f"{'OPTIMIZATION RESULTS':^70}")
    print("-" * 70)
    print(f"Status: {status}")
    print(f"Scenarios (N): {N}")
    print(f"Objective Value: {cost:.4f}")
    print(f"Max Constraint Violation (zeta): {zeta_val:.6f}")
    print(f"Complexity (k): {k} support constraints")
    if degeneracy:
        print(f"Risk Bounds (99%): [unreliable, {eps_upper:.4f}]")
    else:
        print(f"Risk Bounds (99%): [{eps_lower:.4f}, {eps_upper:.4f}]")
    print("-" * 70)

    print()
    print("BEAMLET INTENSITIES (Dwell Times):")
    print("-" * 70)
    print(f"  Min: {np.min(intensities):.2f}")
    print(f"  Max: {np.max(intensities):.2f}")
    print(f"  Mean: {np.mean(intensities):.2f}")
    print(f"  Std: {np.std(intensities):.2f}")
    print(f"  Active (>0.1): {np.sum(intensities > 0.1)}/{len(intensities)}")

    if structure_doses:
        print()
        print("DOSE STATISTICS (Nominal Position):")
        print("-" * 70)
        for name, dose_vals in structure_doses.items():
            if len(dose_vals) > 0:
                print(f"  {name}:")
                print(f"    min={np.min(dose_vals):.2f} Gy, "
                      f"mean={np.mean(dose_vals):.2f} Gy, "
                      f"max={np.max(dose_vals):.2f} Gy")
                if 'prostate' in name.lower():
                    d95 = np.percentile(dose_vals, 5)
                    print(f"    D95={d95:.2f} Gy (target: {D_min:.1f} Gy)")

    print()
    print("SCENARIO APPROACH GUARANTEES:")
    print("-" * 70)
    print(f"  With {(1-beta)*100:.0f}% confidence:")
    print(f"  Probability of constraint violation ε ∈ [{eps_lower:.4f}, {eps_upper:.4f}]")
    print(f"  This means: in >{(1-eps_upper)*100:.1f}% of implant procedures,")
    print(f"  dose constraints will be satisfied.")

    # Save results
    solution_data = {
        'intensities': intensities.tolist(),
        'objective_value': float(cost),
        'max_zeta': float(zeta_val),
        'N': int(N),
        'k': int(k),
        'eps_lower': float(eps_lower),
        'eps_upper': float(eps_upper),
        'degeneracy': bool(degeneracy),
        'rho': float(rho),
        'tau': float(tau),
        'target_dose': float(target_dose),
        'D_min': float(D_min),
    }

    for name, dose_vals in structure_doses.items():
        if len(dose_vals) > 0:
            solution_data[f'{name}_dose_min'] = float(np.min(dose_vals))
            solution_data[f'{name}_dose_mean'] = float(np.mean(dose_vals))
            solution_data[f'{name}_dose_max'] = float(np.max(dose_vals))
            if 'prostate' in name.lower():
                solution_data[f'{name}_D95'] = float(np.percentile(dose_vals, 5))

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(solution_data, f, indent=2)

    np.savetxt(os.path.join(results_dir, 'solution.csv'), x, delimiter=',')
    print(f"\nResults saved to: {results_dir}")

    # ===== VISUALIZATIONS =====
    print()
    print("Generating visualizations...")

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

    # ===== Plot 1: Dwell Time Distribution =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(intensities, bins=25, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(x=np.mean(intensities), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(intensities):.1f}')
    ax1.set_xlabel('Dwell Time (s)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Dwell Time Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== Plot 2: Dose Distribution by Structure =====
    ax2 = fig.add_subplot(gs[0, 1])
    if structure_doses:
        structure_data = []
        structure_labels = []
        structure_colors = []

        color_map = {
            'Prostate': COLORS['prostate'],
            'Rectum': COLORS['rectum'],
            'Bladder': COLORS['bladder'],
            'Urethra': COLORS['urethra']
        }

        for name, dose_vals in structure_doses.items():
            if len(dose_vals) > 0:
                structure_data.append(dose_vals)
                structure_labels.append(name)
                structure_colors.append(color_map.get(name, COLORS['normal']))

        if structure_data:
            bp2 = ax2.boxplot(structure_data, patch_artist=True, tick_labels=structure_labels)
            for patch, color in zip(bp2['boxes'], structure_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add constraint lines
            ax2.axhline(y=target_dose, color=COLORS['prostate'], linestyle='--',
                       linewidth=2, label=f'Prescription ({target_dose} Gy)')
            ax2.axhline(y=D_min, color=COLORS['prostate'], linestyle=':',
                       linewidth=2, label=f'D95 target ({D_min:.1f} Gy)')

            ax2.set_ylabel('Dose (Gy)', fontsize=11)
            ax2.set_title('Dose Distribution by Structure', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=7, loc='upper right')
            ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Dose data not available', ha='center', va='center')
        ax2.set_title('Dose Distribution', fontsize=12, fontweight='bold')

    # ===== Plot 3: Dose-Volume Histogram (DVH) =====
    ax3 = fig.add_subplot(gs[0, 2])
    if structure_doses:
        max_dose = max(np.max(d) for d in structure_doses.values() if len(d) > 0)
        dose_bins = np.linspace(0, max_dose * 1.1, 100)

        for name, dose_vals in structure_doses.items():
            if len(dose_vals) > 0:
                hist, _ = np.histogram(dose_vals, bins=dose_bins)
                cumulative = 100 * (1 - np.cumsum(hist) / len(dose_vals))
                color = {'Prostate': COLORS['prostate'], 'Rectum': COLORS['rectum'],
                        'Bladder': COLORS['bladder'], 'Urethra': COLORS['urethra']}.get(name, COLORS['normal'])
                ax3.plot(dose_bins[:-1], cumulative, color=color,
                        linewidth=2, label=name)

        ax3.axvline(x=D_min, color=COLORS['prostate'], linestyle=':',
                   alpha=0.7, label=f'D95 ({D_min:.0f} Gy)')

        ax3.set_xlabel('Dose (Gy)', fontsize=11)
        ax3.set_ylabel('Volume (%)', fontsize=11)
        ax3.set_title('Dose-Volume Histogram (DVH)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
    else:
        ax3.text(0.5, 0.5, 'DVH data not available', ha='center', va='center')
        ax3.set_title('Dose-Volume Histogram', fontsize=12, fontweight='bold')

    # ===== Plot 4: Setup Scenarios 3D =====
    ax4 = fig.add_subplot(gs[0, 3], projection='3d')
    ax4.scatter(scenarios[:, 0], scenarios[:, 1], scenarios[:, 2],
               c='steelblue', s=30, alpha=0.6)
    ax4.scatter([0], [0], [0], c='red', s=100, marker='*', label='Nominal')
    ax4.set_xlabel('Lateral (mm)', fontsize=9)
    ax4.set_ylabel('A-P (mm)', fontsize=9)
    ax4.set_zlabel('S-I (mm)', fontsize=9)
    ax4.set_title(f'Positioning Scenarios (N={N})', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)

    # ===== Plot 5: Active Dwell Positions =====
    ax5 = fig.add_subplot(gs[1, 0])
    active_threshold = 0.1
    active_mask = intensities > active_threshold
    ax5.bar(range(len(intensities)), intensities, color=['steelblue' if a else 'lightgray' for a in active_mask],
           alpha=0.7)
    ax5.axhline(y=active_threshold, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Dwell Position Index', fontsize=11)
    ax5.set_ylabel('Dwell Time (s)', fontsize=11)
    ax5.set_title(f'Dwell Times ({np.sum(active_mask)}/{len(intensities)} active)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # ===== Plot 6: Risk Bounds =====
    ax6 = fig.add_subplot(gs[1, 1])
    k_range = np.arange(1, min(N//2, 80))
    eps_lowers = []
    eps_uppers = []
    for k_val in k_range:
        el, eu = quantify_risk(k_val, N, beta)
        eps_lowers.append(el)
        eps_uppers.append(eu)

    ax6.fill_between(k_range, eps_lowers, eps_uppers, alpha=0.3, color='blue')
    ax6.plot(k_range, eps_lowers, 'b--', linewidth=1.5, label='Lower bound')
    ax6.plot(k_range, eps_uppers, 'b-', linewidth=1.5, label='Upper bound')
    ax6.axvline(x=k, color='red', linestyle='-', linewidth=2, label=f'k={k}')
    ax6.scatter([k], [eps_lower], c='red', s=100, zorder=5)
    ax6.scatter([k], [eps_upper], c='red', s=100, zorder=5)
    ax6.set_xlabel('Complexity (k)', fontsize=11)
    ax6.set_ylabel('Risk (ε)', fontsize=11)
    ax6.set_title(f'Campi-Garatti Risk Bounds\n(N={N}, 99% confidence)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ===== Plot 7: Dose Profile =====
    ax7 = fig.add_subplot(gs[1, 2])
    if doses is not None:
        ax7.plot(range(len(doses)), doses, 'b-', linewidth=1, alpha=0.7)
        ax7.axhline(y=target_dose, color='red', linestyle='--', linewidth=2,
                   label=f'Target ({target_dose} Gy)')
        ax7.axhline(y=D_min, color='orange', linestyle=':', linewidth=2,
                   label=f'D95 ({D_min:.0f} Gy)')

        # Mark structure boundaries
        if anatomy:
            for name, info in anatomy.items():
                ax7.axvline(x=info['start'], color='gray', linestyle=':', alpha=0.5)

        ax7.set_xlabel('Voxel Index', fontsize=11)
        ax7.set_ylabel('Dose (Gy)', fontsize=11)
        ax7.set_title('Dose Profile Across All Voxels', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Dose data not available', ha='center', va='center')
        ax7.set_title('Dose Profile', fontsize=12, fontweight='bold')

    # ===== Plot 8: Constraint Satisfaction by Structure =====
    ax8 = fig.add_subplot(gs[1, 3])
    if structure_doses:
        constraint_names = []
        constraint_values = []
        constraint_limits = []
        colors = []

        for name, dose_vals in structure_doses.items():
            if len(dose_vals) > 0:
                if 'prostate' in name.lower():
                    # Target: D95 must be >= D_min
                    d95 = np.percentile(dose_vals, 5)
                    constraint_names.append(f'{name}\nD95')
                    constraint_values.append(d95)
                    constraint_limits.append(D_min)
                    colors.append('green' if d95 >= D_min else 'red')
                else:
                    # OAR: max must be <= limit
                    max_dose = np.max(dose_vals)
                    limit = oar_limits.get(name, 100.0)
                    constraint_names.append(f'{name}\nmax')
                    constraint_values.append(max_dose)
                    constraint_limits.append(limit)
                    colors.append('green' if max_dose <= limit else 'orange')

        if constraint_names:
            bars = ax8.bar(constraint_names, constraint_values, color=colors, alpha=0.7)

            # Add limit markers
            for i, (val, limit) in enumerate(zip(constraint_values, constraint_limits)):
                ax8.plot([i-0.4, i+0.4], [limit, limit], 'k--', linewidth=2)

            ax8.set_ylabel('Dose (Gy)', fontsize=11)
            ax8.set_title('Constraint Satisfaction', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')
    else:
        ax8.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax8.set_title('Constraint Satisfaction', fontsize=12, fontweight='bold')

    # ===== Plot 9: Scenario Statistics =====
    ax9 = fig.add_subplot(gs[2, 0])
    scenario_stats = pd.DataFrame(scenarios, columns=['Δx', 'Δy', 'Δz'])
    ax9.hist([scenario_stats['Δx'], scenario_stats['Δy'], scenario_stats['Δz']],
            bins=15, alpha=0.7, label=['Lateral', 'A-P', 'S-I'])
    ax9.set_xlabel('Shift (mm)', fontsize=11)
    ax9.set_ylabel('Frequency', fontsize=11)
    ax9.set_title('Scenario Distribution by Direction', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')

    # ===== Plot 10: Dose Heatmap =====
    ax10 = fig.add_subplot(gs[2, 1])
    if doses is not None:
        # Reshape to 2D for visualization
        n = int(np.sqrt(len(doses)))
        if n * n == len(doses):
            dose_grid = doses.reshape(n, n)
        else:
            # Pad to square
            side = int(np.ceil(np.sqrt(len(doses))))
            dose_grid = np.zeros(side * side)
            dose_grid[:len(doses)] = doses
            dose_grid = dose_grid.reshape(side, side)

        im = ax10.imshow(dose_grid, cmap='hot', aspect='auto')
        ax10.set_xlabel('Voxel Column', fontsize=11)
        ax10.set_ylabel('Voxel Row', fontsize=11)
        ax10.set_title('Dose Distribution Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax10, shrink=0.8, label='Dose (Gy)')
    else:
        ax10.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax10.set_title('Dose Heatmap', fontsize=12, fontweight='bold')

    # ===== Plot 11: Dwell Time Map =====
    ax11 = fig.add_subplot(gs[2, 2])
    n = int(np.sqrt(len(intensities)))
    if n * n >= len(intensities):
        side = int(np.ceil(np.sqrt(len(intensities))))
        intensity_grid = np.zeros(side * side)
        intensity_grid[:len(intensities)] = intensities
        intensity_grid = intensity_grid.reshape(side, side)

        im = ax11.imshow(intensity_grid, cmap='YlOrRd', aspect='auto')
        ax11.set_xlabel('Position Column', fontsize=11)
        ax11.set_ylabel('Position Row', fontsize=11)
        ax11.set_title('Dwell Time Map', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax11, shrink=0.8, label='Time (s)')
    else:
        ax11.bar(range(len(intensities)), intensities, color='steelblue', alpha=0.7)
        ax11.set_title('Dwell Times', fontsize=12, fontweight='bold')

    # ===== Plot 12: Summary Box =====
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    # Get dose statistics
    prostate_d95 = 0
    prostate_mean = 0
    if structure_doses and 'Prostate' in structure_doses:
        prostate_d95 = np.percentile(structure_doses['Prostate'], 5)
        prostate_mean = np.mean(structure_doses['Prostate'])

    oar_max = {}
    for name in ['Rectum', 'Bladder', 'Urethra']:
        if name in structure_doses and len(structure_doses[name]) > 0:
            oar_max[name] = np.max(structure_doses[name])

    summary_text = f"""
    PROSTATE BRACHYTHERAPY
    (TROTS Dataset)
    ======================

    Problem Size:
      {n_beamlets} dwell positions
      {n_voxels} voxels
      {N} scenarios
      3D uncertainty (±5mm)

    Prescription:
      Target: {target_dose} Gy
      D95 ≥ {D_min:.1f} Gy

    Results:
      Prostate D95: {prostate_d95:.1f} Gy
      Prostate mean: {prostate_mean:.1f} Gy
"""
    for name, val in oar_max.items():
        limit = oar_limits.get(name, 100)
        summary_text += f"      {name} max: {val:.1f} Gy (≤{limit})\n"

    summary_text += f"""
    Scenario Approach:
      k = {k} support constraints
      Risk: [{eps_lower:.4f}, {eps_upper:.4f}]

      99% confidence that ≥{(1-eps_upper)*100:.1f}%
      of procedures satisfy limits.
    """

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='steelblue'))

    plt.suptitle("Prostate Brachytherapy - Robust Treatment Planning (TROTS Dataset)",
                 fontsize=14, fontweight='bold', y=0.98)
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        pass

    # Save figure
    fig_path = os.path.join(results_dir, 'visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {fig_path}")

    try:
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    except Exception:
        plt.close()

    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
