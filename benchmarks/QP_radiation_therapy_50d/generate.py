#!/usr/bin/env python3
# Requires mat73 (not in requirements.txt): pip install mat73
"""
Prostate Brachytherapy Benchmark Data Generator (TROTS Dataset)

This benchmark models robust radiation therapy treatment planning for prostate cancer
brachytherapy under catheter placement uncertainty. The scenario approach provides
quantifiable risk bounds for organ-at-risk (OAR) protection.

Data Source:
    TROTS (The Rotterdam Optimisation Test Set) — Prostate Brachytherapy Patient 01
    Breedveld et al., "Data for TROTS — The Radiotherapy Optimisation Test Set"
    https://doi.org/10.5281/zenodo.2541834

    To regenerate from source, download Prostate_BT.zip from the TROTS Zenodo
    repository and place Prostate_BT_01.mat in data/

Clinical Context:
    Prostate LDR brachytherapy implants radioactive seeds via catheters inserted
    through the perineum. Catheter placement uncertainty of +/-5mm arises from:
    - Needle deflection during insertion
    - Prostate deformation from needle insertion
    - Seed migration post-implant

Usage:
    python generate.py [--seed SEED]
"""

import numpy as np
import pandas as pd
import argparse
import os

# =============================================================================
# CLINICAL CONFIGURATION — Prostate LDR Brachytherapy
# =============================================================================

N_BEAMLETS = 50       # Sampled from 55 TROTS catheters/channels
N_TUMOR = 20          # Prostate (PTV) voxels, sampled from 7750
N_OAR1 = 30           # Rectum voxels, sampled from 5198
N_OAR2 = 30           # Bladder voxels, sampled from 5226
N_NORMAL = 20         # Unspecified tissue voxels, sampled from 12647
N_VOXELS = N_TUMOR + N_OAR1 + N_OAR2 + N_NORMAL  # 100

# Dose prescription (Gy) — Prostate LDR brachytherapy (I-125)
DOSE_PRESCRIBED = 145.0   # Standard prescription dose
DOSE_MIN_TUMOR = 137.75   # Minimum tumor dose (95% of prescription)
DOSE_MAX_OAR1 = 100.0     # Maximum rectum dose
DOSE_MAX_OAR2 = 120.0     # Maximum bladder dose

# Catheter placement uncertainty (mm)
SETUP_SHIFTS = [-5, -2.5, 0, 2.5, 5]  # 5-level grid
N_RANDOM_SCENARIOS = 75


def load_trots_data(data_dir):
    """
    Load dose-influence matrices from TROTS Prostate_BT_01.mat.

    Returns dict of {structure_name: D_matrix} where D[v,b] is the dose
    deposited in voxel v by unit intensity of beamlet/channel b.
    """
    import mat73

    filepath = os.path.join(data_dir, 'Prostate_BT_01.mat')
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"TROTS data file not found: {filepath}\n"
            "Download Prostate_BT.zip from https://doi.org/10.5281/zenodo.2541834\n"
            "and place Prostate_BT_01.mat in the data/ directory."
        )

    print(f"Loading {filepath}...")
    data = mat73.loadmat(filepath)

    structures = {}
    matrix_data = data['data']['matrix']

    # Map TROTS structure names to our categories
    target_names = ['Prostate']
    oar1_names = ['Rectum']        # full rectum (not partial)
    oar2_names = ['Bladder']       # full bladder
    normal_names = ['Unspecified Tissue']

    for m in matrix_data:
        if isinstance(m, dict):
            name = m.get('Name', 'Unknown')
            A = m.get('A', None)
            if A is not None and hasattr(A, 'shape') and len(A.shape) == 2:
                structures[name] = A.astype(np.float64)

    # Select the structures we need
    result = {}
    for name in target_names:
        if name in structures:
            result['target'] = structures[name]
            print(f"  Target ({name}): {structures[name].shape}")
    for name in oar1_names:
        if name in structures:
            result['oar1'] = structures[name]
            print(f"  OAR1 ({name}): {structures[name].shape}")
    for name in oar2_names:
        if name in structures:
            result['oar2'] = structures[name]
            print(f"  OAR2 ({name}): {structures[name].shape}")
    for name in normal_names:
        if name in structures:
            result['normal'] = structures[name]
            print(f"  Normal ({name}): {structures[name].shape}")

    for key in ['target', 'oar1', 'oar2', 'normal']:
        if key not in result:
            raise ValueError(f"Structure '{key}' not found in TROTS data")

    return result


def sample_voxels(D, n_target, seed=42):
    """
    Sample voxels from a large dose matrix, prioritizing high-variance voxels.

    High-variance voxels lie at dose gradients and are most sensitive to
    optimization — they represent the clinically important boundary regions.
    """
    if D.shape[0] <= n_target:
        return D, np.arange(D.shape[0])

    rng = np.random.RandomState(seed)
    variance = np.var(D, axis=1)

    # Take top 50% by variance (boundary/important voxels)
    n_important = min(n_target // 2, D.shape[0] // 4)
    important_idx = np.argsort(variance)[-n_important:]

    # Random sample the rest
    remaining_idx = np.setdiff1d(np.arange(D.shape[0]), important_idx)
    n_random = min(n_target - n_important, len(remaining_idx))
    random_idx = rng.choice(remaining_idx, n_random, replace=False)

    selected_idx = np.sort(np.concatenate([important_idx, random_idx]))
    return D[selected_idx], selected_idx


def sample_beamlets(D, n_target, seed=42):
    """
    Sample beamlets from a large dose matrix, prioritizing high-contribution beamlets.
    """
    if D.shape[1] <= n_target:
        return D, np.arange(D.shape[1])

    rng = np.random.RandomState(seed)
    total_dose = np.sum(D, axis=0)

    # Take top 70% by dose contribution
    n_important = int(0.7 * n_target)
    important_idx = np.argsort(total_dose)[-n_important:]

    remaining_idx = np.setdiff1d(np.arange(D.shape[1]), important_idx)
    n_random = n_target - n_important
    random_idx = rng.choice(remaining_idx, n_random, replace=False)

    selected_idx = np.sort(np.concatenate([important_idx, random_idx]))
    return D[:, selected_idx], selected_idx


def apply_catheter_shift(D_nominal, shift_x, shift_y, shift_z, seed=42):
    """
    Apply a catheter placement shift to the dose matrix.

    In brachytherapy, catheter displacement changes the dose distribution:
    - Dose from each seed/channel is approximately 1/r^2
    - Small shifts can cause large local dose changes near sources
    - Y-direction (anterior-posterior) is most critical: rectum is posterior,
      bladder is anterior/superior

    Perturbation model (multiplicative):
        D_shifted[v,b] = D_nominal[v,b] * voxel_factor * beamlet_factor * noise

    where factors depend on shift direction and structure location.
    """
    rng = np.random.RandomState(seed + abs(hash((shift_x, shift_y, shift_z))) % 10000)
    D_shifted = D_nominal.copy()
    shift_magnitude = np.sqrt(shift_x**2 + shift_y**2 + shift_z**2) / 5.0

    # Tumor: coverage degrades with any shift (seeds move away from target)
    for v in range(N_TUMOR):
        coverage_factor = 1 - shift_magnitude * rng.uniform(0.05, 0.25)
        D_shifted[v, :] *= coverage_factor

    # Rectum (OAR1): posterior shifts increase dose
    oar1_start = N_TUMOR
    for v in range(oar1_start, oar1_start + N_OAR1):
        posterior_effect = 1 + (shift_y / 5) * rng.uniform(0.10, 0.35)
        lateral_effect = 1 + abs(shift_x) / 10 * rng.uniform(0, 0.05)
        D_shifted[v, :] *= posterior_effect * lateral_effect

    # Bladder (OAR2): anterior shifts increase dose
    oar2_start = N_TUMOR + N_OAR1
    for v in range(oar2_start, oar2_start + N_OAR2):
        anterior_effect = 1 + (-shift_y / 5) * rng.uniform(0.08, 0.30)
        si_effect = 1 + abs(shift_z) / 10 * rng.uniform(0, 0.08)
        D_shifted[v, :] *= anterior_effect * si_effect

    return D_shifted


def generate_scenarios(seed=42):
    """Generate catheter placement uncertainty scenarios (grid + random)."""
    rng = np.random.RandomState(seed)
    scenarios = []

    # Grid scenarios: 5^3 = 125
    for dx in SETUP_SHIFTS:
        for dy in SETUP_SHIFTS:
            for dz in SETUP_SHIFTS:
                scenarios.append([dx, dy, dz])

    # Random scenarios
    for _ in range(N_RANDOM_SCENARIOS):
        dx = np.clip(rng.normal(0, 2.5), -5, 5)
        dy = np.clip(rng.normal(0, 2.5), -5, 5)
        dz = np.clip(rng.normal(0, 2.5), -5, 5)
        scenarios.append([dx, dy, dz])

    return np.array(scenarios)


def build_qp_matrices(D_nominal):
    """
    Build QP objective: minimize weighted dose deviations.

    (1/2) x' Q x + c' x  where Q = 2 D' W D, c = -2 target' W D
    """
    w_tumor = 10.0
    w_oar = 1.0
    w_normal = 0.1

    weights = np.zeros(N_VOXELS)
    weights[:N_TUMOR] = w_tumor
    weights[N_TUMOR:N_TUMOR + N_OAR1] = w_oar
    weights[N_TUMOR + N_OAR1:N_TUMOR + N_OAR1 + N_OAR2] = w_oar
    weights[N_TUMOR + N_OAR1 + N_OAR2:] = w_normal

    W = np.diag(weights)

    target = np.zeros(N_VOXELS)
    target[:N_TUMOR] = DOSE_PRESCRIBED

    Q = 2 * D_nominal.T @ W @ D_nominal
    Q = (Q + Q.T) / 2  # enforce exact symmetry

    c = -2 * target @ W @ D_nominal

    # Normalize Q to improve conditioning (real clinical data can have large range)
    scale_factor = np.max(np.diag(Q))
    if scale_factor > 1:
        Q = Q / scale_factor
        c = c / scale_factor

    # Add regularization for numerical stability
    Q += 1e-4 * np.eye(N_BEAMLETS)

    return Q, c


def build_hard_constraints(max_intensity=150.0):
    """
    Hard constraints on beamlet intensities: 0 <= x <= max_intensity.

    In form G @ x + h <= 0.
    """
    G_rows = []
    h_values = []

    # Non-negativity: -x_b <= 0
    for b in range(N_BEAMLETS):
        row = np.zeros(N_BEAMLETS)
        row[b] = -1
        G_rows.append(row)
        h_values.append(0)

    # Upper bound: x_b - max_intensity <= 0
    for b in range(N_BEAMLETS):
        row = np.zeros(N_BEAMLETS)
        row[b] = 1
        G_rows.append(row)
        h_values.append(-max_intensity)

    return np.array(G_rows), np.array(h_values)


def build_scenario_constraints(D_nominal, scenarios, seed=42):
    """
    Build affine scenario-dependent dose constraints.

    Constraints:
      OAR max dose:   D(delta)[v,:] @ x <= D_max  -->  D @ x - D_max <= 0
      Tumor min dose: D(delta)[v,:] @ x >= D_min  --> -D @ x + D_min <= 0

    Affine parametrization via finite differences:
      A(delta) = A0 + grad_x*delta[0] + grad_y*delta[1] + grad_z*delta[2]
    """
    a_d_rows = []
    b_d_rows = []

    # Compute gradients via finite differences at +/-5mm
    D_plus_x = apply_catheter_shift(D_nominal, 5, 0, 0, seed)
    D_minus_x = apply_catheter_shift(D_nominal, -5, 0, 0, seed)
    D_plus_y = apply_catheter_shift(D_nominal, 0, 5, 0, seed)
    D_minus_y = apply_catheter_shift(D_nominal, 0, -5, 0, seed)
    D_plus_z = apply_catheter_shift(D_nominal, 0, 0, 5, seed)
    D_minus_z = apply_catheter_shift(D_nominal, 0, 0, -5, seed)

    # Rectum (OAR1) max dose constraints
    oar1_start = N_TUMOR
    for v in range(oar1_start, oar1_start + N_OAR1):
        base_coeffs = D_nominal[v, :]
        grad_x = (D_plus_x[v, :] - D_minus_x[v, :]) / 10
        grad_y = (D_plus_y[v, :] - D_minus_y[v, :]) / 10
        grad_z = (D_plus_z[v, :] - D_minus_z[v, :]) / 10

        row_parts = []
        for b in range(N_BEAMLETS):
            coeff_str = f'{base_coeffs[b]:.6f}'
            if abs(grad_x[b]) > 1e-6:
                coeff_str += f' + {grad_x[b]:.6f}*delta[0]'
            if abs(grad_y[b]) > 1e-6:
                coeff_str += f' + {grad_y[b]:.6f}*delta[1]'
            if abs(grad_z[b]) > 1e-6:
                coeff_str += f' + {grad_z[b]:.6f}*delta[2]'
            row_parts.append(coeff_str)

        a_d_rows.append(', '.join(row_parts))
        b_d_rows.append(f'{-DOSE_MAX_OAR1}')

    # Bladder (OAR2) max dose constraints
    oar2_start = N_TUMOR + N_OAR1
    for v in range(oar2_start, oar2_start + N_OAR2):
        base_coeffs = D_nominal[v, :]
        grad_x = (D_plus_x[v, :] - D_minus_x[v, :]) / 10
        grad_y = (D_plus_y[v, :] - D_minus_y[v, :]) / 10
        grad_z = (D_plus_z[v, :] - D_minus_z[v, :]) / 10

        row_parts = []
        for b in range(N_BEAMLETS):
            coeff_str = f'{base_coeffs[b]:.6f}'
            if abs(grad_x[b]) > 1e-6:
                coeff_str += f' + {grad_x[b]:.6f}*delta[0]'
            if abs(grad_y[b]) > 1e-6:
                coeff_str += f' + {grad_y[b]:.6f}*delta[1]'
            if abs(grad_z[b]) > 1e-6:
                coeff_str += f' + {grad_z[b]:.6f}*delta[2]'
            row_parts.append(coeff_str)

        a_d_rows.append(', '.join(row_parts))
        b_d_rows.append(f'{-DOSE_MAX_OAR2}')

    # Tumor min dose constraints: -D @ x + D_min <= 0
    for v in range(N_TUMOR):
        base_coeffs = -D_nominal[v, :]
        grad_x = -(D_plus_x[v, :] - D_minus_x[v, :]) / 10
        grad_y = -(D_plus_y[v, :] - D_minus_y[v, :]) / 10
        grad_z = -(D_plus_z[v, :] - D_minus_z[v, :]) / 10

        row_parts = []
        for b in range(N_BEAMLETS):
            coeff_str = f'{base_coeffs[b]:.6f}'
            if abs(grad_x[b]) > 1e-6:
                coeff_str += f' + {grad_x[b]:.6f}*delta[0]'
            if abs(grad_y[b]) > 1e-6:
                coeff_str += f' + {grad_y[b]:.6f}*delta[1]'
            if abs(grad_z[b]) > 1e-6:
                coeff_str += f' + {grad_z[b]:.6f}*delta[2]'
            row_parts.append(coeff_str)

        a_d_rows.append(', '.join(row_parts))
        b_d_rows.append(f'{DOSE_MIN_TUMOR}')

    return a_d_rows, b_d_rows


def save_benchmark_files(scenarios, Q, c, G, h, a_d_rows, b_d_rows, D_nominal):
    """Save all benchmark files to the data/ directory."""
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    pd.DataFrame(scenarios).to_csv(
        os.path.join(data_dir, 'scenarios.csv'), header=False, index=False)
    print(f"Saved scenarios.csv: {len(scenarios)} scenarios")

    with open(os.path.join(data_dir, 'A_d.csv'), 'w') as f:
        for row in a_d_rows:
            f.write(row + '\n')
    print(f"Saved A_d.csv: {len(a_d_rows)} constraint rows")

    with open(os.path.join(data_dir, 'b_d.csv'), 'w') as f:
        for row in b_d_rows:
            f.write(row + '\n')
    print("Saved b_d.csv")

    pd.DataFrame(Q).to_csv(
        os.path.join(data_dir, 'Q.csv'), header=False, index=False)
    print(f"Saved Q.csv: {Q.shape[0]}x{Q.shape[1]}")

    with open(os.path.join(data_dir, 'c.csv'), 'w') as f:
        for val in c:
            f.write(f'{val}\n')
    print("Saved c.csv")

    pd.DataFrame(G).to_csv(
        os.path.join(data_dir, 'G.csv'), header=False, index=False)
    print(f"Saved G.csv: {G.shape[0]} hard constraints")

    with open(os.path.join(data_dir, 'h.csv'), 'w') as f:
        for val in h:
            f.write(f'{val}\n')
    print("Saved h.csv")

    pd.DataFrame(D_nominal).to_csv(
        os.path.join(data_dir, 'D_nominal.csv'), header=False, index=False)
    print(f"Saved D_nominal.csv: {D_nominal.shape[0]}x{D_nominal.shape[1]}")

    # Anatomy info
    with open(os.path.join(data_dir, 'anatomy.txt'), 'w') as f:
        f.write(f'n_tumor = {N_TUMOR}\n')
        f.write(f'n_oar1 = {N_OAR1}\n')
        f.write(f'n_oar2 = {N_OAR2}\n')
        f.write(f'n_normal = {N_NORMAL}\n')
        f.write(f'n_beamlets = {N_BEAMLETS}\n')
        f.write(f'dose_prescribed = {DOSE_PRESCRIBED}\n')
        f.write(f'dose_min_tumor = {DOSE_MIN_TUMOR}\n')
        f.write(f'dose_max_oar1 = {DOSE_MAX_OAR1}\n')
        f.write(f'dose_max_oar2 = {DOSE_MAX_OAR2}\n')
    print("Saved anatomy.txt")

    # Parameters
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'w') as f:
        f.write("# Prostate LDR Brachytherapy Benchmark (TROTS Dataset)\n")
        f.write("# Robust optimization under catheter placement uncertainty\n")
        f.write("#\n")
        f.write(f"# Data: TROTS Prostate_BT_01 (Breedveld et al.)\n")
        f.write(f"# Beamlets: {N_BEAMLETS} (sampled from 55 channels)\n")
        f.write(f"# Voxels: {N_VOXELS} (prostate: {N_TUMOR}, rectum: {N_OAR1}, bladder: {N_OAR2})\n")
        f.write(f"# Prescribed dose: {DOSE_PRESCRIBED} Gy\n")
        f.write(f"# Min tumor dose: {DOSE_MIN_TUMOR} Gy (95% of prescription)\n")
        f.write(f"# Max rectum dose: {DOSE_MAX_OAR1} Gy\n")
        f.write(f"# Max bladder dose: {DOSE_MAX_OAR2} Gy\n")
        f.write(f"# Catheter shifts: {SETUP_SHIFTS} mm\n")
        f.write("#\n")
        f.write("rho = 1.0\n")
        f.write("tau = 0.1\n")
        f.write("confidence = 0.999999\n")
    print("Saved parameters.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Generate prostate brachytherapy benchmark from TROTS data'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("Prostate Brachytherapy Benchmark Generator (TROTS Data)")
    print("=" * 70)
    print()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')

    # Load real TROTS dose matrices
    trots = load_trots_data(data_dir)

    # Sample beamlets (50 from 55) — applied to all structures
    print(f"\nSampling {N_BEAMLETS} beamlets from {trots['target'].shape[1]}...")
    _, beamlet_idx = sample_beamlets(trots['target'], N_BEAMLETS, seed=args.seed)

    # Sample voxels for each structure
    print(f"Sampling voxels...")
    D_target, _ = sample_voxels(trots['target'][:, beamlet_idx], N_TUMOR, seed=args.seed)
    D_oar1, _ = sample_voxels(trots['oar1'][:, beamlet_idx], N_OAR1, seed=args.seed + 1)
    D_oar2, _ = sample_voxels(trots['oar2'][:, beamlet_idx], N_OAR2, seed=args.seed + 2)
    D_normal, _ = sample_voxels(trots['normal'][:, beamlet_idx], N_NORMAL, seed=args.seed + 3)

    print(f"  Prostate (target): {D_target.shape}")
    print(f"  Rectum (OAR1): {D_oar1.shape}")
    print(f"  Bladder (OAR2): {D_oar2.shape}")
    print(f"  Normal tissue: {D_normal.shape}")

    # Combine into nominal dose matrix
    D_nominal = np.vstack([D_target, D_oar1, D_oar2, D_normal])
    print(f"\nCombined dose matrix: {D_nominal.shape}")
    print(f"  Dose range: [{D_nominal.min():.4f}, {D_nominal.max():.4f}]")

    # Generate scenarios
    print(f"\nGenerating scenarios...")
    scenarios = generate_scenarios(args.seed)
    print(f"  {len(scenarios)} scenarios (5^3 grid + {N_RANDOM_SCENARIOS} random)")

    # Build QP matrices
    print("\nBuilding QP objective matrices...")
    Q, c = build_qp_matrices(D_nominal)

    # Build hard constraints
    print("Building hard constraints (non-negativity + bounds)...")
    G, h = build_hard_constraints(max_intensity=150.0)

    # Build scenario constraints
    print("Building scenario-dependent dose constraints...")
    a_d_rows, b_d_rows = build_scenario_constraints(D_nominal, scenarios, args.seed)
    print(f"  {len(a_d_rows)} dose constraints")
    print(f"    - Rectum: {N_OAR1} voxels")
    print(f"    - Bladder: {N_OAR2} voxels")
    print(f"    - Tumor min: {N_TUMOR} voxels")

    # Save files
    print("\nSaving benchmark files...")
    save_benchmark_files(scenarios, Q, c, G, h, a_d_rows, b_d_rows, D_nominal)

    print()
    print("=" * 70)
    print("Benchmark generation complete!")
    print("=" * 70)
    print()
    print("Problem Dimensions:")
    print(f"  Decision variables: {N_BEAMLETS}")
    print(f"  Uncertainty dimensions: 3 (catheter displacement)")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Scenario constraints: {len(a_d_rows)}")
    print(f"  Hard constraints: {len(h)}")
    print()
    print("Next: Run run.py to solve, then plot.py to visualize")


if __name__ == '__main__':
    main()
