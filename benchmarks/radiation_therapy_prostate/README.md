# Prostate Brachytherapy: Robust Treatment Planning (TROTS Dataset)

## The Story

**Regional Cancer Center** treats over 800 prostate cancer patients annually with brachytherapy—a highly effective technique that delivers radiation directly from radioactive sources implanted within the prostate. Dr. Michael Torres, Chief Brachytherapist, faces a critical challenge with each procedure: positioning dozens of radioactive seeds with millimeter precision while protecting the urethra (which passes directly through the prostate) and the adjacent rectum and bladder.

Unlike external beam therapy where the patient can be repositioned between fractions, **brachytherapy implants are permanent**. Once the seeds are placed during the surgical procedure, any positioning errors become locked in for the lifetime of the treatment. Dr. Torres needs a treatment plan that is **robust to implantation uncertainty**—one that achieves therapeutic dose coverage even when seeds shift by ±5mm during the procedure.

**This benchmark uses real clinical dose-influence matrices from the TROTS dataset.**

---

## Data Source

**TROTS: The Radiotherapy Optimisation Test Set**
- Reference: Breedveld et al., "The TROTS dataset: Open radiotherapy optimization data"
- Source: Zenodo ([doi:10.5281/zenodo.XXXX](https://zenodo.org))
- Patient: Prostate_BT_01 (Prostate Brachytherapy)
- Original size: 55 dwell positions, 7750+ prostate voxels, 5000+ OAR voxels

The dataset contains real dose-influence matrices computed from patient CT scans, representing the dose each voxel receives from unit activity at each dwell position.

---

## Clinical Background

### Prostate Brachytherapy

Brachytherapy ("short-distance therapy") delivers radiation from sources placed directly inside or next to the tumor. For prostate cancer, two main techniques exist:

1. **Permanent seed implant (LDR)**: Iodine-125 or Palladium-103 seeds remain in the prostate permanently, delivering radiation over weeks to months.

2. **High-dose-rate (HDR)**: A single radioactive source is temporarily inserted through catheters, stopping at multiple "dwell positions" to deliver the prescribed dose.

This benchmark models HDR brachytherapy, where the optimization variables are the dwell times at each position.

### Treatment Planning Challenge

The treatment planner must simultaneously achieve:

1. **Target Coverage**: Deliver ≥145 Gy prescription dose to the entire prostate (D95 ≥ 137.75 Gy)
2. **Urethra Sparing**: Keep urethra dose below 160 Gy (passes through center of prostate)
3. **Rectum Sparing**: Keep posterior rectum wall below 100 Gy
4. **Bladder Sparing**: Keep anterior bladder wall below 120 Gy
5. **Dose Conformality**: Minimize dose to healthy tissue

### Implantation Uncertainty

Positioning errors during brachytherapy arise from:

| Error Type | Magnitude | Source |
|------------|-----------|--------|
| Needle deflection | 2-5 mm | Tissue resistance, pelvic bone |
| Organ deformation | 3-8 mm | Needle insertion compresses prostate |
| Seed migration | 1-3 mm | Seeds can shift after placement |
| Patient movement | 2-4 mm | During the 1-2 hour procedure |

**Clinical References:**
- Stock, R.G. et al. (2000). "A modified technique allowing interactive ultrasound-guided three-dimensional transperineal prostate implantation." Int J Radiat Oncol Biol Phys.
- Nath, R. et al. (2009). "AAPM recommendations on dose prescription and reporting methods for permanent interstitial brachytherapy for prostate cancer." Medical Physics.

---

## Problem Description

Optimize dwell times at 50 positions (sampled from 55) to deliver 145 Gy to the prostate while protecting adjacent organs under ±5mm implantation uncertainty.

### Anatomy (from TROTS patient)

| Structure | Voxels | Dose Constraint | Clinical Rationale |
|-----------|--------|-----------------|-------------------|
| Prostate | 100 | ≥137.75 Gy (D95) | Tumor control |
| Rectum | 50 | ≤100 Gy (max) | Avoid rectal bleeding |
| Bladder | 50 | ≤120 Gy (max) | Avoid urinary toxicity |
| Urethra | 50 | ≤160 Gy (max) | Avoid stricture |

### Optimization Problem

```
minimize    (1/2) x' Q x + c' x + tau ||x - x_ref||₂ + rho * max(zeta)

subject to  (for each scenario i = 1,...,200):
    D(δⁱ)[v,:] @ x ≤ D_max[v] + zeta    ∀v ∈ {rectum, bladder, urethra}
    -D(δⁱ)[v,:] @ x ≤ -D_min + zeta     ∀v ∈ prostate

hard constraints:
    x[p] ≥ 0           ∀p ∈ dwell positions (physical)
    x[p] ≤ 100         ∀p ∈ dwell positions (time limit)
```

Where:
- x = dwell times at each position (seconds)
- D(δ) = dose-influence matrix under positioning shift δ
- δ = (Δx, Δy, Δz) positioning error in mm
- zeta = maximum constraint violation (slack)

---

## Uncertainty Characterization

**3-dimensional uncertainty** δ = (Δx, Δy, Δz):

| Index | Parameter | Range | Clinical Basis |
|-------|-----------|-------|----------------|
| δ[0] | Lateral shift (x) | ±5 mm | Needle insertion angle |
| δ[1] | Anterior-posterior (y) | ±5 mm | Organ deformation |
| δ[2] | Superior-inferior (z) | ±5 mm | Catheter depth |

**200 scenarios** combining:
- 125 grid points (5³ systematic coverage)
- 75 random samples (uncertain positions)

---

## File Structure

| File | Description | Dimensions |
|------|-------------|------------|
| `test_and_visualize.py` | Solves QP, generates 12-panel visualization | - |
| `scenarios.csv` | Positioning shift scenarios | 200 × 3 |
| `A_d.csv` | Affine constraint coefficients | 250 × 50 |
| `b_d.csv` | Constraint RHS values | 250 × 1 |
| `Q.csv` | Quadratic objective matrix | 50 × 50 |
| `c.csv` | Linear objective vector | 50 × 1 |
| `G.csv` | Hard constraint matrix (bounds) | 100 × 50 |
| `h.csv` | Hard constraint RHS | 100 × 1 |
| `D_nominal.csv` | Nominal dose-influence matrix | 250 × 50 |
| `anatomy.txt` | Structure definitions | - |
| `parameters.txt` | Optimization parameters | - |

---

## Running the Benchmark

### Solve and Visualize

```bash
cd benchmarks/radiation_therapy_prostate
python test_and_visualize.py
```

Produces:
- Optimal dwell times
- Dose distributions to prostate and OARs
- Dose-Volume Histograms (DVH)
- Risk bounds from scenario approach
- 12-panel visualization saved to `results/visualization.png`

---

## Expected Results

| Metric | Expected Range | Clinical Interpretation |
|--------|----------------|------------------------|
| Complexity (k) | 20-50 | Constraint-active at boundary scenarios |
| Risk bound (ε) | 0.05-0.20 | With 99% confidence, <20% violation probability |
| Prostate D95 | 137-145 Gy | Adequate target coverage |
| Rectum D_max | 80-100 Gy | Within tolerance |
| Bladder D_max | 100-120 Gy | Within tolerance |
| Urethra D_max | 140-160 Gy | Within tolerance (inside target) |
| Active dwell positions | 30-45 | Clinically deliverable |

The scenario approach provides **finite-sample guarantees**: with high confidence, the optimized plan will satisfy dose constraints for a bounded fraction of all possible implant positionings.

---

## Visualization Guide

The 12-panel visualization includes:

1. **Dwell Time Distribution**: Histogram of optimized dwell times
2. **Dose Distribution by Structure**: Box plots of dose to each organ
3. **Dose-Volume Histogram (DVH)**: Cumulative dose curves (clinical standard)
4. **Positioning Scenarios 3D**: Scatter plot of the 200 shift combinations
5. **Active Dwell Positions**: Bar chart showing which positions are used
6. **Risk Bounds**: Campi-Garatti bounds as function of complexity k
7. **Dose Profile**: Dose across all voxels with structure boundaries
8. **Constraint Satisfaction**: Bar chart showing constraint margins
9. **Scenario Statistics**: Distribution of shifts by direction
10. **Dose Heatmap**: 2D visualization of dose distribution
11. **Dwell Time Map**: 2D visualization of dwell times
12. **Summary Statistics**: Key metrics and risk bounds

---

## Technical Details

### Data Processing

The TROTS MATLAB files contain:
- `data.matrix[i].A`: Dose-influence matrix for structure i
- `data.matrix[i].Name`: Structure name
- `solutionX`: Reference solution from original optimization

Processing steps:
1. Load MATLAB v7.3 HDF5 file using `mat73`
2. Extract dose matrices for Prostate, Rectum, Bladder, Urethra
3. Sample voxels (prioritizing boundary voxels with high dose variance)
4. Sample beamlets (prioritizing high-contribution positions)
5. Generate scenarios by perturbing dose matrices
6. Build affine constraint expressions

### Scenario Generation

Since TROTS provides nominal dose matrices, we simulate positioning uncertainty:

```python
D_perturbed[v,b] = D_nominal[v,b] * (1 + sensitivity * delta / max_delta) * noise
```

Where:
- `sensitivity` varies by voxel (higher at organ boundaries)
- `delta` is the positioning shift
- `noise` adds 2% random variation

---

## References

1. Breedveld, S. et al. "The TROTS dataset: Open radiotherapy optimization data." (Zenodo)

2. Stock, R.G. et al. (2000). "A modified technique allowing interactive ultrasound-guided three-dimensional transperineal prostate implantation." Int J Radiat Oncol Biol Phys.

3. Nath, R. et al. (2009). "AAPM recommendations on dose prescription and reporting methods for permanent interstitial brachytherapy for prostate cancer." Medical Physics.

4. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions of uncertain convex programs." SIAM J. Optimization.

5. Garatti, S. & Campi, M.C. (2024). "Risk and Complexity in Scenario Optimization." Mathematical Programming.

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust optimization with finite-sample risk guarantees for medical applications using real clinical data.*
