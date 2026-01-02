# Liver SBRT: Robust Treatment Planning (TROTS Dataset)

## The Story

**Midwest Radiation Oncology** specializes in treating liver metastases with SBRT (Stereotactic Body Radiation Therapy). Dr. James Park, Lead Physicist, faces a unique challenge: the liver moves 10-15mm with each breath, making precise tumor targeting extremely difficult. Unlike static tumors, liver lesions follow complex 3D trajectories during the breathing cycle.

During the 3-5 fraction SBRT course, respiratory motion causes significant uncertainty in where the tumor will be at any moment. Dr. Park needs a treatment plan that is **robust to respiratory motion**—one that delivers ablative doses (50+ Gy) to the tumor while protecting the spinal cord, kidneys, and gastrointestinal organs across all breathing phases.

**This benchmark uses real clinical dose-influence matrices from the TROTS dataset.**

---

## Data Source

**TROTS: The Radiotherapy Optimisation Test Set**
- Reference: Breedveld et al., "The TROTS dataset: Open radiotherapy optimization data"
- Source: Zenodo
- Patient: Liver_01 (Liver SBRT)
- Original size: 1,118 beamlets, 5,364 PTV voxels, multiple OAR structures

---

## Clinical Background

### Liver SBRT

SBRT delivers high radiation doses (typically 10-18 Gy per fraction) over 3-5 treatments. For liver tumors, this achieves local control rates of 80-95%, comparable to surgical resection but without the surgical risks.

Key challenges:
1. **Respiratory motion**: Liver moves 10-25mm during breathing
2. **Liver tolerance**: Must preserve >700cc of uninvolved liver
3. **Adjacent OARs**: Stomach, duodenum, kidneys, and spinal cord
4. **High dose conformality**: Ablative doses must be precisely targeted

### Treatment Planning Challenge

The treatment planner must simultaneously achieve:

1. **Target Coverage**: Deliver ≥47.5 Gy (95% of 50 Gy) to the PTV
2. **Spinal Cord Protection**: Keep max dose below 25 Gy
3. **Kidney Sparing**: Keep mean kidney dose below 18 Gy
4. **Liver Preservation**: Limit mean uninvolved liver dose to 30 Gy
5. **Motion Robustness**: Account for ±8mm respiratory motion

### Motion Uncertainty in Liver SBRT

Respiratory motion characteristics:

| Motion Type | Magnitude | Source |
|------------|-----------|--------|
| Diaphragmatic (S-I) | 10-25 mm | Primary breathing motion |
| Anterior-Posterior | 5-12 mm | Rib cage expansion |
| Lateral | 3-8 mm | Asymmetric breathing |
| Baseline drift | 2-5 mm | Over treatment course |

**Clinical References:**
- Seppenwoolde, Y. et al. (2002). "Precise and real-time measurement of 3D tumor motion in lung due to breathing and heartbeat, measured during radiotherapy." Int J Radiat Oncol Biol Phys.
- Eccles, C.L. et al. (2016). "Respiratory motion in liver SBRT." Medical Physics.

---

## Problem Description

Optimize 50 beamlet intensities (sampled from 1,118) to deliver 50 Gy to a liver tumor while protecting adjacent organs under ±8mm respiratory motion uncertainty.

### Anatomy (from TROTS patient)

| Structure | Voxels | Dose Constraint | Clinical Rationale |
|-----------|--------|-----------------|-------------------|
| PTV | 80 | ≥47.5 Gy (D95) | Ablative tumor control |
| Spinal Cord | 1 (mean) | ≤25 Gy (max) | Avoid myelopathy |
| Kidney (R) | 1 (mean) | ≤18 Gy (mean) | Preserve renal function |
| Kidney (L) | 1 (mean) | ≤18 Gy (mean) | Preserve renal function |

### Optimization Problem

```
minimize    (1/2) x' Q x + c' x + tau ||x - x_ref||₂ + rho * max(zeta)

subject to  (for each scenario i = 1,...,150):
    D(δⁱ)[v,:] @ x ≤ D_max[v] + zeta    ∀v ∈ OARs
    -D(δⁱ)[v,:] @ x ≤ -D_min + zeta     ∀v ∈ PTV

hard constraints:
    x[b] ≥ 0           ∀b ∈ beamlets (physical)
    x[b] ≤ 100         ∀b ∈ beamlets (linac limit)
```

---

## Uncertainty Characterization

**3-dimensional uncertainty** δ = (Δx, Δy, Δz) modeling respiratory motion:

| Index | Parameter | Range | Clinical Basis |
|-------|-----------|-------|----------------|
| δ[0] | Lateral (x) | ±8 mm | Asymmetric breathing |
| δ[1] | Anterior-posterior (y) | ±8 mm | Rib cage expansion |
| δ[2] | Superior-inferior (z) | ±8 mm | Diaphragmatic motion |

**150 scenarios** combining:
- 125 grid points (5³ systematic coverage)
- 25 random samples (interpolated positions)

The larger shift range (±8mm vs ±5mm for prostate) reflects the significant respiratory motion in liver treatments.

---

## File Structure

| File | Description | Dimensions |
|------|-------------|------------|
| `test_and_visualize.py` | Solves QP, generates visualization | - |
| `scenarios.csv` | Respiratory motion scenarios | 150 × 3 |
| `A_d.csv` | Affine constraint coefficients | N × 50 |
| `b_d.csv` | Constraint RHS values | N × 1 |
| `Q.csv` | Quadratic objective matrix | 50 × 50 |
| `c.csv` | Linear objective vector | 50 × 1 |
| `G.csv` | Hard constraint matrix | 100 × 50 |
| `h.csv` | Hard constraint RHS | 100 × 1 |
| `D_nominal.csv` | Nominal dose-influence matrix | - |
| `anatomy.txt` | Structure definitions | - |
| `parameters.txt` | Optimization parameters | - |

---

## Running the Benchmark

```bash
cd benchmarks/radiation_therapy_liver
python test_and_visualize.py
```

---

## Expected Results

| Metric | Expected Range | Clinical Interpretation |
|--------|----------------|------------------------|
| Complexity (k) | 15-40 | OAR constraints active at motion extremes |
| Risk bound (ε) | 0.08-0.25 | <25% violation probability at 99% confidence |
| PTV D95 | 45-50 Gy | Adequate tumor coverage |
| Spinal Cord max | 20-25 Gy | Within tolerance |
| Kidney mean | 10-18 Gy | Within tolerance |

---

## References

1. Breedveld, S. et al. "The TROTS dataset: Open radiotherapy optimization data."

2. Seppenwoolde, Y. et al. (2002). "Precise and real-time measurement of 3D tumor motion." Int J Radiat Oncol Biol Phys.

3. Eccles, C.L. et al. (2016). "Respiratory motion in liver SBRT." Medical Physics.

4. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions."

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust optimization for radiation therapy under respiratory motion uncertainty.*
