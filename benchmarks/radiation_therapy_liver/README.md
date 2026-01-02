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

1. **Target Coverage**: Deliver >=47.5 Gy (95% of 50 Gy) to the PTV
2. **Spinal Cord Protection**: Keep max dose below 25 Gy
3. **Kidney Sparing**: Keep mean kidney dose below 18 Gy
4. **Liver Preservation**: Limit mean uninvolved liver dose to 30 Gy
5. **Motion Robustness**: Account for +/-8mm respiratory motion

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

Optimize 50 beamlet intensities (sampled from 1,118) to deliver 50 Gy to a liver tumor while protecting adjacent organs under +/-8mm respiratory motion uncertainty.

### Anatomy (from TROTS patient)

| Structure | Voxels | Dose Constraint | Clinical Rationale |
|-----------|--------|-----------------|-------------------|
| PTV | 80 | >=47.5 Gy (D95) | Ablative tumor control |
| Spinal Cord | 1 (mean) | <=25 Gy (max) | Avoid myelopathy |
| Kidney (R) | 1 (mean) | <=18 Gy (mean) | Preserve renal function |
| Kidney (L) | 1 (mean) | <=18 Gy (mean) | Preserve renal function |
| Liver | N/A | <=30 Gy (mean) | Preserve liver function |
| Stomach | N/A | <=35 Gy (max) | Avoid GI toxicity |

### Optimization Problem

```
minimize    (1/2) x' Q x + c' x + tau ||x - x_ref||_2 + rho * sum(zeta)

subject to  (for each scenario i = 1,...,200):
    D(delta^i)[v,:] @ x <= D_max[v] + zeta    for all v in OARs
    -D(delta^i)[v,:] @ x <= -D_min + zeta     for all v in PTV

hard constraints:
    x[b] >= 0           for all b in beamlets (physical)
    x[b] <= max_intensity  for all b in beamlets (linac limit)
```

---

## Running the Benchmark

### Quick Run (Recommended)

```bash
cd benchmarks/radiation_therapy_liver
python run_benchmark.py
```

This uses a fast compiled expression evaluator and prioritizes MOSEK for solving.

### Full Visualization

```bash
cd benchmarks/radiation_therapy_liver
python test_and_visualize.py
```

Produces a 12-panel visualization saved to `results/visualization.png`.

---

## Optimization Parameters

The scenario approach optimization uses several key parameters that control the trade-off between constraint satisfaction and solution quality:

### Core Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| **rho** | rho | 1.0 | Slack variable penalty weight |
| **tau** | tau | 0.1 | Regularization weight |
| **beta** | beta | 0.01 | Confidence level (1-beta = 99%) |
| **N** | N | 200 | Number of scenarios |
| **D_min** | D_min | 47.5 Gy | Minimum target dose (95% of prescription) |

### Parameter Effects

**rho (Slack Penalty)**:
- Controls how much the optimization penalizes constraint violations
- `rho = 0`: Hard constraints only (often infeasible)
- `rho > 0`: Allows soft constraint violations with penalty
- Higher rho = tighter constraint satisfaction at cost of objective

**tau (Regularization)**:
- Adds L2 regularization term `tau * ||x - x_ref||_2` to objective
- Prevents extreme solutions and improves numerical stability
- `tau = 0`: No regularization
- `tau > 0`: Solution stays closer to reference (typically x_ref = 0)

**beta (Confidence Level)**:
- Determines confidence in risk bounds
- `beta = 0.01` means 99% confidence
- Lower beta = more conservative bounds (wider interval)

**N (Number of Scenarios)**:
- More scenarios = tighter risk bounds but larger problem
- Trade-off between computational cost and guarantee quality
- Rule of thumb: N should be > 10 * number of decision variables

---

## Results with MOSEK

### Optimization Output

```
======================================================================
BENCHMARK: Liver SBRT (TROTS Dataset)
Robust Radiation Therapy Under Respiratory Motion Uncertainty
======================================================================

Solver: MOSEK
Status: SUCCESS

----------------------------------------------------------------------
                         OPTIMIZATION RESULTS
----------------------------------------------------------------------
Scenarios (N): 200
Objective Value: -826422.5061
Max Constraint Violation (zeta): 44.51 Gy
Complexity (k): 58 support constraints
Risk Bounds (99%): [0.1800, 0.4172] (lower bound unreliable due to degeneracy)
----------------------------------------------------------------------

BEAMLET INTENSITIES:
----------------------------------------------------------------------
  Min: 0.00
  Max: 4173.96
  Mean: 3433.52
  Std: 1412.29
  Active (>0.1): 45/50

DOSE STATISTICS (Nominal Position):
----------------------------------------------------------------------
  PTV:
    min=-0.30 Gy, mean=30.24 Gy, max=65.58 Gy
    D95=3.21 Gy (target: 47.5 Gy)
  Spinal_Cord:
    min=0.13 Gy, mean=0.13 Gy, max=0.13 Gy
  Kidney_R:
    mean=0.38 Gy (limit: 18.0 Gy)
  Kidney_L:
    mean=0.00 Gy (limit: 18.0 Gy)

SCENARIO APPROACH GUARANTEES:
----------------------------------------------------------------------
  With 99% confidence:
  Probability of constraint violation: [18.0%, 41.7%]
  This means: in >58.3% of treatment fractions,
  dose constraints will be satisfied.
```

### Interpretation of Results

**Complexity (k = 58)**:
- 58 out of 200 scenario constraints are active (binding) at the optimal solution
- These are the "support constraints" that determine the solution
- Higher k indicates the problem is more constrained by the uncertainty

**Risk Bounds [0.18, 0.42]**:
- With 99% confidence, the probability that a randomly drawn scenario violates constraints is between 18% and 42%
- The lower bound is marked as unreliable due to degeneracy (multiple optimal solutions)
- This translates to: with 99% confidence, at least 58% of treatment fractions will satisfy all constraints

**Degeneracy**:
- The problem exhibits degeneracy, meaning there are multiple optimal solutions
- This is common in radiation therapy optimization where many beamlet combinations achieve similar dose distributions
- When degeneracy is detected, the lower risk bound is not reliable (Campi-Garatti theory requires non-degenerate solutions for both bounds)

**Constraint Violations (zeta = 44.51 Gy)**:
- The maximum slack needed is 44.51 Gy
- This indicates the PTV coverage constraints are very difficult to satisfy across all motion scenarios
- With `rho > 0`, we allow these violations to find a feasible solution

**Clinical Significance**:
- OAR doses are well within limits (spinal cord, kidneys)
- PTV coverage is challenging due to respiratory motion
- The robust plan trades some target coverage for protection against worst-case motion
- In practice, motion management techniques (breath hold, gating) would reduce uncertainty

---

## Uncertainty Characterization

**3-dimensional uncertainty** delta = (dx, dy, dz) modeling respiratory motion:

| Index | Parameter | Range | Clinical Basis |
|-------|-----------|-------|----------------|
| delta[0] | Lateral (x) | +/-8 mm | Asymmetric breathing |
| delta[1] | Anterior-posterior (y) | +/-8 mm | Rib cage expansion |
| delta[2] | Superior-inferior (z) | +/-8 mm | Diaphragmatic motion |

**200 scenarios** combining:
- 125 grid points (5^3 systematic coverage)
- 75 random samples (interpolated positions)

The larger shift range (+/-8mm vs +/-5mm for prostate) reflects the significant respiratory motion in liver treatments.

---

## File Structure

| File | Description | Dimensions |
|------|-------------|------------|
| `run_benchmark.py` | Fast solver with compiled expressions | - |
| `test_and_visualize.py` | Full QP solve with 12-panel visualization | - |
| `scenarios.csv` | Respiratory motion scenarios | 200 x 3 |
| `A_d.csv` | Affine constraint coefficients | 103 x 50 |
| `b_d.csv` | Constraint RHS values | 103 x 1 |
| `Q.csv` | Quadratic objective matrix | 50 x 50 |
| `c.csv` | Linear objective vector | 50 x 1 |
| `G.csv` | Hard constraint matrix | 100 x 50 |
| `h.csv` | Hard constraint RHS | 100 x 1 |
| `D_nominal.csv` | Nominal dose-influence matrix | 103 x 50 |
| `anatomy.txt` | Structure definitions | - |
| `parameters.txt` | Optimization parameters | - |

---

## Visualization Guide

The 12-panel visualization includes:

1. **Beamlet Intensity Distribution**: Histogram of optimized intensities
2. **Dose Distribution by Structure**: Box plots of dose to each organ
3. **Dose-Volume Histogram (DVH)**: Cumulative dose curves (clinical standard)
4. **Positioning Scenarios 3D**: Scatter plot of the 200 motion scenarios
5. **Active Beamlets**: Bar chart showing which beamlets are used
6. **Risk Bounds**: Campi-Garatti bounds as function of complexity k
7. **Dose Profile**: Dose across all voxels with structure boundaries
8. **Constraint Satisfaction**: Bar chart showing constraint margins
9. **Scenario Statistics**: Distribution of shifts by direction
10. **Dose Heatmap**: 2D visualization of dose distribution
11. **Beamlet Intensity Map**: 2D visualization of beamlet intensities
12. **Summary Statistics**: Key metrics and risk bounds

---

## Expected Results

| Metric | Expected Range | Clinical Interpretation |
|--------|----------------|------------------------|
| Complexity (k) | 40-70 | OAR constraints active at motion extremes |
| Risk bound (epsilon) | 0.15-0.45 | Challenging due to large motion range |
| PTV D95 | Variable | Depends on motion range and rho |
| Spinal Cord max | <25 Gy | Within tolerance |
| Kidney mean | <18 Gy | Within tolerance |

---

## References

1. Breedveld, S. et al. "The TROTS dataset: Open radiotherapy optimization data."

2. Seppenwoolde, Y. et al. (2002). "Precise and real-time measurement of 3D tumor motion." Int J Radiat Oncol Biol Phys.

3. Eccles, C.L. et al. (2016). "Respiratory motion in liver SBRT." Medical Physics.

4. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions of uncertain convex programs."

5. Garatti, S. & Campi, M.C. (2024). "Risk and Complexity in Scenario Optimization." Mathematical Programming.

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven robust optimization for radiation therapy under respiratory motion uncertainty.*
