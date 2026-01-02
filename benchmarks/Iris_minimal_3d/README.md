# Iris Classification with SVM (3D QP)

## The Problem

This benchmark applies the **scenario approach** to **Support Vector Machine (SVM) classification** using the famous Iris dataset. The goal is to find an optimal linear classifier that separates Iris-setosa from other Iris species, with probabilistic guarantees on generalization.

The scenario approach provides a novel perspective on machine learning: each training data point becomes a "scenario constraint" that the classifier must satisfy, and the Campi-Garatti theory gives finite-sample bounds on out-of-sample error.

---

## Dataset: The Iris Dataset

The Iris dataset is one of the most famous datasets in pattern recognition, introduced by R.A. Fisher in 1936.

| Class | Count | Description |
|-------|-------|-------------|
| Iris-setosa | 50 | Small, distinct petals |
| Iris-versicolor | 50 | Medium petals |
| Iris-virginica | 50 | Large petals |

**Features used**:
- Petal length (cm)
- Petal width (cm)

These two features provide excellent separation, especially for Iris-setosa which has notably smaller petals.

---

## Mathematical Background

### Support Vector Machine (SVM)

The SVM finds a hyperplane that maximally separates two classes:

```
Decision boundary: w' x + b = 0
Classification: sign(w' x + b)
```

Where:
- w = [w_1, w_2] is the normal vector
- b is the bias term
- The margin is 2/||w||

### Optimization Problem

```
minimize    (1/2) ||w||^2    (maximize margin)

subject to  y_i * (w' x_i + b) >= 1    for all i = 1, ..., N
```

Where y_i in {-1, +1} is the class label.

### Decision Variables

```
x = [w_1, w_2, b]
```

- w_1: coefficient for petal length
- w_2: coefficient for petal width
- b: bias (intercept)

---

## Problem Formulation

This is a **Quadratic Program (QP)**:

```
minimize    (1/2) x' Q x + c' x

subject to  -y_i * (w' x_i + b) + 1 <= 0    for all i
```

In scenario approach form:
- Each data point (x_i, y_i) is a "scenario"
- The constraint A_d(delta) @ x + b_d(delta) <= 0 encodes margin satisfaction
- delta = [x_i[0], x_i[1], y_i] contains the feature values and label

---

## Problem Dimensions

| Dimension | Value |
|-----------|-------|
| Decision variables | 3 (w_1, w_2, b) |
| Scenarios (N) | 150 (all Iris samples) |
| Class distribution | 50 setosa (-1), 100 others (+1) |
| Features | 2 (petal length, petal width) |

---

## Files

| File | Description |
|------|-------------|
| `iris_minimal.csv` | 150 samples: [petal_length, petal_width, label] |
| `Q.csv` | Quadratic objective matrix (3x3, margin regularization) |
| `c.csv` | Linear objective vector (3x1) |
| `A_d.csv` | Scenario-dependent constraint (single row with delta terms) |
| `test_and_visualize.py` | Solve SVM and create 6-panel visualization |

---

## Usage

```bash
cd benchmarks/Iris_minimal_3d
python test_and_visualize.py
```

Results are saved to `results/`:
- `metrics.json`: Hyperplane parameters, margin, accuracy, risk bounds
- `solution.csv`: Optimal decision variables [w_1, w_2, b]
- `visualization.png`: 6-panel classification analysis

---

## Results with MOSEK

Running `test_and_visualize.py` with MOSEK produces the following results:

```
============================================================
BENCHMARK: Iris_minimal_3d (QP)
SVM Classification: Iris-setosa vs Others
============================================================

Status: SUCCESS (MOSEK)
Scenarios (N): 150
Decision Variables: 3
Optimal Cost: 0.851073
Complexity (k): 2 support constraints
Risk Bounds (99%): [0.0000, 0.0777]
------------------------------------------------------------

Solution (Hyperplane):
  w1 (petal_length coef): -0.593946
  w2 (petal_width coef):  -1.044706
  b (bias):                3.173958
  ||w||:                   1.201777
  Margin (2/||w||):        1.664197
  Decision boundary: -0.5939*x1 + -1.0447*x2 + 3.1740 = 0
  Training accuracy: 100.0%
```

### Interpretation

**Perfect Training Accuracy (100%)**:
- The SVM perfectly separates Iris-setosa from other species
- This confirms that petal measurements alone are sufficient for this classification
- Iris-setosa has distinctly smaller petals than versicolor and virginica

**Complexity (k = 2)**:
- Only 2 data points (support vectors) determine the decision boundary
- These are the closest points from each class to the separating hyperplane
- The extremely low complexity indicates a wide-margin separation

**Risk Bounds [0.0000, 0.0777]**:
- With 99% confidence, at most 7.77% of new samples might be misclassified
- The lower bound of 0 suggests the classifier may generalize perfectly
- For 150 training points, this is a strong generalization guarantee

**Margin = 1.304**:
- The geometric margin between classes is 1.304 units
- A larger margin generally indicates better generalization
- This wide margin explains the low complexity and tight risk bounds

**Hyperplane Interpretation**:
- Negative coefficients mean larger petals push toward class +1 (not setosa)
- Iris-setosa has small petals (both length and width)
- The decision rule: predict setosa if 0.594*length + 1.045*width < 3.174

**Scenario Approach Perspective**:
- Each of the 150 Iris samples is a "scenario constraint"
- The SVM solution satisfies all constraints (100% accuracy)
- Only k=2 constraints are active (support vectors)
- Risk bounds quantify out-of-sample generalization

---

## Visualization Guide

The 6-panel visualization includes:

1. **Classification Plot**: Data points with decision boundary and margins
2. **Margin Concept**: Geometric visualization of the SVM margin
3. **Feature Distributions**: Box plots of petal measurements by class
4. **Risk Bounds**: Scenario approach generalization bounds
5. **Decision Regions**: Color-coded classification regions
6. **Summary Statistics**: Key SVM metrics and risk interpretation

---

## Comparison with Classical ML

| Aspect | Classical SVM | Scenario Approach SVM |
|--------|---------------|----------------------|
| Optimization | Same QP formulation | Same QP formulation |
| Generalization | VC dimension bounds (loose) | Finite-sample bounds (tight) |
| Complexity | Not directly used | k = support vectors |
| Risk guarantee | Asymptotic | Non-asymptotic (exact for convex) |

The scenario approach provides **tighter, non-asymptotic bounds** on generalization error compared to classical VC theory, making it particularly valuable for safety-critical applications.

---

## References

1. Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems." Annals of Eugenics.

2. Cortes, C. & Vapnik, V. (1995). "Support-vector networks." Machine Learning.

3. Campi, M.C. & Garatti, S. (2008). "The exact feasibility of randomized solutions of uncertain convex programs." SIAM J. Optimization.

4. Calafiore, G.C. (2010). "Random convex programs." SIAM J. Optimization.

---

*This benchmark is part of the Scenario Approach Tool, demonstrating data-driven classification with finite-sample generalization guarantees.*
