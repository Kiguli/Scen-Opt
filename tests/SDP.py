import cvxpy as cp
import numpy as np

# Problem Data
np.random.seed(42)
n = 4  # Matrix size (decision variable is X ∈ ℝⁿˣⁿ)
m = 3  # Number of constraints

C = np.random.rand(n, n)  # Cost matrix
C = 0.5 * (C + C.T)  # Make it symmetric
A_matrices = [np.random.rand(n, n) for _ in range(m)]  # Constraint matrices
A_matrices = [0.5 * (A + A.T) for A in A_matrices]  # Ensure symmetry
b = np.random.rand(m, 1)  # Constraint vector

delta_A_matrices = [0.1 * np.random.rand(n, n) for _ in range(m)]  # Uncertainty
delta_A_matrices = [0.5 * (ΔA + ΔA.T) for ΔA in delta_A_matrices]
delta_b = 0.1 * np.random.rand(m, 1)  # Uncertainty bound on b

lambda_reg = 0.1  # Regularization parameter
mu_relax = 1.0    # Relaxation penalty

# Decision Variable (Symmetric PSD matrix)
X = cp.Variable((n, n), symmetric=True)
s = cp.Variable((m, 1), nonneg=True)  # Slack variables

# Robust Constraints: (A_i + ΔA_i) • X ≤ (b_i - Δb_i) + s_i
constraints = [
    cp.trace(A_matrices[i] @ X) <= (b[i] - delta_b[i] + s[i]) for i in range(m)
]

# Positive Semidefiniteness Constraint
constraints.append(X >> 0)  # X ⪰ 0 (X is PSD)

# Objective Function (SDP with Regularization & Relaxation)
objective = cp.Minimize(cp.trace(C @ X) + lambda_reg * cp.norm(X, 'fro') + mu_relax * cp.sum(s))

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print results
print("Optimal X:\n", X.value)
print("Optimal cost:", prob.value)
