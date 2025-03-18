import cvxpy as cp
import numpy as np

# Problem Data
np.random.seed(42)
n = 5  # Number of variables
m = 3  # Number of constraints

A = np.random.rand(m, n)  # Constraint matrix
b = np.random.rand(m, 1)  # Constraint vector
c = np.random.rand(n, 1)  # Cost vector

delta_A = 0.1 * np.random.rand(m, n)  # Uncertainty bound on A
delta_b = 0.1 * np.random.rand(m, 1)  # Uncertainty bound on b

lambda_reg = 0.1  # Regularization parameter
mu_relax = 1.0    # Relaxation penalty

# Variables
x = cp.Variable((n, 1))
s = cp.Variable((m, 1), nonneg=True)  # Slack variables

# Robust Constraints (worst-case approach)
A_robust = A + delta_A
b_robust = b - delta_b

constraints = [
    A_robust @ x <= b_robust + s,  # Relaxed robust constraints
    x >= 0  # Non-negativity
]

# Use constraints.append() for scenario constraints

# Objective Function
objective = cp.Minimize(c.T @ x + lambda_reg * cp.norm(x, 2) + mu_relax * cp.sum(s))

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print results
print("Optimal x:\n", x.value)
print("Optimal cost:", prob.value)
