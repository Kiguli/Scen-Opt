import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Create two scalar optimization variables.
x = cp.Variable()
y = cp.Variable()

# Create two constraints.
constraints = [x - y >= 1,
               x - y >= 1,
               x - y >= 0]

# Form objective.
obj = cp.Minimize((x - y)**2)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()

# The optimal dual variable (Lagrange multiplier) for
# a constraint is stored in constraint.dual_value.
print("x", x.value)
print("y", y.value)
print("solution", prob.value)
print("optimal (x - y >= 1) dual variable", constraints[0].dual_value)
print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
print("optimal (x - y >= 0) dual variable", constraints[2].dual_value)
print("x - y value:", (x - y).value)

print(constraints)

active = []
for constraint in constraints:
    if constraint.dual_value > 0:
        active.append(constraint)

print(active)
prob2 = cp.Problem(obj, active)
print(prob2)
prob2.solve()
print("x", x.value)
print("y", y.value)
print("solution", prob.value)
print("optimal (x - y >= 1) dual variable", active[0].dual_value)
print("optimal (x - y >= 1) dual variable", active[1].dual_value)
print("x - y value:", (x - y).value)
