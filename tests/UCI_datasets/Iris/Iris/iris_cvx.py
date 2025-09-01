import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
X = df.iloc[:, :4].values
y = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values

import cvxpy as cp
import numpy as np

n, d = X.shape
K = 3
C = 1.0
alpha = 1.0
beta = 1.0

# Variables
A = [cp.Variable((d,d), PSD=True) for _ in range(K)]
b = [cp.Variable(d) for _ in range(K)]
c = [cp.Variable() for _ in range(K)]

# Slack variables: one per (i,k) with k != y_i
xi = {}
constraints = []
for i in range(n):
    Xi = np.outer(X[i], X[i])   # dxd
    yi = int(y[i])
    for k in range(K):
        if k == yi:
            continue
        xi[(i,k)] = cp.Variable(nonneg=True)
        expr = cp.trace((A[yi]-A[k]) @ Xi) + (b[yi]-b[k]) @ X[i] + (c[yi]-c[k])
        constraints.append( expr >= 1 - xi[(i,k)] )

# Objective
obj = alpha * sum(cp.trace(Ak) for Ak in A) \
      + beta * sum(cp.sum_squares(bk) for bk in b) \
      + C * sum(xi[(i,k)] for (i,k) in xi)

prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver=cp.MOSEK)   # SCS or MOSEK (if you have it)