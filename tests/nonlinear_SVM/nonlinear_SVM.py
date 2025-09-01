# svm_cvxpy_example.py
# Requires: pip install numpy cvxpy matplotlib
import numpy as np
import cvxpy as cp

def rbf_kernel(X, Y, gamma):
    # X: nxd, Y: mxd
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    dists = XX + YY - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * dists)

# --- synthetic concentric circles dataset (same layout as original example)
X_pos = np.array([[0.0, 0.6], [0.5, 0.0], [-0.5, 0.0]])
X_neg = np.array([[1.5, 0.0], [-1.5, 0.0], [0.0, -1.5]])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

n = X.shape[0]
C = 1.0
gamma = 1.0

# Build kernel and P matrix
K = rbf_kernel(X, X, gamma)
P = np.outer(y, y) * K

# numerical jitter to keep P numerically PSD
jitter = 1e-9
P = (P + P.T) / 2.0 + jitter * np.eye(n)

# CVXPY variable
alpha = cp.Variable(n)

# objective: minimize 0.5 * alpha^T P alpha - 1^T alpha
objective = cp.Minimize(0.5 * cp.quad_form(alpha, P) - cp.sum(alpha))

# constraints: 0 <= alpha <= C, sum(alpha * y) == 0
constraints = [alpha >= 0, alpha <= C, y @ alpha == 0]

prob = cp.Problem(objective, constraints)

# Solve (default solver is fine; can pass solver=cp.OSQP or cp.SCS if you prefer)
prob.solve(verbose=False)  # set verbose=True for solver logs

# Extract solution
alpha_val = alpha.value
if alpha_val is None:
    raise RuntimeError("CVXPY failed to find a solution. Try a different solver or add jitter.")

# Numerical cleanup: set tiny alphas to zero
tol = 1e-6
alpha_val[np.abs(alpha_val) < tol] = 0.0

# support vectors indices
sv_idx = np.where(alpha_val > 0)[0]
print("Support vector indices:", sv_idx)
print("Alphas (nonzero):", alpha_val[sv_idx])

# Compute bias b using KKT on 0 < alpha < C
sv_margin = np.where((alpha_val > tol) & (alpha_val < C - tol))[0]
if len(sv_margin) > 0:
    b_vals = []
    for i in sv_margin:
        s = np.sum(alpha_val * y * K[:, i])
        b_vals.append(y[i] - s)
    b = np.mean(b_vals)
else:
    # fallback: use any support vector (typical when no alpha strictly in (0,C))
    if len(sv_idx) == 0:
        raise RuntimeError("No support vectors found - something is wrong.")
    i = sv_idx[0]
    b = y[i] - np.sum(alpha_val * y * K[:, i])

print("Bias b:", b)

# Prediction function
def predict(x_new):
    k = rbf_kernel(X, x_new.reshape(1, -1), gamma).ravel()
    val = np.sum(alpha_val * y * k) + b
    return np.sign(val), val

# Test a point
pt = np.array([0.0, 1.0])
print("Prediction for", pt, ":", predict(pt))

# Optional: plot decision boundary and data (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    xx = np.linspace(-2, 2, 200)
    yy = np.linspace(-2, 2, 200)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    # compute decision values on grid
    K_grid = rbf_kernel(grid, X, gamma)  # m x n
    vals = (K_grid @ (alpha_val * y)) + b
    ZZ = vals.reshape(XX.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(XX, YY, ZZ, levels=50, alpha=0.8)
    plt.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=2)  # decision boundary
    plt.scatter(X_pos[:,0], X_pos[:,1], c='white', edgecolor='k', label='+1')
    plt.scatter(X_neg[:,0], X_neg[:,1], c='black', edgecolor='k', label='-1')
    # highlight support vectors
    plt.scatter(X[sv_idx,0], X[sv_idx,1], s=150, facecolors='none', edgecolors='yellow', linewidths=2, label='SV')
    plt.legend()
    plt.title('SVM (dual via CVXPY) decision surface')
    plt.show()
except Exception as e:
    print("Plotting skipped (matplotlib not available or error):", e)