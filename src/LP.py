import cvxpy as cp
import numpy as np

def solve_lp(A, b, c, T, P, norm_type=2):
    """
    Solves a linear programming problem with optional robust and regularization constraints.

    Parameters:
    A (numpy.ndarray): Coefficient matrix for the constraints.
    b (numpy.ndarray): Right-hand side vector for the constraints.
    c (numpy.ndarray): Coefficient vector for the objective function.
    T (float): Regularization parameter for the norm term in the objective function.
    P (float): Penalty parameter for the slack variables in the objective function.
    norm_type (int, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).

    Returns:
    tuple: A tuple containing:
        - x (numpy.ndarray): Optimal solution vector.
        - s (numpy.ndarray): Optimal slack variables vector.
        - cost (float): Optimal value of the objective function.
    """
    n = A.shape[1]  # Number of variables
    m = A.shape[0]  # Number of constraints

    # Variables
    x = cp.Variable((n, 1))
    if P != 0:
        s = cp.Variable((m, 1), nonneg=True)  # Slack variables
    else:
        s = np.zeros((m,1))

    constraints = [
        A @ x + b <= s,  # Relaxed robust constraints
    ]

    # Objective Function
    objective = cp.Minimize(c.T @ x + T * cp.norm(x, norm_type) + P * cp.sum(s))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    #Simplify results
    x = x.value
    if P != 0:
        s = s.value
    cost = prob.value

    # Return results
    return x, s, cost

# Example usage
A = np.array([[2,1],[3,2],[-1,0],[0,-1]])
b = np.array([[-100],[-120],[0],[0]])
c = np.array([-5,-3])
T = 0.0
P = 0.0
norm_type = 2

optimal_x, optimal_s, optimal_cost = solve_lp(A, b, c, T, P, norm_type)
print("Optimal x:\n", optimal_x)
print("Optimal s:\n", optimal_s)
print("Optimal cost:", optimal_cost)