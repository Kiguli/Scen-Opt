import cvxpy as cp
import numpy as np

def solve_qp(deltas, A, b, c, Q, T=0, P=0, norm_type=2,solver=None):
    """
        Solves a quadratic programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        A (function): Function that returns the coefficient matrix for the constraints given a delta.
        b (function): Function that returns the right-hand side vector for the constraints given a delta.
        c (numpy.ndarray): Coefficient vector for the objective function.
        Q (numpy.ndarray): Quadratic cost matrix for the objective function.
        T (float): Regularization parameter for the norm term in the objective function.
        P (float): Penalty parameter for the slack variables in the objective function.
        norm_type (int or str, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Optimal solution vector.
            - s (numpy.ndarray): Optimal slack variables vector.
            - cost (float): Optimal value of the objective function.
        """
    # Check Q is positive semi-definite and symmetric
    print(np.linalg.eigvals(Q))
    assert np.all(np.linalg.eigvals(Q) >= 0), "Q needs to be positive semi-definite"
    assert (Q==Q.T).all(), "Q needs to be symmetric"

    n = A(deltas[0]).shape[1]  # Number of variables
    m = A(deltas[0]).shape[0]  # Number of constraints
    num_of_deltas = deltas.shape[1] # Number of deltas per row
    size_of_deltas = deltas.shape[0] # Number of rows

    # Variables
    x = cp.Variable((n, 1))
    if P != 0:
        s = cp.Variable((m, 1), nonneg=True)  # Slack variables
    else:
        s = np.zeros((m,1))

    constraints = []
    for i in range(size_of_deltas):
        constraints.append(A(deltas[i]) @ x + b(deltas[i]) <= s)  # Relaxed robust constraints

    # Objective Function
    objective = cp.Minimize(cp.quad_form(x, Q) + c.T @ x + T * cp.norm(x, norm_type) + P * cp.sum(s))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    #Simplify results
    x = x.value
    if P != 0:
        s = s.value
    cost = prob.value

    # Return results
    return x, s, cost