import cvxpy as cp
import numpy as np
from src.Miscellaneous import get_active

def solve_lp(deltas, A_d, b_d, G, h, c, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2,solver=None):
    """
        Solves a linear programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        A_d (function): Function that returns the coefficient matrix for the constraints given a delta.
        b_d (function): Function that returns the right-hand side vector for the constraints given a delta.
        G (numpy.ndarray): coefficient matrix for the hard constraints.
        h (numpy.ndarray): right-hand side vector for the hard constraints.
        c (numpy.ndarray): Coefficient vector for the objective function.
        tau (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Regularization parameter for the regularization term in the objective to stay near some point.
        rho (float): Penalty parameter for the slack variables in the objective function.
        norm_type (int, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Optimal solution vector.
            - zeta (numpy.ndarray): Optimal slack variables vector.
            - cost (float): Optimal value of the objective function.
        """
    n = A_d(deltas[0]).shape[1]  # Number of variables
    m = A_d(deltas[0]).shape[0]  # Number of constraints
    try:
        num_of_deltas = deltas.shape[1]  # Number of deltas per row
    except IndexError:
        num_of_deltas = 1 #TODO: check A_d and b_d don't include delta[i] where i>num_deltas
    try:
        N = deltas.shape[0]  # Number of row
    except IndexError:
        raise ValueError("The input `deltas` must have at least one row.")

    # Variables
    x = cp.Variable((n, 1))
    if rho != 0:
        zeta = cp.Variable((m, 1), nonneg=True)  # Adjust size based on constraints
    else:
        zeta = np.zeros((m, 1))

    constraints = []
    for i in range(N):
        constraints.append(A_d(deltas[i]) @ x + b_d(deltas[i]) <= zeta)  # Add each row separately

    if not (G.size == 0 or h.size == 0):
        m = G.shape[0]  # Number of constraints
        non_risk_constraints = G @ x + h <= 0
        constraints.append(non_risk_constraints)  # hard constraints
    else:
        non_risk_constraints = []

    # Objective Function
    objective = cp.Minimize(c.T @ x + tau * cp.norm(x-x_ref, norm_type) + rho * cp.sum(zeta))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"LP did not solve to optimality. Status: {prob.status}, Objective: {prob.value}, x: {x_out}")

    if rho != 0.0:
        zeta_out = zeta.value
        #print(zeta)
    else:
        zeta_out = np.zeros((m,1))

    cost_out = prob.value
    print(cost_out)
    # =====================================
    #SOLVE FOR ACTIVE CONSTRAINTS
    # =====================================

    # Find the active constraints
    complexity, active, degeneracy = get_active(constraints, non_risk_constraints, prob, objective, rho, solver)

    # Return results
    return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy