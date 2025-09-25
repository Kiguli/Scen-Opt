import cvxpy as cp
import numpy as np
from src.Miscellaneous import get_active


def solve_sdp1(deltas, F_d, E, c, Q, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None):
    """
        Solves a semidefinite programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        F_d (function): Function that returns the matrices for the constraints given a delta.
        E (numpy.ndarray): matrices for the hard constraints.
        c (numpy.ndarray): Coefficient vector for the objective function.
        Q (numpy.ndarray): Quadratic cost matrix for the objective function.
        tau (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Reference point for the norm term in the objective function.
        rho (float): Penalty parameter for the slack variables in the objective function.
        norm_type (int or str, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Optimal solution vector.
            - zeta (numpy.ndarray): Optimal slack variables vector.
            - cost (float): Optimal value of the objective function.
        """

    # TODO: need to change all this...
    # Check Q is positive semi-definite and symmetric
    # print(np.linalg.eigvals(Q)) #TODO: add eigenvalues to errors if not PSD

    assert np.all(np.linalg.eigvals(Q) >= 0), "Q needs to be positive semi-definite"
    assert (Q == Q.T).all(), "Q needs to be symmetric"
    n = Q.shape[1]
    m = list(F_d(deltas[0]).values())[0].shape[0]

    print(n,m)

    try:
        num_of_deltas = deltas.shape[1]  # Number of deltas per row
    except IndexError:
        num_of_deltas = 1  # TODO: check A_d and b_d don't include delta[i] where i>num_deltas
    try:
        N = deltas.shape[0]  # Number of row
    except IndexError:
        raise ValueError("The input `deltas` must have at least one row.")

    # Variables
    x = cp.Variable(n)
    if rho != 0:
        zeta = cp.Variable((m, m), nonneg=True)  # Slack variables
    else:
        zeta = np.zeros((m, m))

    print("writing F")

    constraints = []
    for i in range(N):
        F_dict = F_d(deltas[i])  # Dictionary of submatrices for this delta
        expr = None
        for k, Fk in F_dict.items():
            #assert np.all(np.linalg.eigvals(Fk) >= 0), "\\(F_j(\\delta)\\) need to be positive semi-definite and symmetric"
            assert (Fk == Fk.T).all(), "\\(F_j(\\delta)\\) need to be symmetric"
            if k == '0':
                term = Fk
            else:
                xk = x[int(k)-1]
                term = xk * Fk   # scalar-variable times numpy matrix is fine
            expr = term if expr is None else expr + term
        constraints.append(expr << zeta)

    print("writing E")

    if (E):
        expr = None
        for k, Fk in E.items():
            #assert np.all(np.linalg.eigvals(Fk) >= 0), "\\(F_j\\) need to be positive semi-definite and symmetric"
            assert (Fk == Fk.T).all(), "\\(F_j\\) needs to be symmetric"
            if k == '0':
                term = Fk
            else:
                xk = x[int(k) - 1]
                term = xk*Fk   # scalar-variable times numpy matrix is fine
            expr = term if expr is None else expr + term
        non_risk_constraints = expr << 0
        constraints.append(non_risk_constraints)  # hard constraints

    else:
        non_risk_constraints = []

    # Objective Function
    objective = cp.Minimize((1 / 2) * cp.quad_form(x, Q) + c.T @ x + tau * cp.norm(x - x_ref, norm_type) + rho * cp.sum(zeta))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"SDP did not solve to optimality. Status: {prob.status}, Objective: {prob.value}, x: {x_out}")

    if rho != 0.0:
        zeta_out = zeta.value
        # print(s)
    else:
        zeta_out = np.zeros((m, 1))

    cost_out = prob.value

    # Find the active constraints
    complexity, active, degeneracy = get_active(constraints, non_risk_constraints, prob, objective, rho, solver)

    # Return results
    return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy


def solve_sdp2(deltas, C, A_d, b_d, G, h, tau=0.0, X_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None):
    """
        Solves a semidefinite programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        A_d (dictionary of function): Function that returns the matrices for the constraints given a delta.
        G (dictionary of numpy.ndarray): matrices for the hard constraints.
        b_d (dictionary of function): Function that returns the vector for the constraints given a delta.
        h (dictionary of numpy.ndarray): vector for the hard constraints.
        C (numpy.ndarray): cost matrix for the objective function.
        tau (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Reference point for the norm term in the objective function.
        rho (float): Penalty parameter for the slack variables in the objective function.
        norm_type (int or str, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Optimal solution vector.
            - zeta (numpy.ndarray): Optimal slack variables vector.
            - cost (float): Optimal value of the objective function.
        """

    # TODO: need to change all this...
    # Check Q is positive semi-definite and symmetric
    # print(np.linalg.eigvals(Q)) #TODO: add eigenvalues to errors if not PSD
    assert np.all(np.linalg.eigvals(C) >= 0), "C needs to be positive semi-definite and symmetric"
    assert (C == C.T).all(), "C needs to be positive semi-definite and symmetric"
    n = C.shape[1]
    m = C.shape[0]

    try:
        num_of_deltas = deltas.shape[1]  # Number of deltas per row
    except IndexError:
        num_of_deltas = 1
    try:
        N = deltas.shape[0]  # Number of row
    except IndexError:
        raise ValueError("The input `deltas` must have at least one row.")

    # Variables
    X = cp.Variable((n,n), symmetric=True)
    constraints = []
    constraints.append(X >> 0)  # X must be positive semidefinite

    if rho != 0:
        zeta = cp.Variable((m, 1), nonneg=True)  # Slack variables
    else:
        zeta = np.zeros((m, 1))

    for i in range(N):
        A_dict = A_d(deltas[i])  # Dictionary of submatrices for this delta
        #TODO: assert A_d symmetric
        b_dict = b_d(deltas[i])  # Dictionary of sub-vectors for this delta
        for key in A_dict.keys():
            constraints.append(cp.trace(A_dict[key] @ X) + b_dict[key] == zeta)

    if G and h:
        non_risk_constraints = []
        for key in G.keys():
            #TODO: assert G symmetric
            non_risk_constraints += [cp.trace(G[key] @ X) + h[key] == 0]

        constraints.append(non_risk_constraints)  # hard constraints

    else:
        non_risk_constraints = []

    # Objective Function
    objective = cp.Minimize(cp.trace(C @ X) + rho * cp.sum(zeta) + tau * cp.norm(X - X_ref, norm_type))


    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    X_out = X.value

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"SDP did not solve to optimality. Status: {prob.status}, Objective: {prob.value}, X: {X_out}")

    if rho != 0.0:
        zeta_out = zeta.value
        # print(zeta)
    else:
        zeta_out = np.zeros((m, 1))

    cost_out = prob.value

    # Find the active constraints
    complexity, active, degeneracy = get_active(constraints, non_risk_constraints, prob, objective, rho, solver)

    # Return results
    return X_out, zeta_out, cost_out, N, complexity, constraints, degeneracy