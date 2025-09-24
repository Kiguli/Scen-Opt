import cvxpy as cp
import numpy as np


def solve_sdp1(deltas, F_d, F, c, Q, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None):
    """
        Solves a semidefinite programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        F_d (function): Function that returns the matrices for the constraints given a delta.
        F (numpy.ndarray): matrices for the hard constraints.
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
            - s (numpy.ndarray): Optimal slack variables vector.
            - s_h (numpy.ndarray): Optimal slack variables vector for hard constraints.
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
        size_of_deltas = deltas.shape[0]  # Number of row
    except IndexError:
        raise ValueError("The input `deltas` must have at least one row.")

    print(size_of_deltas)
    # Variables
    x = cp.Variable(n)
    if rho != 0:
        zeta = cp.Variable((m, m), nonneg=True)  # Slack variables
    else:
        zeta = np.zeros((m, m))

    print("writing F")

    constraints = []
    for i in range(size_of_deltas):
        F_dict = F_d(deltas[i])  # Dictionary of submatrices for this delta
        expr = None
        for k, Fk in F_dict.items():
            #assert np.all(np.linalg.eigvals(Fk) >= 0), "\\(F_j(\\delta)\\) need to be positive semi-definite and symmetric"
            assert (Fk == Fk.T).all(), "\\(F_j(\\delta)\\) need to be positive semi-definite and symmetric"
            if k == '0':
                term = Fk
            else:
                xk = x[int(k)-1]
                term = xk * Fk   # scalar-variable times numpy matrix is fine
            expr = term if expr is None else expr + term
        constraints.append(expr << zeta)

    print("writing E")

    if (F):
        expr = None
        for k, Fk in F.items():
            #assert np.all(np.linalg.eigvals(Fk) >= 0), "\\(F_j\\) need to be positive semi-definite and symmetric"
            assert (Fk == Fk.T).all(), "\\(F_j\\) needs to be positive semi-definite and symmetric"
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
    active, degeneracy = get_active_SDP(constraints, non_risk_constraints, prob, objective, rho, solver)

    # Return results
    return x_out, zeta_out, cost_out, size_of_deltas, active, constraints, degeneracy


def solve_sdp2(deltas, C, A_da, A_a, b_da, b_a, tau=0.0, X_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None):
    """
        Solves a semidefinite programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        A_da (dictionary of function): Function that returns the matrices for the constraints given a delta.
        A_a (dictionary of numpy.ndarray): matrices for the hard constraints.
        b_da (dictionary of function): Function that returns the vector for the constraints given a delta.
        b_a (dictionary of numpy.ndarray): vector for the hard constraints.
        C (numpy.ndarray): cost matrix for the objective function.
        tau (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Reference point for the norm term in the objective function.
        rho (float): Penalty parameter for the slack variables in the objective function.
        norm_type (int or str, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Optimal solution vector.
            - s (numpy.ndarray): Optimal slack variables vector.
            - s_h (numpy.ndarray): Optimal slack variables vector for hard constraints.
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
        size_of_deltas = deltas.shape[0]  # Number of row
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

    for i in range(size_of_deltas):
        A_dict = A_da(deltas[i])  # Dictionary of submatrices for this delta
        b_dict = b_da(deltas[i])  # Dictionary of sub-vectors for this delta
        for key in A_dict.keys():
            constraints.append(cp.trace(A_dict[key] @ X) + b_dict[key] == zeta)

    if A_a and b_a:
        non_risk_constraints = []
        for key in A_a.keys():
            non_risk_constraints += [cp.trace(A_a[key] @ X) + b_a[key] == 0]

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
    active, degeneracy = get_active_SDP(constraints, non_risk_constraints, prob, objective, rho, solver)

    #TODO: get_active_SDP2 ???

    # Return results
    return X_out, zeta_out, cost_out, size_of_deltas, active, constraints, degeneracy


def test_active_SDP(prob, objective, active, rho=0.0, solver=None):
    """
        Solves a linear programming problem with optional robust and regularization constraints.

        Parameters:
        prob: CVXPY problem instance for optimal solution.
        objective: CVXPY objective function.
        active: list of active constraints.
        rho (float): Penalty parameter for the slack variables in the objective function.
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        bool: True if all assertions pass, False if any assertion fails.
    """
    try:
        prob2 = cp.Problem(objective, active)
        prob2.solve(solver=solver)

        # Assert that the objective values are the same
        assert np.isclose(prob.value, prob2.value), f"Objective values differ: {prob.value} vs {prob2.value}"

        # Assert that the solutions are the same
        assert np.allclose(prob.variables()[0].value,
                           prob2.variables()[
                               0].value), f"Solutions for x differ: {prob.variables()[0].value} vs {prob2.variables()[0].value}"

        if rho != 0:
            assert np.allclose(prob.variables()[1].value, prob2.variables()[
                1].value), f"Solutions for s differ: {prob.variables()[1].value} vs {prob2.variables()[1].value}"

        # Return True if all assertions pass
        return True

    except AssertionError as e:
        print(f"Assertion failed: {e}")
        # Return False if any assertion fails
        return False


def get_active_SDP(constraints, non_risk_constraints, prob, objective, rho=0.0, solver=None, threshold=1e-8):
    """
        Finds the active constraints of the SDP.

        Parameters:
        constraints (list): list of constraints.
        non_risk_constraints (list): list of constraints that should not be included in the error calculation.
        prob: CVXPY problem instance for optimal solution.
        objective: CVXPY objective function.
        rho (float,optional): Penalty parameter for the slack variables in the objective function. Default is 0.0.
        solver (str, optional): The solver to use for the optimization problem. Default is None.
        threshold (float, optional): Threshold for the solver checking active constraints. Default is 1e-8.

        Returns:
        active (list): list of active constraints.
    """

    # TODO: make threshold a global parameter?
    degeneracy = False
    active = []
    for constraint in constraints:
        #TODO: print constraint duals for 'normal' problem and see what they are
        if constraint.dual_value.any() > threshold:  # Check if any dual value is positive
            # print(constraint.dual_value)
            active.append(constraint)

    # If solution changes then likely to have degeneracy
    if not test_active_SDP(prob, objective, active, rho=rho, solver=solver):
        degeneracy = True
        print("Active constraints are not valid. Lower bound not viable likely due to degeneracy.")
        # loop through all constraints and make a support list from them
        active = constraints
        # Iteratively remove constraints from active if test_active_SDP returns True when they are removed
        changed = True
        while changed:
            changed = False
            for constraint in active[:]:
                temp_active = [c for c in active if c != constraint]
                if test_active_SDP(prob, objective, temp_active, rho=rho, solver=solver):
                    active.remove(constraint)
                    changed = True
                    break  # Restart loop since active has changed
        # At the end, active contains only constraints whose removal makes test_active_SDP return False
        if not test_active_SDP(prob, objective, active, rho=rho, solver=solver):
            raise ValueError("Error calculating support list! Degeneracy present as active constraints != constraints.")
    else:
        # If solution does not change then likely to be non-degenerate, check for true support list as solvers can be incorrect
        drop = []
        for a in active:
            if test_active_SDP(prob, objective, [constraint for constraint in active if constraint != a], rho=rho,
                               solver=solver):
                drop.append(a)
        if not test_active_SDP(prob, objective, [constraint for constraint in active if constraint not in drop], rho=rho,
                               solver=solver):
            degeneracy = True
            print(
                "Reduced version of active constraints are not valid. Lower bound not viable likely due to degeneracy.")
            # Iteratively remove constraints from active if test_active_SDP returns True when they are removed
            changed = True
            while changed:
                changed = False
                for constraint in active[:]:
                    temp_active = [c for c in active if c != constraint]
                    if test_active_SDP(prob, objective, temp_active, rho=rho, solver=solver):
                        active.remove(constraint)
                        changed = True
                        break  # Restart loop since active has changed
            # At the end, active contains only constraints whose removal makes test_active_SDP return False
            if not test_active_SDP(prob, objective, active, rho=rho, solver=solver):
                raise ValueError(
                    "Error calculating support list! Degeneracy present as reduced active constraints != active constraints.")
        else:
            active = [constraint for constraint in active if constraint not in drop]

    if non_risk_constraints in active:
        active.remove(non_risk_constraints)

    return active, degeneracy
