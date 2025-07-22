import cvxpy as cp
import numpy as np


def solve_sdp1(deltas, F_d, F, c, Q, T=0.0, x_ref=np.array([0.0]), P=0.0, norm_type=2, solver=None):
    """
        Solves a semidefinite programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        F_d (function): Function that returns the matrices for the constraints given a delta.
        F (numpy.ndarray): matrices for the hard constraints.
        c (numpy.ndarray): Coefficient vector for the objective function.
        Q (numpy.ndarray): Quadratic cost matrix for the objective function.
        T (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Reference point for the norm term in the objective function.
        P (float): Penalty parameter for the slack variables in the objective function.
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
    n = list(F_d(deltas[0]).values())[0].shape[1]
    m = list(F_d(deltas[0]).values())[0].shape[0]

    try:
        num_of_deltas = deltas.shape[1]  # Number of deltas per row
    except IndexError:
        num_of_deltas = 1  # TODO: check A_d and b_d don't include delta[i] where i>num_deltas
    try:
        size_of_deltas = deltas.shape[0]  # Number of row
    except IndexError:
        raise ValueError("The input `deltas` must have at least one row.")

    # Variables
    x = cp.Variable((n, 1))
    if P != 0:
        s = cp.Variable((m, 1), nonneg=True)  # Slack variables
    else:
        s = np.zeros((m, 1))

    constraints = []
    for i in range(size_of_deltas):
        F_dict = F_d(deltas[i])  # Dictionary of submatrices for this delta
        expr = None
        for k, Fk in F_dict.items():
            assert np.all(np.linalg.eigvals(Fk) >= 0), "\\(F_j(\delta)\\) need to be positive semi-definite and symmetric"
            assert (Fk == Fk.T).all(), "\\(F_j(\delta)\\) need to be positive semi-definite and symmetric"
            if k == '0':
                term = Fk
            else:
                term = Fk * x[int(k)-1][0] if x.shape[0] > 1 else Fk * x[0]  # Use x_k if x is multidimensional
            expr = term if expr is None else expr + term
        constraints.append(expr <= s)

    if (F):
        expr = None
        for k, Fk in F.items():
            assert np.all(np.linalg.eigvals(Fk) >= 0), "\\(F_j\\) need to be positive semi-definite and symmetric"
            assert (Fk == Fk.T).all(), "\\(F_j\\) needs to be positive semi-definite and symmetric"
            if k == '0':
                term = Fk
            else:
                term = Fk * x[int(k) - 1][0] if x.shape[0] > 1 else Fk * x[0]  # Use x_k if x is multidimensional
            expr = term if expr is None else expr + term
        non_risk_constraints = expr <= 0
        constraints.append(non_risk_constraints)  # hard constraints

    else:
        non_risk_constraints = []


    print("Pass phase 3")
    # Objective Function
    objective = cp.Minimize((1 / 2) * cp.quad_form(x, Q) + c.T @ x + T * cp.norm(x - x_ref, norm_type) + P * cp.sum(s))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if P != 0.0:
        s_out = s.value
        # print(s)
    else:
        s_out = np.zeros((m, 1))

    cost_out = prob.value

    # Find the active constraints
    active, degeneracy = get_active_SDP(constraints, non_risk_constraints, prob, objective, P, solver)

    # Return results
    return x_out, s_out, cost_out, size_of_deltas, active, constraints, degeneracy


def solve_sdp2(deltas, C, A_da, A_a, T=0.0, x_ref=np.array([0.0]), P=0.0, norm_type=2, solver=None):
    """
        Solves a semidefinite programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        A_da (function): Function that returns the matrices for the constraints given a delta.
        A_a (numpy.ndarray): matrices for the hard constraints.
        C (numpy.ndarray): cost matrix for the objective function.
        T (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Reference point for the norm term in the objective function.
        P (float): Penalty parameter for the slack variables in the objective function.
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
    print("In new function")
    n = list(A_da(deltas[0]).values())[0].shape[1]
    m = list(A_da(deltas[0]).values())[0].shape[0]
    print(n)
    print(m)

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

    if P != 0:
        s = cp.Variable((m, 1), nonneg=True)  # Slack variables
    else:
        s = np.zeros((m, 1))

    print("Pass phase 1")

    for i in range(size_of_deltas):
        A_dict = A_da(deltas[i])  # Dictionary of submatrices for this delta
        expr = None
        #TODO: logic for Trace(A_j(delta)X)
        constraints.append(expr <= s)

    print("Pass phase 2")

    if (A_a):
        expr = None
        #TODO: logic for Trace(A_jX)

        non_risk_constraints = expr <= 0
        constraints.append(non_risk_constraints)  # hard constraints

    else:
        non_risk_constraints = []

    print("Pass phase 3")
    # Objective Function
    objective = cp.Minimize(
        cp.trace(C @ X) + T * cp.norm(x - x_ref, norm_type) + P * cp.sum(s))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if P != 0.0:
        s_out = s.value
        # print(s)
    else:
        s_out = np.zeros((m, 1))

    cost_out = prob.value

    # Find the active constraints
    active, degeneracy = get_active_SDP(constraints, non_risk_constraints, prob, objective, P, solver)

    # Return results
    return x_out, s_out, cost_out, size_of_deltas, active, constraints, degeneracy


def test_active_SDP(prob, objective, active, P=0.0, solver=None):
    """
        Solves a linear programming problem with optional robust and regularization constraints.

        Parameters:
        prob: CVXPY problem instance for optimal solution.
        objective: CVXPY objective function.
        active: list of active constraints.
        P (float): Penalty parameter for the slack variables in the objective function.
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

        if P != 0:
            assert np.allclose(prob.variables()[1].value, prob2.variables()[
                1].value), f"Solutions for s differ: {prob.variables()[1].value} vs {prob2.variables()[1].value}"

        # Return True if all assertions pass
        return True

    except AssertionError as e:
        print(f"Assertion failed: {e}")
        # Return False if any assertion fails
        return False


def get_active_SDP(constraints, non_risk_constraints, prob, objective, P=0.0, solver=None, threshold=1e-8):
    """
        Finds the active constraints of the SDP.

        Parameters:
        constraints (list): list of constraints.
        non_risk_constraints (list): list of constraints that should not be included in the error calculation.
        prob: CVXPY problem instance for optimal solution.
        objective: CVXPY objective function.
        P (float,optional): Penalty parameter for the slack variables in the objective function. Default is 0.0.
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
    if not test_active_SDP(prob, objective, active, P=P, solver=solver):
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
                if test_active_SDP(prob, objective, temp_active, P=P, solver=solver):
                    active.remove(constraint)
                    changed = True
                    break  # Restart loop since active has changed
        # At the end, active contains only constraints whose removal makes test_active_SDP return False
        if not test_active_SDP(prob, objective, active, P=P, solver=solver):
            raise ValueError("Error calculating support list! Degeneracy present as active constraints != constraints.")
    else:
        # If solution does not change then likely to be non-degenerate, check for true support list as solvers can be incorrect
        drop = []
        for a in active:
            if test_active_SDP(prob, objective, [constraint for constraint in active if constraint != a], P=P,
                               solver=solver):
                drop.append(a)
        if not test_active_SDP(prob, objective, [constraint for constraint in active if constraint not in drop], P=P,
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
                    if test_active_SDP(prob, objective, temp_active, P=P, solver=solver):
                        active.remove(constraint)
                        changed = True
                        break  # Restart loop since active has changed
            # At the end, active contains only constraints whose removal makes test_active_SDP return False
            if not test_active_SDP(prob, objective, active, P=P, solver=solver):
                raise ValueError(
                    "Error calculating support list! Degeneracy present as reduced active constraints != active constraints.")
        else:
            active = [constraint for constraint in active if constraint not in drop]

    if non_risk_constraints in active:
        active.remove(non_risk_constraints)

    return active, degeneracy
