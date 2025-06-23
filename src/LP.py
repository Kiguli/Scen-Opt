import cvxpy as cp
import numpy as np

def solve_lp(deltas, A_d, b_d, A, b, c, T=0.0, x_ref=np.array([0.0]), P=0.0, norm_type=2,solver=None):
    """
        Solves a linear programming problem with optional robust and regularization constraints.

        Parameters:
        deltas (numpy.ndarray): Collected deltas that should be added to constraints.
        A_d (function): Function that returns the coefficient matrix for the constraints given a delta.
        b_d (function): Function that returns the right-hand side vector for the constraints given a delta.
        A (numpy.ndarray): coefficient matrix for the hard constraints.
        b (numpy.ndarray): right-hand side vector for the hard constraints.
        c (numpy.ndarray): Coefficient vector for the objective function.
        T (float): Regularization parameter for the norm term in the objective function.
        x_ref (numpy.ndarray): Regularization parameter for the regularization term in the objective to stay near some point.
        P (float): Penalty parameter for the slack variables in the objective function.
        norm_type (int, optional): Type of norm to use in the objective function. Default is 2 (Euclidean norm).
        solver (str, optional): The solver to use for the optimization problem. Default is None.

        Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Optimal solution vector.
            - s (numpy.ndarray): Optimal slack variables vector.
            - cost (float): Optimal value of the objective function.
        """
    assert A_d.size != 0, "A(delta) cannot be empty."
    assert b_d.size != 0, "b(delta) cannot be empty."

    n = A_d(deltas[0]).shape[1]  # Number of variables
    m = A_d(deltas[0]).shape[0]  # Number of constraints
    try:
        num_of_deltas = deltas.shape[1]  # Number of deltas per row
    except IndexError:
        num_of_deltas = 1 #TODO: check A_d and b_d don't include delta[i] where i>num_deltas
    try:
        size_of_deltas = deltas.shape[0]  # Number of row
    except IndexError:
        raise ValueError("The input `deltas` must have at least one row.")

    # Variables
    x = cp.Variable((n, 1))
    if P != 0:
        s = cp.Variable((m, 1), nonneg=True)  # Adjust size based on constraints
    else:
        s = np.zeros((m, 1))

    constraints = []
    for i in range(size_of_deltas):
        constraints.append(A_d(deltas[i]) @ x + b_d(deltas[i]) <= s)  # Add each row separately

    if not (A.size == 0 or b.size == 0):
        m = A.shape[0]  # Number of constraints
        non_risk_constraints = A @ x + b <= 0
        constraints.append(non_risk_constraints)  # hard constraints
    else:
        non_risk_constraints = []

    # Objective Function
    objective = cp.Minimize(c.T @ x + T * cp.norm(x-x_ref, norm_type) + P * cp.sum(s))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value
    if P != 0.0:
        s_out = s.value
        #print(s)
    else:
        s_out = np.zeros((m,1))

    cost_out = prob.value

    # =====================================
    #SOLVE FOR ACTIVE CONSTRAINTS
    # =====================================

    # Find the active constraints
    active = get_active_LP(constraints, non_risk_constraints, prob, objective, P, solver)

    # Return results
    return x_out, s_out, cost_out, size_of_deltas, active, constraints

def test_active_LP(prob, objective, active, P=0.0, solver=None):
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
                           prob2.variables()[0].value), f"Solutions for x differ: {prob.variables()[0].value} vs {prob2.variables()[0].value}"

        if P != 0:
            assert np.allclose(prob.variables()[1].value, prob2.variables()[1].value), f"Solutions for s differ: {prob.variables()[1].value} vs {prob2.variables()[1].value}"

        # Return True if all assertions pass
        return True

    except AssertionError as e:
        print(f"Assertion failed: {e}")
        # Return False if any assertion fails
        return False

def get_active_LP(constraints, non_risk_constraints, prob, objective, P=0.0, solver=None, threshold=1e-8):
    """
        Finds the active constraints of the LP.

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

    #TODO: make threshold a global parameter?
    active = []
    for constraint in constraints:
        #print(max(constraint.dual_value))
        if max(constraint.dual_value) > threshold:  # Check if any dual value is positive
            active.append(constraint)

    #If solution changes then likely to have degeneracy
    if not test_active_LP(prob, objective, active, P=P, solver=solver):
        print("Active constraints are not valid. Lower bound not viable likely due to degeneracy.")
        # loop through all constraints and make a support list from them
        active = constraints
        for a in active:
            if test_active_LP(prob, objective, [constraint for constraint in active if constraint != a], P=P, solver=solver):
                active.remove(a)
        if not test_active_LP(prob, objective, active, P=P, solver=solver):
            raise ValueError("Error calculating support list after finding degeneracy.")
    else:
        #If solution does not change then likely to be non-degenerate, check for true support list as solvers can be incorrect
        for a in active:
            if test_active_LP(prob, objective, [constraint for constraint in active if constraint != a], P=P, solver=solver):
                active.remove(a)
        if not test_active_LP(prob, objective, active, P=P, solver=solver):
            raise ValueError("Error calculating support list.")

    if non_risk_constraints in active:
        active.remove(non_risk_constraints)

    return active