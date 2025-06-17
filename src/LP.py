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
    n = A_d(deltas[0]).shape[1]  # Number of variables
    m = A_d(deltas[0]).shape[0]  # Number of constraints
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
        constraints.append(A_d(deltas[i]) @ x + b_d(deltas[i]) <= s)  # Relaxed robust constraints

    m = A.shape[0]  # Number of constraints
    if P != 0:
        s_h = cp.Variable((m, 1), nonneg=True)  # Slack variables
    else:
        s_h = np.zeros((m,1))

    constraints.append(A @ x + b <= s_h) # hard constraints

    # Objective Function
    objective = cp.Minimize(c.T @ x + T * cp.norm(x-x_ref, norm_type) + P * cp.sum(s) + P*cp.sum(s_h))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value
    if P != 0:
        s_out = s.value
        print(s)
    else:
        s_out = np.zeros((m,1))
    if P != 0:
        s_h_out = s_h.value  # TODO: check how this should work...
        print(s_h)
    else:
        s_h_out = np.zeros((m,1))

    cost_out = prob.value

    # ======================================
    #SOLVE FOR ACTIVE CONSTRAINTS
    # =====================================

    active = []
    for constraint in constraints:
        if constraint.dual_value > 0: #TODO: add a tolerance for different solvers
            print(constraint.dual_value)
            active.append(constraint)

    if test_active_LP(prob, objective, active, P=P, solver=solver):
        pass
        # TODO: return an error about how the active constraints are not active constraints, probably a solver error.
    else:
        drop = []
        for a in active:
            if not test_active_LP(prob, objective, [constraint for constraint in active if constraint != a], P=P, solver=solver):
                print(a)
                drop.append(a)
        print(drop)
        active = [a for a in active if a not in drop]

    test_active_LP(prob, objective, active, P=P, solver=solver)

    # Return results #TODO: check if active includes non-delta constraints
    return x_out, s_out, s_h_out, cost_out, size_of_deltas, active, constraints

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
        print("check objective values are the same:")
        print(prob.value)
        print(prob2.value)

        # Assert that the solutions are the same
        assert np.allclose(prob.variables()[0].value,
                           prob2.variables()[0].value), f"Solutions for x differ: {prob.variables()[0].value} vs {prob2.variables()[0].value}"
        print("check solutions are the same:")
        print(prob.variables()[0].value)
        print(prob2.variables()[0].value)

        if P != 0:
            assert np.allclose(prob.variables()[1].value, prob2.variables()[1].value), f"Solutions for s differ: {prob.variables()[1].value} vs {prob2.variables()[1].value}"
            print("check solutions are the same:")
            print(prob.variables()[1].value)
            print(prob2.variables()[1].value)

        # Return True if all assertions pass
        return True

    except AssertionError as e:
        print(f"Assertion failed: {e}")
        # Return False if any assertion fails
        return False