import cvxpy as cp
import numpy as np
from src.Miscellaneous import get_active


def solve_sdp(deltas, F_d, E, c, Q, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None):
    r"""Solve a semidefinite program with LMI scenario constraints via the scenario approach.

    Minimizes :math:`\tfrac{1}{2} x^\top Q x + c^\top x + \tau \|x - x_{\text{ref}}\|_p + \rho \sum \zeta_i`
    subject to scenario LMI constraints
    :math:`F_0(\delta_i) + \sum_j x_j F_j(\delta_i) \preceq \zeta_i I`
    and a hard LMI constraint :math:`E_0 + \sum_j x_j E_j \preceq 0`.

    Parameters
    ----------
    deltas : numpy.ndarray
        Scenario samples with shape ``(N,)`` or ``(N, q)`` where *N* is the
        number of scenarios and *q* is the uncertainty dimension.
    F_d : callable
        Function mapping a scenario to a dict of symmetric matrices keyed by
        variable index (``'0'`` for the constant term, ``'1'`` for
        :math:`x_1`, etc.):  ``F_d(delta) -> {'0': F0, '1': F1, ...}``.
    E : dict or None
        Dict of symmetric matrices for the hard LMI constraint, same key
        format as *F_d* output. Pass ``None`` if there are no hard constraints.
    c : numpy.ndarray
        Linear objective coefficient vector with shape ``(n,)``.
    Q : numpy.ndarray
        Quadratic cost matrix with shape ``(n, n)``. Must be positive
        semidefinite and symmetric.
    tau : float, optional
        Regularization strength toward *x_ref*. Default is ``0.0``.
    x_ref : numpy.ndarray, optional
        Reference point for the regularization term. Default is ``[0.0]``.
    rho : float, optional
        Penalty on slack variables. When ``0.0`` the scenario constraints are
        hard. Default is ``0.0``.
    norm_type : int, optional
        Norm type for the regularization term (1, 2, etc.). Default is ``2``.
    solver : str or None, optional
        CVXPY solver name. ``None`` for automatic selection.

    Returns
    -------
    x : numpy.ndarray
        Optimal decision variable vector.
    zeta : numpy.ndarray
        Optimal slack variable values (one per scenario).
    cost : float
        Optimal objective value.
    N : int
        Number of scenarios used.
    complexity : int
        Number of active (support) scenario constraints.
    constraints : list
        CVXPY constraint objects from the problem formulation.
    degeneracy : bool
        ``True`` if degeneracy was detected during support identification.

    Raises
    ------
    ValueError
        If the solver does not reach an optimal or near-optimal status.
    AssertionError
        If *Q* or any :math:`F_j(\delta)` matrix is not symmetric.
    """

    # TODO: need to change all this...
    # Check Q is positive semi-definite and symmetric
    # print(np.linalg.eigvals(Q)) #TODO: add eigenvalues to errors if not PSD

    assert np.all(np.linalg.eigvals(Q) >= 0), "Q needs to be positive semi-definite"
    assert (Q == Q.T).all(), "Q needs to be symmetric"
    n = Q.shape[1]
    m = list(F_d(deltas[0]).values())[0].shape[0]

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
        zeta = cp.Variable(N, nonneg=True)  # One slack per scenario
    else:
        zeta = np.zeros(N)

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
        constraints.append(expr << zeta[i] * np.eye(m))  # Per-scenario scalar slack

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
    else:
        zeta_out = np.zeros(N)

    cost_out = prob.value

    # Find the active constraints
    complexity, active, degeneracy = get_active(constraints, non_risk_constraints, prob, objective, rho, solver)

    # Return results
    return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy