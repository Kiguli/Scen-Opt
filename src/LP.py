import cvxpy as cp
import numpy as np
from src.Miscellaneous import get_active

def solve_lp(deltas, A_d, b_d, G, h, c, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2,solver=None):
    r"""Solve a linear program with scenario constraints via the scenario approach.

    .. math::

        \min \quad c^\top x \;+\; \tau \|x - x_{\text{ref}}\|_p \;+\; \rho \sum_i \zeta_i

    .. math::

        \text{s.t.} \quad A(\delta_i)\, x + b(\delta_i) \;\leq\; \zeta_i, \quad i = 1,\ldots,N

    .. math::

        G\, x + h \;\leq\; 0

    Parameters
    ----------
    deltas : numpy.ndarray
        Scenario samples with shape ``(N,)`` or ``(N, q)`` where *N* is the
        number of scenarios and *q* is the uncertainty dimension.
    A_d : callable
        Function mapping a scenario to the constraint coefficient matrix,
        ``A_d(delta) -> (m_s, n)`` array.
    b_d : callable
        Function mapping a scenario to the constraint right-hand-side vector,
        ``b_d(delta) -> (m_s, 1)`` array.
    G : numpy.ndarray
        Coefficient matrix for hard (non-scenario) constraints.
        Pass ``np.array([])`` if there are none.
    h : numpy.ndarray
        Right-hand-side vector for hard constraints.
        Pass ``np.array([])`` if there are none.
    c : numpy.ndarray
        Linear objective coefficient vector with shape ``(n, 1)``.
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
    """
    n = A_d(deltas[0]).shape[1]  # Number of variables
    m_scenario = A_d(deltas[0]).shape[0]  # Number of scenario constraints
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
        zeta = cp.Variable(N, nonneg=True)  # One slack per scenario
    else:
        zeta = np.zeros(N)

    constraints = []
    for i in range(N):
        constraints.append(A_d(deltas[i]) @ x + b_d(deltas[i]) <= zeta[i])  # Per-scenario slack

    if not (G.size == 0 or h.size == 0):
        non_risk_constraints = G @ x + h <= 0
        constraints.append(non_risk_constraints)  # hard constraints
    else:
        non_risk_constraints = []

    # Reshape x_ref to match the decision variable's (n, 1) shape. The form
    # arrives variously as a scalar (default), a 1-D array (legacy CSV), or
    # a 2-D array (matrix modal). We accept all three and size-check.
    _xr = np.asarray(x_ref, dtype=float).reshape(-1)
    if _xr.size == 1:
        # Broadcast a scalar reference across every coordinate of x.
        x_ref = np.full((n, 1), float(_xr[0]))
    elif _xr.size == n:
        x_ref = _xr.reshape(n, 1)
    else:
        raise ValueError(
            f"x̄ has {_xr.size} entries but d = {n}. Provide either a scalar "
            f"(broadcast across all coordinates) or a length-{n} vector."
        )

    # Objective Function
    objective = cp.Minimize(c.T @ x + tau * cp.norm(x - x_ref, norm_type) + rho * cp.sum(zeta))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"LP did not solve to optimality. Status: {prob.status}, Objective: {prob.value}, x: {x_out}")

    if rho != 0.0:
        zeta_out = zeta.value
    else:
        zeta_out = np.zeros(N)

    cost_out = prob.value
    print(cost_out)
    # =====================================
    #SOLVE FOR ACTIVE CONSTRAINTS
    # =====================================

    # Find the active constraints
    complexity, active, degeneracy = get_active(constraints, non_risk_constraints, prob, objective, rho, solver)

    # Return results
    return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy