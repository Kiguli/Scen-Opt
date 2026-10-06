import cvxpy as cp
import numpy as np
from src.Miscellaneous import get_support, check_scenarios, regularization_term

def solve_lp(deltas, A_d, b_d, G, h, c, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None, include_slack=None):
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
    norm_type : int, float or str, optional
        Order p of the vector norm in the regularization term: any number
        ``p >= 1``, ``'inf'`` or ``'fro'`` (Euclidean). Default is ``2``.
    solver : str or None, optional
        CVXPY solver name. ``None`` for automatic selection.
    include_slack : bool or None, optional
        Whether to add the slack variables ζ_i (the relaxation formulation).
        ``None`` adds them only when ``rho != 0``.

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
        Cardinality of the support list (violated plus retained active scenario constraints).
    constraints : list
        Scenario constraint objects (one per scenario; hard constraints are kept separate).
    degeneracy : bool
        ``True`` if degeneracy was detected during support identification.

    Raises
    ------
    ValueError
        If ``deltas`` is empty, or the solver does not reach an optimal or
        near-optimal status.
    """
    N = check_scenarios(deltas)  # Number of scenarios
    n = A_d(deltas[0]).shape[1]  # Number of variables
    m_scenario = A_d(deltas[0]).shape[0]  # Number of scenario constraints

    # Variables
    x = cp.Variable((n, 1))
    # Introduce a non-negative slack ζ_i whenever the caller has selected the
    # relaxation formulation (`include_slack=True`), even if the user has set
    # ρ = 0. Previously we silently dropped the slack at ρ = 0, which made
    # the problem look like the robust formulation; for relaxation with ρ = 0
    # the LP is genuinely unbounded (ζ_i can absorb any constraint violation
    # at zero cost) and the solver should report it as such.
    if include_slack is None:
        include_slack = (rho != 0)
    if include_slack:
        zeta = cp.Variable(N, nonneg=True)  # One slack per scenario
    else:
        zeta = np.zeros(N)

    constraints = []
    for i in range(N):
        # b(δ) as a column: a 1-D (m,) array would otherwise be broadcast to (m, m).
        b_i = np.asarray(b_d(deltas[i]), dtype=float).reshape(-1, 1)
        constraints.append(A_d(deltas[i]) @ x + b_i <= zeta[i])  # Per-scenario slack

    # Hard constraints are kept separate from the scenario constraints: they
    # are always enforced but never candidates for the support list.
    if not (G.size == 0 or h.size == 0):
        non_risk_constraints = [np.atleast_2d(G) @ x + np.asarray(h, dtype=float).reshape(-1, 1) <= 0]
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
    objective = cp.Minimize(c.T @ x + regularization_term(tau, x, x_ref, norm_type) + rho * cp.sum(zeta))

    # Solve the problem
    prob = cp.Problem(objective, constraints + non_risk_constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"LP did not solve to optimality. Status: {prob.status}, Objective: {prob.value}, x: {x_out}")

    if include_slack:
        zeta_out = zeta.value
    else:
        zeta_out = np.zeros(N)

    cost_out = prob.value

    # Find the support list
    complexity, support, degeneracy = get_support(
        constraints, non_risk_constraints, prob, objective, solver=solver)

    # Return results
    return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy