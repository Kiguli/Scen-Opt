import cvxpy as cp
import numpy as np
from src.Miscellaneous import get_support, check_scenarios, check_psd, regularization_term


def solve_sdp(deltas, F_d, E, c, Q, tau=0.0, x_ref=np.array([0.0]), rho=0.0, norm_type=2, solver=None, include_slack=None):
    r"""Solve a semidefinite program with LMI scenario constraints via the scenario approach.

    .. math::

        \min \quad \tfrac{1}{2}\, x^\top Q\, x \;+\; c^\top x \;+\; \tau \|x - x_{\text{ref}}\|_p \;+\; \rho \sum_i \zeta_i

    .. math::

        \text{s.t.} \quad F_0(\delta_i) + \sum_j x_j\, F_j(\delta_i) \;\preceq\; \zeta_i\, I, \quad i = 1,\ldots,N

    .. math::

        E_0 + \sum_j x_j\, E_j \;\preceq\; 0

    Parameters
    ----------
    deltas : numpy.ndarray
        Scenario samples with shape ``(N,)`` or ``(N, q)`` where *N* is the
        number of scenarios and *q* is the uncertainty dimension.
    F_d : callable
        Function mapping a scenario to a dict of symmetric matrices keyed by
        variable index (``'0'`` for the constant term, ``'1'`` for
        :math:`x_1`, etc.):  ``F_d(delta) -> {'0': F0, '1': F1, ...}``.
        Integer keys (``0, 1, ...``) work too.
    E : dict or None
        Dict of symmetric matrices for the hard LMI constraint, same key
        format as *F_d* output. Pass ``None`` if there are no hard constraints.
    c : numpy.ndarray
        Linear objective coefficient vector with shape ``(n,)``.
    Q : numpy.ndarray or None
        Quadratic cost matrix with shape ``(n, n)``. Must be positive
        semidefinite and symmetric. ``None`` or an empty array means no
        quadratic term.
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
        If the solver does not reach an optimal or near-optimal status.
    AssertionError
        If *Q* or any :math:`F_j(\delta)` matrix is not symmetric.
    """

    n = np.asarray(c).reshape(-1).size  # Number of variables
    # Check Q is symmetric positive semi-definite (up to a small tolerance);
    # an empty Q means there is no quadratic term.
    Q = np.zeros((n, n)) if Q is None or np.size(Q) == 0 else check_psd(Q)
    N = check_scenarios(deltas)  # Number of scenarios
    m = list(F_d(deltas[0]).values())[0].shape[0]

    # Variables
    x = cp.Variable(n)
    # See solve_lp for the rationale: respect include_slack independently of
    # whether ρ happens to be 0, so relaxation with ρ = 0 is correctly
    # reported as unbounded rather than silently coerced into robust.
    if include_slack is None:
        include_slack = (rho != 0)
    if include_slack:
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
            if int(k) == 0:  # keys may be '0', '1', ... or 0, 1, ...
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
            if int(k) == 0:  # keys may be '0', '1', ... or 0, 1, ...
                term = Fk
            else:
                xk = x[int(k) - 1]
                term = xk*Fk   # scalar-variable times numpy matrix is fine
            expr = term if expr is None else expr + term
        # Hard constraints are kept separate from the scenario constraints:
        # always enforced but never candidates for the support list.
        non_risk_constraints = [expr << 0]

    else:
        non_risk_constraints = []

    # x is a vector here, so x_ref must be a scalar or a length-n vector: a
    # (n, 1) column would be broadcast against x into an (n, n) matrix.
    _xr = np.asarray(x_ref, dtype=float).reshape(-1)
    if _xr.size == 1:
        x_ref = float(_xr[0])
    elif _xr.size == n:
        x_ref = _xr
    else:
        raise ValueError(
            f"x̄ has {_xr.size} entries but d = {n}. Provide either a scalar "
            f"(broadcast across all coordinates) or a length-{n} vector."
        )

    # Objective Function. The quadratic term is left out when Q = 0: CVXPY
    # would still add a cone for it, which solvers without second-order cones
    # (e.g. SDPA) cannot handle.
    quadratic = (1 / 2) * cp.quad_form(x, cp.psd_wrap(Q)) if np.any(Q) else 0
    objective = cp.Minimize(quadratic + c.T @ x + regularization_term(tau, x, x_ref, norm_type) + rho * cp.sum(zeta))

    # Solve the problem
    prob = cp.Problem(objective, constraints + non_risk_constraints)
    prob.solve(solver=solver)

    # Simplify results
    x_out = x.value

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"SDP did not solve to optimality. Status: {prob.status}, Objective: {prob.value}, x: {x_out}")

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