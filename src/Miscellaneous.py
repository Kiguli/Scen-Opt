import cvxpy as cp
import numpy as np
import pandas as pd

SOLVER_CAPABILITIES = {
    'CLARABEL': ['LP', 'QP', 'SDP'], 'SCS': ['LP', 'QP', 'SDP'],
    # OSQP is kept off the LP list: it fails on the scenario LPs we build
    # (the slack reformulation produces degenerate QPs OSQP can't handle).
    'OSQP': ['QP'], 'ECOS': ['LP', 'QP'],
    'CVXOPT': ['LP', 'QP', 'SDP'], 'GLOP': ['LP'], 'GLPK': ['LP'],
    'GLPK_MI': ['LP'], 'SCIPY': ['LP'], 'HIGHS': ['LP', 'QP'],
    'SCIP': ['LP', 'QP'], 'CBC': ['LP'],
    'DAQP': ['LP', 'QP'], 'PIQP': ['LP', 'QP'],
    'PROXQP': ['LP', 'QP'],
    'MOSEK': ['LP', 'QP', 'SDP'], 'GUROBI': ['LP', 'QP'],
    'CPLEX': ['LP', 'QP'], 'SDPA': ['SDP'],
}


def _cvxpy_capabilities(name):
    """Program types a solver missing from SOLVER_CAPABILITIES can handle, from CVXPY's own metadata.

    LP needs any conic or QP interface, QP needs a QP interface or second-order
    cones, and SDP needs positive semidefinite cones.
    """
    from cvxpy.constraints import PSD, SOC
    from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC, SOLVER_MAP_QP
    conic, qp = SOLVER_MAP_CONIC.get(name), SOLVER_MAP_QP.get(name)
    supported = conic.SUPPORTED_CONSTRAINTS if conic else []
    types = []
    if conic or qp:
        types.append('LP')
    if qp or SOC in supported:
        types.append('QP')
    if PSD in supported:
        types.append('SDP')
    return types

def get_solvers():
    """Retrieve installed CVXPY solvers and their supported problem types.

    Returns
    -------
    solver_dict : dict
        Mapping of solver name to list of supported types
        (``'LP'``, ``'QP'``, ``'SDP'``).

    See Also
    --------
    `CVXPY solver list <https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver>`_
    """
    installed = cp.installed_solvers()
    solver_dict = {s: SOLVER_CAPABILITIES[s] if s in SOLVER_CAPABILITIES else _cvxpy_capabilities(s)
                   for s in installed}
    return {s: types for s, types in solver_dict.items() if types}

def check_scenarios(deltas):
    """Return the number of scenarios N, raising ValueError when there are none."""
    if deltas is None or np.size(deltas) == 0:
        raise ValueError("The input `deltas` must have at least one row.")
    return len(deltas)


def check_psd(Q, tol=1e-9):
    """Return Q symmetrised, after checking it is symmetric positive semidefinite.

    Both checks allow a small relative tolerance, so rank-deficient matrices
    such as D.T @ D, whose zero eigenvalues come out as about -1e-16, pass.
    Those tiny negative eigenvalues are then rounded up to zero, so the
    returned matrix is exactly positive semidefinite.
    """
    Q = np.asarray(Q, dtype=float)
    scale = max(1.0, float(np.abs(Q).max()))
    assert np.allclose(Q, Q.T, rtol=0, atol=tol * scale), "Q must be symmetric"
    Q = (Q + Q.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Q)
    assert eigvals.min() >= -tol * scale, f"Q must be positive semi-definite (eigenvalues: {eigvals})"
    if eigvals.min() < 0:
        Q = (eigvecs * np.maximum(eigvals, 0)) @ eigvecs.T
        Q = (Q + Q.T) / 2
    return Q


def regularization_term(tau, x, x_ref, norm_type):
    """Return tau * ||x - x_ref||_p, or 0 when tau = 0.

    The term is left out when tau = 0, so a robust or relaxed LP stays a pure
    LP: with the default p = 2 the norm would add a second-order cone, which
    LP-only solvers cannot handle. The norm is the vector p-norm of
    x - x_ref, so any p >= 1 works, and ``'fro'`` is the Euclidean norm.
    """
    if tau == 0:
        return 0
    diff = x - x_ref
    if len(diff.shape) == 2:
        diff = diff[:, 0]  # (n, 1) column -> vector
    return tau * cp.norm(diff, 2 if norm_type == 'fro' else norm_type)


def get_norm_types():
    """Retrieve available norm types for the regularization term.

    Returns
    -------
    list
        Common norm types: ``1`` (L1), ``2`` (L2), ``'inf'`` (L-infinity)
        and ``'fro'`` (the same as L2 for the vector ``x - x_ref``). Any
        other number ``p >= 1`` also works.
    """
    norm_types = [1, 2, "inf", "fro"]
    return norm_types

def load_file(file_path):
    """Read a data file and convert it to a NumPy array.

    Supports CSV, TXT, XLSX, and JSON formats. A ``.txt`` file may be
    separated by whitespace or by commas; a ``.json`` file holds a list of
    rows (one per scenario) or a table in pandas' default JSON format.

    Parameters
    ----------
    file_path : str
        Path to the input file. Must have a supported extension
        (``.csv``, ``.txt``, ``.xlsx``, ``.json``).

    Returns
    -------
    numpy.ndarray or None
        Array containing the file data, or ``None`` if the file cannot be
        read or has an unsupported extension (the error is printed).
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path,header=None).values  # Read CSV using pandas and convert to NumPy
        elif file_path.endswith('.txt'):
            try:
                data = np.loadtxt(file_path)  # whitespace-separated numbers
            except ValueError:
                data = np.loadtxt(file_path, delimiter=',')  # comma-separated numbers
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, engine='openpyxl',header=None).values  # Read Excel file and convert to NumPy
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path).values  # a list of rows, or pandas' column format
        else:
            raise ValueError("Unsupported file format")

        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_support(objective, support, non_risk, ref, solver=None):
    """Check whether the scenario constraints in `support`, together with the
    hard constraints in `non_risk` (always enforced), reproduce the reference
    optimal value.

    Support membership is decided on the optimal *value*, which is well
    defined even when the optimizer is not unique. A scenario is of support
    exactly when its removal changes the optimum: a genuinely violated
    scenario changes it by its penalty contribution, an active support
    constraint by the relaxation its removal permits, while a redundant or
    only-spuriously-slack scenario leaves it unchanged. Comparing solution
    vectors instead would misfire on problems with a non-unique optimum
    (a flat optimal face), which is why only the value is used.

    Parameters
    ----------
    objective : cvxpy.Minimize
        The original objective function.
    support : list
        Candidate support list of scenario constraint objects.
    non_risk : list
        Hard (non-scenario) constraints, always enforced.
    ref : dict
        Reference optimum with keys ``cost`` and ``tol``.
    solver : str or None, optional
        CVXPY solver name.

    Returns
    -------
    bool
        ``True`` if the reduced problem reproduces the reference value.
    """
    try:
        prob2 = cp.Problem(objective, list(support) + list(non_risk))
        prob2.solve(solver=solver)
        if prob2.value is None or prob2.status not in ("optimal", "optimal_inaccurate"):
            return False
        return bool(np.isclose(ref["cost"], prob2.value, rtol=ref["tol"], atol=1e-9))
    except Exception:
        # Solver failure on the reduced problem: candidate not validated
        # (conservative — the tested constraint is kept).
        return False


def _is_slack(constraint, atol=1e-6, rtol=1e-6):
    """Return ``True`` when `constraint` is strictly slack (satisfied with
    positive margin, hence neither active nor violated) at the currently
    stored solution.

    Interior-point solvers routinely leave small nonzero duals on inactive
    constraints, so a constraint passing the dual screen is not necessarily a
    support scenario. The activity margin -- ``min(rhs - lhs)`` for scalar or
    vector inequalities, the smallest eigenvalue for PSD constraints -- decides:
    a margin clearly above the scale-aware tolerance means the dual was
    spurious. Must be called while the original solution is still stored on
    the variables (before any re-solve overwrites them). Errs on the side of
    ``False`` (treated as active) when the margin cannot be evaluated.
    """
    try:
        if isinstance(constraint, cp.constraints.PSD):
            X = np.asarray(constraint.args[0].value, dtype=float)
            margin = float(np.min(np.linalg.eigvalsh((X + X.T) / 2.0)))
            scale = float(np.max(np.abs(X)))
        else:
            lhs = np.asarray(constraint.args[0].value, dtype=float)
            rhs = np.asarray(constraint.args[1].value, dtype=float)
            margin = float(np.min(rhs - lhs))
            scale = max(float(np.max(np.abs(lhs))), float(np.max(np.abs(rhs))))
        return margin > atol + rtol * scale
    except Exception:
        return False


def _prune(objective, candidates, non_risk, ref, solver=None):
    """Greedily drop scenario constraints whose removal leaves the optimum
    unchanged, returning an irreducible support list."""
    keep = list(candidates)
    i = 0
    while i < len(keep):
        if test_support(objective, keep[:i] + keep[i + 1:], non_risk, ref, solver=solver):
            keep.pop(i)
        else:
            i += 1
    return keep


def get_support(constraints, non_risk_constraints, prob, objective, solver=None, threshold=1e-8, sol_tol=1e-6):
    """Identify the support list of a solved scenario problem.

    The support list comprises every scenario constraint whose removal changes
    the optimal value: all genuinely violated constraints together with an
    irreducible subset of the active ones (cf. the support-list definition).
    Candidates are screened by dual value -- under relaxation the dual
    variables of a violated constraint sum to rho, so both violated and active
    constraints are captured (these are the *support scenarios*) -- then
    greedily pruned by re-solving and comparing the optimal value. Hard
    constraints are always enforced in re-solves and never counted.

    Non-degeneracy is equivalent to the support scenarios forming a support
    list, so the prune doubles as a degeneracy test. Interior-point solvers
    may leave a small nonzero dual on constraints that are neither active nor
    violated, so each screened constraint's activity margin is recorded at the
    optimum (see ``_is_slack``) and only the removal of a genuinely active or
    violated constraint raises ``degeneracy`` -- the support scenarios are
    then reducible, more than one support list exists, and the lower risk
    bound is not certified. Removals of spuriously screened (strictly slack)
    constraints are discounted as solver dual noise.

    If the screened set fails to reproduce the
    optimum (inaccurate solver duals, or a degenerate instance), the support
    list is instead recovered by pruning the full scenario-constraint list.
    That recovery yields a valid support list but cannot certify it is of
    minimum cardinality, so ``degeneracy`` is raised conservatively and the
    two-sided lower risk bound should not be trusted in that case.

    Parameters
    ----------
    constraints : list
        Scenario constraints only (one constraint object per scenario).
    non_risk_constraints : list
        Hard constraints; always enforced in re-solves, never counted.
    prob : cvxpy.Problem
        The solved CVXPY problem instance.
    objective : cvxpy.Minimize
        The CVXPY objective function.
    solver : str or None, optional
        CVXPY solver name.
    threshold : float, optional
        Dual-value threshold for screening candidates. Default ``1e-8``.
    sol_tol : float, optional
        Relative tolerance for judging whether a re-solve reproduces the
        reference optimal value. Set above solver value-reproducibility
        (~1e-8 relative) and below the smallest meaningful support
        contribution. Default ``1e-6``.

    Returns
    -------
    complexity : int
        Cardinality of the support list (*k* in the scenario approach).
    support : list
        The support list of CVXPY scenario constraint objects.
    degeneracy : bool
        ``True`` when the support list is not certified minimal -- either
        pruning removed a genuinely active or violated constraint (the support
        scenarios were reducible) or the dual screen failed and the list was
        recovered from the full constraint set. In both cases the lower risk
        bound is not certified.

    Raises
    ------
    ValueError
        If no valid support list can be determined.
    """
    degeneracy = False
    non_risk = list(non_risk_constraints) if isinstance(non_risk_constraints, (list, tuple)) \
        else [non_risk_constraints]
    ref = {"cost": prob.value, "tol": sol_tol}

    def is_candidate(constraint):
        dv = constraint.dual_value
        return dv is not None and np.max(np.abs(np.asarray(dv))) > threshold

    screened = [c for c in constraints if is_candidate(c)]
    # Activity margins must be captured now: the re-solves inside
    # test_support/_prune overwrite the shared variable values.
    spurious = {id(c) for c in screened if _is_slack(c)}

    if test_support(objective, screened, non_risk, ref, solver=solver):
        support = _prune(objective, screened, non_risk, ref, solver=solver)
        support_ids = {id(c) for c in support}
        removed = [c for c in screened if id(c) not in support_ids]
        genuine = [c for c in removed if id(c) not in spurious]
        if genuine:
            # PRUNE removed a constraint that was genuinely active or violated
            # at the optimum: the support scenarios are reducible, more than
            # one support list exists, and the instance is degenerate (its
            # lower risk bound is not certified). Removing a spuriously
            # screened constraint (strictly slack, near-threshold dual left by
            # the solver) says nothing about degeneracy and is discounted.
            degeneracy = True
            print(f"PRUNE removed {len(genuine)} active/violated support "
                  f"scenario(s) ({len(screened)} screened -> {len(support)} "
                  f"irreducible); flagging degeneracy (lower risk bound not "
                  f"certified).")
        elif removed:
            print(f"Discarded {len(removed)} spuriously screened constraint(s) "
                  f"(strictly slack, near-threshold duals); remaining support "
                  f"scenarios are irreducible: non-degenerate.")
    else:
        # The dual-screened candidates did not reproduce the optimum, so the
        # support list is recovered by pruning the full constraint set. Pruning
        # the full set gives a valid but not necessarily minimal support list,
        # and cannot tell us whether the true support list is unique -- whether
        # the screen failed from inaccurate duals or genuine degeneracy. We
        # therefore raise the degeneracy flag conservatively: the two-sided
        # lower risk bound is not trustworthy in this case.
        degeneracy = True
        print("Dual-screened candidates did not reproduce the optimum; "
              "recovering the support list from the full constraint set "
              "(flagging degeneracy: lower risk bound not certified).")
        support = _prune(objective, list(constraints), non_risk, ref, solver=solver)
        if not test_support(objective, support, non_risk, ref, solver=solver):
            raise ValueError("Error calculating support list! Perhaps try another solver?")

    return len(support), support, degeneracy
