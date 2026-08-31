import cvxpy as cp
import numpy as np
import pandas as pd

SOLVER_CAPABILITIES = {
    'CLARABEL': ['LP', 'QP', 'SDP'], 'SCS': ['LP', 'QP', 'SDP'],
    # OSQP and SCIPY are removed from the LP list — both technically claim
    # LP support but fail on the scenario LPs we build (slack reformulation
    # produces degenerate QPs OSQP can't handle; SCIPY's wrapper hits
    # NotImplemented paths for our equality reductions).
    'OSQP': ['QP'], 'ECOS': ['LP', 'QP', 'SDP'],
    'CVXOPT': ['LP', 'QP', 'SDP'], 'GLOP': ['LP'], 'GLPK': ['LP'],
    'GLPK_MI': ['LP'], 'SCIPY': ['QP'], 'HIGHS': ['LP', 'QP'],
    'SCIP': ['LP', 'QP', 'SDP'], 'CBC': ['LP', 'QP'],
    'DAQP': ['LP', 'QP'], 'PIQP': ['LP', 'QP'],
    'PROXQP': ['LP', 'QP'], 'QPALM': ['LP', 'QP'],
    'MOSEK': ['LP', 'QP', 'SDP'], 'GUROBI': ['LP', 'QP', 'SDP'],
    'CPLEX': ['LP', 'QP', 'SDP'], 'SDPA': ['SDP'],
}

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
    solver_dict = {s: SOLVER_CAPABILITIES.get(s, ['LP', 'QP', 'SDP']) for s in installed}
    return solver_dict

def get_norm_types():
    """Retrieve available norm types for the regularization term.

    Returns
    -------
    list
        Supported CVXPY norm types: ``1`` (L1), ``2`` (L2), ``'inf'``
        (L-infinity), ``'fro'`` (Frobenius), ``'nuc'`` (nuclear).
    """
    norm_types = [1, 2, "inf", "fro", "nuc"]
    return norm_types

def load_file(file_path):
    """Read a data file and convert it to a NumPy array.

    Supports CSV, TXT, XLSX, and JSON formats.

    Parameters
    ----------
    file_path : str
        Path to the input file. Must have a supported extension
        (``.csv``, ``.txt``, ``.xlsx``, ``.json``).

    Returns
    -------
    numpy.ndarray or None
        Array containing the file data, or ``None`` if an error occurred.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path,header=None).values  # Read CSV using pandas and convert to NumPy
        elif file_path.endswith('.txt'):
            data = np.loadtxt(file_path)  # Read TXT using NumPy (default to expect floats)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, engine='openpyxl',header=None).values  # Read Excel file and convert to NumPy
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path, orient='record').values  # Read JSON and convert to NumPy
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
    constraints are captured -- then greedily pruned by re-solving and
    comparing the optimal value. Hard constraints are always enforced in
    re-solves and never counted. If the screened set fails to reproduce the
    optimum (inaccurate solver duals or degeneracy), the search restarts from
    the full scenario-constraint list.

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
        ``True`` if the recovery procedure was used.

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

    if test_support(objective, screened, non_risk, ref, solver=solver):
        support = _prune(objective, screened, non_risk, ref, solver=solver)
    else:
        degeneracy = True
        print("Screened candidates do not reproduce the optimum "
              "(inaccurate duals or degeneracy); recovering from the full list.")
        support = _prune(objective, list(constraints), non_risk, ref, solver=solver)
        if not test_support(objective, support, non_risk, ref, solver=solver):
            raise ValueError("Error calculating support list! Perhaps try another solver?")

    return len(support), support, degeneracy
