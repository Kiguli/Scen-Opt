import concurrent.futures
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

def test_active(prob, objective, active, rho=0.0, solver=None):
    """Validate that a constraint subset produces the same optimal solution.

    Re-solves the problem using only the given constraints and checks that the
    objective value and decision variables match the original solution. Used
    internally by :func:`get_active` to verify support set correctness.

    Parameters
    ----------
    prob : cvxpy.Problem
        The original solved CVXPY problem instance.
    objective : cvxpy.Minimize
        The CVXPY objective function.
    active : list
        List of candidate active CVXPY constraint objects.
    rho : float, optional
        Slack variable penalty. Default is ``0.0``.
    solver : str or None, optional
        CVXPY solver name. Default is ``None``.

    Returns
    -------
    bool
        ``True`` if the reduced problem matches the original solution,
        ``False`` otherwise.
    """
    try:
        prob2 = cp.Problem(objective, active)
        prob2.solve(solver=solver)

        # Assert that the objective values are the same
        assert np.isclose(prob.value, prob2.value), f"Objective values differ: {prob.value} vs {prob2.value}"

        # Assert that the solutions are the same
        assert np.allclose(prob.variables()[0].value,
                           prob2.variables()[0].value), f"Solutions for x differ: {prob.variables()[0].value} vs {prob2.variables()[0].value}"

        if rho != 0:
            assert np.allclose(prob.variables()[1].value, prob2.variables()[1].value), f"Solutions for zeta differ: {prob.variables()[1].value} vs {prob2.variables()[1].value}"

        # Return True if all assertions pass
        return True

    except AssertionError as e:
        print(f"Assertion failed: {e}")
        # Return False if any assertion fails
        return False



def get_active(constraints, non_risk_constraints, prob, objective, rho=0.0, solver=None, threshold=1e-8):
    """Identify the support (active) constraints that define the optimal solution.

    Uses dual variable analysis with parallel constraint testing. A constraint
    is initially flagged as active when its dual value exceeds *threshold*.
    The candidate set is then validated and pruned to find a minimal support
    set. Degeneracy is handled by iteratively removing constraints until the
    minimal valid set is found.

    Parameters
    ----------
    constraints : list
        All scenario constraints from the optimization problem.
    non_risk_constraints : list
        Hard constraints to exclude from the support complexity count.
    prob : cvxpy.Problem
        The solved CVXPY problem instance.
    objective : cvxpy.Minimize
        The CVXPY objective function.
    rho : float, optional
        Slack variable penalty parameter. Default is ``0.0``.
    solver : str or None, optional
        CVXPY solver name. Default is ``None``.
    threshold : float, optional
        Dual value threshold for detecting active constraints.
        Default is ``1e-8``.

    Returns
    -------
    complexity : int
        Number of active scenario constraints (*k* in the scenario approach).
    active : list
        List of active CVXPY constraint objects.
    degeneracy : bool
        ``True`` if degeneracy was detected during support identification.

    Raises
    ------
    ValueError
        If a valid support set cannot be determined.
    """

    #TODO: make threshold a global parameter?
    degeneracy = False

    def is_active(constraint, threshold):
        dv = constraint.dual_value
        if dv is None:
            return False
        # Handle both vector and matrix dual values (for SDP)
        if hasattr(dv, 'flatten'):
            return np.max(np.abs(dv.flatten())) > threshold
        return max(dv) > threshold

    with concurrent.futures.ThreadPoolExecutor() as executor:
       results = list(executor.map(lambda c: is_active(c,threshold), constraints))

    active = [c for c, is_act in zip(constraints, results) if is_act]

    # If solution changes then likely to have degeneracy
    if not test_active(prob, objective, active, rho=rho, solver=solver):
        degeneracy = True
        print("Active constraints are not valid. Lower bound not viable likely due to degeneracy.")
        # loop through all constraints and make a support list from them
        active = constraints.copy()
        # Iteratively remove constraints from active if test_active returns True when they are removed
        i = 0
        while i < len(active):
            temp_active = active[:i] + active[i + 1:]
            if test_active(prob, objective, temp_active, rho=rho, solver=solver):
                # Remove constraint and don't increment i
                active.pop(i)
            else:
                i += 1
        # At the end, active contains only constraints whose removal makes test_active return False
        if not test_active(prob, objective, active, rho=rho, solver=solver):
            raise ValueError("Error calculating support list! Perhaps try another solver?")
    else:
        # If solution does not change then likely to be non-degenerate, check for true support list as solvers can be incorrect
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda a: test_active(prob, objective, [constraint for constraint in active if constraint != a],
                                      rho=rho, solver=solver),
                active
            ))
        drop = [a for a, should_drop in zip(active, results) if should_drop]
        if not test_active(prob, objective, [constraint for constraint in active if constraint not in drop], rho=rho,solver=solver):
            degeneracy = True
            print("Reduced version of active constraints are not valid. Lower bound not viable likely due to degeneracy.")  # TODO: in theory can have degeneracy here too! SVM p=0.1 fails here!!
            # Iteratively remove constraints from active if test_active returns True when they are removed
            i = 0
            while i < len(active):
                temp_active = active[:i] + active[i + 1:]
                if test_active(prob, objective, temp_active, rho=rho, solver=solver):
                    # Remove constraint and don't increment i
                    active.pop(i)
                else:
                    i += 1
            # At the end, active contains only constraints whose removal makes test_active return False
            if not test_active(prob, objective, active, rho=rho, solver=solver):
                raise ValueError("Error calculating support list! Perhaps try another solver?")
        else:
            active = [constraint for constraint in active if constraint not in drop]

    if non_risk_constraints in active:
        active.remove(non_risk_constraints)
        complexity = len(active)
    else:
        complexity = len(active)

    return complexity, active, degeneracy