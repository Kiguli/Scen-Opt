import concurrent.futures
import cvxpy as cp
import numpy as np
import pandas as pd

SOLVER_CAPABILITIES = {
    'CLARABEL': ['LP', 'QP', 'SDP'], 'SCS': ['LP', 'QP', 'SDP'],
    'OSQP': ['LP', 'QP'], 'ECOS': ['LP', 'QP', 'SDP'],
    'CVXOPT': ['LP', 'QP', 'SDP'], 'GLOP': ['LP'], 'GLPK': ['LP'],
    'GLPK_MI': ['LP'], 'SCIPY': ['LP', 'QP'], 'HIGHS': ['LP', 'QP'],
    'SCIP': ['LP', 'QP', 'SDP'], 'CBC': ['LP', 'QP'],
    'DAQP': ['LP', 'QP'], 'PIQP': ['LP', 'QP'],
    'PROXQP': ['LP', 'QP'], 'QPALM': ['LP', 'QP'],
    'MOSEK': ['LP', 'QP', 'SDP'], 'GUROBI': ['LP', 'QP', 'SDP'],
    'CPLEX': ['LP', 'QP', 'SDP'], 'SDPA': ['SDP'],
}

def get_solvers():
    """
        Retrieves installed solvers and their supported problem types.

        Returns:
        dict: Mapping of solver name -> list of supported types ('LP', 'QP', 'SDP').

        Full list: https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver
    """
    installed = cp.installed_solvers()
    return {s: SOLVER_CAPABILITIES.get(s, ['LP', 'QP', 'SDP']) for s in installed}

def get_norm_types():
    """
        Retrieves a list of available norm types in cvxpy.

        Returns:
        list: A list of norm types including 1, 2, 'inf', 'fro', and 'nuc'.
    """
    norm_types = [1, 2, "inf", "fro", "nuc"]
    return norm_types

def load_file(file_path):
    """
    Reads a file (TXT, CSV, XLSX, JSON, etc.) and converts it into a NumPy array.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        np.ndarray: NumPy array containing the file's data.
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
    """
        Solves a convex optimization problem with optional robust and regularization constraints.

        Parameters:
        prob: CVXPY problem instance for optimal solution.
        objective: CVXPY objective function.
        active: list of active constraints.
        rho (float): Penalty parameter for the slack variables in the objective function.
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

        if rho != 0:
            assert np.allclose(prob.variables()[1].value, prob2.variables()[1].value), f"Solutions for zeta differ: {prob.variables()[1].value} vs {prob2.variables()[1].value}"

        # Return True if all assertions pass
        return True

    except AssertionError as e:
        print(f"Assertion failed: {e}")
        # Return False if any assertion fails
        return False



def get_active(constraints, non_risk_constraints, prob, objective, rho=0.0, solver=None, threshold=1e-8):
    """
        Finds the active constraints of the Convex optimization problem.

        Parameters:
        constraints (list): list of constraints.
        non_risk_constraints (list): list of constraints that should not be included in the error calculation.
        prob: CVXPY problem instance for optimal solution.
        objective: CVXPY objective function.
        rho (float,optional): Penalty parameter for the slack variables in the objective function. Default is 0.0.
        solver (str, optional): The solver to use for the optimization problem. Default is None.
        threshold (float, optional): Threshold for the solver checking active constraints. Default is 1e-8.
        Returns:
        active (list): list of active constraints.
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