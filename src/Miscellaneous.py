import cvxpy as cp

def get_solvers():
    solver_list = cp.installed_solvers()
    print(solver_list)
    return solver_list