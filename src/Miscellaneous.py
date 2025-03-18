import cvxpy as cp

def get_solvers():
    solver_list = cp.installed_solvers()
    return solver_list

def get_norm_types():
    norm_types = [1,2,"inf","fro","nuc", "or any positive integer"]
    return norm_types