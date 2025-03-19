import cvxpy as cp
import numpy as np
import pandas as pd

def get_solvers():
    """
        Retrieves a list of installed solvers available in cvxpy.

        Returns:
        list: A list of strings representing the names of the installed solvers.
    """
    solver_list = cp.installed_solvers()
    return solver_list

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
