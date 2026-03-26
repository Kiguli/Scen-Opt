import ast
import math
import numpy as np


def generate_matrix_function(expr_matrix_str):
    """Create a function that evaluates a matrix expression for a given delta vector."""
    expr_matrix = ast.literal_eval(expr_matrix_str)

    def matrix_function(delta):
        return np.array([
            [eval(expr, {"delta": delta, "math": math}) for expr in row]
            for row in expr_matrix
        ])

    return matrix_function


def generate_matrix(expr_matrix_str):
    """Evaluate a matrix expression string (no delta dependency) into a numpy array."""
    expr_matrix = ast.literal_eval(expr_matrix_str)
    return np.array([
        [eval(expr, {"math": math}) for expr in row]
        for row in expr_matrix
    ])


def generate_tensor_function(expr_matrix_str):
    """Create a function that evaluates a dict of matrix expressions for a given delta vector."""
    expr_dict = ast.literal_eval(expr_matrix_str)

    def tensor_function(delta):
        return {
            key: np.array([
                [eval(expr, {"delta": delta, "math": math}) for expr in row]
                for row in expr_dict[key]
            ])
            for key in expr_dict
        }

    return tensor_function


def generate_tensor(expr_matrix_str):
    """Evaluate a dict of matrix expression strings (no delta dependency) into numpy arrays."""
    expr_dict = ast.literal_eval(expr_matrix_str)
    return {
        key: np.array([[float(cell) for cell in row] for row in expr_dict[key]])
        for key in expr_dict
    }


def generate_numeric_A_b(rows_A, n_x):
    """Create callables that extract A and b matrices from a flattened scenario row.

    In numeric mode each scenario row is the row-wise flattening of A_i
    concatenated with the row-wise flattening of b_i:
    ``[A_i_flat | b_i_flat]`` with length ``rows_A * n_x + rows_A``.
    """
    a_len = rows_A * n_x

    def A_d(delta):
        return delta[:a_len].reshape(rows_A, n_x)

    def b_d(delta):
        return delta[a_len:].reshape(rows_A, 1)

    return A_d, b_d


def generate_numeric_F(lmi_size, n_x):
    """Create a callable that extracts F_0, ..., F_d LMI matrices from a flattened scenario row.

    In numeric mode each scenario row contains ``n_x + 1`` symmetric matrices
    (F_0, F_1, ..., F_{n_x}), each of size ``lmi_size x lmi_size``, flattened
    row-wise and concatenated: ``[F_0_flat | F_1_flat | ... | F_d_flat]``
    with total length ``(n_x + 1) * lmi_size ** 2``.
    """
    mat_elems = lmi_size * lmi_size

    def F_d(delta):
        return {
            str(j): delta[j * mat_elems:(j + 1) * mat_elems].reshape(lmi_size, lmi_size)
            for j in range(n_x + 1)
        }

    return F_d
