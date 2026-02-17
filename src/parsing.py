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
