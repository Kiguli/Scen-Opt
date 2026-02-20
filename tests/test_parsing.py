import math
import numpy as np

from src.parsing import (
    generate_matrix_function,
    generate_matrix,
    generate_tensor_function,
    generate_tensor,
)


def test_generate_matrix_function_constant():
    """Constant matrix expression (no delta dependency)."""
    fn = generate_matrix_function('[["1", "2"], ["3", "4"]]')
    result = fn([0, 0])
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))


def test_generate_matrix_function_with_delta():
    """Matrix expression parameterized by delta."""
    fn = generate_matrix_function('[["delta[0]", "delta[1]"], ["0", "1"]]')
    result = fn([5, 7])
    np.testing.assert_array_equal(result, np.array([[5, 7], [0, 1]]))


def test_generate_matrix_function_with_math():
    """Matrix expression using math functions."""
    fn = generate_matrix_function('[["math.sin(delta[0])", "math.cos(delta[0])"]]')
    result = fn([0.0])
    np.testing.assert_array_almost_equal(result, np.array([[0.0, 1.0]]))


def test_generate_matrix():
    """Static matrix expression (no delta)."""
    result = generate_matrix('[["3", "4"], ["5", "6"]]')
    np.testing.assert_array_equal(result, np.array([[3, 4], [5, 6]]))


def test_generate_tensor_function():
    """Dict of delta-parameterized matrix expressions."""
    fn = generate_tensor_function(
        '{"0": [["delta[0]", "0"], ["0", "delta[0]"]], "1": [["1", "0"], ["0", "0"]]}'
    )
    result = fn([3.0])
    np.testing.assert_array_equal(result["0"], np.array([[3.0, 0], [0, 3.0]]))
    np.testing.assert_array_equal(result["1"], np.array([[1, 0], [0, 0]]))


def test_generate_tensor():
    """Static dict of matrix expressions (no delta)."""
    result = generate_tensor(
        '{"0": [["1", "0"], ["0", "1"]], "1": [["2", "0"], ["0", "3"]]}'
    )
    np.testing.assert_array_equal(result["0"], np.array([[1, 0], [0, 1]]))
    np.testing.assert_array_equal(result["1"], np.array([[2, 0], [0, 3]]))
