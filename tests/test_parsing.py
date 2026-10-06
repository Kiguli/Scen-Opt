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


def test_numeric_entries_are_accepted():
    """Plain numbers (from JSON or MAT files) work like their string form."""
    np.testing.assert_allclose(generate_matrix('[[0], [1]]'), [[0.0], [1.0]])
    f = generate_matrix_function('[[-1, 2.5], ["delta[0]", 0]]')
    np.testing.assert_allclose(f(np.array([3.0])), [[-1.0, 2.5], [3.0, 0.0]])
    t = generate_tensor_function('{"0": [[1, "delta[0]"]]}')
    np.testing.assert_allclose(t(np.array([2.0]))["0"], [[1.0, 2.0]])


def test_constant_tensor_accepts_expressions():
    """E matrices accept constant expressions like '1/2', as other matrices do."""
    E = generate_tensor('{"0": [["1/2", "0"], ["0", "-1"]]}')
    np.testing.assert_allclose(E["0"], [[0.5, 0.0], [0.0, -1.0]])


def test_scalar_scenario_is_indexable():
    """A scalar scenario (from a 1-D scenarios array) has one component, delta[0]."""
    f = generate_matrix_function('[["delta[0]", "-delta[0]"]]')
    np.testing.assert_allclose(f(0.25), [[0.25, -0.25]])


def test_sympy_style_names():
    """SymPy-style functions, constants and ^ work without the math. prefix."""
    from src.parsing import safe_eval
    d = np.array([0.5, 2.0])
    assert math.isclose(safe_eval("sin(delta[0])", d), math.sin(0.5))
    assert math.isclose(safe_eval("delta[1]^2 + sqrt(delta[1])", d), 4 + math.sqrt(2))
    assert math.isclose(safe_eval("Abs(-pi) + ln(E)", d), math.pi + 1)
    assert math.isclose(safe_eval("math.exp(delta[0])", d), math.exp(0.5))  # still accepted


def test_disallowed_names_are_rejected():
    """Names outside the whitelist are still rejected."""
    import pytest
    from src.parsing import safe_eval
    for bad in ["__import__('os')", "open('x')", "eval('1')", "sin"]:
        with pytest.raises(ValueError):
            safe_eval(bad, np.array([1.0]))


def test_unreadable_entries_give_value_errors():
    """Malformed matrices, overflow, division by zero and non-numeric entries raise ValueError."""
    import pytest
    from src.parsing import safe_eval
    with pytest.raises(ValueError, match="could not read the matrix"):
        generate_matrix('[["1", "2"')
    for bad in ["exp(1000)", "1/0", "1 +"]:
        with pytest.raises(ValueError):
            safe_eval(bad, np.array([1.0]))
    with pytest.raises(ValueError):
        safe_eval(True)
