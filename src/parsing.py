import ast
import math
import operator
import numpy as np

# Untrusted matrix/tensor cells arrive from the public web form, so they are
# evaluated by a restricted AST walker instead of eval(): only numeric
# literals, + - * / // % ** (and ^ as a power, as in SymPy), unary +/-, the
# name `delta` (indexed), the functions and constants below, and
# `math.<fn>(...)` are permitted. Anything else raises ValueError, which
# closes the remote-code-execution path that eval() with an open namespace
# would otherwise expose.
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# SymPy-style names, so cells can say sin(delta[0]) instead of math.sin(delta[0]).
_FUNCTIONS = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'exp': math.exp, 'log': math.log, 'ln': math.log, 'log10': math.log10, 'log2': math.log2,
    'sqrt': math.sqrt, 'abs': abs, 'Abs': abs,
    'floor': math.floor, 'ceiling': math.ceil, 'ceil': math.ceil,
    'min': min, 'Min': min, 'max': max, 'Max': max,
}
_CONSTANTS = {'pi': math.pi, 'e': math.e, 'E': math.e}


def safe_eval(expr, delta=None):
    """Safely evaluate a scalar arithmetic expression over `delta` and `math`.

    Parameters
    ----------
    expr : str or number
        Expression such as ``"delta[0]"``, ``"-delta[3]"``, ``"0.5"``,
        ``"sin(delta[1])"``, ``"delta[0]^2"`` or ``"math.sin(delta[1])"``.
        A plain number (e.g. from a JSON or MAT file) is returned as a float.
    delta : sequence, scalar or None
        The uncertainty vector referenced by ``delta[i]`` (a scalar is
        treated as a vector with one component); ``None`` for constant
        expressions.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the expression uses any construct outside the permitted subset.
    """
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"disallowed constant: {node.value!r}")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name):
            if node.id == "delta":
                if delta is None:
                    raise ValueError("delta is not available in this expression")
                return delta
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Subscript):
            base, index = ev(node.value), int(ev(node.slice))
            if np.ndim(base) == 0:  # a scalar scenario has one component
                base = [base]
            try:
                return base[index]
            except IndexError:
                raise ValueError(f"delta[{index}] does not exist: the scenario has {len(base)} component(s)")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math" \
                    and not node.attr.startswith("_"):
                return getattr(math, node.attr)
            raise ValueError(f"disallowed attribute access: {node.attr}")
        if isinstance(node, ast.Call):
            func = ev(node.func)
            if node.keywords or not callable(func):
                raise ValueError("disallowed call")
            return func(*[ev(a) for a in node.args])
        raise ValueError(f"disallowed expression element: {type(node).__name__}")

    if isinstance(expr, (int, float, np.integer, np.floating)) and not isinstance(expr, bool):
        return float(expr)
    if not isinstance(expr, str):
        raise ValueError(f"matrix entries must be numbers or expressions, not {expr!r}")
    # ^ means power, as in SymPy. It is replaced before parsing so that it gets
    # the precedence of ** (Python's ^ is XOR, which binds more loosely than +).
    try:
        value = ev(ast.parse(expr.replace("^", "**"), mode="eval"))
    except SyntaxError:
        raise ValueError(f"invalid expression {expr!r}")
    except (TypeError, ZeroDivisionError, OverflowError) as error:
        # e.g. a function applied to the wrong arguments, 1/0, or exp(1000)
        raise ValueError(f"cannot evaluate {expr!r}: {error}")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"expression {expr!r} does not evaluate to a number")
    return value


def _read_literal(text):
    """Read a matrix (list of rows) or a dict of matrices from its JSON/Python text."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(f"could not read the matrix {text[:60]!r}: check the brackets and commas")


def generate_matrix_function(expr_matrix_str):
    """Create a function that evaluates a matrix expression for a given delta vector."""
    expr_matrix = _read_literal(expr_matrix_str)

    def matrix_function(delta):
        return np.array([
            [safe_eval(expr, delta) for expr in row]
            for row in expr_matrix
        ])

    return matrix_function


def generate_matrix(expr_matrix_str):
    """Evaluate a matrix expression string (no delta dependency) into a numpy array."""
    expr_matrix = _read_literal(expr_matrix_str)
    return np.array([
        [safe_eval(expr) for expr in row]
        for row in expr_matrix
    ])


def generate_tensor_function(expr_matrix_str):
    """Create a function that evaluates a dict of matrix expressions for a given delta vector."""
    expr_dict = _read_literal(expr_matrix_str)

    def tensor_function(delta):
        return {
            key: np.array([
                [safe_eval(expr, delta) for expr in row]
                for row in expr_dict[key]
            ])
            for key in expr_dict
        }

    return tensor_function


def generate_tensor(expr_matrix_str):
    """Evaluate a dict of matrix expression strings (no delta dependency) into numpy arrays."""
    expr_dict = _read_literal(expr_matrix_str)
    return {
        key: np.array([[safe_eval(cell) for cell in row] for row in expr_dict[key]])
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
