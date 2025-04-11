import ast  # Safer than eval for parsing

def generate_matrix_function(expr_matrix_str):
    # Convert string to list of lists
    expr_matrix = ast.literal_eval(expr_matrix_str)

    def A(delta):
        # Evaluate each expression in the matrix
        return [
            [eval(expr, {"delta": delta, "math": __import__('math')}) for expr in row]
            for row in expr_matrix
        ]

    return A

# Your example string:
s = '[["delta[0]", "delta[1]"], ["0", "0"]]'
A = generate_matrix_function(s)

print("A([1, 2]) =", A([1, 2]))  # Expect [[1, 2], [0, 0]]
print("A([5, 7]) =", A([5, 7]))  # Expect [[5, 7], [0, 0]]