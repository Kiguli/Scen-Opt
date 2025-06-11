def find_active(constraints, sol):
    """
            Solves a linear programming problem with optional robust and regularization constraints.

            Parameters:
            constraints (numpy.ndarray): Collected constraints that we check are active.

            Returns:
                - active (numpy.ndarray): the set of active constraints.
            """
    active = []

    #TODO: 1) check if the solution touches the constraint, e.g. constraint is an equality not inequality.

    #TODO: 2) check within a tolerance?

    # TODO: 3) check only using active constraints gives same solution (should be true)

    #TODO: 4) remove each active constraint in turn and see if the solution changes

    #TODO: 5) check the new support list gives the solution if yes, done and both bounds hold!

    #TODO: 6) if not, sequentially remove elements keeping only ones where the solution changes,
    # then return warning that there is evidence lower bound does not hold

    return active