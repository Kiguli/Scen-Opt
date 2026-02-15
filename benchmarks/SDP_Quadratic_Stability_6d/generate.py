#!/usr/bin/env python3
"""
Generate data for Quadratic_Stability_6d (SDP) Benchmark

This script generates the constraint matrices for a 3x3 coupled oscillator
LPV system with 2D parameter uncertainty.

System Model:
    dx/dt = A(delta) * x

    A(delta) = A_0 + delta[0]*A_1 + delta[1]*A_2

    where delta = [delta[0], delta[1]] in [-1, 1]^2

Lyapunov Stability Condition:
    Find symmetric P > 0 such that:
    A(delta)' P + P A(delta) < 0   for all delta in [-1,1]^2

Decision Variables (6 for 3x3 symmetric P):
    P = [[p11, p12, p13],
         [p12, p22, p23],
         [p13, p23, p33]]

    x = [p11, p12, p13, p22, p23, p33]

The SDP formulation uses:
    F_0(delta) + sum(x_i * F_i(delta)) <= 0   (Lyapunov condition)
    E_0 + sum(x_i * E_i) <= 0                  (P >= epsilon*I)

where F_i matrices encode the Lyapunov equation structure.
"""

import numpy as np
import os

def generate_benchmark(n_scenarios=500, seed=42):
    """Generate all files for the Quadratic_Stability_6d benchmark."""

    np.random.seed(seed)
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # ==========================================================
    # Define the 3x3 LPV System (Coupled Oscillators)
    # ==========================================================

    # Nominal system matrix A_0 (stable coupled oscillator)
    # Two coupled second-order modes with moderate damping
    A_0 = np.array([
        [-0.3,   1.0,   0.0],
        [-2.0,  -1.0,   0.2],
        [ 0.2,   0.0,  -1.2]
    ])

    # Parameter perturbation A_1 (stiffness/coupling uncertainty)
    A_1 = np.array([
        [0.0,   0.0,   0.0],
        [0.5,   0.15,  0.0],
        [0.0,   0.0,   0.3]
    ])

    # Parameter perturbation A_2 (damping uncertainty)
    A_2 = np.array([
        [0.0,   0.0,   0.0],
        [0.0,   0.3,   0.15],
        [0.15,  0.0,   0.45]
    ])

    # Verify nominal stability
    eigs_nominal = np.linalg.eigvals(A_0)
    print(f"Nominal system eigenvalues: {eigs_nominal}")
    assert all(np.real(eigs_nominal) < 0), "Nominal system must be stable!"

    # Check stability at corners of parameter space
    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for d0, d1 in corners:
        A_corner = A_0 + d0 * A_1 + d1 * A_2
        eigs = np.linalg.eigvals(A_corner)
        print(f"Corner ({d0:+d}, {d1:+d}) eigenvalues: {np.round(eigs, 4)}")
        if any(np.real(eigs) >= 0):
            print(f"  WARNING: Unstable at corner ({d0}, {d1})")

    # ==========================================================
    # Generate Scenarios (2D uniform sampling)
    # ==========================================================

    print(f"\nGenerating {n_scenarios} scenarios...")
    scenarios = np.random.uniform(-1, 1, (n_scenarios, 2))

    np.savetxt(os.path.join(data_dir, 'scenarios.csv'),
               scenarios, delimiter=',', fmt='%.10f')
    print(f"Saved scenarios to data/scenarios.csv")

    # ==========================================================
    # Build F matrices for Lyapunov condition
    # ==========================================================
    #
    # The Lyapunov equation A'P + PA = Q can be written as:
    # For symmetric P with elements [p11, p12, p13, p22, p23, p33],
    # we need to express A'P + PA in matrix form.
    #
    # F_0(delta) is the part independent of P (should be non-zero for well-posedness)
    # F_i(delta) for i=1,...,6 are the coefficients for each p_ij

    def compute_lyapunov_matrices(A):
        """
        For A(delta), compute the F_0 and F_1,...,F_6 such that:
        A'P + PA = F_0 + p11*F_1 + p12*F_2 + p13*F_3 + p22*F_4 + p23*F_5 + p33*F_6

        For the SDP, we need A'P + PA <= 0 (negative semidefinite)
        """
        n = 3  # 3x3 system

        # The (i,j) entry of A'P + PA is:
        # sum_k (A[k,i]*P[k,j] + P[i,k]*A[k,j])

        # For a symmetric P, the independent elements are:
        # p11=P[0,0], p12=P[0,1]=P[1,0], p13=P[0,2]=P[2,0],
        # p22=P[1,1], p23=P[1,2]=P[2,1], p33=P[2,2]

        # We'll compute the 3x3 matrix M = A'P + PA symbolically
        # and extract coefficients for each p_ij

        F = {}

        # F_0 is zero (no constant term in A'P + PA)
        F[0] = np.zeros((n, n))

        # Coefficient matrices for each p_ij
        # p11 = P[0,0]
        F[1] = np.zeros((n, n))
        F[1][0, 0] = 2 * A[0, 0]
        F[1][0, 1] = A[1, 0]
        F[1][1, 0] = A[1, 0]
        F[1][0, 2] = A[2, 0]
        F[1][2, 0] = A[2, 0]

        # p12 = P[0,1] = P[1,0]
        F[2] = np.zeros((n, n))
        F[2][0, 0] = 2 * A[0, 1]
        F[2][0, 1] = A[0, 0] + A[1, 1]
        F[2][1, 0] = A[0, 0] + A[1, 1]
        F[2][1, 1] = 2 * A[1, 0]
        F[2][0, 2] = A[2, 1]
        F[2][2, 0] = A[2, 1]
        F[2][1, 2] = A[2, 0]
        F[2][2, 1] = A[2, 0]

        # p13 = P[0,2] = P[2,0]
        F[3] = np.zeros((n, n))
        F[3][0, 0] = 2 * A[0, 2]
        F[3][0, 1] = A[1, 2]
        F[3][1, 0] = A[1, 2]
        F[3][0, 2] = A[0, 0] + A[2, 2]
        F[3][2, 0] = A[0, 0] + A[2, 2]
        F[3][1, 2] = A[1, 0]
        F[3][2, 1] = A[1, 0]
        F[3][2, 2] = 2 * A[2, 0]

        # p22 = P[1,1]
        F[4] = np.zeros((n, n))
        F[4][0, 1] = A[0, 1]
        F[4][1, 0] = A[0, 1]
        F[4][1, 1] = 2 * A[1, 1]
        F[4][1, 2] = A[2, 1]
        F[4][2, 1] = A[2, 1]

        # p23 = P[1,2] = P[2,1]
        F[5] = np.zeros((n, n))
        F[5][0, 1] = A[0, 2]
        F[5][1, 0] = A[0, 2]
        F[5][0, 2] = A[0, 1]
        F[5][2, 0] = A[0, 1]
        F[5][1, 1] = 2 * A[1, 2]
        F[5][1, 2] = A[1, 1] + A[2, 2]
        F[5][2, 1] = A[1, 1] + A[2, 2]
        F[5][2, 2] = 2 * A[2, 1]

        # p33 = P[2,2]
        F[6] = np.zeros((n, n))
        F[6][0, 2] = A[0, 2]
        F[6][2, 0] = A[0, 2]
        F[6][1, 2] = A[1, 2]
        F[6][2, 1] = A[1, 2]
        F[6][2, 2] = 2 * A[2, 2]

        return F

    # Compute F matrices at different parameter values to create expressions
    # F_i(delta) = F_i(A_0) + delta[0]*F_i(A_1) + delta[1]*F_i(A_2)

    F_A0 = compute_lyapunov_matrices(A_0)
    F_A1 = compute_lyapunov_matrices(A_1)
    F_A2 = compute_lyapunov_matrices(A_2)

    # Write F_i.csv files with delta expressions
    print("\nWriting F matrix files...")

    n = 3  # matrix dimension
    n_vars = 6  # decision variables

    for i in range(n_vars + 1):
        filepath = os.path.join(data_dir, f'F_{i}.csv')
        with open(filepath, 'w') as f:
            for row in range(n):
                row_entries = []
                for col in range(n):
                    # Build expression: base + delta[0]*coef1 + delta[1]*coef2
                    base = F_A0[i][row, col]
                    coef1 = F_A1[i][row, col]
                    coef2 = F_A2[i][row, col]

                    parts = []
                    if abs(base) > 1e-12:
                        parts.append(f"{base:.10f}")
                    if abs(coef1) > 1e-12:
                        if coef1 > 0 and parts:
                            parts.append(f"+{coef1:.10f}*delta[0]")
                        else:
                            parts.append(f"{coef1:.10f}*delta[0]")
                    if abs(coef2) > 1e-12:
                        if coef2 > 0 and parts:
                            parts.append(f"+{coef2:.10f}*delta[1]")
                        else:
                            parts.append(f"{coef2:.10f}*delta[1]")

                    if not parts:
                        expr = "0"
                    else:
                        expr = "".join(parts)

                    row_entries.append(expr)
                f.write(",".join(row_entries) + "\n")
        print(f"  Saved F_{i}.csv")

    # ==========================================================
    # Build E matrices for P >= epsilon*I (positive definiteness)
    # ==========================================================

    # Hard constraint: -P + epsilon*I <= 0 (i.e., P >= epsilon*I)
    # E_0 + sum(x_i * E_i) <= 0
    #
    # We want: -p11 + eps <= 0, -p22 + eps <= 0, -p33 + eps <= 0
    # And the matrix should be P - eps*I >= 0
    #
    # Actually for SDP: we need F_0 + sum(x_i F_i) <= 0 (neg semidef)
    # Hard constraint E should enforce P > 0
    #
    # E_0 = epsilon * I (positive for the constraint to be binding)
    # E_i = -contribution of p_ij to the identity test

    epsilon = 0.01  # Minimum eigenvalue of P

    # E_0: baseline (epsilon * I gives us room)
    E = {}
    E[0] = epsilon * np.eye(n)

    # E_1 through E_6: coefficients for -P (since we want P >= eps*I means -P <= -eps*I)
    # The constraint is E_0 + sum(x_i * E_i) <= 0
    # With E_0 = eps*I and E_i = -coefficient of p_ij in P

    # For p11: appears in P[0,0]
    E[1] = np.zeros((n, n))
    E[1][0, 0] = -1.0

    # For p12: appears in P[0,1] and P[1,0]
    E[2] = np.zeros((n, n))
    E[2][0, 1] = -1.0
    E[2][1, 0] = -1.0

    # For p13: appears in P[0,2] and P[2,0]
    E[3] = np.zeros((n, n))
    E[3][0, 2] = -1.0
    E[3][2, 0] = -1.0

    # For p22: appears in P[1,1]
    E[4] = np.zeros((n, n))
    E[4][1, 1] = -1.0

    # For p23: appears in P[1,2] and P[2,1]
    E[5] = np.zeros((n, n))
    E[5][1, 2] = -1.0
    E[5][2, 1] = -1.0

    # For p33: appears in P[2,2]
    E[6] = np.zeros((n, n))
    E[6][2, 2] = -1.0

    # Write E_i.csv files (constant matrices, no delta dependence)
    print("\nWriting E matrix files...")
    for i in range(n_vars + 1):
        filepath = os.path.join(data_dir, f'E_{i}.csv')
        np.savetxt(filepath, E[i], delimiter=',', fmt='%.10f')
        print(f"  Saved E_{i}.csv")

    # ==========================================================
    # Write objective function files
    # ==========================================================

    # Minimize trace(P) = p11 + p22 + p33
    # c = [1, 0, 0, 1, 0, 1] (coefficients for x = [p11, p12, p13, p22, p23, p33])
    c = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    np.savetxt(os.path.join(data_dir, 'c.csv'), c, fmt='%.10f')
    print("\nSaved data/c.csv (objective: minimize trace(P))")

    # Q matrix for quadratic regularization (small for numerical stability)
    Q = 0.001 * np.eye(n_vars)
    np.savetxt(os.path.join(data_dir, 'Q.csv'), Q, delimiter=',', fmt='%.10f')
    print("Saved data/Q.csv (regularization matrix)")

    # ==========================================================
    # Summary
    # ==========================================================

    print("\n" + "="*60)
    print("BENCHMARK GENERATION COMPLETE")
    print("="*60)
    print(f"\nSystem: 3x3 Coupled Oscillator LPV System")
    print(f"Decision variables: 6 (elements of symmetric 3x3 Lyapunov matrix P)")
    print(f"Uncertainty dimensions: 2 (delta[0], delta[1] in [-1,1])")
    print(f"Scenarios: {n_scenarios}")
    print(f"Matrix dimension: {n}x{n}")
    print(f"\nObjective: Minimize trace(P)")
    print(f"Constraint 1: A(delta)'P + PA(delta) <= 0 (stability)")
    print(f"Constraint 2: P >= {epsilon}*I (positive definiteness)")
    print("\nFiles generated:")
    print("  - quadratic_stability_data.csv (scenarios)")
    print("  - F_0.csv through F_6.csv (Lyapunov constraint matrices)")
    print("  - E_0.csv through E_6.csv (P > 0 constraint matrices)")
    print("  - c.csv (objective vector)")
    print("  - Q.csv (regularization matrix)")


if __name__ == '__main__':
    generate_benchmark(n_scenarios=500, seed=42)
