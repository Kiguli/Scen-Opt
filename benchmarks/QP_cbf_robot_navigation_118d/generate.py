"""
QP Robot Navigation Benchmark Data Generator

This script generates scenario data for a 2D robot navigation problem
with a wall obstacle whose face position is uncertain.

A wide rectangular wall blocks the direct path from start to goal,
forcing the robot to arc above it. The wall face position has
Gaussian uncertainty, making this a scenario optimization problem.

Usage:
    python generate.py [--n_scenarios N] [--seed SEED] [--sigma SIGMA]
"""

import numpy as np
import pandas as pd
import argparse

# Configuration
N_SCENARIOS = 500
T_HORIZON = 20  # Number of timesteps
DT = 0.4  # Timestep in seconds (8.0s total)

# Workspace parameters
WORKSPACE_X = 10.0  # 10m wide
WORKSPACE_Y = 8.0   # 8m tall
START_POS = np.array([0.5, 2.5])
GOAL_POS = np.array([9.5, 2.5])

# Robot parameters
R_ROBOT = 0.2  # Robot radius
D_MARGIN = 0.1  # Safety margin
D_SAFE = R_ROBOT + D_MARGIN  # 0.3m clearance from wall face

# Wall obstacle: a wide rectangular barrier extending from the bottom
# The robot must pass above the wall through the gap
WALL = {
    'x_min': 3.0, 'x_max': 7.0,   # Wall extent in x (4m wide)
    'y_min': 0.0, 'y_max': 3.0,   # Wall extent in y (3m tall)
    'face': 'top',                  # Gap is above the wall
    'face_y': 3.0,                  # y-coordinate of the gap face
    'name': 'Wall',
}

# Uncertainty in wall face position
OBSTACLE_SIGMA = 0.05  # Standard deviation in metres

# Control limits
U_MAX = 2.5  # Maximum acceleration (m/s^2)
U_MIN = -2.5  # Minimum acceleration


def generate_scenarios(n_scenarios, sigma, seed=42):
    """
    Generate N scenarios of wall face position perturbations.

    Each scenario perturbs the gap face y-position.
    delta = [wall_face_shift]  (1D Gaussian)
    """
    np.random.seed(seed)
    scenarios = np.random.randn(n_scenarios, 1) * sigma
    return scenarios


def get_dynamics_matrices():
    """
    Return the discrete-time dynamics matrices for a double integrator.
    State: x = [px, py, vx, vy]
    Control: u = [ax, ay]
    """
    A = np.array([
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    B = np.array([
        [0.5 * DT**2, 0],
        [0, 0.5 * DT**2],
        [DT, 0],
        [0, DT]
    ])

    return A, B


def compute_reference_trajectory():
    """
    Compute a straight-line reference trajectory from start to goal.

    The reference represents the desired path (direct route at y=2.5).
    The wall constraint forces the optimizer to deviate upward, making
    the scenario constraints active.

    X-reference uses a bang-bang acceleration profile to match the
    natural dynamics timing for correct constraint activation.
    """
    ref = np.zeros((T_HORIZON, 2))

    # X-trajectory: bang-bang acceleration (accel for ~5 steps, decel for ~5)
    px_vals = [START_POS[0]]
    vx_vals = [0.0]
    switch_step = 5  # Switch from accel to decel
    reached_goal = False

    for t in range(T_HORIZON - 1):
        if reached_goal:
            px_vals.append(GOAL_POS[0])
            vx_vals.append(0.0)
            continue
        ax = U_MAX if t < switch_step else -U_MAX
        new_vx = vx_vals[-1] + ax * DT
        new_px = px_vals[-1] + vx_vals[-1] * DT + 0.5 * ax * DT**2
        if new_px >= GOAL_POS[0] or new_vx <= 0:
            new_px = min(new_px, GOAL_POS[0])
            new_vx = max(new_vx, 0.0)
            if new_vx == 0.0:
                reached_goal = True
        px_vals.append(new_px)
        vx_vals.append(new_vx)

    for t in range(T_HORIZON):
        ref[t, 0] = px_vals[t]
        ref[t, 1] = START_POS[1]  # Straight line at y=2.5

    return ref


def build_qp_matrices():
    """
    Build the QP matrices for the motion planning problem.

    Decision variables (stacked):
    x = [x_0, x_1, ..., x_{T-1}, u_0, u_1, ..., u_{T-2}]

    where x_t = [px_t, py_t, vx_t, vy_t] and u_t = [ax_t, ay_t]

    Total: 4*T + 2*(T-1) = 4*20 + 2*19 = 118 variables
    """
    n_states = 4 * T_HORIZON
    n_controls = 2 * (T_HORIZON - 1)
    n_vars = n_states + n_controls

    ref_traj = compute_reference_trajectory()

    # Build Q matrix (quadratic cost)
    Q = np.zeros((n_vars, n_vars))
    R_control = 1.0   # Control cost weight
    Q_pos = 10.0      # Position tracking weight

    # Control costs: 1/2 * (2*R) * u^2 = R * u^2
    for t in range(T_HORIZON - 1):
        u_idx = n_states + t * 2
        Q[u_idx, u_idx] = 2 * R_control
        Q[u_idx + 1, u_idx + 1] = 2 * R_control

    # Position tracking costs: 1/2 * (2*Q_pos) * (px - ref)^2 with c = -2*Q_pos*ref
    for t in range(T_HORIZON):
        x_idx = t * 4
        Q[x_idx, x_idx] = 2 * Q_pos      # px
        Q[x_idx + 1, x_idx + 1] = 2 * Q_pos  # py

    # Build c vector (linear cost) for tracking reference
    c = np.zeros(n_vars)
    for t in range(T_HORIZON):
        x_idx = t * 4
        c[x_idx] = -2 * Q_pos * ref_traj[t, 0]
        c[x_idx + 1] = -2 * Q_pos * ref_traj[t, 1]

    return Q, c, n_vars, n_states, n_controls


def build_hard_constraints():
    """
    Build the hard constraint matrices G and h.

    Hard constraints:
    1. Dynamics: x_{t+1} = A x_t + B u_t
    2. Initial state: x_0 = [start_pos, 0, 0]
    3. Control limits: u_min <= u_t <= u_max
    4. Workspace bounds: 0 <= px <= WORKSPACE_X, 0 <= py <= WORKSPACE_Y
    5. Goal constraint: final position within tolerance of goal
    """
    n_states = 4 * T_HORIZON
    n_controls = 2 * (T_HORIZON - 1)
    n_vars = n_states + n_controls

    A_dyn, B_dyn = get_dynamics_matrices()

    G_rows = []
    h_values = []

    # 1. Dynamics constraints (equality as two inequalities)
    for t in range(T_HORIZON - 1):
        for state_dim in range(4):
            row = np.zeros(n_vars)
            row[(t+1)*4 + state_dim] = 1
            for j in range(4):
                row[t*4 + j] = -A_dyn[state_dim, j]
            for j in range(2):
                row[n_states + t*2 + j] = -B_dyn[state_dim, j]

            G_rows.append(row)
            h_values.append(0)
            G_rows.append(-row)
            h_values.append(0)

    # 2. Initial state constraint
    initial_state = np.array([START_POS[0], START_POS[1], 0, 0])
    for state_dim in range(4):
        row = np.zeros(n_vars)
        row[state_dim] = 1
        G_rows.append(row)
        h_values.append(-initial_state[state_dim])

        row = np.zeros(n_vars)
        row[state_dim] = -1
        G_rows.append(row)
        h_values.append(initial_state[state_dim])

    # 3. Control limits
    for t in range(T_HORIZON - 1):
        for ctrl_dim in range(2):
            u_idx = n_states + t*2 + ctrl_dim
            row = np.zeros(n_vars)
            row[u_idx] = 1
            G_rows.append(row)
            h_values.append(-U_MAX)

            row = np.zeros(n_vars)
            row[u_idx] = -1
            G_rows.append(row)
            h_values.append(U_MIN)

    # 4. Workspace bounds
    workspace_limits = [WORKSPACE_X, WORKSPACE_Y]
    for t in range(T_HORIZON):
        for pos_dim in range(2):
            x_idx = t*4 + pos_dim
            row = np.zeros(n_vars)
            row[x_idx] = 1
            G_rows.append(row)
            h_values.append(-workspace_limits[pos_dim])

            row = np.zeros(n_vars)
            row[x_idx] = -1
            G_rows.append(row)
            h_values.append(0)

    # 5. Goal constraint
    goal_tol = 0.1
    final_state_idx = (T_HORIZON - 1) * 4
    for pos_dim in range(2):
        x_idx = final_state_idx + pos_dim
        goal_val = GOAL_POS[pos_dim]
        row = np.zeros(n_vars)
        row[x_idx] = 1
        G_rows.append(row)
        h_values.append(-(goal_val + goal_tol))
        row = np.zeros(n_vars)
        row[x_idx] = -1
        G_rows.append(row)
        h_values.append(goal_val - goal_tol)

    return np.array(G_rows), np.array(h_values)


def build_scenario_constraints(scenarios):
    """
    Build scenario-dependent wall avoidance constraints.

    At timesteps where the reference trajectory is within the wall's x-range,
    enforce that the robot passes above the wall:
        -py + face_y + d_safe + delta[0] <= 0
    i.e., py >= face_y + d_safe + delta[0]

    In the tool's format: A(delta) @ x + b(delta) <= zeta
    """
    n_states = 4 * T_HORIZON
    n_controls = 2 * (T_HORIZON - 1)
    n_vars = n_states + n_controls

    ref_traj = compute_reference_trajectory()

    X_MARGIN = 0.5  # Extend constraint activation slightly beyond wall

    a_d_rows = []
    b_d_rows = []

    x_lo = WALL['x_min'] - X_MARGIN
    x_hi = WALL['x_max'] + X_MARGIN

    for t in range(T_HORIZON):
        ref_x = ref_traj[t, 0]

        if ref_x < x_lo or ref_x > x_hi:
            continue

        py_idx = t * 4 + 1

        # Robot must be above: -py + face_y + d_safe + delta[0] <= 0
        a_d_entries = ['0'] * n_vars
        a_d_entries[py_idx] = '-1.000000'

        constant = WALL['face_y'] + D_SAFE
        b_d = f'1.000000*delta[0] + {constant:.6f}'

        a_d_rows.append(', '.join(a_d_entries))
        b_d_rows.append(b_d)

    return a_d_rows, b_d_rows


def save_benchmark_files(scenarios, Q, c, G, h, a_d_rows, b_d_rows):
    """Save all benchmark files to data/ subdirectory."""
    import os
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    pd.DataFrame(scenarios).to_csv(os.path.join(data_dir, 'scenarios.csv'), header=False, index=False)
    print(f"Saved data/scenarios.csv: {len(scenarios)} scenarios")

    with open(os.path.join(data_dir, 'A_d.csv'), 'w') as f:
        for row in a_d_rows:
            f.write(row + '\n')
    print(f"Saved data/A_d.csv: {len(a_d_rows)} constraint rows")

    with open(os.path.join(data_dir, 'b_d.csv'), 'w') as f:
        for row in b_d_rows:
            f.write(row + '\n')
    print(f"Saved data/b_d.csv")

    pd.DataFrame(Q).to_csv(os.path.join(data_dir, 'Q.csv'), header=False, index=False)
    print(f"Saved data/Q.csv: {Q.shape[0]}x{Q.shape[1]}")

    with open(os.path.join(data_dir, 'c.csv'), 'w') as f:
        for val in c:
            f.write(f'{val}\n')
    print(f"Saved data/c.csv")

    pd.DataFrame(G).to_csv(os.path.join(data_dir, 'G.csv'), header=False, index=False)
    print(f"Saved data/G.csv: {G.shape[0]} hard constraints")

    with open(os.path.join(data_dir, 'h.csv'), 'w') as f:
        for val in h:
            f.write(f'{val}\n')
    print(f"Saved data/h.csv")

    # Save parameters
    with open(os.path.join(benchmark_dir, 'parameters.txt'), 'w') as f:
        f.write("# QP Robot Navigation Benchmark Parameters\n")
        f.write(f"# Workspace: {WORKSPACE_X}m x {WORKSPACE_Y}m\n")
        f.write(f"# Start: {START_POS}\n")
        f.write(f"# Goal: {GOAL_POS}\n")
        f.write(f"# Time horizon: {T_HORIZON} steps, dt={DT}s ({T_HORIZON * DT}s total)\n")
        f.write(f"# Wall: x=[{WALL['x_min']}, {WALL['x_max']}], y=[{WALL['y_min']}, {WALL['y_max']}]\n")
        f.write(f"# Wall face uncertainty: sigma={OBSTACLE_SIGMA}m\n")
        f.write(f"# Robot radius: {R_ROBOT}m\n")
        f.write(f"# Safety margin: {D_MARGIN}m\n")
        f.write(f"# Safety distance from face: {D_SAFE}m\n\n")
        f.write("rho = 0\n")
        f.write("tau = 0\n")
        f.write("confidence = 0.99\n")
    print("Saved parameters.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate Robot Navigation benchmark data')
    parser.add_argument('--n_scenarios', type=int, default=N_SCENARIOS,
                        help=f'Number of scenarios (default: {N_SCENARIOS})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sigma', type=float, default=OBSTACLE_SIGMA,
                        help=f'Wall face uncertainty std (default: {OBSTACLE_SIGMA})')
    args = parser.parse_args()

    sigma = args.sigma

    print("=" * 60)
    print("QP Robot Navigation Benchmark Generator")
    print("Wall obstacle configuration")
    print("=" * 60)
    print()
    print(f"  Workspace: {WORKSPACE_X}m x {WORKSPACE_Y}m")
    print(f"  Start: {START_POS}, Goal: {GOAL_POS}")
    print(f"  Time horizon: {T_HORIZON} steps, dt={DT}s ({T_HORIZON * DT}s total)")
    print(f"  Robot radius: {R_ROBOT}m")
    print()
    print(f"  Wall: x=[{WALL['x_min']}, {WALL['x_max']}], "
          f"y=[{WALL['y_min']}, {WALL['y_max']}], gap above at y={WALL['face_y']}")
    print()
    print(f"  Uncertainty: sigma={sigma}m (Gaussian)")
    print(f"  Safety distance from face: {D_SAFE}m")
    print()

    print("Generating wall face perturbation scenarios...")
    scenarios = generate_scenarios(args.n_scenarios, sigma, args.seed)

    print("Building QP matrices...")
    Q, c, n_vars, n_states, n_controls = build_qp_matrices()
    print(f"  Decision variables: {n_vars} ({n_states} states + {n_controls} controls)")

    print("Building hard constraints...")
    G, h = build_hard_constraints()
    print(f"  Hard constraints: {len(h)}")

    print("Building scenario-dependent wall constraints...")
    a_d_rows, b_d_rows = build_scenario_constraints(scenarios)
    print(f"  Wall constraints per scenario: {len(a_d_rows)}")

    print("\nSaving benchmark files...")
    save_benchmark_files(scenarios, Q, c, G, h, a_d_rows, b_d_rows)

    print("\n" + "=" * 60)
    print("Benchmark generation complete!")
    print("=" * 60)
    print(f"Total decision variables: {n_vars}")
    print(f"Wall constraints per scenario: {len(a_d_rows)}")
    print(f"Hard constraints: {len(h)}")
    print(f"Scenarios: {args.n_scenarios}")
    print()
    print("Run 'python run.py' to solve and 'python plot.py' to visualize.")


if __name__ == '__main__':
    main()
