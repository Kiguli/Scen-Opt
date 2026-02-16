"""
Economic Dispatch with Renewable Uncertainty Benchmark Data Generator

This script generates scenario data for the economic dispatch problem with
uncertain wind and solar generation.

Data Source:
- Synthetic data based on NREL WIND Toolkit statistical properties
- Reference: https://www.nrel.gov/grid/wind-toolkit.html
- Solar: https://nsrdb.nrel.gov/

Usage:
    python generate.py [--n_scenarios N] [--seed SEED]
"""

import os
import numpy as np
import pandas as pd
import argparse

# Configuration
N_SCENARIOS = 150
T_HORIZON = 24  # 24 hours

# System parameters
GENERATORS = [
    {'name': 'Gas1', 'capacity': 400, 'min_output': 100, 'cost': 40, 'ramp': 200},
    {'name': 'Gas2', 'capacity': 300, 'min_output': 75,  'cost': 50, 'ramp': 150},
    {'name': 'Coal', 'capacity': 500, 'min_output': 200, 'cost': 30, 'ramp': 100},
]
N_GENERATORS = len(GENERATORS)

# Renewable capacities
WIND_CAPACITY = 200   # MW
SOLAR_CAPACITY = 150  # MW

# Demand profile (normalized to peak = 1)
# Typical residential + commercial pattern with morning and evening peaks
DEMAND_PROFILE = np.array([
    0.65, 0.60, 0.58, 0.57, 0.58, 0.62,  # 00:00 - 05:00
    0.70, 0.80, 0.88, 0.92, 0.95, 0.96,  # 06:00 - 11:00
    0.94, 0.92, 0.90, 0.88, 0.90, 0.95,  # 12:00 - 17:00
    1.00, 0.98, 0.92, 0.85, 0.78, 0.70,  # 18:00 - 23:00
])
PEAK_DEMAND = 1000  # MW


def generate_wind_scenarios(n_scenarios, seed=42):
    """
    Generate wind power scenarios using Weibull distribution.
    Wind capacity factor varies by hour with temporal correlation.
    """
    np.random.seed(seed)

    # Weibull shape parameter (typical for wind)
    shape = 2.0

    # Scale parameter varies by hour (wind often stronger at night)
    hour_scale = 0.3 + 0.1 * np.cos(2 * np.pi * np.arange(24) / 24 + np.pi)

    scenarios = np.zeros((n_scenarios, T_HORIZON))

    for i in range(n_scenarios):
        # Generate base wind pattern
        base_scale = np.random.uniform(0.8, 1.2)  # Day-specific variation

        for t in range(T_HORIZON):
            scale = hour_scale[t] * base_scale

            # Weibull sample for capacity factor
            cf = np.random.weibull(shape) * scale

            # Clip to [0, 1]
            cf = np.clip(cf, 0, 1)

            # Add temporal correlation with previous hour
            if t > 0:
                cf = 0.7 * cf + 0.3 * scenarios[i, t-1] / WIND_CAPACITY

            scenarios[i, t] = cf * WIND_CAPACITY

    return scenarios


def generate_solar_scenarios(n_scenarios, seed=42):
    """
    Generate solar power scenarios using Beta distribution.
    Solar follows clear-sky envelope with cloud variability.
    """
    np.random.seed(seed + 1000)

    # Clear-sky envelope (sunrise ~6am, sunset ~6pm)
    hours = np.arange(T_HORIZON)
    clear_sky = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    clear_sky[hours < 6] = 0
    clear_sky[hours >= 18] = 0

    scenarios = np.zeros((n_scenarios, T_HORIZON))

    for i in range(n_scenarios):
        # Day-specific cloudiness (0 = clear, 1 = overcast)
        cloudiness = np.random.beta(2, 5)  # Tends toward clearer days

        for t in range(T_HORIZON):
            if clear_sky[t] > 0:
                # Cloud-adjusted capacity factor
                cloud_factor = 1 - cloudiness * np.random.uniform(0.5, 1.0)
                cf = clear_sky[t] * cloud_factor

                # Add some randomness
                cf = cf * np.random.uniform(0.9, 1.1)
                cf = np.clip(cf, 0, 1)

                scenarios[i, t] = cf * SOLAR_CAPACITY
            else:
                scenarios[i, t] = 0

    return scenarios


def generate_demand():
    """Generate the demand profile (deterministic)."""
    return DEMAND_PROFILE * PEAK_DEMAND


def build_lp_matrices(demand):
    """
    Build the LP matrices for the economic dispatch problem.

    Decision variables: P[m,t] for m in generators, t in hours
    Arranged as: [P_0_0, P_0_1, ..., P_0_23, P_1_0, ..., P_2_23]
    Total: N_GENERATORS * T_HORIZON = 72 variables
    """
    n_vars = N_GENERATORS * T_HORIZON

    # Index mapping
    def idx(m, t):
        return m * T_HORIZON + t

    # Objective: minimize total generation cost
    c = np.zeros(n_vars)
    for m, gen in enumerate(GENERATORS):
        for t in range(T_HORIZON):
            c[idx(m, t)] = gen['cost']

    # Scenario constraints: power balance
    # sum_m P[m,t] + wind_t + solar_t = demand_t
    # Rewritten as: sum_m P[m,t] = demand_t - wind_t - solar_t
    # As inequality: sum_m P[m,t] - (demand_t - wind_t - solar_t) <= zeta
    #                -(sum_m P[m,t] - (demand_t - wind_t - solar_t)) <= zeta

    # For the scenario approach:
    # A @ x + b(delta) <= zeta
    # where delta = [wind_0, ..., wind_23, solar_0, ..., solar_23]

    a_d_rows = []
    b_d_rows = []

    for t in range(T_HORIZON):
        # Upper bound: sum_m P[m,t] <= demand_t - wind_t - solar_t + zeta
        # A @ x + b <= zeta  where b = -(demand_t - wind_t - solar_t) = wind_t + solar_t - demand_t
        a_row = ['0'] * n_vars
        for m in range(N_GENERATORS):
            a_row[idx(m, t)] = '1'
        a_d_rows.append(', '.join(a_row))
        # b = wind_t + solar_t - demand_t = delta[t] + delta[24+t] - demand_t
        b_d_rows.append(f'delta[{t}] + delta[{24+t}] + {-demand[t]:.2f}')

        # Lower bound: -(sum_m P[m,t]) <= -(demand_t - wind_t - solar_t) + zeta
        # A @ x + b <= zeta where A = -P, b = demand_t - wind_t - solar_t
        a_row = ['0'] * n_vars
        for m in range(N_GENERATORS):
            a_row[idx(m, t)] = '-1'
        a_d_rows.append(', '.join(a_row))
        b_d_rows.append(f'-delta[{t}] - delta[{24+t}] + {demand[t]:.2f}')

    # Hard constraints
    G_rows = []
    h_values = []

    # Convention: Gx + h <= 0, i.e. Gx <= -h
    for m, gen in enumerate(GENERATORS):
        for t in range(T_HORIZON):
            # Upper bound: P[m,t] <= capacity  →  P[m,t] - capacity <= 0
            row = [0] * n_vars
            row[idx(m, t)] = 1
            G_rows.append(row)
            h_values.append(-gen['capacity'])

            # Lower bound: P[m,t] >= min_output  →  -P[m,t] + min_output <= 0
            row = [0] * n_vars
            row[idx(m, t)] = -1
            G_rows.append(row)
            h_values.append(gen['min_output'])

        # Ramp constraints
        for t in range(T_HORIZON - 1):
            # P[m,t+1] - P[m,t] <= ramp  →  P[m,t+1] - P[m,t] - ramp <= 0
            row = [0] * n_vars
            row[idx(m, t+1)] = 1
            row[idx(m, t)] = -1
            G_rows.append(row)
            h_values.append(-gen['ramp'])

            # P[m,t] - P[m,t+1] <= ramp  →  P[m,t] - P[m,t+1] - ramp <= 0
            row = [0] * n_vars
            row[idx(m, t)] = 1
            row[idx(m, t+1)] = -1
            G_rows.append(row)
            h_values.append(-gen['ramp'])

    G = np.array(G_rows)
    h = np.array(h_values)

    return c, a_d_rows, b_d_rows, G, h


def load_parameters():
    """Load rho and other parameters from parameters.txt."""
    params = {}
    params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.txt')
    with open(params_path, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                params[key.strip()] = float(val.split('#')[0].strip())
    return params


def save_benchmark_files(wind_scenarios, solar_scenarios, c, a_d_rows, b_d_rows, G, h, demand):
    """Save all benchmark files with shared-slack augmentation into data/.

    The data files encode the augmented formulation directly:
        Original:  min c'P + rho * 1'zeta   s.t. A(d)P + b(d) <= zeta, GP+h <= 0, zeta >= 0
        Augmented: min c_aug' x_aug          s.t. A_aug(d) x_aug + b(d) <= 0, G_aug x_aug + h_aug <= 0

    where x_aug = [P; zeta] in R^(n + m), with:
        c_aug = [c; rho * 1_m]
        A_aug = [A, -I_m]        (slack absorbs scenario violations)
        G_aug = [[G, 0]; [0, -I_m]]   (original hard constraints + zeta >= 0)
        h_aug = [h; 0]
    """
    params = load_parameters()
    rho = params.get('rho', 100.0)

    n_scenarios = len(wind_scenarios)
    n_vars = len(c)
    m_scenario = len(a_d_rows)
    n_aug = n_vars + m_scenario

    # Ensure data directory exists
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(benchmark_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Combine scenarios: [wind_0, ..., wind_23, solar_0, ..., solar_23]
    scenarios = np.hstack([wind_scenarios, solar_scenarios])

    # Save scenarios
    pd.DataFrame(scenarios).to_csv(os.path.join(data_dir, 'scenarios.csv'), header=False, index=False)
    print(f"Saved data/scenarios.csv: {n_scenarios} scenarios x 48 variables")

    # ---- Augment A_d: append -I_m slack columns ----
    a_d_aug_rows = []
    for i, row_str in enumerate(a_d_rows):
        slack_cols = ['0'] * m_scenario
        slack_cols[i] = '-1'
        a_d_aug_rows.append(row_str + ', ' + ', '.join(slack_cols))

    with open(os.path.join(data_dir, 'A_d.csv'), 'w') as f:
        for row in a_d_aug_rows:
            f.write(row + '\n')
    print(f"Saved data/A_d.csv: {m_scenario} x {n_aug} (augmented)")

    # ---- b_d.csv: unchanged ----
    with open(os.path.join(data_dir, 'b_d.csv'), 'w') as f:
        for row in b_d_rows:
            f.write(row + '\n')
    print(f"Saved data/b_d.csv")

    # ---- c.csv: augmented cost vector [c; rho * 1_m] ----
    c_aug = list(c) + [rho] * m_scenario
    with open(os.path.join(data_dir, 'c.csv'), 'w') as f:
        for val in c_aug:
            f.write(f'{val}\n')
    print(f"Saved data/c.csv: {n_aug} values (augmented)")

    # ---- G.csv and h.csv: augmented hard constraints ----
    # Pad original G with zero slack columns, then add zeta >= 0 rows
    G_aug = np.block([
        [G, np.zeros((G.shape[0], m_scenario))],
        [np.zeros((m_scenario, n_vars)), -np.eye(m_scenario)]
    ])
    h_aug = np.concatenate([h, np.zeros(m_scenario)])

    pd.DataFrame(G_aug).to_csv(os.path.join(data_dir, 'G.csv'), header=False, index=False)
    print(f"Saved data/G.csv: {G_aug.shape[0]} x {n_aug} (augmented)")

    with open(os.path.join(data_dir, 'h.csv'), 'w') as f:
        for val in h_aug:
            f.write(f'{val}\n')
    print(f"Saved data/h.csv: {len(h_aug)} values (augmented)")

    print(f"  Augmented formulation: x_aug = [P; zeta] in R^{n_aug}")

    # Save demand profile
    with open(os.path.join(data_dir, 'demand.csv'), 'w') as f:
        for t, d in enumerate(demand):
            f.write(f'{t},{d:.2f}\n')
    print("Saved data/demand.csv")

    # Save generator info
    with open(os.path.join(data_dir, 'generators.txt'), 'w') as f:
        f.write("# Generator Parameters\n")
        f.write("# Name, Capacity(MW), MinOutput(MW), Cost($/MWh), RampRate(MW/h)\n")
        for gen in GENERATORS:
            f.write(f"{gen['name']}, {gen['capacity']}, {gen['min_output']}, "
                    f"{gen['cost']}, {gen['ramp']}\n")
    print("Saved data/generators.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate Power Dispatch benchmark data')
    parser.add_argument('--n_scenarios', type=int, default=N_SCENARIOS,
                        help=f'Number of scenarios (default: {N_SCENARIOS})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("Generating Power Dispatch benchmark...")
    print(f"  Time horizon: {T_HORIZON} hours")
    print(f"  Generators: {N_GENERATORS}")
    print(f"  Renewables: Wind {WIND_CAPACITY} MW, Solar {SOLAR_CAPACITY} MW")
    print(f"  Peak demand: {PEAK_DEMAND} MW")
    print()

    # Generate demand
    demand = generate_demand()

    # Generate renewable scenarios
    print("Generating wind scenarios (Weibull distribution)...")
    wind_scenarios = generate_wind_scenarios(args.n_scenarios, args.seed)

    print("Generating solar scenarios (Beta distribution)...")
    solar_scenarios = generate_solar_scenarios(args.n_scenarios, args.seed)

    # Build LP matrices
    print("Building LP matrices...")
    c, a_d_rows, b_d_rows, G, h = build_lp_matrices(demand)

    # Save files
    print("\nSaving benchmark files...")
    save_benchmark_files(wind_scenarios, solar_scenarios, c, a_d_rows, b_d_rows, G, h, demand)

    # Print summary
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    print(f"Decision variables: {N_GENERATORS * T_HORIZON}")
    print(f"Scenarios: {args.n_scenarios}")
    print(f"Scenario constraints: {len(a_d_rows)} (power balance at each hour)")
    print(f"Hard constraints: {len(h)}")
    print(f"\nRenewable generation statistics:")
    print(f"  Wind - Mean: {np.mean(wind_scenarios):.1f} MW, "
          f"Std: {np.std(wind_scenarios):.1f} MW")
    print(f"  Solar - Mean: {np.mean(solar_scenarios):.1f} MW, "
          f"Std: {np.std(solar_scenarios):.1f} MW")
    print(f"  Combined avg penetration: "
          f"{(np.mean(wind_scenarios) + np.mean(solar_scenarios)) / np.mean(demand) * 100:.1f}%")


if __name__ == '__main__':
    main()
