<p align="center">
  <img src="docs/_static/logo.svg" alt="Scen-Opt" width="400">
</p>

<h1 align="center">A Scenario Optimization Toolbox for Data-Driven Convex Programming</h1>

Scen-Opt is an open-source software tool for data-driven convex optimization using the scenario approach of Campi and Garatti. It solves **Linear Programs (LP)**, **Quadratic Programs (QP)**, and **Semidefinite Programs (SDP)** using only sampled uncertainty realizations, providing rigorous probabilistic guarantees on out-of-sample performance without requiring knowledge of the underlying probability distribution.

The tool is implemented as a Python Flask web application with a modern JavaScript frontend, offering an intuitive graphical interface for specifying problems, uploading data, and inspecting results. It supports 27+ convex optimization solvers through CVXPY, including MOSEK, CLARABEL, SCS, and OSQP.

## Installation

**Prerequisites:** Python 3.12+ and pip.

```bash
# Clone the repository
git clone https://github.com/Kiguli/Scen-Opt.git
cd Scen-Opt

# Install dependencies
pip install -r requirements.txt

# Start the application
python3 app.py
```

The server starts at `http://127.0.0.1:5000`. Open this URL in a browser to access the web interface.

**Optional:** For best performance on SDP problems, install [MOSEK](https://www.mosek.com/) (free academic license available). See [MOSEK Support](#mosek-support) for license configuration.

## Quick Start

### Web Interface

1. Run `python3 app.py` and open `http://127.0.0.1:5000`
2. Select the program type tab (LP, QP, or SDP)
3. Enter or upload constraint matrices, objective vectors, and scenarios
4. Choose a solver (if using MOSEK, you will be prompted to upload a license file)
5. Set solver parameters and press **Solve**
6. Download results as `.json` or `.mat` using the buttons in the results panel

### One-Shot Benchmark Upload

Every benchmark under `benchmarks/*/data/` ships two self-contained JSON files that can be pasted straight into the UI:

- **`program_symbolic.json`** — matrices use `delta[k]` expressions. Pair with the benchmark's `scenarios.csv` and pick **Symbolic** input mode.
- **`program_numeric.json`** — matrices are pre-evaluated per sample. Pair with `scenarios_numeric.csv` (each row carries row-wise flattened `A_i | b_i` for LP/QP, or `F_{0,i} | F_{1,i} | … | F_{d,i}` for SDP) and pick **Numeric** input mode.

Steps:

1. Click **Upload Program** (next to the LP/QP/SDP tabs).
2. Paste the contents of `program_symbolic.json` (or `program_numeric.json`) into the paste box, or select the file in the upload input.
3. Under **Scenarios**, upload the matching `scenarios.csv` (symbolic) or `scenarios_numeric.csv` (numeric).
4. Press **Solve**.

## Problem Formulations

Each program type supports four configurations: robust, robust + relaxation, robust + regularization, and the general form combining all three.

### Linear Program (LP)

```
min   c'x + tau * ||x - x_ref||_p + rho * sum(zeta)
s.t.  A(delta_i) x + b(delta_i) <= zeta_i,   i = 1,...,N    (scenario constraints)
      G x + h <= 0                                           (hard constraints)
      zeta_i >= 0
```

### Quadratic Program (QP)

```
min   (1/2) x'Qx + c'x + tau * ||x - x_ref||_p + rho * sum(zeta)
s.t.  A(delta_i) x + b(delta_i) <= zeta_i,   i = 1,...,N    (scenario constraints)
      G x + h <= 0                                           (hard constraints)
      zeta_i >= 0
```

where Q is positive semidefinite.

### Semidefinite Program (SDP)

```
min   (1/2) x'Qx + c'x + tau * ||x - x_ref||_p + rho * sum(zeta)
s.t.  F_0(delta_i) + sum_j x_j F_j(delta_i) <= zeta_i * I,  i = 1,...,N   (scenario LMIs)
      E_0 + sum_j x_j E_j <= 0                                             (hard LMI)
```

where the scenario constraints are Linear Matrix Inequalities (LMIs) parameterized by uncertainty samples.

### Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Scenarios | delta_i | Sampled uncertainty realizations (N samples) |
| Slack penalty | rho | Penalty on constraint relaxation (rho = 0 means hard scenario constraints) |
| Regularization | tau | Regularization strength toward reference point x_ref |
| Norm type | p | Norm for regularization (1, 2, inf, fro, nuc) |
| Confidence | beta | Confidence parameter for risk bounds (e.g. beta = 0.01 for 99% confidence) |

### Risk Quantification

After solving, the tool computes the **complexity** k (number of support constraints) and the scenario approach **risk bounds** [epsilon_lower, epsilon_upper]. These bound the probability that a new unseen scenario would violate the solution, providing certified probabilistic guarantees on out-of-sample performance.

### Downloading Results

After solving, two buttons appear in the results panel:

- **Download .JSON** -- Downloads the full result data (optimal solution, cost, risk bounds, timing, etc.) as a JSON file directly from the browser.
- **Download .MAT** -- Downloads the same data as a MATLAB `.mat` file, generated server-side.

The results include optimization solve time and risk computation time alongside the standard outputs.

## Benchmarks

The `benchmarks/` directory contains 12 case studies spanning all three program types, each with pre-generated data, solver scripts, and paper figures.

| Benchmark | Type | d | q | m_s | m_h | rho | tau | N | beta | k | epsilon_lower | epsilon_upper | Time (s) | Data Source |
|-----------|------|---|---|-----|-----|-----|-----|---|------|---|---------------|---------------|----------|-------------|
| Half Width | LP | 2 | 1 | 2 | 0 | 0 | 0 | 100 | 1e-6 | 2 | 0 | 0.2085 | 1.2 | --- |
| Growth Bound | LP | 12 | 6 | 3 | 6 | 0 | 0 | 3127 | 1e-6 | 6 | 0 | 0.0103 | 1276.8 | --- |
| Inventory | LP | 11 | 15 | 6 | 17 | 100 | 0 | 500 | 1e-6 | 7 | 0 | 0.0666 | 4.0 | --- |
| Portfolio CVaR | LP | 13 | 12 | 4 | 27 | 0.016 | 0 | 1255 | 1e-6 | 3 | 0 | 0.0203 | 22.0 | Yahoo Finance |
| Power Dispatch | LP | 120 | 48 | 48 | 330 | 100 | 0 | 150 | 1e-6 | 34 | 0.0791 | 0.4496 | 2.9 | --- |
| Iris SVM | QP | 3 | 3 | 1 | 0 | 0 | 0 | 150 | 1e-6 | 2 | 0 | 0.1441 | 1.2 | UCI Iris |
| Robot Navigation | QP | 118 | 1 | 3 | 320 | 0 | 0 | 500 | 1e-6 | 1 | 0 | 0.0403 | 1.6 | --- |
| Radiation Therapy | QP | 50 | 3 | 80 | 100 | 0 | 0.1 | 200 | 1e-6 | 5 | 0 | 0.1414 | 112.3 | TROTS |
| LPV Stability | SDP | 3 | 1 | 2 | 2 | 1 | 0 | 100 | 1e-6 | 1 | 0 | 0.1858 | 1.7 | --- |
| Quadratic Stability | SDP | 6 | 2 | 3 | 3 | 0 | 0 | 500 | 1e-6 | 1 | 0 | 0.0403 | 2.2 | --- |
| Covariance Estimation | SDP | 15 | 5 | 5 | 5 | 0 | 0 | 178 | 1e-6 | 7 | 0 | 0.1780 | 2.1 | UCI Wine |
| Min. Encl. Ellipsoid | SDP | 15 | 5 | 1 | 5 | 0 | 0 | 569 | 1e-6 | 5 | 0 | 0.0518 | 235.6 | UCI Breast Cancer |

d = decision variables, q = uncertainty dimension, m_s = scenario constraints per sample, m_h = hard constraints, N = number of scenarios, k = complexity (support constraints). Solve times on a desktop with an Intel Core Ultra 9 285K (24 cores) and 64 GB RAM running Windows 11.

### Benchmark Descriptions

**Linear Programs:**

- **Half Width** (`LP_half_width_2d`) -- Find the smallest interval enclosing 100 points sampled from [0, 1]. A minimal introductory example with 2 decision variables.
- **Growth Bound** (`LP_growth_bound_12d`) -- Compute a data-driven growth bound for finite abstraction of a 3D vehicle dynamics system from 3127 sampled trajectories (Kazemi et al. 2024).
- **Inventory** (`LP_inventory_11d`) -- Optimize daily ordering of 5 perishable produce items under uncertain yield, demand, and warehouse packing. Budget $12,000, warehouse 800 cu.ft.
- **Portfolio CVaR** (`LP_portfolio_cvar_13d`) -- Minimize Conditional Value-at-Risk (95%) for a pension fund across 8 ETF asset classes using the Rockafellar-Uryasev formulation. Real market data from Yahoo Finance.
- **Power Dispatch** (`LP_power_dispatch_120d`) -- Schedule 3 thermal generators over 24 hours to meet demand with uncertain wind and solar generation. 120 decision variables, 330 hard constraints.

**Quadratic Programs:**

- **Iris SVM** (`QP_Iris_minimal_3d`) -- Hard-margin SVM separating Iris Setosa from other species using petal measurements from the UCI Iris dataset.
- **Robot Navigation** (`QP_cbf_robot_navigation_118d`) -- Navigate a point robot from start to goal around a wall obstacle with uncertain wall position. 20 timesteps, double-integrator dynamics, 118 decision variables.
- **Radiation Therapy** (`QP_radiation_therapy_50d`) -- Design a prostate brachytherapy plan under catheter placement uncertainty using TROTS dose-influence data. 50 dwell positions, 100 voxels, 80 dose constraints per scenario.

**Semidefinite Programs:**

- **LPV Stability** (`SDP_LPV_stability_3d`) -- Certify quadratic stability of a 2x2 LPV damped oscillator by finding a common Lyapunov matrix P over the scheduling parameter range.
- **Quadratic Stability** (`SDP_Quadratic_Stability_6d`) -- Find a Lyapunov matrix for a 3x3 coupled oscillator with two uncertain parameters, certifying stability over a 2D parameter space.
- **Covariance Estimation** (`SDP_covariance_wine_15d`) -- Compute a robust minimum-trace covariance matrix dominating all subsample covariances drawn from the UCI Wine dataset (5 features).
- **Min. Encl. Ellipsoid** (`SDP_ellipsoid_enclosure_15d`) -- Find the tightest ellipsoid containing data points from the UCI Breast Cancer dataset (5 features, 569 samples).

### Benchmark Directory Structure

Each benchmark follows a standardized layout:

```
<benchmark_name>/
├── README.md           Problem description, formulation, and usage
├── parameters.txt      Solver parameters (rho, tau, confidence)
├── generate.py         Generate/regenerate scenarios and data
├── run.py              Solve the program and print results
├── plot.py             Generate the paper figure
├── data/
│   ├── program_symbolic.json  Upload-Program definition (symbolic mode)
│   ├── program_numeric.json   Upload-Program definition (numeric mode)
│   ├── scenarios.csv   Sampled uncertainty realizations
│   ├── scenarios_numeric.csv  Per-row-flattened matrices for numeric mode
│   ├── c.csv           Objective vector
│   └── ...             Additional problem-specific data
└── results/
    ├── metrics.json    Solver output (cost, risk bounds, complexity)
    ├── solution.csv    Optimal decision vector
    └── *.png / *.pdf   Paper figures
```

### Running a Benchmark

```bash
cd benchmarks/<benchmark_name>

# Regenerate scenarios (optional, data already provided)
python generate.py

# Solve the program
python run.py

# Generate the paper figure
python plot.py
```

## Project Structure

```
Scen-Opt/
├── app.py                  Flask application (routes, matrix parsing, solver dispatch)
├── requirements.txt        Python dependencies
├── Dockerfile              Container image for local deployment
├── DOCKER.md               Docker setup instructions
├── src/
│   ├── LP.py               Linear programming solver
│   ├── QP.py               Quadratic programming solver
│   ├── SDP.py              Semidefinite programming solver
│   ├── Risk.py             Scenario approach risk bound computation
│   ├── Miscellaneous.py    Utilities (active constraint detection, file loading)
│   ├── parsing.py          Shared matrix/tensor expression parsing
│   └── mosek_solve.py      Subprocess entry point for MOSEK solves
├── templates/
│   └── index.html          Web interface (single-page application)
├── static/
│   ├── css/styles.css      Stylesheet
│   └── js/main.js          Frontend logic (matrix editors, file upload, results display)
├── benchmarks/             12 benchmark case studies (see above)
└── tests/                  pytest test suite (42 tests)
```

### Core Modules

#### `app.py` -- Flask Server

The main entry point. Routes:

- `GET /` -- Serves the web interface with available solver list
- `POST /solve` -- Parses uploaded matrices and scenario data, dispatches to the appropriate solver, and returns results as JSON. When a MOSEK license is uploaded, the solve runs in an isolated subprocess for concurrent user support. Results include `solve_time` and `risk_time` fields.
- `POST /parse-file` -- Parses uploaded binary files (MAT, Excel, Parquet) and returns content as JSON for the frontend
- `POST /download-mat` -- Converts JSON result data to a MATLAB `.mat` file download

Supports matrix expressions containing the `delta` variable (e.g., `[[delta[0] + delta[1], 0], [0, 1]]`) that are evaluated at each scenario point. Accepts file uploads in JSON, CSV, TXT, TSV, NPY, NPZ, MAT, Excel, and Parquet formats.

#### `src/LP.py` -- `solve_lp()`

Solves the scenario LP using CVXPY. Takes scenario-dependent constraint functions `A_d(delta)` and `b_d(delta)`, hard constraint matrices `G` and `h`, objective vector `c`, and parameters `rho`, `tau`, `x_ref`, `norm_type`. When `rho > 0`, introduces non-negative slack variables zeta with per-scenario relaxation. Returns the optimal solution `x`, slacks `zeta`, cost, number of scenarios, complexity, constraints, and degeneracy flag.

#### `src/QP.py` -- `solve_qp()`

Extends the LP solver with a quadratic objective `(1/2) x'Qx`. Validates that Q is positive semidefinite and symmetric before solving. Same constraint structure and return values as the LP solver.

#### `src/SDP.py` -- `solve_sdp()`

Solves semidefinite programs with LMI constraints. Each scenario produces a matrix inequality `F_0(delta) + sum_j x_j F_j(delta) <= zeta_i * I`. The hard constraint is an LMI `E_0 + sum_j x_j E_j <= 0`. All F and E matrices must be symmetric. Uses scalar decision variables `x` (not matrix variables).

#### `src/Risk.py` -- `quantify_risk(k, N, beta)`

Computes the scenario approach risk bounds given the complexity k (support constraint count), number of scenarios N, and confidence parameter beta. Uses the regularized incomplete beta function via JAX to compute both lower and upper bounds on the violation probability epsilon through bisection. Returns `(epsilon_lower, epsilon_upper)`.

#### `src/Miscellaneous.py` -- `get_support()`

Identifies the support list of a solved scenario program: the constraints whose removal changes the optimal value (all violated constraints together with an irreducible subset of the active ones). Candidates are screened by dual value, then greedily pruned by re-solving and comparing the optimal value (via the helper `test_support()`); hard constraints are always enforced and never counted. If the screened set fails to reproduce the optimum, a recovery pass restarts from the full scenario list and flags degeneracy. Also provides `get_solvers()` to list installed CVXPY solvers and `load_file()` for reading data files.

## Supported File Formats

| Format | Extensions | Usage |
|--------|-----------|-------|
| CSV | `.csv` | Matrices, vectors, scenarios |
| JSON | `.json` | One-shot program definitions, matrices |
| MATLAB | `.mat` | One-shot program definitions |
| Text | `.txt`, `.tsv` | Matrices, parameter files |
| NumPy | `.npy`, `.npz` | Arrays |
| Excel | `.xlsx`, `.xls` | Matrices, scenarios |
| Parquet | `.parquet` | Large datasets |

## MOSEK Support

[MOSEK](https://www.mosek.com/) is a commercial solver with free academic licenses that is particularly effective for SDP problems. Scen-Opt supports MOSEK with the following features:

- **License upload:** When MOSEK is selected as the solver, a modal prompts you to upload your `mosek.lic` file. The license is **not** stored on the server.
- **Browser caching:** After the first upload, the license is cached in your browser's session storage. Subsequent solves skip the modal automatically. A cache indicator and "Clear" button appear below the solver dropdown.
- **Local license:** If MOSEK is installed locally with a system license (e.g., at `~/mosek/mosek.lic`), click "Skip" in the modal to use it directly.
- **Concurrent users:** Each MOSEK solve with an uploaded license runs in an isolated subprocess with the license written to an ephemeral `/tmp` directory (RAM-backed, auto-purged). Multiple users can solve simultaneously without conflicts.

## Docker

The easiest way to run Scen-Opt locally without managing Python dependencies is with Docker.

```bash
# Build the image
docker build -t scen-opt .

# Run the container
docker run -p 5000:5000 scen-opt
```

Open `http://localhost:5000` in your browser. See [DOCKER.md](DOCKER.md) for MOSEK license mounting, custom port mapping, and worker configuration.

### Publishing as a GitHub Package

To publish the Docker image to GitHub Container Registry so others can pull it directly:

1. Create a Personal Access Token with `write:packages` scope at [GitHub Settings > Tokens](https://github.com/settings/tokens)

2. Log in to the registry:
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
   ```

3. Build, tag, and push:
   ```bash
   docker build -t ghcr.io/kiguli/scen-opt:latest .
   docker push ghcr.io/kiguli/scen-opt:latest
   ```

4. Make the package public (optional): go to the package settings at `https://github.com/users/Kiguli/packages/container/package/scen-opt` and set visibility to **Public**.

Users can then run the tool with a single command:

```bash
docker run -p 5000:5000 ghcr.io/kiguli/scen-opt:latest
```

## Renaming the GitHub Repository (from Scen-O-Con to Scen-Opt)

The codebase, docs, CI, and Docker labels have all been moved to the name **Scen-Opt**. The GitHub repository itself still needs to be renamed on the web side. One-off migration steps:

1. **Rename on GitHub.** Go to `https://github.com/Kiguli/Scen-O-Con/settings` → top of the page → change the repository name to `Scen-Opt` → **Rename**. GitHub will keep the old `Scen-O-Con` URL working indefinitely via automatic redirects, but prefer the new name going forward.
2. **Update your local remote URL.**
   ```bash
   cd path/to/local/checkout
   git remote set-url origin https://github.com/Kiguli/Scen-Opt.git
   git remote -v   # verify
   ```
3. **Update any open forks / clones.** Collaborators should run the same `git remote set-url` command once.
4. **Update Docker / ghcr image tags** if you've published images — rebuild and push under the new name:
   ```bash
   docker build -t ghcr.io/kiguli/scen-opt:latest .
   docker push ghcr.io/kiguli/scen-opt:latest
   ```
   (The package appears as a new one under `Scen-Opt`; delete the old `scen-o-con` package from GitHub → Packages if desired.)
5. **Deployment host (if used).** The deploy workflow at `.github/workflows/deploy.yml` and the systemd unit are already named `scen-opt`. If an older `scen-o-con.*` systemd unit or nginx server name is still live, rename them:
   ```bash
   sudo systemctl stop scen-o-con && sudo systemctl disable scen-o-con
   sudo mv /srv/scen-o-con.woodingben.com /srv/scen-opt.woodingben.com
   sudo mv /etc/systemd/system/scen-o-con.service /etc/systemd/system/scen-opt.service
   sudo systemctl daemon-reload && sudo systemctl enable --now scen-opt
   ```
6. **Browser bookmarks, pip/poetry caches, IDE workspace references** — update anywhere that hard-codes the old URL. The GitHub redirect covers cloning and browsing; it does **not** cover `pip install git+...` pinning or CI secrets tied to the old name.

Sphinx docs, the web UI title (`templates/index.html`), Dockerfile labels, and the README are all updated already — no further code changes are needed after step 1 on GitHub.

## Testing

The test suite uses [pytest](https://docs.pytest.org/) with 42 tests covering all core modules: solvers (LP, QP, SDP), risk quantification, matrix/tensor parsing, file loading utilities, and Flask route integration.

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests run automatically on push and pull request via GitHub Actions (see `.github/workflows/test.yml`).

## License

This work was supported in part by an EPSRC Doctoral Prize Research Fellowship at Newcastle University.
