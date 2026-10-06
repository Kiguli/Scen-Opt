#!/usr/bin/env python3
"""Subprocess entry point for MOSEK solves.

Sets MOSEKLM_LICENSE_FILE *before* importing CVXPY so that MOSEK 11
reads the license path on first import. This enables concurrent
users to each supply their own license without conflicts.

Usage (from the repository root):
    echo '<json>' | python -m src.mosek_solve /tmp/mosek_<uuid>/mosek.lic
"""
import os
import sys

# CRITICAL: set license path before any CVXPY / MOSEK imports
if len(sys.argv) > 1:
    os.environ["MOSEKLM_LICENSE_FILE"] = sys.argv[1]

import json
import numpy as np

# Now safe to import the solve code (which imports CVXPY → MOSEK). It is the
# same code the app uses, so both paths report results identically.
from src.solve_form import solve_form


def _safe_tolist(v):
    """Convert numpy types to JSON-serialisable Python types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def main():
    # Redirect stdout → stderr so that stray print() calls from solver
    # code (LP.py, Miscellaneous.py, etc.) don't corrupt the JSON output.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    input_data = json.loads(sys.stdin.read())
    form = input_data["form"]
    scenarios_raw = input_data.get("scenarios")
    scenarios = np.array(scenarios_raw, dtype=float) if scenarios_raw is not None else None

    result = solve_form(form, scenarios, solver="MOSEK")
    json.dump(result, real_stdout, default=_safe_tolist)


if __name__ == "__main__":
    real_stdout = sys.stdout
    try:
        main()
    except Exception as exc:
        sys.stdout = real_stdout
        json.dump({"error": str(exc)}, real_stdout)
        sys.exit(0)
