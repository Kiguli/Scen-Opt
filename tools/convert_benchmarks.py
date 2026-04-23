#!/usr/bin/env python3
"""Convert benchmark data folders to the new Upload-Program JSON schema.

For each ``benchmarks/*/data/`` directory, this script:

* reads the existing ``benchmark.json`` (which we treat as the
  canonical symbolic definition),
* writes ``program_symbolic.json`` — identical content, normalised to
  include ``mode: "symbolic"`` plus dimension scalars so the
  Upload-Program modal can accept it verbatim,
* writes ``program_numeric.json`` — the numeric counterpart with
  symbolic fields stripped and ``mode: "numeric"`` + ``n_x`` / ``rows_A``
  / ``rows_G`` / ``lmi_size`` / ``lmi_e_size`` scalars present,
* writes ``scenarios_numeric.csv`` — per-scenario row-wise flattened
  matrices (LP/QP: ``[A_i | b_i]``; SDP: ``[F_{0,i} | … | F_{d,i}]``) so
  a user can select Numeric mode in the UI and upload this file as the
  scenarios data directly.

Individual matrix CSVs (``c.csv``, ``A_d.csv``, …, ``F_*.csv``,
``E_*.csv``), the old ``benchmark.json`` and ``benchmark.mat`` uploads
are then removed — the two JSON files replace them for the UI flow.

Run from the project root::

    python tools/convert_benchmarks.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
#  Symbolic evaluation helpers (duplicated from src/parsing.py so the script
#  has no runtime dependency on the Flask app beyond numpy).
# --------------------------------------------------------------------------- #

def eval_expression_matrix(expr_matrix, delta):
    return np.array([
        [eval(expr, {"delta": delta, "math": math})
         if isinstance(expr, str) else float(expr)
         for expr in row]
        for row in expr_matrix
    ], dtype=float)


def eval_expression_tensor(expr_tensor, delta):
    """Evaluate a dict-of-matrices at one scenario value."""
    return {
        key: eval_expression_matrix(expr_tensor[key], delta)
        for key in expr_tensor
    }


# --------------------------------------------------------------------------- #
#  Shape helpers
# --------------------------------------------------------------------------- #

def _rows(mat):
    return len(mat) if isinstance(mat, list) else 0


def _cols(mat):
    return len(mat[0]) if isinstance(mat, list) and mat and isinstance(mat[0], list) else 0


def _dict_rows(coll):
    return _rows(coll[next(iter(coll))]) if isinstance(coll, dict) and coll else 0


# --------------------------------------------------------------------------- #
#  Per-benchmark conversion
# --------------------------------------------------------------------------- #

INDIVIDUAL_CSV_PATTERNS = (
    'A_d.csv', 'b_d.csv', 'c.csv', 'Q.csv', 'G.csv', 'h.csv', 'C.csv',
    'F_*.csv', 'E_*.csv',
)
LEGACY_UPLOAD_FILES = ('benchmark.json', 'benchmark.mat')


def load_scenarios(path):
    """Read scenarios.csv (one sample per row) as an ``(N, q)`` float array."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(cell) for cell in line.split(',')])
    return np.array(rows, dtype=float)


def convert_one(data_dir: str) -> dict:
    report = {"dir": data_dir}
    bench_path = os.path.join(data_dir, 'benchmark.json')
    if not os.path.isfile(bench_path):
        report["status"] = "skipped (no benchmark.json)"
        return report

    with open(bench_path, 'r', encoding='utf-8') as f:
        program = json.load(f)

    ptype = (program.get('type') or '').upper()
    if ptype not in ('LP', 'QP', 'SDP'):
        report["status"] = f"skipped (unknown type: {ptype!r})"
        return report

    # ── Build SYMBOLIC program.json (canonical full definition) ──
    sym = dict(program)
    sym['mode'] = 'symbolic'
    # Infer + record dimensions so the Upload-Program modal has everything.
    if ptype in ('LP', 'QP'):
        d = _rows(sym.get('c'))
        m = _rows(sym.get('A_d'))
        n = _rows(sym.get('G')) if sym.get('G') else 0
        sym.setdefault('n_x', d)
        sym.setdefault('rows_A', m)
        sym['rows_G'] = n
    else:  # SDP
        d = _rows(sym.get('c'))
        lmi = _dict_rows(sym.get('F_d')) if sym.get('F_d') else 0
        lmi_e = _dict_rows(sym.get('E')) if sym.get('E') else 0
        sym.setdefault('n_x', d)
        sym.setdefault('lmi_size', lmi)
        sym['lmi_e_size'] = lmi_e

    # ── Build NUMERIC program.json (strip symbolic expressions) ──
    numeric = {k: v for k, v in sym.items()
               if k not in ('A_d', 'b_d', 'F_d')}
    numeric['mode'] = 'numeric'

    # ── Compute scenarios_numeric.csv by evaluating symbolic matrices at
    #    each scenario sample, then flattening row-wise and concatenating.
    scen_path = os.path.join(data_dir, 'scenarios.csv')
    scen_numeric_path = os.path.join(data_dir, 'scenarios_numeric.csv')
    if os.path.isfile(scen_path):
        deltas = load_scenarios(scen_path)
        N = deltas.shape[0]
        numeric_rows = []
        try:
            if ptype in ('LP', 'QP'):
                A_expr = sym.get('A_d')
                b_expr = sym.get('b_d')
                if A_expr is None or b_expr is None:
                    raise RuntimeError("A_d/b_d missing")
                for i in range(N):
                    d_vec = deltas[i]
                    A_i = eval_expression_matrix(A_expr, d_vec)
                    b_i = eval_expression_matrix(b_expr, d_vec)
                    numeric_rows.append(
                        np.concatenate([A_i.reshape(-1), b_i.reshape(-1)])
                    )
            else:  # SDP
                F_expr = sym.get('F_d')
                if F_expr is None:
                    raise RuntimeError("F_d missing")
                keys = sorted(F_expr.keys(), key=lambda k: int(k))
                for i in range(N):
                    d_vec = deltas[i]
                    parts = []
                    for k in keys:
                        F_ki = eval_expression_matrix(F_expr[k], d_vec)
                        parts.append(F_ki.reshape(-1))
                    numeric_rows.append(np.concatenate(parts))
        except Exception as exc:
            numeric_rows = []
            report["scenarios_numeric_error"] = str(exc)
        if numeric_rows:
            arr = np.array(numeric_rows)
            np.savetxt(scen_numeric_path, arr, delimiter=',')
            report["scenarios_numeric"] = arr.shape

    # ── Write the two program JSON files ──
    with open(os.path.join(data_dir, 'program_symbolic.json'), 'w',
              encoding='utf-8') as f:
        json.dump(sym, f, indent=2)
    with open(os.path.join(data_dir, 'program_numeric.json'), 'w',
              encoding='utf-8') as f:
        json.dump(numeric, f, indent=2)

    # ── Remove old uploadable artefacts and per-matrix CSVs ──
    removed = []
    for name in LEGACY_UPLOAD_FILES:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            os.remove(p)
            removed.append(name)
    for pattern in INDIVIDUAL_CSV_PATTERNS:
        for p in glob.glob(os.path.join(data_dir, pattern)):
            os.remove(p)
            removed.append(os.path.basename(p))
    report["removed"] = removed
    report["status"] = "ok"
    return report


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    bench_dirs = sorted(glob.glob(os.path.join(root, 'benchmarks', '*', 'data')))
    for d in bench_dirs:
        report = convert_one(d)
        relpath = os.path.relpath(d, root)
        print(f"{relpath:55s}  {report.get('status', '?')}")
        if report.get('scenarios_numeric'):
            print(f"  scenarios_numeric.csv -> shape {report['scenarios_numeric']}")
        if report.get('removed'):
            print(f"  removed: {', '.join(report['removed'])}")


if __name__ == '__main__':
    sys.exit(main())
