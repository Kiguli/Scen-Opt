"""Shared loader for benchmark ``data/program_symbolic.json`` files.

This replaces the per-matrix CSV reads each ``run.py`` used to do — a
single JSON file now holds the whole problem definition in the exact
schema the Upload-Program modal consumes.

Usage from any benchmark's ``run.py``::

    from benchmarks._loader import load_symbolic_program
    prog = load_symbolic_program(data_dir)

The returned object is a dict with numpy-ready fields::

    prog['type']    → 'LP' / 'QP' / 'SDP'
    prog['c']       → (d, 1) ndarray
    prog['Q']       → (d, d) ndarray            (QP/SDP only)
    prog['A_d']     → callable δ → (m, d)       (LP/QP symbolic)
    prog['b_d']     → callable δ → (m, 1)       (LP/QP symbolic)
    prog['G']       → (n, d) ndarray or empty
    prog['h']       → (n, 1) ndarray or empty
    prog['F_d']     → callable δ → dict         (SDP symbolic)
    prog['E']       → dict of (k, k) ndarrays   (SDP hard LMI) or {}
    prog['n_x']     → int
    prog['rows_A']  → int                       (LP/QP)
    prog['rows_G']  → int
    prog['lmi_size']   → int                    (SDP)
    prog['lmi_e_size'] → int                    (SDP)
    prog['rho'], prog['tau'], prog['confidence'], prog['p'], prog['option']
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, Optional

import numpy as np


def _as_numeric_matrix(m):
    return np.array([[float(cell) for cell in row] for row in m], dtype=float)


def _expr_matrix_function(expr_matrix) -> Callable[[np.ndarray], np.ndarray]:
    def f(delta):
        return np.array([
            [eval(cell, {"delta": delta, "math": math})
             if isinstance(cell, str) else float(cell)
             for cell in row]
            for row in expr_matrix
        ], dtype=float)
    return f


def _expr_tensor_function(expr_tensor) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    funcs = {k: _expr_matrix_function(v) for k, v in expr_tensor.items()}
    def f(delta):
        return {k: funcs[k](delta) for k in funcs}
    return f


def load_symbolic_program(data_dir: str) -> dict:
    """Load and parse ``data_dir/program_symbolic.json``.

    Converts symbolic matrices to callables, numeric matrices to numpy
    arrays, and scalar parameters to Python numbers. Missing optional
    fields come back as empty arrays / dicts / sensible defaults so the
    caller can pass them straight to ``solve_lp`` / ``solve_qp`` /
    ``solve_sdp``.
    """
    path = os.path.join(data_dir, 'program_symbolic.json')
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    out: dict = {
        'type':        (raw.get('type') or '').upper(),
        'option':      raw.get('option', 'robust'),
        'rho':         float(raw['rho']) if raw.get('rho') not in (None, '') else 0.0,
        'tau':         float(raw['tau']) if raw.get('tau') not in (None, '') else 0.0,
        'confidence':  float(raw['confidence']) if raw.get('confidence') not in (None, '') else 1e-6,
        'p':           raw.get('p', '2'),
    }

    if raw.get('c') is not None:
        out['c'] = _as_numeric_matrix(raw['c'])
    if raw.get('Q') is not None:
        out['Q'] = _as_numeric_matrix(raw['Q'])
    if raw.get('x_ref') is not None:
        out['x_ref'] = np.asarray(raw['x_ref'], dtype=float).reshape(-1)

    # LP / QP
    if out['type'] in ('LP', 'QP'):
        out['A_d'] = _expr_matrix_function(raw['A_d']) if raw.get('A_d') else None
        out['b_d'] = _expr_matrix_function(raw['b_d']) if raw.get('b_d') else None
        out['G']   = _as_numeric_matrix(raw['G']) if raw.get('G') else np.array([])
        out['h']   = _as_numeric_matrix(raw['h']) if raw.get('h') else np.array([])
        out['n_x']    = int(raw.get('n_x')    or (len(raw.get('c') or []) if raw.get('c') else 0))
        out['rows_A'] = int(raw.get('rows_A') or (len(raw.get('A_d') or []) if raw.get('A_d') else 0))
        out['rows_G'] = int(raw.get('rows_G') or (len(raw.get('G')   or []) if raw.get('G')   else 0))

    # SDP
    elif out['type'] == 'SDP':
        out['F_d'] = _expr_tensor_function(raw['F_d']) if raw.get('F_d') else None
        if raw.get('E'):
            out['E'] = {k: _as_numeric_matrix(v) for k, v in raw['E'].items()}
        else:
            out['E'] = {}
        out['n_x']        = int(raw.get('n_x')        or (len(raw.get('c') or []) if raw.get('c') else 0))
        out['lmi_size']   = int(raw.get('lmi_size')   or (len(next(iter(raw['F_d'].values())))
                                                          if raw.get('F_d') else 0))
        out['lmi_e_size'] = int(raw.get('lmi_e_size') or (len(next(iter(raw['E'].values())))
                                                          if raw.get('E') else 0))
    return out


def load_scenarios_csv(data_dir: str, filename: Optional[str] = None) -> np.ndarray:
    """Read a scenarios CSV (one sample per row) into an ``(N, q)`` array.

    Auto-picks ``scenarios.csv`` unless a specific ``filename`` is given.
    """
    name = filename or 'scenarios.csv'
    path = os.path.join(data_dir, name)
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(cell) for cell in line.split(',')])
    return np.array(rows, dtype=float)
