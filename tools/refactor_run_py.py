#!/usr/bin/env python3
"""Rewrite each benchmark's run.py to use ``benchmarks._loader``.

We don't attempt a line-by-line edit — instead, after verifying the file
uses the standard ``solve_lp`` / ``solve_qp`` / ``solve_sdp`` pattern,
we patch the single data-loading block that matters.

Strategy:
1. Find the line that assigns ``scenarios = load_file(...)``. Replace the
   block from there through the last CSV load (``h = load_matrix(...)``
   or similar) with a single call to ``load_symbolic_program`` plus the
   tab-appropriate unpacking.
2. Insert ``from benchmarks._loader import load_symbolic_program`` after
   the existing ``sys.path.insert`` line.
3. Leave everything else alone.
"""
from __future__ import annotations

import glob
import os
import re
import sys


LP_REPLACE = """    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c, A_d, b_d = prog['c'], prog['A_d'], prog['b_d']
    G, h = prog.get('G', np.array([])), prog.get('h', np.array([]))
"""

QP_REPLACE = """    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c, Q, A_d, b_d = prog['c'], prog['Q'], prog['A_d'], prog['b_d']
    G, h = prog.get('G', np.array([])), prog.get('h', np.array([]))
"""

SDP_REPLACE = """    # ── Load program definition (via shared _loader) ──
    prog = load_symbolic_program(data_dir)
    c = prog['c']
    Q = prog.get('Q', np.array([]))
    F_d = prog['F_d']
    E = prog.get('E', {})
"""


# Regex that matches a contiguous block of CSV-loading lines. Each line
# starts with 4-space indent and is either a load_* call or a small helper
# like "G = np.array([])". The block ends at the first blank / non-matching
# line. Anchored immediately after the scenarios line.
CSV_LOAD_LINE = re.compile(
    r"^[ \t]*(?:"
    r"c|Q|A_d|b_d|G|h|F_\w+|E|E_\w+|F_funcs|F_d"
    r")\s*=.*\n",
    re.M,
)
BLOCK_BREAK = re.compile(r"^\s*(?:[A-Za-z_][\w]*\s*=\s*|[^ ])", re.M)


def _is_loading_line(line: str) -> bool:
    """True for any line that's part of a per-matrix data-loading block."""
    s = line.strip()
    if s == "" or s.startswith("#"):
        return True   # allow blanks / comments inside the block
    # Simple ``c = load_vector(...)`` / ``Q = load_matrix(...)`` etc.
    if re.match(r"^(c|Q|A_d|b_d|G|h|F_d|E|F_funcs)\s*=\s*"
                r"(?:load_|parse_expression|np\.array\(\[\]\)|\{\})", s):
        return True
    # ``for i in range(...):`` and its indented body, emitting F_funcs[...] or E[...]
    if re.match(r"^for\s+\w+\s+in\s+range\(", s):
        return True
    if re.match(r"^(F_funcs|E|F_d)\[", s):
        return True
    # One-liner or block ``def F_d(delta): ...``
    if re.match(r"^def\s+F_d\s*\(", s):
        return True
    # ``return {...}`` inside def F_d
    if re.match(r"^return\s+\{", s):
        return True
    return False


def strip_loader_block(src: str) -> str:
    """Strip every CSV-loading line immediately after
    ``scenarios = load_file(...)``. Consumes a contiguous chunk of
    blank/comment/CSV-loading lines; stops at the first statement that
    isn't recognisable as part of the data-loading preamble.
    """
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if re.match(r"^[ \t]*scenarios\s*=\s*load_file\(.*\)\s*$", line):
            j = i + 1
            while j < len(lines) and _is_loading_line(lines[j]):
                j += 1
            i = j
            continue
        i += 1
    return "\n".join(out)


def infer_tab(src: str) -> str:
    if "solve_lp(" in src:
        return "LP"
    if "solve_qp(" in src:
        return "QP"
    if "solve_sdp(" in src:
        return "SDP"
    return ""


def insert_import(src: str) -> str:
    if "from benchmarks._loader" in src:
        return src
    # Insert right after whichever ``sys.path.insert(...)`` line the file
    # uses to reach the project root — any arg form, one line.
    return re.sub(
        r"(^[ \t]*sys\.path\.insert\([^\n]+\)\s*\n)",
        r"\1from benchmarks._loader import load_symbolic_program\n",
        src,
        count=1,
        flags=re.M,
    )


def insert_loader(src: str, tab: str) -> str:
    repl = {"LP": LP_REPLACE, "QP": QP_REPLACE, "SDP": SDP_REPLACE}[tab]
    return re.sub(
        r"(^[ \t]*scenarios\s*=\s*load_file\(.*?\)\s*\n)",
        r"\1" + repl,
        src,
        count=1,
        flags=re.M,
    )


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_files = sorted(glob.glob(os.path.join(root, 'benchmarks', '*', 'run.py')))
    for path in run_files:
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        tab = infer_tab(src)
        rel = os.path.relpath(path, root)
        if not tab:
            print(f"{rel:60s} SKIP (no solve_xx call)")
            continue
        if "load_symbolic_program" in src:
            print(f"{rel:60s} already refactored")
            continue
        new = strip_loader_block(src)
        new = insert_import(new)
        new = insert_loader(new, tab)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
        print(f"{rel:60s} patched ({tab})")


if __name__ == '__main__':
    sys.exit(main())
