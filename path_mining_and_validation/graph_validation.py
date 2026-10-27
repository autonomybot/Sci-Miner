"""Graph-wide validation of the science graph (e3sm.csv) against the Fortran sources.

Every edge of the graph is checked, not only the pathways of the case studies:

  contains      file -> symbol      symbol is an identifier of that Fortran file
  encodes       file -> equation    >= SRC_TOKEN_MATCH of the equation's identifiers are in that file
  hasVariable   symbol -> equation  (a) symbol occurs in the equation text,
  hasConstant                       (b) symbol is an identifier of the equation's source file
  has           root -> file        file exists in the Fortran folder

A chance baseline re-pairs every edge with a random file, which shows how much
of the agreement is explained by common names such as `t` or `dt`.
"""

import random
import re
from collections import defaultdict

import pandas as pd

import config
import verification
from graph_io import is_scientific_symbol, norm_symbol


def _symbol_in_text(norm_sym: str, eq: str) -> bool:
    body = verification.equation_body(eq).lower()
    body = body.replace("{", "").replace("}", "").replace("\\", "")
    return re.search(rf"(?<![a-z0-9_]){re.escape(norm_sym)}(?![a-z0-9_])", body) is not None


def validate_graph(graph, fortran, seed=config.SEED):
    rng = random.Random(seed)
    files_in_dir = sorted(fortran.ids)
    df = graph.df
    rows = []

    # --- root -> file
    has = df[df.relationship == "has"]
    rows.append({"relationship": "has", "check": "file exists in Fortran folder", "edges": len(has),
                 "passed": int(has.target.isin(fortran.ids).sum()), "chance": float("nan")})

    # --- contains
    c = df[df.relationship == "contains"]
    c_ok, c_ok_sci, c_n_sci, chance_ok, c_len = [], 0, 0, 0, 0
    per_file = defaultdict(lambda: [0, 0])
    for f, raw in zip(c.source, c.target):
        n = norm_symbol(raw)
        ok = fortran.has_file(f) and fortran.has_identifier(f, n)
        c_ok.append(ok)
        per_file[f][0] += 1
        per_file[f][1] += ok
        if is_scientific_symbol(n):
            c_n_sci += 1
            c_ok_sci += ok
            c_len += fortran.has_file(f) and fortran.has_identifier_lenient(f, n)
            chance_ok += fortran.has_identifier(rng.choice(files_in_dir), n)
    rows.append({"relationship": "contains", "check": "symbol is an identifier of its file (all)",
                 "edges": len(c), "passed": int(sum(c_ok)), "chance": float("nan")})
    rows.append({"relationship": "contains", "check": "symbol is an identifier of its file (identifier-like symbols)",
                 "edges": c_n_sci, "passed": c_ok_sci, "chance": chance_ok / max(c_n_sci, 1)})

    rows.append({"relationship": "contains", "check": "same, ignoring underscores/sub-script marks (lenient)",
                 "edges": c_n_sci, "passed": c_len, "chance": float("nan")})

    # --- encodes
    e = df[df.relationship == "encodes"]
    e_ok, e_frac, e_file_ok = [], [], 0
    chance_e = 0
    for f, eq in zip(e.source, e.target):
        ids = verification.equation_identifiers(eq)
        e_file_ok += fortran.has_file(f)
        if not fortran.has_file(f) or not ids:
            e_ok.append(False)
            e_frac.append(0.0)
            continue
        frac = sum(fortran.has_identifier(f, t) for t in ids) / len(ids)
        e_frac.append(frac)
        e_ok.append(frac >= config.SRC_TOKEN_MATCH)
        g = rng.choice(files_in_dir)
        chance_e += (sum(fortran.has_identifier(g, t) for t in ids) / len(ids)) >= config.SRC_TOKEN_MATCH
    rows.append({"relationship": "encodes", "check": "source file exists", "edges": len(e), "passed": e_file_ok,
                 "chance": float("nan")})
    rows.append({"relationship": "encodes", "check": f">= {int(config.SRC_TOKEN_MATCH * 100)}% of equation identifiers found in its file",
                 "edges": len(e), "passed": int(sum(e_ok)), "chance": chance_e / max(len(e), 1)})

    # --- hasVariable / hasConstant
    for rel in ("hasVariable", "hasConstant"):
        h = df[df.relationship == rel]
        lex, in_src, n_src, chance_h, n_sci, in_len = 0, 0, 0, 0, 0, 0
        for sym, eq in zip(h.source, h.target):
            n = norm_symbol(sym)
            lex += _symbol_in_text(n, eq)
            if not is_scientific_symbol(n):
                continue
            n_sci += 1
            files = [f for f in graph.eq_files.get(eq, ()) if fortran.has_file(f)]
            if files:
                n_src += 1
                in_src += any(fortran.has_identifier(f, n) for f in files)
                in_len += any(fortran.has_identifier_lenient(f, n) for f in files)
                chance_h += fortran.has_identifier(rng.choice(files_in_dir), n)
        rows.append({"relationship": rel, "check": "symbol occurs in the equation text (all)", "edges": len(h),
                     "passed": lex, "chance": float("nan")})
        rows.append({"relationship": rel, "check": "symbol is an identifier of the equation's file (identifier-like, file found)",
                     "edges": n_src, "passed": in_src, "chance": chance_h / max(n_src, 1)})
        rows.append({"relationship": rel, "check": "same, lenient identifier match",
                     "edges": n_src, "passed": in_len, "chance": float("nan")})

    table = pd.DataFrame(rows)
    table["rate"] = table.passed / table.edges.clip(lower=1)

    # per-file extraction precision (contains edges) for the distribution figure
    pf = pd.DataFrame([{"file": f, "n": n, "precision": ok / n} for f, (n, ok) in per_file.items()])
    eq_frac = pd.Series(e_frac, name="identifier_coverage")
    return table, pf, eq_frac


def symbolic_pass_rate(graph, n_sample=config.SYMBOLIC_SAMPLE, seed=config.SEED):
    """Share of a random equation sample that parses into 'lhs = rhs' with SymPy."""
    rng = random.Random(seed)
    eqs = sorted(graph.eq_files)
    sample = rng.sample(eqs, min(n_sample, len(eqs)))
    ok = [verification.symbolic_check(e) for e in sample]
    return len(sample), sum(ok)
