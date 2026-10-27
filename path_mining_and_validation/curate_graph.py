"""Repair e3sm.csv against the Fortran sources -> data/output/e3sm.csv (the original is backed up as data/output/e3sm_original.csv and is the input here).

 Repairs, in order:

  1. file -> symbol (`contains`): the extracted symbol is re-mapped to the identifier that really exists in that
     Fortran file (case / underscore variants, or the Fortran name that the LLM listed next to the symbol in
     data/intermediate/math.txt). Symbols with no counterpart in the file are removed.
  2. file -> equation (`encodes`): an equation is kept only if a Fortran statement (window of <= 3 statements)
     contains >= 50 % of its identifiers; the edge then carries the source line range.
  3. symbol -> equation (`hasVariable` / `hasConstant`): kept only if the (re-mapped) identifier occurs in the
     matched statement itself, i.e. the variable is traceable to a line.
  4. root -> file (`has`): only files that exist in the Fortran folder.
  5. Fortran-extracted case study: the lake-turbulence chain is read directly from LakeTemperatureMod.F90 and
     added with exact line numbers (origin = fortran_extracted), because the LLM extraction missed it.

IMPORTANT - circularity: the repaired graph is built with the same Fortran files that validate it, so its edge-level
agreement is high *by construction*. The honest quality numbers are the retention figures printed at the end
(how much of the original graph survived) and the held-out numerical replay in verification.
"""

import os
import re
from collections import defaultdict

import pandas as pd

import config
import graph_io
import provenance
from graph_io import is_scientific_symbol, norm_symbol

OUT_CSV = os.path.join(config.ROOT, "data", "output", "e3sm.csv")   # the original is kept as e3sm_original.csv
MATH_TXT = os.path.join(config.ROOT, "data", "intermediate", "math.txt")
LAKE_FILE = "LakeTemperatureMod.F90"

# statements of the lake chain (left-hand side -> role); looked up verbatim in the Fortran file
LAKE_LHS = ["rhow", "drhodz", "n2", "num", "den", "ri", "ke", "kme", "savedtke1"]


def names_from_math_txt():
    """file -> {norm symbol: Fortran name} from the 'Name (symbol)' lines the LLM produced."""
    out = defaultdict(dict)
    if not os.path.exists(MATH_TXT):
        return out
    text = open(MATH_TXT, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"===== File: (.*?) =====", text)[1:]
    for fname, block in zip(blocks[0::2], blocks[1::2]):
        sec = re.search(r"###\s*(?:Global\s+|Local\s+)?Variables:(.*?)(?=###|={5,}|\Z)", block, re.S)
        for line in (sec.group(1).splitlines() if sec else []):
            m = re.match(r"\s*[-*]?\s*(.+?)\s*\((.+)\)\s*$", line)
            if m:
                out[fname.strip()][norm_symbol(m.group(2))] = norm_symbol(m.group(1))
    return out


def make_resolver(fortran, names):
    """resolve(file, norm_symbol) -> identifier present in the file, or None."""
    flat_cache = {}

    def flat_map(f):
        if f not in flat_cache:
            m = {}
            for i in fortran.ids.get(f, ()):
                m.setdefault(re.sub(r"[^a-z0-9]", "", i), i)
            flat_cache[f] = m
        return flat_cache[f]

    def resolve(f, n):
        ids = fortran.ids.get(f)
        if not ids:
            return None
        if n in ids:
            return n
        alt = flat_map(f).get(re.sub(r"[^a-z0-9]", "", n))
        if alt:
            return alt
        nm = names.get(f, {}).get(n)
        if nm and nm in ids:
            return nm
        return None

    return resolve


def lake_case_study(loc):
    """Rows for the lake chain, read from LakeTemperatureMod.F90 (statement text and line numbers are the file's own)."""
    stmts, _ = loc.statements(LAKE_FILE)
    rows, seen = [], set()
    for s in stmts:
        if s.lhs in LAKE_LHS and s.lhs not in seen and "=" in s.text and not s.is_decl and 340 <= s.start <= 430:
            seen.add(s.lhs)
            rows.append(s)
    return rows


def curate(verbose=True):
    A = graph_io.ScienceGraph(os.path.join(config.ROOT, "data", "output", "e3sm_original.csv"))
    F = graph_io.FortranIndex()
    loc = provenance.SourceLocator(F)
    resolve = make_resolver(F, names_from_math_txt())
    eq_trace, _ = provenance.trace_equations(A, F, loc)
    n_orig = {"contains": 0, "encodes": 0, "var": 0, "has": 0}
    out = []

    # 4. root -> file
    for f in sorted(A.files):
        if F.has_file(f):
            out.append((config.ROOT_NODE if hasattr(config, "ROOT_NODE") else "E3SM", f, "has", f, "", ""))
    n_orig["has"] = int((A.df.relationship == "has").sum())

    # 1. file -> symbol
    kept_contains = set()
    n_orig["contains"] = int((A.df.relationship == "contains").sum())
    for f, syms in A.file_symbols.items():
        for raw in syms:
            n = norm_symbol(raw)
            if not is_scientific_symbol(n):
                continue
            r = resolve(f, n)
            if r and (f, r) not in kept_contains:
                kept_contains.add((f, r))
                out.append((f, r, "contains", f, "", ""))

    # 2. + 3. equations and their variables
    n_orig["encodes"] = int((A.df.relationship == "encodes").sum())
    n_orig["var"] = int(A.df.relationship.isin(["hasVariable", "hasConstant"]).sum())
    kept_eq = eq_trace[eq_trace.status.isin(["located", "partial"])]
    n_var_kept = 0
    for r in kept_eq.itertuples(index=False):
        stmts, _ = loc.statements(r.file)
        win = [s for s in stmts if s.start >= r.start and s.end <= r.end]
        ids = set().union(*(s.ids for s in win)) if win else set()
        out.append((r.file, r.equation, "encodes", r.file, int(r.start), int(r.end)))
        for n, kind in sorted(A.eq_symbols.get(r.equation, ())):
            if not is_scientific_symbol(n):
                continue
            ident = resolve(r.file, n)
            if ident and ident in ids:
                out.append((ident, r.equation, "hasVariable" if kind == "var" else "hasConstant", r.file, int(r.start), int(r.end)))
                n_var_kept += 1
                if (r.file, ident) not in kept_contains:       # variable proven present in the file by its own line
                    kept_contains.add((r.file, ident))
                    out.append((r.file, ident, "contains", r.file, int(r.start), int(r.start)))

    # 5. Fortran-extracted lake chain
    lake = lake_case_study(loc)
    for s in lake:
        eq_text = " ".join(s.text.split())
        out.append((LAKE_FILE, eq_text, "encodes", LAKE_FILE, s.start, s.end))
        for t in sorted(s.ids):
            if t in ("max", "abs", "sqrt", "exp", "c", "j", "fc") or len(t) < 2:
                continue
            out.append((t, eq_text, "hasVariable", LAKE_FILE, s.start, s.end))
    if lake:
        out.append(("E3SM", LAKE_FILE, "has", LAKE_FILE, "", ""))

    df = pd.DataFrame(out, columns=["source", "target", "relationship", "file", "line_start", "line_end"]).drop_duplicates()
    df.to_csv(OUT_CSV, index=False)
    stats = {
        "original_rows_unique": A.n_unique_rows,
        "curated_rows": len(df),
        "contains_orig": n_orig["contains"], "contains_kept": int((df.relationship == "contains").sum()),
        "encodes_orig": n_orig["encodes"], "encodes_kept": int((df.relationship == "encodes").sum()) - len(lake),
        "var_orig": n_orig["var"], "var_kept": n_var_kept,
        "lake_statements_added": len(lake),
    }
    if verbose:
        for k, v in stats.items():
            print(f"{k:28s} {v}")
        print("wrote", OUT_CSV)
    return df, stats


if __name__ == "__main__":
    curate()
