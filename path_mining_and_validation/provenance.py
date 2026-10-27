"""Line-level provenance: which Fortran statement does each graph equation / variable come from?

For every equation of the science graph the source file named by the graph's
`encodes` edge is parsed into Fortran statements (continuation lines joined,
comments removed). The statement - or short window of consecutive statements -
that best explains the equation is reported with its line numbers:

    coverage = share of the equation's identifiers found in the matched statement(s)

    located        coverage >= 0.8
    partial        0.5 <= coverage < 0.8
    unlocated      coverage < 0.5   (identifiers exist, but never together in one place)
    no_identifier  equation has no multi-letter identifier to match (pure symbols)
    file_missing   the named file is not in the Fortran folder

A chance baseline repeats the search in a random *other* file.
"""

import random
import re
from collections import defaultdict

import pandas as pd

import config
import verification
from graph_io import is_scientific_symbol, norm_symbol

_IDENT = re.compile(r"[a-z_][a-z0-9_]*")
_NUM = re.compile(r"(?<![a-z_0-9])\d+\.?\d*(?:[ed][+-]?\d+)?|\.\d+(?:[ed][+-]?\d+)?")
_TYPE_START = re.compile(r"^\s*(real|integer|logical|character|complex|type|class|double\s+precision|parameter)\b")
_ASSIGN = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*(?:\([^=]*\))?\s*(?:%\s*[a-z_][a-z0-9_]*\s*(?:\([^=]*\))?)*\s*=(?!=)")
MAX_WINDOW = 3


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            break
        out.append(ch)
    return "".join(out)


class Statement:
    __slots__ = ("start", "end", "text", "ids", "nums", "lhs", "is_decl")

    def __init__(self, start, end, text):
        t = text.lower()
        self.start, self.end, self.text = start, end, text.strip()
        self.ids = set(_IDENT.findall(t))
        self.nums = set()
        for x in _NUM.findall(t):
            try:
                self.nums.add(round(float(x.replace("d", "e")), 12))
            except ValueError:
                pass
        m = _ASSIGN.match(t)
        self.lhs = m.group(1) if m and "::" not in t else None
        self.is_decl = bool(_TYPE_START.match(t)) or "::" in t


def parse_statements(path):
    """Fortran file -> list of Statement (continuations joined, comments dropped)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    stmts, buf, start = [], [], None
    for no, raw in enumerate(lines, start=1):
        code = _strip_comment(raw).rstrip()
        if not code.strip() or code.lstrip().startswith("#"):
            continue
        body = code.strip()
        if buf and body.startswith("&"):
            body = body[1:]
        cont = body.endswith("&")
        if cont:
            body = body[:-1]
        if start is None:
            start = no
        buf.append(body)
        if not cont:
            stmts.append(Statement(start, no, " ".join(buf)))
            buf, start = [], None
    return stmts


class SourceLocator:
    """Caches parsed files and answers equation / variable location queries."""

    def __init__(self, fortran):
        self.fortran = fortran
        self._cache = {}

    def statements(self, name):
        if name not in self._cache:
            stmts = parse_statements(f"{self.fortran.folder}/{name}") if self.fortran.has_file(name) else []
            index = defaultdict(list)
            for i, s in enumerate(stmts):
                for t in s.ids:
                    index[t].append(i)
            self._cache[name] = (stmts, index)
        return self._cache[name]

    # ------------------------------------------------------------ equations
    def locate_equation(self, eq, name):
        """Best statement window for `eq` in file `name`.

        Returns dict(status, coverage, start, end, window, lhs_match, number_hits, snippet)."""
        if not self.fortran.has_file(name):
            return {"status": "file_missing", "coverage": 0.0}
        ids = verification.equation_identifiers(eq)
        if not ids:
            return {"status": "no_identifier", "coverage": 0.0}
        stmts, index = self.statements(name)
        cand = defaultdict(int)
        for t in ids:
            for i in index.get(t, ()):
                cand[i] += 1
        if not cand:
            return {"status": "unlocated", "coverage": 0.0}

        eq_nums = set()
        for x in _NUM.findall(verification.equation_body(eq).lower()):
            try:
                eq_nums.add(round(float(x.replace("d", "e")), 12))
            except ValueError:
                pass
        body = verification.equation_body(eq)
        parts = verification.split_top_level_equals(body)
        eq_lhs = None
        if parts:
            lhs_ids = [t for t in re.sub(r"[{}\\]", "", parts[0]).lower().replace(" ", "")[:60].split() or []]
            m = _IDENT.findall(re.sub(r"\\[A-Za-z]+|[{}]", "", parts[0]).lower())
            eq_lhs = m[0] if m else None

        best = None
        top_starts = sorted(cand, key=lambda i: -cand[i])[:40]
        for i in top_starts:
            for w in range(1, MAX_WINDOW + 1):
                win = stmts[i:i + w]
                if len(win) < w:
                    break
                got = set().union(*(s.ids for s in win))
                cov = len(ids & got) / len(ids)
                lhs_ok = bool(eq_lhs and win[0].lhs == eq_lhs)
                nums_ok = len(eq_nums & set().union(*(s.nums for s in win)))
                # prefer high coverage, then matching assignment target, matching numbers, then compactness
                key = (round(cov, 6), lhs_ok, nums_ok, -w, -len(win[0].ids))
                if best is None or key > best[0]:
                    best = (key, i, w, cov, lhs_ok, nums_ok)
        _, i, w, cov, lhs_ok, nums_ok = best
        win = stmts[i:i + w]
        status = "located" if cov >= 0.8 else "partial" if cov >= 0.5 else "unlocated"
        return {"status": status, "coverage": cov, "start": win[0].start, "end": win[-1].end, "window": w,
                "lhs_match": lhs_ok, "number_hits": nums_ok, "n_numbers": len(eq_nums),
                "snippet": " ".join(s.text for s in win)[:300]}

    # ------------------------------------------------------------ variables
    def locate_variable(self, norm_sym, name):
        """Declaration line (or first use) of a symbol in file `name`."""
        stmts, index = self.statements(name)
        ident = norm_sym if norm_sym in index else None
        if ident is None:  # lenient: ignore underscores
            flat = re.sub(r"[^a-z0-9]", "", norm_sym)
            for k in index:
                if k.replace("_", "") == flat:
                    ident = k
                    break
        if ident is None:
            return {"status": "absent"}
        uses = [stmts[i] for i in index[ident]]
        decl = [s for s in uses if s.is_decl]
        if decl:
            return {"status": "declared", "line": decl[0].start, "n_uses": len(uses), "snippet": decl[0].text[:200]}
        return {"status": "used_only", "line": uses[0].start, "n_uses": len(uses), "snippet": uses[0].text[:200]}


# --------------------------------------------------------------- whole graph
def trace_equations(graph, fortran, locator=None):
    """One row per (file, equation) `encodes` edge."""
    loc = locator or SourceLocator(fortran)
    rows = []
    for eq, files in graph.eq_files.items():
        for f in sorted(files):
            r = loc.locate_equation(eq, f)
            rows.append({"file": f, "equation": eq, **r})
    return pd.DataFrame(rows), loc


def trace_variables(graph, fortran, locator):
    """One row per (file, symbol) `contains` edge with the declaration line."""
    rows = []
    for f, syms in graph.file_symbols.items():
        if not fortran.has_file(f):
            rows.append({"file": f, "n_missing_file": len(syms)})
            continue
        for raw in syms:
            n = norm_symbol(raw)
            if not is_scientific_symbol(n):
                continue
            rows.append({"file": f, "symbol": raw, "norm": n, **locator.locate_variable(n, f)})
    return pd.DataFrame(rows)


def trace_variable_in_equation(graph, eq_trace, locator):
    """For each located equation, is each of its linked variables inside the matched statement?"""
    rows = []
    loc_eq = eq_trace[eq_trace.status.isin(["located", "partial"])]
    for r in loc_eq.itertuples(index=False):
        stmts, _ = locator.statements(r.file)
        win = [s for s in stmts if s.start >= r.start and s.end <= r.end]
        ids = set().union(*(s.ids for s in win)) if win else set()
        flat = {i.replace("_", "") for i in ids}
        for n, kind in graph.eq_symbols.get(r.equation, ()):
            if not is_scientific_symbol(n):
                continue
            rows.append({"file": r.file, "line": r.start, "symbol": n, "kind": kind,
                         "in_statement": n in ids or re.sub(r"[^a-z0-9]", "", n) in flat})
    return pd.DataFrame(rows)


def chance_baseline(graph, fortran, locator, n=1500, seed=config.SEED):
    """Same equation search, but in a random other file."""
    rng = random.Random(seed)
    files = sorted(fortran.ids)
    eqs = [(eq, f) for eq, fs in graph.eq_files.items() for f in fs if fortran.has_file(f)]
    sample = rng.sample(eqs, min(n, len(eqs)))
    out = []
    for eq, f in sample:
        g = rng.choice([x for x in rng.sample(files, 5) if x != f] or files)
        out.append(locator.locate_equation(eq, g))
    return pd.DataFrame(out)
