"""Path recovery: Sci-Miner equation chains vs definition-use chains vs backward program slicing.

Both classical baselines are computed on the same Fortran sources Sci-Miner is validated against
(data/input/fortran), with the same set of generic index symbols removed, so differences come from the method and
not from the input.

  def-use chain    shortest sequence of assignment statements s1..sk such that s1 uses the source variable,
                   s_i defines a variable used by s_{i+1}, and s_k defines the target variable
                   (breadth-first search on the name-based def-use graph).
  backward slice   all assignment statements that can influence the target (transitive closure of defs/uses), either over the
                   whole model ("global") or restricted to the files that define the target ("file-scoped").
                   A slice answers "what affects the target", not "how is the source connected to it".

Metrics per query
  found            a chain exists
  n_stmts          statements on the chain (slice: size of the slice)
  max_var_degree   largest number of statements that define/use a variable on the chain  (hub exposure)
  same_file_hops   share of consecutive chain hops located in the same file (continuity analogue of Phys)
  contains_source  slice reaches the source variable
  precision        |Sci-Miner path statements found in the slice| / |slice|  (how much of the slice is the mined path)
  recall           share of Sci-Miner path statements that lie in the slice / that equal a DU-chain statement

The Sci-Miner side is read from results/path_mining_validation (case_studies.csv, case_path_code_lines.csv,
benchmark_paths.csv) -- rerun path_mining_and_validation/run_all.py to refresh it.
usage:  python path_recovery_comparison.py [--rebuild]
"""
import argparse, os, re, sys, pickle
from collections import defaultdict, deque
import numpy as np
import pandas as pd

import common

ROOT = common.ROOT
F90 = os.path.join(ROOT, "data", "input", "fortran")
RES = os.path.join(ROOT, "results", "path_mining_validation")
CACHE = os.path.join(common.OUT, "fortran_defuse.pkl")

sys.path.insert(0, os.path.join(ROOT, "path_mining_and_validation"))
try:
    import config
    INDEX = set(config.INDEX_SYMBOLS)
except Exception:
    INDEX = {"i", "j", "k", "l", "m", "n", "c", "p", "g", "t", "d", "e", "f", "s", "r", "u", "v", "w", "x", "y", "z"}

INTRINSICS = set("""abs exp log log10 sqrt min max sin cos tan tanh asin acos atan atan2 sinh cosh sign mod nint int real dble
float sum size shape lbound ubound present merge any all count maxval minval floor ceiling epsilon huge tiny
if then else endif do enddo while where elsewhere endwhere select case end call return stop write print read
and or not eq ne lt le gt ge eqv neqv true false r8 r4 i4 i8 dp pi""".split())
KEYWORD_START = re.compile(r"^(do|call|if\s*\(.*\)\s*then|else|end|select|case|where|use|implicit|integer|real|logical|"
                           r"character|type|public|private|interface|contains|subroutine|function|module|program|"
                           r"allocate|deallocate|write|print|read|open|close|return|stop|exit|cycle|goto|data|parameter|"
                           r"intent|dimension|save|common|external|namelist|nullify|forall)\b")
NUM = re.compile(r"(?<![\w%])\d+\.?\d*(?:[ed][+-]?\d+)?(?:_\w+)?|\.\d+(?:[ed][+-]?\d+)?(?:_\w+)?", re.I)
DOTOP = re.compile(r"\.(?:and|or|not|eq|ne|lt|le|gt|ge|eqv|neqv|true|false)\.", re.I)
STR = re.compile(r"'[^']*'|\"[^\"]*\"")
IDENT = re.compile(r"[a-z_]\w*(?:%[a-z_]\w*)*")
ASSIGN = re.compile(r"^([a-z_]\w*(?:\([^=]*?\))?(?:%[a-z_]\w*(?:\([^=]*?\))?)*)\s*=(?![=>])\s*(.+)$")


def _strip_inline_if(s):
    m = re.match(r"^if\s*\(", s)
    if not m:
        return s
    depth, i = 0, m.end() - 1
    for i in range(m.end() - 1, len(s)):
        depth += (s[i] == "(") - (s[i] == ")")
        if depth == 0:
            break
    rest = s[i + 1:].strip()
    return "" if rest.startswith("then") else rest


def _names(expr):
    e = DOTOP.sub(" ", STR.sub(" ", expr.lower()))
    e = NUM.sub(" ", e)
    out = []
    for tok in IDENT.findall(e):
        last = tok.split("%")[-1]
        if last not in INTRINSICS:
            out.append(last)
    return out


def parse_file(path):
    try:
        raw = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except OSError:
        return []
    stmts, buf, start = [], "", 0
    for ln, line in enumerate(raw, 1):
        s = re.sub(r"!.*$", "", line) if "'" not in line and '"' not in line else line.split("!")[0]
        s = s.strip()
        if not s:
            continue
        if not buf:
            start = ln
        if s.endswith("&"):
            buf += " " + s[:-1].lstrip("&").strip()
            continue
        buf += " " + s.lstrip("&").strip()
        for part in buf.split(";"):
            part = _strip_inline_if(part.strip().lower())
            if not part or KEYWORD_START.match(part) or "::" in part:
                buf = ""
                continue
            m = ASSIGN.match(part)
            if m:
                lhs_txt = m.group(1)
                base = re.sub(r"\([^()]*\)", "", re.sub(r"\([^()]*\)", "", lhs_txt))      # drop (nested) subscripts
                base_names = _names(base)
                if base_names:
                    d = base_names[-1]
                    subs = set(_names(lhs_txt)) - {d}
                    rhs = set(_names(m.group(2)))
                    # variables used: right-hand side plus subscripts of the left-hand side (a self-reference stays a use)
                    stmts.append((start, d, frozenset(rhs | subs), part[:160]))
        buf = ""
    return stmts


def build(rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        return pickle.load(open(CACHE, "rb"))
    files, stmts = [], []
    for fn in sorted(os.listdir(F90)):
        if fn.lower().endswith((".f90", ".f", ".f95", ".f03")):
            for ln, d, u, txt in parse_file(os.path.join(F90, fn)):
                files.append(fn); stmts.append((len(stmts), fn, ln, d, u, txt))
    pickle.dump(stmts, open(CACHE, "wb"))
    return stmts


class DU:
    def __init__(self, stmts):
        self.s = stmts
        self.defs, self.users = defaultdict(list), defaultdict(list)
        for sid, fn, ln, d, u, txt in stmts:
            self.defs[d].append(sid)
            for v in u:
                self.users[v].append(sid)

    def deg(self, v): return len(self.defs.get(v, ())) + len(self.users.get(v, ()))

    def chain(self, src, tgt, drop_index=True, max_nodes=2_000_000):
        """BFS over variables; edge u -> d for every statement defining d from u."""
        banned = (INDEX if drop_index else set()) - {src, tgt}
        prev, q = {src: None}, deque([src])
        while q and len(prev) < max_nodes:
            v = q.popleft()
            if v == tgt:
                break
            for sid in self.users.get(v, ()):
                d = self.s[sid][3]
                if d in banned or d in prev:
                    continue
                prev[d] = (v, sid)
                q.append(d)
        if tgt not in prev:
            return None
        out, v = [], tgt
        while prev[v] is not None:
            v, sid = prev[v][0], prev[v][1]
            out.append(sid)
        return out[::-1]

    def slice(self, tgt, files=None, drop_index=True):
        banned = (INDEX if drop_index else set()) - {tgt}
        seen, work, st = {tgt}, [tgt], set()
        while work:
            v = work.pop()
            for sid in self.defs.get(v, ()):
                if files is not None and self.s[sid][1] not in files:
                    continue
                st.add(sid)
                for u in self.s[sid][4]:
                    if u not in seen and u not in banned:
                        seen.add(u); work.append(u)
        return st, seen


def norm_stmt(t): return re.sub(r"[\s_]", "", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    stmts = build(a.rebuild)
    G = DU(stmts)
    print(f"parsed {len(stmts):,} assignment statements, {len(G.defs):,} defined variables")

    # ----- paper case studies -------------------------------------------------------------------------------
    cs = pd.read_csv(os.path.join(RES, "case_studies.csv"))
    cs = cs[cs["graph"] == "e3sm.csv"]
    lines = pd.read_csv(os.path.join(RES, "case_path_code_lines.csv"))
    fortran_names = {  # label -> (source, target) as written in the Fortran source (graph labels use paper notation)
        "Lake chain (Fortran)": ("rhow", "savedtke1"), "Lake chain: density to Richardson": ("rhow", "ri"),
        "Lake chain: temperature to buoyancy": ("t_lake", "n2"), "Fig.5A lake pathway": ("rhow", "savedtke1"),
        "Fig.5B methane pathway": ("t_soisno", "conc_ch4"), "Soil temperature to methane": ("t_soisno", "conc_ch4"),
        "Density to Richardson number": ("rhow", "ri"), "Mixing energy to buoyancy": ("tke", "n2"),
        "von Karman to Richardson number": ("vkc", "ri")}
    rows = []
    for _, r in cs.iterrows():
        label = r["case"]
        if label not in fortran_names:
            continue
        s, t = fortran_names[label]
        ch = G.chain(s, t)
        sl_g, vars_g = G.slice(t)
        tfiles = {G.s[i][1] for i in G.defs.get(t, ())}
        sl_f, vars_f = G.slice(t, files=tfiles)
        mined = lines[(lines["case"] == f"e3sm.csv::{label}")]
        mined_txt = {norm_stmt(x) for x in mined["snippet"].astype(str)}
        in_slice = lambda S: sum(norm_stmt(G.s[i][5]) in mined_txt or any(m.startswith(norm_stmt(G.s[i][5])[:40]) for m in mined_txt) for i in S)
        du_txt = {norm_stmt(G.s[i][5])[:40] for i in (ch or [])}
        row = dict(query=label, source=s, target=t, sci_miner=r["status"] if r["status"] != "path found" else "path found",
                   sci_miner_equations=r.get("n_equations"), sci_miner_V=r.get("V"), sci_miner_status=r.get("status_class"),
                   du_found=ch is not None, du_stmts=len(ch) if ch else np.nan,
                   du_max_var_degree=max([G.deg(G.s[i][3]) for i in ch] + [G.deg(s)]) if ch else np.nan,
                   du_same_file_hops=(np.mean([G.s[a_][1] == G.s[b][1] for a_, b in zip(ch[:-1], ch[1:])]) if ch and len(ch) > 1 else (1.0 if ch else np.nan)),
                   slice_global_stmts=len(sl_g), slice_global_vars=len(vars_g), slice_global_has_source=s in vars_g,
                   slice_file_stmts=len(sl_f), slice_file_has_source=s in vars_f)
        if len(mined):
            n_m = len(mined)
            row["mined_stmts_located"] = int((mined["status"] == "located").sum())
            row["slice_file_precision"] = sum(1 for m in mined_txt if any(norm_stmt(G.s[i][5])[:40] == m[:40] for i in sl_f)) / max(len(sl_f), 1)
            row["slice_file_recall"] = sum(1 for m in mined_txt if any(norm_stmt(G.s[i][5])[:40] == m[:40] for i in sl_f)) / max(len(mined_txt), 1)
            row["du_recall_of_mined"] = len({m[:40] for m in mined_txt} & du_txt) / max(len(mined_txt), 1)
        rows.append(row)
        if ch:
            row["du_chain"] = " || ".join(f"{G.s[i][1]}:{G.s[i][2]} {G.s[i][5][:70]}" for i in ch)
    cases = pd.DataFrame(rows)
    cases.to_csv(os.path.join(common.OUT, "path_recovery_cases.csv"), index=False)

    # ----- the 150 random benchmark queries ------------------------------------------------------------------
    bp = pd.read_csv(os.path.join(RES, "benchmark_paths.csv"))
    bq = bp[bp["rank"] == 1].drop_duplicates("query")
    bs = pd.read_csv(os.path.join(RES, "benchmark_summary.csv")).set_index("metric")["value"]
    sm_connected = float(bs["connected_share"])
    bsum = []
    for _, r in bq.iterrows():
        ch = G.chain(str(r["source"]).lower(), str(r["target"]).lower())
        bsum.append(dict(query=r["query"], source=r["source"], target=r["target"], sm_equations=r["n_equations"], sm_V=r["V"],
                         sm_status=r["status"], sm_max_degree=r["max_degree"], du_found=ch is not None,
                         du_stmts=len(ch) if ch else np.nan,
                         du_max_var_degree=max(G.deg(G.s[i][3]) for i in ch) if ch else np.nan,
                         du_same_file_hops=(np.mean([G.s[a_][1] == G.s[b][1] for a_, b in zip(ch[:-1], ch[1:])]) if ch and len(ch) > 1 else (1.0 if ch else np.nan))))
    bd = pd.DataFrame(bsum)
    bd.to_csv(os.path.join(common.OUT, "path_recovery_benchmark.csv"), index=False)
    summ = pd.DataFrame([
        dict(method="Sci-Miner (equation chain)", queries=len(bd), connected_share=sm_connected,   # share of all benchmark queries
             median_steps=bd["sm_equations"].median(), median_max_degree=bd["sm_max_degree"].median(),
             verified_share=(bd["sm_status"] == "verified").mean(), partially_or_verified=bd["sm_status"].isin(["verified", "partially_verified"]).mean()),
        dict(method="Definition-use chain (name-based, on the Sci-Miner-connected queries)", queries=len(bd), connected_share=bd["du_found"].mean(),
             median_steps=bd["du_stmts"].median(), median_max_degree=bd["du_max_var_degree"].median(),
             verified_share=np.nan, partially_or_verified=np.nan)])
    summ.to_csv(os.path.join(common.OUT, "path_recovery_summary.csv"), index=False)
    for _, r in cases.iterrows():
        if isinstance(r.get("du_chain"), str) and r["query"] in ("Lake chain (Fortran)", "Fig.5B methane pathway"):
            print(r["query"], "->", r["du_chain"])
    print(cases.drop(columns=["du_chain"], errors="ignore").round(3).to_string(index=False))
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
