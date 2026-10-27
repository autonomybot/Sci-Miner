"""Chain-of-equations verification (paper Eq. 7).

Three static checks are implemented here; the fourth (numerical replay
against E3SM output) lives in numerical_validation.py.

  S_src  source provenance : each equation is traceable to a Fortran file that
                             really contains most of the equation's identifiers
  S_var  variable consistency : each variable on the path occurs in the source
                             file(s) of the equations it connects
  S_sym  symbolic consistency : each equation parses into a valid
                             left-hand-side = right-hand-side form (SymPy)
"""

import re
import warnings
from functools import lru_cache

import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

import config
from graph_io import norm_symbol

warnings.filterwarnings("ignore")

_FUNCS = {"exp", "ln", "log", "sin", "cos", "tan", "sqrt", "min", "max", "abs", "mod", "frac", "left", "right",
          "cdot", "times", "partial", "sum", "int", "text", "mathrm", "if", "then", "else", "end", "do", "sign",
          "boxed", "oxed", "begin", "quad", "real", "dt", "dz"}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def equation_body(eq: str) -> str:
    """Strip markdown/LaTeX delimiters and the leading `code`: prefix."""
    s = eq.strip()
    m = re.match(r"`([^`]*)`\s*:\s*(.*)", s, re.S)
    if m:
        s = m.group(1)                                 # Fortran-style code part
    s = re.sub(r"^\\\(|\\\)$", "", s.strip())
    s = s.strip().strip("$").strip()
    s = re.sub(r"^\\boxed\{|^oxed\{", "", s)
    return s


def equation_identifiers(eq: str) -> set:
    """Identifiers of an equation, comparable with Fortran names
    (`k_{rate}` -> `k_rate`; LaTeX commands, 1-letter tokens and function words removed)."""
    s = equation_body(eq)
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    s = s.replace("{", "").replace("}", "")
    out = set()
    for t in _TOKEN.findall(s):
        t = t.lower().strip("_")
        if len(t) >= 2 and t not in _FUNCS:
            out.add(t)
    return out


def equation_src_support(eq: str, graph, fortran):
    """(supported, best_fraction): share of the equation's identifiers present in
    the best of the files that encode it."""
    ids = equation_identifiers(eq)
    files = [f for f in graph.eq_files.get(eq, ()) if fortran.has_file(f)]
    if not files or not ids:
        return False, 0.0
    best = max(sum(1 for t in ids if fortran.has_identifier(f, t)) / len(ids) for f in files)
    return best >= config.SRC_TOKEN_MATCH, best


def split_top_level_equals(s: str):
    """Split on the first '=' that is not part of <=, >=, ==, /= and not inside braces."""
    depth = 0
    for i, ch in enumerate(s):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth <= 0:
            prev = s[i - 1] if i else ""
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if prev in "<>/=!" or nxt == "=":
                continue
            return s[:i], s[i + 1:]
    return None


def _parse_side(side: str, latex: bool):
    side = side.strip()
    if not side:
        return None
    # array sections / index lists carry no algebraic meaning: a(1:n), f(i,j,k) -> a(i)
    for _ in range(3):
        side = re.sub(r"\(([^()]*[:,][^()]*)\)", "(i)", side)
    if latex:
        return parse_latex(side)
    side = re.sub(r"_(?:r8|r4|rk|r16|dp|sp|kind\w*)\b", "", side, flags=re.I)   # Fortran kind suffixes: 1._r8
    side = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", side)                      # 1.0d-3 -> 1.0e-3
    side = side.replace("^", "**").replace("π", "pi").replace("%", "_")   # derived-type access a%b -> a_b
    return parse_expr(side, transformations=standard_transformations + (implicit_multiplication_application,))


@lru_cache(maxsize=None)
def symbolic_check(eq: str) -> bool:
    """True if both sides of the top-level '=' parse into SymPy expressions."""
    body = equation_body(eq)
    parts = split_top_level_equals(body)
    if parts is None:
        return False
    is_latex = "\\" in body or "{" in body
    for attempt_latex in ([is_latex, not is_latex] if is_latex else [False]):
        try:
            lhs, rhs = (_parse_side(p, attempt_latex) for p in parts)
            if lhs is None or rhs is None:
                continue
            if isinstance(lhs, sympy.Basic) and isinstance(rhs, sympy.Basic):
                return True
        except Exception:
            continue
    return False


def verify_path(path_nodes, graph, fortran, s_num=None):
    """Return the Eq.(7) components and class for one path (list of graph nodes)."""
    eqs = [n[2:] for n in path_nodes if n.startswith("e:")]
    syms = [n[2:] for n in path_nodes if n.startswith("s:")]

    src = [equation_src_support(e, graph, fortran)[0] for e in eqs]
    s_src = sum(src) / len(src) if src else 0.0

    ok_vars = 0
    for v in syms:
        # files of the equations adjacent to this variable on the path
        idx = path_nodes.index("s:" + v)
        adj = [path_nodes[i][2:] for i in (idx - 1, idx + 1) if 0 <= i < len(path_nodes) and path_nodes[i].startswith("e:")]
        files = {f for e in adj for f in graph.eq_files.get(e, ()) if fortran.has_file(f)}
        base = v.split("/")[0]
        ok_vars += any(fortran.has_identifier(f, base) for f in files)
    s_var = ok_vars / len(syms) if syms else 0.0

    sym = [symbolic_check(e) for e in eqs]
    s_sym = sum(sym) / len(sym) if sym else 0.0

    comps = {"var": (config.W_VAR, s_var)}
    if eqs:  # a path made only of variable-variable links has no equation to audit
        comps["src"] = (config.W_SRC, s_src)
        comps["sym"] = (config.W_SYM, s_sym)
    if s_num is not None:
        comps["num"] = (config.W_NUM, s_num)
    wsum = sum(w for w, _ in comps.values())
    V = sum(w * v for w, v in comps.values()) / wsum

    if V >= config.VERIFIED_MIN and all(v >= config.COMPONENT_MIN for _, v in comps.values()):
        label = "verified"
    elif V >= config.PARTIAL_MIN:
        label = "partially_verified"
    else:
        label = "rejected"
    return {"S_src": s_src, "S_var": s_var, "S_sym": s_sym,
            "S_num": s_num if s_num is not None else float("nan"), "V": V, "status": label}
