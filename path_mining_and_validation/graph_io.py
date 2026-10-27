"""Loading and cleaning of the science graph (e3sm.csv) and indexing of the
Fortran sources used as ground truth for provenance checks."""

import math
import os
import re
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

import config

_LATEX_CMD = re.compile(r"\\([A-Za-z]+)")


def norm_symbol(s: str) -> str:
    """Canonical lower-case identifier for a symbol as written in the CSV.

    `\\(\\rho_w\\`, `$\\kappa$` and `rho_w` map to forms comparable with Fortran
    identifiers (Fortran is case-insensitive)."""
    s = str(s).strip().strip("`")
    s = s.replace("\\(", "").replace("\\)", "").replace("$", "")
    s = _LATEX_CMD.sub(lambda m: m.group(1), s)          # \rho -> rho
    s = s.replace("{", "").replace("}", "").replace("\\", "").strip()
    return s.lower()


def is_scientific_symbol(norm: str) -> bool:
    """False for numbers, operators, LaTeX words and multi-word descriptions."""
    if not norm or len(norm) > 40 or not re.search(r"[a-z]", norm):
        return False
    if norm in config.LATEX_STOPWORDS:
        return False
    return re.fullmatch(r"[a-z_][a-z0-9_]*(?:[/'^][a-z0-9_]+)?", norm) is not None


class ScienceGraph:
    """e3sm.csv converted into typed dictionaries plus a bipartite
    symbol-equation graph (the scientific node set V_sci of the paper)."""

    def __init__(self, path=None):
        raw = pd.read_csv(path or config.EDGES_CSV)
        self.n_raw_rows = len(raw)
        self.n_nan_rows = int(raw.isna().any(axis=1).sum())
        raw = raw.iloc[:, :3]                      # curated csv carries extra provenance columns
        df = raw.dropna().drop_duplicates()
        df.columns = ["source", "target", "relationship"]
        self.df = df
        self.n_unique_rows = len(df)

        self.file_symbols = defaultdict(set)     # file -> raw symbols it contains
        self.eq_files = defaultdict(set)         # equation text -> files that encode it
        self.file_equations = defaultdict(set)
        self.eq_symbols = defaultdict(set)       # equation -> {(norm symbol, kind)}
        self.raw_by_norm = defaultdict(Counter)  # norm -> raw spelling counts
        self.symbol_files = defaultdict(set)     # norm -> files declaring it

        for r in df.itertuples(index=False):
            if r.relationship == "contains":
                self.file_symbols[r.source].add(r.target)
                n = norm_symbol(r.target)
                self.raw_by_norm[n][r.target] += 1
                self.symbol_files[n].add(r.source)
            elif r.relationship == "encodes":
                self.eq_files[r.target].add(r.source)
                self.file_equations[r.source].add(r.target)
            elif r.relationship in ("hasVariable", "hasConstant"):
                n = norm_symbol(r.source)
                self.raw_by_norm[n][r.source] += 1
                self.eq_symbols[r.target].add((n, "var" if r.relationship == "hasVariable" else "const"))
        self.files = set(self.file_symbols) | set(self.file_equations)
        self.eq_links = set()                    # direct equation-equation links (none in e3sm.csv)
        self.var_links = set()                   # direct variable-variable links (export graph only)
        self._build_search_graph()

    def _build_search_graph(self):
        """Bipartite graph: symbol nodes ('s:<norm>') and equation nodes ('e:<text>')."""
        G = nx.Graph()
        kept, dropped = set(), set()
        for eq, syms in self.eq_symbols.items():
            if eq not in self.eq_files:
                continue  # an equation without a source file cannot be audited
            for n, kind in syms:
                if is_scientific_symbol(n):
                    kept.add(n)
                    G.add_edge("s:" + n, "e:" + eq, kind=kind)
                else:
                    dropped.add(n)
        for a, b in getattr(self, "var_links", ()):
            na, nb = norm_symbol(a), norm_symbol(b)
            if is_scientific_symbol(na) and is_scientific_symbol(nb) and na != nb:
                G.add_edge("s:" + na, "s:" + nb, kind="var_link")
        for a, b in self.eq_links:
            if a in self.eq_files and b in self.eq_files and a != b:
                G.add_edge("e:" + a, "e:" + b, kind="link")
        self.G = G
        self.n_dropped_symbols = len(dropped)
        self.degree = dict(G.degree())
        for u, v, d in G.edges(data=True):
            d["w"] = 1.0 + config.HUB_EDGE_WEIGHT * (math.log1p(self.degree[u]) + math.log1p(self.degree[v])) / 2.0

    def display(self, node: str) -> str:
        if node.startswith("e:"):
            return node[2:]
        c = self.raw_by_norm.get(node[2:])
        return node[2:] if not c else c.most_common(1)[0][0]

    def symbol_node(self, name: str):
        n = "s:" + norm_symbol(name)
        return n if n in self.G else None

    def stats(self) -> dict:
        G = self.G
        sym = [n for n in G if n.startswith("s:")]
        eq = [n for n in G if n.startswith("e:")]
        comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
        return {
            "raw_rows": self.n_raw_rows,
            "rows_with_missing_fields": self.n_nan_rows,
            "unique_valid_rows": self.n_unique_rows,
            "files": len(self.files),
            "equations_with_source_file": len(self.eq_files),
            "scientific_symbols_kept": len(sym),
            "symbols_dropped_(markup/numbers/descriptions)": self.n_dropped_symbols,
            "equations_in_search_graph": len(eq),
            "edges_in_search_graph": G.number_of_edges(),
            "connected_components": len(comps),
            "largest_component_nodes": comps[0],
            "largest_component_share": round(comps[0] / G.number_of_nodes(), 4),
            "median_symbol_degree": float(pd.Series([G.degree(n) for n in sym]).median()),
            "max_symbol_degree": max(G.degree(n) for n in sym),
        }


class ExportGraph(ScienceGraph):
    """The typed Neo4j export (nodes.csv + edges.csv) used for the paper figures.

    Node types are System/File/Equation/Variable; Equation-Variable and
    Equation-Equation edges (hasVariable, relatesTo) form the scientific graph,
    File hasEquation/hasGlobalVariable edges give provenance."""

    SCI_RELS = {"hasVariable", "relatesTo"}

    def __init__(self, nodes_csv=None, edges_csv=None):
        base = os.path.join(config.ROOT, "data", "output")
        nodes = pd.read_csv(nodes_csv or os.path.join(base, "nodes.csv"))
        edges = pd.read_csv(edges_csv or os.path.join(base, "edges.csv"))
        self.n_raw_rows = len(edges)
        self.n_nan_rows = int(edges.isna().any(axis=1).sum())
        edges = edges.dropna(subset=["source", "target"]).drop_duplicates()
        self.n_unique_rows = len(edges)
        label = dict(zip(nodes.id, nodes.label.astype(str)))
        ntype = dict(zip(nodes.id, nodes.type))

        self.file_symbols = defaultdict(set)
        self.eq_files = defaultdict(set)
        self.file_equations = defaultdict(set)
        self.eq_symbols = defaultdict(set)
        self.raw_by_norm = defaultdict(Counter)
        self.symbol_files = defaultdict(set)
        self.eq_links = set()
        self.var_links = set()
        self.df = edges

        for r in edges.itertuples(index=False):
            a, b, rel = r.source, r.target, r.relationship
            if a not in label or b not in label:
                continue
            ta, tb = ntype[a], ntype[b]
            if ta == "File" and rel in ("hasEquation", "hasGlobalVariable"):
                if tb == "Equation":
                    self.eq_files[label[b]].add(label[a])
                    self.file_equations[label[a]].add(label[b])
                elif tb == "Variable":
                    self.file_symbols[label[a]].add(label[b])
                    n = norm_symbol(label[b])
                    self.raw_by_norm[n][label[b]] += 1
                    self.symbol_files[n].add(label[a])
            elif rel in self.SCI_RELS:
                if ta == "Equation" and tb == "Variable":
                    eq, var = a, b
                elif ta == "Variable" and tb == "Equation":
                    eq, var = b, a
                elif ta == "Equation" and tb == "Equation":
                    self.eq_links.add((label[a], label[b]))
                    continue
                elif ta == "Variable" and tb == "Variable":
                    self.var_links.add((label[a], label[b]))
                    continue
                else:
                    continue
                n = norm_symbol(label[var])
                self.raw_by_norm[n][label[var]] += 1
                self.eq_symbols[label[eq]].add((n, "var"))
        self.files = set(self.file_symbols) | set(self.file_equations)
        self._build_search_graph()


_COMMENT = re.compile(r"!.*")
_IDENT = re.compile(r"[a-z_][a-z0-9_]*")


class FortranIndex:
    """Lower-cased identifier set per Fortran file (comments removed)."""

    def __init__(self, folder=None):
        self.folder = folder or config.FORTRAN_DIR
        self.ids = {}
        for name in os.listdir(self.folder):
            p = os.path.join(self.folder, name)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read().lower()
            except OSError:
                continue
            code = "\n".join(_COMMENT.sub("", ln) for ln in text.splitlines())
            self.ids[name] = set(_IDENT.findall(code))
        # underscore-insensitive view: `k_rate`, `krate` and `kRate` all collapse to `krate`
        self.collapsed = {k: {i.replace("_", "") for i in v} for k, v in self.ids.items()}

    def has_file(self, name):
        return name in self.ids

    def has_identifier(self, name, ident):
        return ident in self.ids.get(name, ())

    def has_identifier_lenient(self, name, ident):
        """Exact match, or a match after dropping underscores / LaTeX sub-script marks."""
        return ident in self.ids.get(name, ()) or re.sub(r"[^a-z0-9]", "", ident) in self.collapsed.get(name, ())
