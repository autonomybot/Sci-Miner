"""Shortest-path discovery and ranking of equation chains (paper Eqs. 5-6).

For a source and target symbol we generate K loop-free candidate paths in the
bipartite symbol-equation graph (Yen's algorithm on hub-penalised edge
weights), then rank them with

    S(P) = (alpha*V(P) + beta*Spec(P) + gamma*Phys(P)) / (1 + lambda*|P|)

  |P|      number of equations on the path
  V(P)     verification score (Eq. 7, verification.py)
  Spec(P)  mean over path nodes of 1 / (1 + log(1 + deg(node)))  (Eq. 6: hubs contribute less)
  Phys(P)  continuity: share of consecutive equation hand-offs whose shared symbol
           is declared (a `contains` edge) in the source file of at least one of the
           two equations
"""

import math
from itertools import islice

import networkx as nx

import config
import verification


def _allowed_graph(graph, source, target):
    """View of the search graph without generic index symbols (except the endpoints)."""
    banned = [n for n in graph.G if n.startswith("s:") and n[2:] in config.INDEX_SYMBOLS and n not in (source, target)]
    return nx.restricted_view(graph.G, banned, [])


def candidate_paths(graph, source, target, k=None):
    H = _allowed_graph(graph, source, target)
    if source not in H or target not in H or not nx.has_path(H, source, target):
        return []
    gen = nx.shortest_simple_paths(H, source, target, weight="w")
    out = []
    for p in islice(gen, k or config.K_CANDIDATES):
        if sum(n.startswith("e:") for n in p) <= config.MAX_EQUATIONS:
            out.append(p)
    return out


def specificity(path, graph):
    return sum(1.0 / (1.0 + math.log1p(graph.degree[n])) for n in path) / len(path)


def continuity(path, graph):
    """Fraction of equation-to-equation hand-offs whose shared symbol is declared in a source file."""
    hops, good = 0, 0
    for i, node in enumerate(path):
        if not node.startswith("s:") or i == 0 or i == len(path) - 1:
            continue
        a, b = path[i - 1], path[i + 1]
        if not (a.startswith("e:") and b.startswith("e:")):
            continue
        hops += 1
        files = set(graph.eq_files.get(a[2:], ())) | set(graph.eq_files.get(b[2:], ()))
        good += bool(files & graph.symbol_files.get(node[2:], set()))
    return good / hops if hops else 1.0   # no intermediate hand-off: nothing to break


def n_equations(path):
    return sum(n.startswith("e:") for n in path)


def score_path(path, graph, fortran, s_num=None):
    ver = verification.verify_path(path, graph, fortran, s_num=s_num)
    spec, phys = specificity(path, graph), continuity(path, graph)
    L = n_equations(path)
    S = (config.ALPHA * ver["V"] + config.BETA * spec + config.GAMMA * phys) / (1.0 + config.LAMBDA * L)
    return {**ver, "Spec": spec, "Phys": phys, "n_equations": L, "n_nodes": len(path), "score": S,
            "max_degree": max(graph.degree[n] for n in path)}


def mine(graph, fortran, source_name, target_name, k=None, s_num=None):
    """Return ranked candidates (best first) for a (source, target) query, or [] if unresolved/unconnected."""
    s, t = graph.symbol_node(source_name), graph.symbol_node(target_name)
    if s is None or t is None:
        return []
    rows = []
    for rank0, p in enumerate(candidate_paths(graph, s, t, k)):
        r = score_path(p, graph, fortran, s_num=s_num)
        r.update({"path": p, "hop_rank": rank0 + 1})
        rows.append(r)
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows
