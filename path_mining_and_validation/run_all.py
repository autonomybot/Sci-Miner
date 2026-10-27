"""Run every experiment and write all tables / raw results to RESULT_DIR.

    python run_all.py            # full run (about 5-10 minutes)
    python make_figures.py       # figures from the saved results

Outputs (RESULT_DIR):
  graph_statistics.csv            size / cleaning of both graphs
  graph_vs_fortran.csv            edge-level agreement of e3sm.csv with the Fortran sources
  equation_provenance.csv         every equation -> file, line range, coverage, source snippet
  equation_provenance_summary.csv status counts vs random-file baseline
  variable_provenance.csv         every (file, symbol) -> declaration line
  variable_provenance_summary.csv
  variable_in_statement.csv       equation variables located inside the matched statement
  per_file_provenance.csv         traceability per Fortran file
  symbolic_check_summary.csv
  benchmark_paths.csv             random-pair path mining (all ranked candidates)
  benchmark_summary.csv           ranked vs naive shortest path
  case_studies.csv                paper case studies on both graphs
  numerical_replay.csv, surrogates_lake.csv, surrogates_methane.csv, data_plausibility.csv
  raw_numeric.npz                 arrays used by the figures
"""

import json
import os
import random
import time

import networkx as nx
import numpy as np
import pandas as pd

import config
import graph_io
import graph_validation
import numerical_validation as nv
import path_mining as pm
import provenance
import verification

OUT = config.RESULT_DIR
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    return df


# ------------------------------------------------------------------ 1. graphs
log("loading graphs and Fortran sources")
fortran = graph_io.FortranIndex()
A = graph_io.ScienceGraph()          # data/output/e3sm.csv
B = graph_io.ExportGraph()           # data/output/nodes.csv + edges.csv (typed Neo4j export)
save(pd.DataFrame({"e3sm.csv": A.stats(), "neo4j export": B.stats()}).rename_axis("metric").reset_index(),
     "graph_statistics.csv")

# --------------------------------------------- 2. graph-wide validation vs Fortran
log("edge-level validation of e3sm.csv against the Fortran files")
edge_table, per_file_contains, eq_cov = graph_validation.validate_graph(A, fortran)
save(edge_table, "graph_vs_fortran.csv")

log("line-level provenance of every equation")
locator = provenance.SourceLocator(fortran)
eq_trace, _ = provenance.trace_equations(A, fortran, locator)
save(eq_trace, "equation_provenance.csv")
chance = provenance.chance_baseline(A, fortran, locator)
order = ["located", "partial", "unlocated", "no_identifier", "file_missing"]
summ = pd.DataFrame({
    "equations": eq_trace.status.value_counts().reindex(order).fillna(0).astype(int),
    "share": eq_trace.status.value_counts(normalize=True).reindex(order).fillna(0),
    "chance_share": chance.status.value_counts(normalize=True).reindex(order).fillna(0),
}).rename_axis("status").reset_index()
save(summ, "equation_provenance_summary.csv")
loc = eq_trace[eq_trace.status == "located"]
extra = {
    "located_with_matching_assignment_target": float(loc.lhs_match.astype(bool).mean()),
    "located_with_>=1_numeric_literal_match": float((loc.number_hits.fillna(0) > 0).mean()),
    "median_statement_window": float(loc.window.median()),
}
json.dump(extra, open(os.path.join(OUT, "equation_provenance_extra.json"), "w"), indent=1)
np.save(os.path.join(OUT, "chance_coverage.npy"), chance.coverage.to_numpy())

log("per-file traceability")
pf = eq_trace.groupby("file").agg(equations=("equation", "size"),
                                  located=("status", lambda s: (s == "located").sum()),
                                  partial=("status", lambda s: (s == "partial").sum()))
pf["traced_share"] = (pf.located + pf.partial) / pf.equations
pf = pf.sort_values("equations", ascending=False).reset_index()
save(pf, "per_file_provenance.csv")

log("variable provenance (declaration lines)")
var_trace = provenance.trace_variables(A, fortran, locator)
save(var_trace, "variable_provenance.csv")
vt = var_trace.dropna(subset=["status"])
vsum = vt.status.value_counts().reindex(["declared", "used_only", "absent"]).fillna(0).astype(int)
save(pd.DataFrame({"status": vsum.index, "symbols": vsum.values, "share": vsum.values / vsum.sum()}),
     "variable_provenance_summary.csv")
log("variables inside the matched statement")
vin = provenance.trace_variable_in_equation(A, eq_trace, locator)
save(vin, "variable_in_statement.csv")

log("symbolic consistency on a random equation sample")
n_s, ok_s = graph_validation.symbolic_pass_rate(A)
save(pd.DataFrame([{"sampled_equations": n_s, "parse_to_lhs=rhs": ok_s, "rate": ok_s / n_s}]), "symbolic_check_summary.csv")

# ----------------------------------------------------------------- 3. numerics
log("numerical replay and surrogate validation")
lake, zl = nv.load_lake()
replay = save(nv.replay_lake_chain(lake, zl), "numerical_replay.csv")
s_lake, extra_lake = nv.lake_surrogates(lake)
s_meth, extra_meth = nv.methane_surrogates(lake)
save(s_lake, "surrogates_lake.csv")
save(s_meth, "surrogates_methane.csv")
plaus = pd.DataFrame([
    {"column": c, "min": lake[c].min(), "median": lake[c].median(), "max": lake[c].max(),
     "corr_with_TLAKE_level1": float(np.corrcoef(lake[c], lake["TLAKE[levlak=1,lndgrid=1]"])[0, 1])}
    for c in ["TKE1[lndgrid=1]", "LAKE_WS[lndgrid=1]", "LAKE_KS[lndgrid=1]", "TLAKE[levlak=1,lndgrid=1]"]])
save(plaus, "data_plausibility.csv")

raw = {"T1": lake["TLAKE[levlak=1,lndgrid=1]"].to_numpy(), "tke1": lake["TKE1[lndgrid=1]"].to_numpy(),
       "time": pd.to_datetime(lake["timestamp"], errors="coerce").astype("int64").to_numpy()}
T = nv.stack(lake, "TLAKE", nv.NLEVLAK, "levlak")[:, :nv.NACT]
f = nv.stack(lake, "LAKEICEFRAC", nv.NLEVLAK, "levlak")[:, :nv.NACT]
raw["rho_obs"] = nv.stack(lake, "LAKE_RHOW", nv.NLEVLAK, "levlak")[:, :nv.NACT].ravel()
raw["rho_rep"] = ((1 - f) * 1000.0 * (1.0 - 1.9549e-05 * np.abs(T - nv.TDMAX) ** 1.68) + f * nv.DENICE).ravel()
n2 = nv.stack(lake, "LAKE_N2", nv.NLEVLAK, "levlak")[:, :nv.NACT]
rho_m = nv.stack(lake, "LAKE_RHOW", nv.NLEVLAK, "levlak")[:, :nv.NACT]
raw["n2_obs"] = n2[:, :-1].ravel()
raw["n2_rep"] = (nv.GRAV / rho_m[:, :-1] * (rho_m[:, 1:] - rho_m[:, :-1]) / (zl[1:nv.NACT] - zl[:nv.NACT - 1])[None, :]).ravel()
for tag, (x, y, info) in {"lake1": extra_lake[1]}.items():
    tr, te, p = info
    raw[f"{tag}_x"], raw[f"{tag}_y"], raw[f"{tag}_te"] = x, y, te
    raw[f"{tag}_paper"] = nv.lake_compact(x)
    raw[f"{tag}_refit"] = p if p is not None else np.full_like(x, np.nan)
    raw[f"{tag}_mlp"] = np.full_like(x, np.nan)
    raw[f"{tag}_mlp"][te] = nv._mlp(x[tr], y[tr], x[te])
mj = sorted(extra_meth)[0]
x, y, info = extra_meth[mj]
tr, te, p = info
raw["meth_level"] = np.array([mj])
raw["meth_x"], raw["meth_y"], raw["meth_te"] = x, y, te
raw["meth_paper"] = nv.methane_compact(x)
raw["meth_refit"] = p if p is not None else np.full_like(x, np.nan)
raw["meth_mlp"] = np.full_like(x, np.nan)
raw["meth_mlp"][te] = nv._mlp(x[tr], y[tr], x[te])
np.savez_compressed(os.path.join(OUT, "raw_numeric.npz"), **raw)

# S_num for the lake chain = mean R^2 over the replayable steps
s_num_lake = float(replay.S_num.fillna(0).mean())   # unusable replay steps count as unverified
log(f"lake-chain S_num over {replay.replayable.sum()} replayable steps = {s_num_lake:.4f}")

# ------------------------------------------------------ 4. random-pair benchmark
log("random-pair path-mining benchmark on e3sm.csv")
rng = random.Random(config.SEED)
comp = max(nx.connected_components(A.G), key=len)
pool = [n for n in comp if n.startswith("s:") and n[2:] not in config.INDEX_SYMBOLS
        and A.degree[n] >= config.MIN_SYMBOL_DEGREE]
pool.sort()
pairs = set()
while len(pairs) < config.N_RANDOM_PAIRS:
    a, b = rng.sample(pool, 2)
    pairs.add((a, b))
rows, naive_rows = [], []
for qi, (a, b) in enumerate(sorted(pairs)):
    cands = pm.mine(A, fortran, a[2:], b[2:])
    for r in cands:
        rows.append({"query": qi, "source": a[2:], "target": b[2:], **{k: v for k, v in r.items() if k != "path"},
                     "path": " | ".join(A.display(n)[:60] for n in r["path"])})
    # naive baseline: plain hop-count shortest path on the same restricted graph
    try:
        H = pm._allowed_graph(A, a, b)
        p = nx.shortest_path(H, a, b)
        r = pm.score_path(p, A, fortran)
        naive_rows.append({"query": qi, "n_candidates": len(cands), **{k: v for k, v in r.items()}})
    except nx.NetworkXNoPath:
        pass
    if qi % 25 == 0:
        log(f"  pair {qi}/{len(pairs)}")
bench = save(pd.DataFrame(rows), "benchmark_paths.csv")
naive = pd.DataFrame(naive_rows)
save(naive, "benchmark_naive.csv")

best = bench[bench["rank"] == 1]
n_conn = bench["query"].nunique()
bsum = {
    "queries": len(pairs), "connected": int(n_conn), "connected_share": n_conn / len(pairs),
    "median_equations_best": float(best.n_equations.median()),
    "median_hub_degree_best": float(best.max_degree.median()),
    "median_hub_degree_naive": float(naive.max_degree.median()),
    "mean_V_best": float(best.V.mean()), "mean_V_naive": float(naive.V.mean()),
    "mean_score_best": float(best.score.mean()), "mean_score_naive": float(naive.score.mean()),
    "verified_share_best": float((best.status == "verified").mean()),
    "partial_share_best": float((best.status == "partially_verified").mean()),
    "rejected_share_best": float((best.status == "rejected").mean()),
    "verified_share_naive": float((naive.status == "verified").mean()),
    "ranking_changed_top_path": float((bench.groupby("query").apply(lambda g: g.loc[g["rank"].idxmin(), "hop_rank"]) != 1).mean()),
    "candidate_verified_share": float((bench.status == "verified").mean()),
}
json.dump(bsum, open(os.path.join(OUT, "benchmark_summary.json"), "w"), indent=1)
save(pd.DataFrame([bsum]).T.reset_index().rename(columns={"index": "metric", 0: "value"}), "benchmark_summary.csv")

# ----------------------------------------------------------- 5. case studies
log("paper case studies")
cs_rows, cs_paths = [], {}
for gname, G in (("e3sm.csv", A), ("neo4j export", B)):
    for label, s, t in config.CASE_STUDIES:
        ns, nt = G.symbol_node(s), G.symbol_node(t)
        row = {"graph": gname, "case": label, "source": s, "target": t,
               "source_in_graph": ns is not None, "target_in_graph": nt is not None}
        if ns is None or nt is None:
            row["status"] = "endpoint missing"
            cs_rows.append(row)
            continue
        cands = pm.mine(G, fortran, s, t, s_num=s_num_lake if label.startswith("Lake chain") else None)
        if not cands:
            row["status"] = "no path"
            cs_rows.append(row)
            continue
        r = cands[0]
        row.update({k: v for k, v in r.items() if k not in ("path",)})
        row["status_class"] = row.pop("status")
        row["status"] = "path found"
        row["path"] = " | ".join(G.display(n)[:70] for n in r["path"])
        cs_rows.append(row)
        cs_paths[f"{gname}::{label}"] = r["path"]
case = save(pd.DataFrame(cs_rows), "case_studies.csv")
pd.to_pickle({"paths": cs_paths}, os.path.join(OUT, "case_paths.pkl"))

# line-level provenance of the equations on the best case-study paths (for the path-to-code figure)
code_rows = []
et = eq_trace.set_index(["file", "equation"])
for key, path in cs_paths.items():
    if not key.startswith("e3sm.csv"):
        continue
    for n in path:
        if n.startswith("e:"):
            for fl in sorted(A.eq_files.get(n[2:], ())):
                if (fl, n[2:]) in et.index:
                    r = et.loc[(fl, n[2:])]
                    code_rows.append({"case": key, "equation": n[2:], "file": fl, "status": r.status,
                                      "start": r.get("start"), "end": r.get("end"), "coverage": r.coverage,
                                      "snippet": r.get("snippet")})
save(pd.DataFrame(code_rows), "case_path_code_lines.csv")

# also line-level provenance for benchmark best paths
bp = []
for r in best.itertuples():
    pass
log("done")
