"""Domain-expert review of mined pathways.

No expert judgement exists in this repository, so none is invented here.  This script

  make   writes  results/expert_review_sheet.csv : one row per mined pathway equation (Sci-Miner chains for the case
         studies plus a random sample of benchmark chains) with blank rating columns for 2+ independent raters, and
         the matching def-use-chain statements so raters judge both methods blind to which one produced a chain.
  score  reads the filled sheet(s) and reports, per method, the share of chains rated physically correct / novel and
         Cohen's kappa between raters.

usage:  python expert_review.py make [--n 30]
        python expert_review.py score rater1.csv rater2.csv ...
Rating scale (columns to fill):  correct  (0 = wrong, 1 = partly, 2 = correct physical dependence)
                                  novel    (0 = textbook/trivial, 1 = known to specialists, 2 = not documented)
"""
import sys, os
import numpy as np
import pandas as pd

import common

RES = os.path.join(common.ROOT, "results", "path_mining_validation")
SHEET = os.path.join(common.OUT, "expert_review_sheet.csv")


def make(n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    cs = pd.read_csv(os.path.join(RES, "case_studies.csv"))
    cs = cs[(cs["graph"] == "e3sm.csv") & (cs["status"] == "path found")]
    for _, r in cs.iterrows():
        rows.append(dict(chain_id=f"case:{r['case']}", method="Sci-Miner", source=r["source"], target=r["target"], steps=r["path"]))
    bp = pd.read_csv(os.path.join(RES, "benchmark_paths.csv"))
    bq = bp[bp["rank"] == 1].drop_duplicates("query")
    for _, r in bq.iloc[rng.permutation(len(bq))[:n]].iterrows():
        rows.append(dict(chain_id=f"bench:{int(r['query'])}", method="Sci-Miner", source=r["source"], target=r["target"], steps=r["path"]))
    du = os.path.join(common.OUT, "path_recovery_cases.csv")
    if os.path.exists(du):
        for _, r in pd.read_csv(du).iterrows():
            if isinstance(r.get("du_chain"), str):
                rows.append(dict(chain_id=f"case:{r['query']}", method="Def-use chain", source=r["source"], target=r["target"], steps=r["du_chain"]))
    df = pd.DataFrame(rows)
    df["blind_id"] = rng.permutation(len(df))                 # raters see only blind_id; the key stays in this file
    df = df.sort_values("blind_id")
    for c in ("rater", "correct", "novel", "comment"):
        df[c] = ""
    df.to_csv(SHEET, index=False)
    df.drop(columns=["method", "chain_id"]).to_csv(os.path.join(common.OUT, "expert_review_sheet_BLIND.csv"), index=False)
    print(f"wrote {len(df)} chains -> {SHEET} (+ _BLIND.csv for raters, without the method column)")


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(files):
    key = pd.read_csv(SHEET)[["blind_id", "method", "chain_id"]]
    rs = [pd.read_csv(f).merge(key, on="blind_id") for f in files]
    for r in rs:
        r["correct"] = pd.to_numeric(r["correct"], errors="coerce"); r["novel"] = pd.to_numeric(r["novel"], errors="coerce")
    allr = pd.concat(rs)
    out = allr.groupby("method").agg(rated=("correct", "count"), correct_mean=("correct", "mean"),
                                     share_correct=("correct", lambda x: (x == 2).mean()), novel_mean=("novel", "mean")).reset_index()
    print(out.round(3).to_string(index=False))
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            m = rs[i].merge(rs[j], on="blind_id").dropna(subset=["correct_x", "correct_y"])
            print(f"kappa(correct) rater{i + 1} vs rater{j + 1}: {cohen_kappa(m.correct_x, m.correct_y):.3f} (n={len(m)})")
    out.to_csv(os.path.join(common.OUT, "expert_review_scores.csv"), index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make":
        make(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 30)
    elif len(sys.argv) > 2 and sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        print(__doc__)
