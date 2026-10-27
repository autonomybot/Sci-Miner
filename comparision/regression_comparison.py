"""Surrogate comparison on the mined E3SM pathways.

For each case (lake, methane) x feature set (univariate, pathway, ...) x split scheme
(random 70/30, chronological 70/30, 5 contiguous time blocks, leave-one-site-out) every model is fitted on the
training part only and scored on the held-out part:

    Pathway ln-form | FEX (1-var) | FEX (multivariate) | SINDy (STLSQ) | PySR (if installed) | MLP

Stochastic learners (FEX, MLP, PySR) are repeated over seeds; the table reports mean +- sd over seeds and folds.

usage:  python regression_comparison.py [--quick] [--cases methane lake] [--schemes chrono blocked5 ...]
"""
import argparse, json, time, sys, os
import numpy as np
import pandas as pd

import common, models

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="small search budgets, 1 seed (smoke test)")
ap.add_argument("--cases", nargs="+", default=list(common.CASES))
ap.add_argument("--schemes", nargs="+", default=["random", "chrono", "blocked5", "site"])
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--max-train", type=int, default=6000, help="subsample training rows for PySR")
a = ap.parse_args()
TAG = '_'.join(a.cases) + ('_quick' if a.quick else '')
SEEDS = [0] if a.quick else list(range(a.seeds))
FEX_KW = dict(iters=4, batch=4, steps=60, finetune=150) if a.quick else dict(iters=15, batch=6, steps=80, finetune=400)


def build(name, names, seed):
    if name == "Pathway ln-form": return models.PathwayForm(names)
    if name == "FEX (1-var)": return models.FEX(names[:1], seed=seed, **FEX_KW)
    if name == "FEX": return models.FEX(names, seed=seed, **FEX_KW)
    if name == "SINDy (STLSQ)": return models.SINDy(names)
    if name == "PySR": return models.PySR(names, niterations=5 if a.quick else 40, seed=seed)
    if name == "MLP": return models.MLP(seed=seed, max_epochs=15 if a.quick else 150)
    raise ValueError(name)


STOCH = {"FEX (1-var)", "FEX", "MLP", "PySR"}
rows, exprs = [], {}
for case in a.cases:
    spec = common.CASES[case]
    df = spec["load"]()
    y_all = df[spec["target"]].values
    print(f"[{case}] {len(df)} rows, sites {sorted(df['site'].unique())}", flush=True)
    for scheme in a.schemes:
        splits = common.make_splits(df, scheme)
        if not splits:
            continue
        for fs, feats in spec["sets"].items():
            X_all = df[feats].values.astype(float)
            lineup = ["Pathway ln-form", "FEX (1-var)", "SINDy (STLSQ)", "MLP"]
            if len(feats) > 1:
                lineup.insert(2, "FEX")
            if models.PySR.available():
                lineup.insert(-1, "PySR")
            for mname in lineup:
                if mname == "FEX (1-var)" and len(feats) > 1:
                    continue            # 1-var FEX is evaluated on the univariate sets only
                t0 = time.time()
                for fold, (tr, te) in enumerate(splits):
                    if len(tr) > a.max_train and mname == "PySR":
                        tr = np.sort(np.random.default_rng(0).choice(tr, a.max_train, replace=False))
                    for seed in (SEEDS if mname in STOCH else [0]):
                        try:
                            m = build(mname, feats, seed).fit(X_all[tr], y_all[tr])
                            r_te = common.metrics(y_all[te], m.predict(X_all[te]))
                            r_tr = common.metrics(y_all[tr], m.predict(X_all[tr]))
                            ok = True
                        except Exception as e:                      # a failed fit is reported, never silently dropped
                            r_te = r_tr = dict(R2=np.nan, RMSE=np.nan, MAE=np.nan, NRMSE=np.nan); ok = False; print("  FAIL", mname, e)
                        rows.append(dict(case=case, feature_set=fs, n_features=len(feats), scheme=scheme, fold=fold, seed=seed,
                                         model=mname, n_train=len(tr), n_test=len(te), ok=ok, **{f"test_{k}": v for k, v in r_te.items()},
                                         **{f"train_{k}": v for k, v in r_tr.items()}))
                        if fold == 0 and seed == SEEDS[0] and ok:
                            exprs[f"{case}|{fs}|{scheme}|{mname}"] = m.expr()
                print(f"  {case:8s} {scheme:9s} {fs:16s} {mname:16s} {time.time() - t0:6.1f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(common.OUT, f"regression_raw_{TAG}.csv"), index=False)
json.dump(exprs, open(os.path.join(common.OUT, f"regression_expressions_{TAG}.json"), "w"), indent=1)
g = res.groupby(["case", "feature_set", "scheme", "model"])
summ = g.agg(n_runs=("fold", "size"), test_R2_mean=("test_R2", "mean"), test_R2_sd=("test_R2", "std"),
             test_RMSE_mean=("test_RMSE", "mean"), test_NRMSE_mean=("test_NRMSE", "mean"),
             train_R2_mean=("train_R2", "mean")).reset_index()
summ.to_csv(os.path.join(common.OUT, f"regression_summary_{TAG}.csv"), index=False)
print(summ.round(4).to_string(index=False))
