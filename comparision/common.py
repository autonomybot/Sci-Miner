"""Shared data loading, splits and metrics for the baseline comparison.

Two E3SM pathway cases are defined, each with one or more *univariate* candidate
feature sets and a *pathway* feature set (every variable that
appears in the mined equation chain and is available in the hourly output):

  lake     rho_w  (LAKE_RHOW, level 1)   <- chain rhow -> drhodz -> ke -> kme -> savedtke1
  methane  t_soisno (TSOI, level 1)      <- Eq. (methane-temperature-pathway)

Variables that are an algebraic function of the target (e.g. CH4_DIAG_TFACT = Q10^((T-Tb)/10),
CH4_DIAG_SOURCE, LAKE_N2 = g/rho*drhodz) are excluded from the pathway sets: they would
reveal the target and make every model look good for the wrong reason.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = os.path.join(ROOT, "data", "validation")
LAKE_CSV = os.path.join(VAL, "Hourly lake-physics output from the E3SM lake-site simulation.csv")
LAND_CSV = os.path.join(VAL, "Hourly land-model output from the E3SM lake-site simulation.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def _read(path, cols):
    have = set(pd.read_csv(path, nrows=0).columns)
    keep = [c for c in cols if c in have]
    return pd.read_csv(path, usecols=keep)[keep]   # usecols alone returns file order


def load_methane(site=1):
    """Hourly soil-temperature / methane records of lndgrid=<site>, chronological order."""
    g = f"lndgrid={site}]"
    lk = ["timestamp", f"TSOI[levgrnd=1,{g}", f"CONC_CH4_SAT[levgrnd=1,{g}", f"CONC_CH4_SAT[levgrnd=2,{g}",
          f"CH4_DIAG_DIFFUS[levgrnd=1,{g}", f"CH4_DIAG_EPS_GAS[levgrnd=1,{g}", f"CH4_DIAG_BASE_DECOMP[levgrnd=1,{g}"]
    ld = ["timestamp", f"CH4_AERE_DEPTH_SAT[levgrnd=1,{g}", f"CH4_OXID_DEPTH_SAT[levgrnd=1,{g}",
          f"CH4_EBUL_DEPTH_SAT[levgrnd=1,{g}"]
    a = _read(LAKE_CSV, lk).merge(_read(LAND_CSV, ld), on="timestamp", how="inner")
    a = a.rename(columns={lk[1]: "t_soisno", lk[2]: "conc1", lk[3]: "conc2", lk[4]: "diffus", lk[5]: "eps_gas",
                          lk[6]: "base_decomp", ld[1]: "aere", ld[2]: "oxid", ld[3]: "ebul"})
    # rate of change of C in layer 1 (centred difference, units per hour step)
    a["dcdt"] = np.gradient(a["conc1"].values)
    a["dcdz"] = a["conc2"] - a["conc1"]
    a["loss"] = a["aere"] + a["oxid"] + a["ebul"]
    a = a.dropna().reset_index(drop=True)
    a = a[(a["t_soisno"] > 150) & (a["t_soisno"] < 400) & (a["conc1"] > 0)].reset_index(drop=True)
    a["site"] = site
    return a


def load_lake(site):
    g = f"lndgrid={site}]"
    cols = ["timestamp", f"LAKE_RHOW[levlak=1,{g}", f"TKE1[{g}", f"LAKE_KME[levlak=1,{g}", f"LAKE_DRHODZ[levlak=1,{g}",
            f"LAKE_RI[levlak=1,{g}", f"LAKE_WS[{g}", f"LAKE_KS[{g}"]
    a = _read(LAKE_CSV, cols)
    a.columns = ["timestamp", "rho_w", "tke1", "kme1", "drhodz1", "ri1", "ws", "ks"]
    a = a.replace([np.inf, -np.inf], np.nan).dropna()
    num = a.drop(columns="timestamp")
    a = a[(num.abs() < 1e30).all(axis=1) & (a["tke1"] > 0)].reset_index(drop=True)
    a["site"] = site
    return a


# case -> (loader, target, {feature-set name: feature list}, sites)
CASES = {
    "methane": dict(load=lambda: pd.concat([load_methane(1)], ignore_index=True), target="t_soisno", sites=[1],
                    sets={"univariate": ["conc1"],
                          "pathway": ["conc1", "conc2", "dcdt", "diffus", "eps_gas", "base_decomp", "loss"],
                          # D(T) and base_decomp(T) are themselves temperature-dependent in E3SM, so the
                          # same set without them shows how much of the fit is that implicit T information
                          "pathway_noT": ["conc1", "conc2", "dcdt", "eps_gas", "loss"]}),
    "lake": dict(load=lambda: pd.concat([load_lake(s) for s in (1, 3, 4)], ignore_index=True), target="rho_w",
                 sites=[1, 3, 4],
                 sets={"univariate": ["tke1"],
                       # NOTE: the exported TKE1 field has mean 284 K and r=0.99 with TLAKE level 1, i.e. it behaves
                       # like a temperature, not like savedtke1. kme1 (x cwat = savedtke1) is the true chain variable.
                       "univariate_kme1": ["kme1"], "pathway": ["tke1", "kme1", "drhodz1", "ri1", "ws", "ks"]}),
}


def make_splits(df, scheme, seed=0):
    """Return a list of (train_idx, test_idx).  Index positions refer to `df` (chronological per site)."""
    n = len(df)
    if scheme == "random":                       # the split used in the paper (optimistic: autocorrelated)
        p = np.random.default_rng(seed).permutation(n)
        k = int(.7 * n)
        return [(np.sort(p[:k]), np.sort(p[k:]))]
    if scheme == "chrono":                       # first 70 % of every site -> train, last 30 % -> test
        tr, te = [], []
        for s in df["site"].unique():
            ix = np.flatnonzero(df["site"].values == s)
            k = int(.7 * len(ix))
            tr += list(ix[:k]); te += list(ix[k:])
        return [(np.array(tr), np.array(te))]
    if scheme == "blocked5":                     # 5 contiguous time blocks, each held out once (per site)
        folds = []
        for f in range(5):
            tr, te = [], []
            for s in df["site"].unique():
                ix = np.flatnonzero(df["site"].values == s)
                b = np.array_split(ix, 5)
                te += list(b[f]); tr += [i for j, bb in enumerate(b) if j != f for i in bb]
            folds.append((np.array(tr), np.array(te)))
        return folds
    if scheme == "site":                         # leave one site out
        sites = df["site"].unique()
        if len(sites) < 2:
            return []
        return [(np.flatnonzero(df["site"].values != s), np.flatnonzero(df["site"].values == s)) for s in sites]
    raise ValueError(scheme)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, np.nanmean(y))
    err = p - y
    sst = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return dict(R2=float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"), RMSE=rmse,
                MAE=float(np.mean(np.abs(err))), NRMSE=rmse / float(np.ptp(y)) if np.ptp(y) > 0 else float("nan"))
