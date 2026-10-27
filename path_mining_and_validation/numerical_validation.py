"""Numerical replay and surrogate validation against the E3SM lake-site output.

1. Equation-chain replay (S_num).  The lake pathway rho_w -> N2 -> Ri -> kme -> savedtke1 is
   re-computed from the Fortran formulas (LakeTemperatureMod.F90) using model *inputs*
   written to the history file, and compared with the model's own *outputs*.
2. Surrogate validation.  The current compact one-variable relations in
   ``function_approximation/`` are evaluated on the E3SM output and compared with
   a neural-network baseline on the held-out last 30 % (chronological split,
   no shuffling).  The path-mining replay is a separate diagnostic from the
   manuscript's fixed random-split tables.
"""

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")

# constants read from the Fortran sources (elm_varcon.F90, LakeCon.F90, LakeTemperatureMod.F90)
GRAV, VKC, TDMAX, DENICE = 9.80616, 0.4, 277.0, 917.0
TKWAT, CWAT, P0 = 0.57, 4.188e6, 1.0
N2MIN, DEPTHCRIT, MIXFACT = 7.5e-5, 25.0, 10.0
NLEVLAK = 10          # layers written to the history file
NACT = 3              # layers actually simulated at this lake site: levels 4-10 of the history file
                      # hold uninitialised values (LAKE_RHOW ~0-3800, TLAKE ~0.01-316) and fill values (1e36)
FILL = 1e30


def metrics(y, yhat):
    """MAE, RMSE, NRMSE (RMSE / range), R^2 and Pearson r on finite pairs."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 3:
        return dict(n=len(y), MAE=np.nan, RMSE=np.nan, NRMSE=np.nan, R2=np.nan, r=np.nan)
    err = y - yhat
    ss_tot = np.sum((y - y.mean()) ** 2)
    rng = y.max() - y.min()
    return dict(n=len(y), MAE=np.mean(np.abs(err)), RMSE=np.sqrt(np.mean(err ** 2)),
                NRMSE=np.sqrt(np.mean(err ** 2)) / rng if rng > 0 else np.nan,
                R2=1 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else np.nan,
                r=np.corrcoef(y, yhat)[0, 1] if y.std() > 0 and yhat.std() > 0 else np.nan)


# -------------------------------------------------------------------- loading
def _cols(prefix, levels, level_name, grid=1):
    return [f"{prefix}[{level_name}={j},lndgrid={grid}]" for j in range(1, levels + 1)]


def load_lake(grid=1):
    """Hourly lake-physics output for one lake (lndgrid=`grid`), plus static layer depths."""
    want = ["timestamp"]
    want += _cols("TLAKE", NLEVLAK, "levlak", grid) + _cols("LAKEICEFRAC", NLEVLAK, "levlak", grid)
    want += _cols("LAKE_RHOW", NLEVLAK, "levlak", grid) + _cols("LAKE_N2", NLEVLAK, "levlak", grid)
    want += _cols("LAKE_RI", NLEVLAK, "levlak", grid) + _cols("LAKE_KME", NLEVLAK, "levlak", grid)
    want += _cols("TSOI", 15, "levgrnd", grid) + _cols("CONC_CH4_SAT", 15, "levgrnd", grid)
    want += [f"LAKE_WS[lndgrid={grid}]", f"LAKE_KS[lndgrid={grid}]", f"TKE1[lndgrid={grid}]"]
    df = pd.read_csv(config.LAKE_CSV, usecols=want)
    zl = pd.read_csv(config.LAND_CSV, usecols=_cols("ZLAKE", NLEVLAK, "levlak", grid), nrows=1).iloc[0].to_numpy(float)
    return df, zl


def stack(df, prefix, levels, level_name, grid=1):
    return df[_cols(prefix, levels, level_name, grid)].to_numpy(float)


# ----------------------------------------------------------------- chain replay
def replay_lake_chain(df, zl, grid=1):
    """Re-compute each link of rho_w -> N2 -> Ri -> kme -> savedtke1 on the NACT active layers and score it."""
    g = lambda pre: stack(df, pre, NLEVLAK, "levlak", grid)[:, :NACT]
    T, f, rho_m, n2_m, ri_m, kme_m = g("TLAKE"), g("LAKEICEFRAC"), g("LAKE_RHOW"), g("LAKE_N2"), g("LAKE_RI"), g("LAKE_KME")
    ws = df[f"LAKE_WS[lndgrid={grid}]"].to_numpy(float)[:, None]
    ks = df[f"LAKE_KS[lndgrid={grid}]"].to_numpy(float)[:, None]
    tke1 = df[f"TKE1[lndgrid={grid}]"].to_numpy(float)
    z = zl[None, :NACT]
    ok = lambda a: np.where(np.abs(a) < FILL, a, np.nan)   # drop 1e36 fill values

    # (1) density of water + ice fraction
    rho = (1 - f) * 1000.0 * (1.0 - 1.9549e-05 * np.abs(T - TDMAX) ** 1.68) + f * DENICE
    # (2) Brunt-Vaisala frequency from the model's own density profile (isolates this equation)
    drhodz = (rho_m[:, 1:] - rho_m[:, :-1]) / (z[:, 1:] - z[:, :-1])
    n2 = GRAV / rho_m[:, :-1] * drhodz
    # (3) Richardson number from the model's own N2
    num = 40.0 * ok(n2_m)[:, :-1] * (VKC * z[:, :-1]) ** 2
    den = np.maximum(ws ** 2 * np.exp(-2.0 * ks * z[:, :-1]), 1e-10)
    ri = (-1.0 + np.sqrt(np.maximum(1.0 + num / den, 0.0))) / 20.0
    # (4) molecular + eddy + Fang-Stefan diffusivity from the model's own Ri and N2
    ke = VKC * ws * z[:, :-1] / P0 * np.exp(-ks * z[:, :-1]) / (1.0 + 37.0 * ok(ri_m)[:, :-1] ** 2)
    fang = 1.039e-08 * np.maximum(ok(n2_m)[:, :-1], N2MIN) ** (-0.43)
    kme = TKWAT / CWAT + ke + fang
    if z[0, -1] >= DEPTHCRIT:
        kme = kme * MIXFACT
    # (5) value stored for the next time step
    savedtke1 = kme_m[:, 0] * CWAT

    rows = []
    # The lake-scalar columns must be physically plausible before a replay that needs them is meaningful:
    # ws is a velocity scale (m/s) and savedtke1 = kme*cwat spans ~1-4e4 W/m/K.
    ws_ok = bool(np.nanmax(np.abs(ws)) < 100.0)
    ks_ok = bool(np.nanmax(np.abs(ks)) < 10.0)
    tke_ok = bool(np.nanmax(tke1) > 500.0 or np.nanmin(tke1) < 250.0)

    def add(eq, y, yhat, note="", usable=True):
        rows.append({"step": eq, **metrics(y, yhat), "note": note, "replayable": usable})

    add("rho_w = 1000(1-1.9549e-5|T-T_dmax|^1.68)(1-f_ice)+917 f_ice", rho_m.ravel(), rho.ravel(), "layers 1-3")
    add("N2 = (g/rho_w) d(rho_w)/dz", ok(n2_m)[:, :-1].ravel(), n2.ravel(), "layers 1-2")
    add("Ri = [-1+sqrt(1+40 N2 (k z)^2 / (ws^2 exp(-2 ks z)))]/20", ok(ri_m)[:, :-1].ravel(), ri.ravel(),
        "layers 1-2" + ("" if ws_ok and ks_ok else f"; LAKE_WS range {np.nanmin(ws):.1f}-{np.nanmax(ws):.1f}, LAKE_KS up to {np.nanmax(ks):.4f} are not a velocity/attenuation scale"),
        usable=ws_ok and ks_ok)
    free = (f[:, :-1] == 0) & (T[:, :-1] > 273.15)
    add("kme = km + ke + fang", kme_m[:, :-1][free], kme[free], "ice-free layers 1-2", usable=ws_ok and ks_ok)
    add("savedtke1 = kme(1) cwat", tke1, savedtke1,
        f"TKE1 field spans {np.nanmin(tke1):.1f}-{np.nanmax(tke1):.1f}; kme(1)*cwat spans {np.nanmin(savedtke1):.1f}-{np.nanmax(savedtke1):.0f}", usable=tke_ok)
    tab = pd.DataFrame(rows)
    tab["S_num"] = np.where(tab["replayable"], tab["R2"].clip(lower=0.0, upper=1.0), np.nan)
    return tab


# --------------------------------------------------------------- surrogates
LAKE_ALPHA = 263.80004341472073
LAKE_BETA = 11003756692.57215
LAKE_GAMMA = -7180533845.867309
METHANE_SCALE = 1.0e-4
METHANE_COEF = np.array([262.2210021366842, 16.366112506569223, -2.4179199478853537])


def lake_compact(x, alpha=LAKE_ALPHA, beta=LAKE_BETA, gamma=LAKE_GAMMA):
    """Current constrained Eq. surrogate, with S = kme_1*cwat."""
    x = np.asarray(x, float)
    q = (beta / (x - gamma) - 1.0) / 37.0
    r = np.sqrt(np.maximum(q, 1.0e-300))
    return alpha / (r * (10.0 * r + 1.0))


def methane_compact(x, coef=METHANE_COEF):
    """Current one-variable quadratic in C/(1e-4 mol m^-3), evaluated in kelvin."""
    u = np.asarray(x, float) / METHANE_SCALE
    return coef[0] + coef[1] * u + coef[2] * u**2


def _fit_methane(x, y):
    """Refit the current quadratic on a training segment for a diagnostic only."""
    u = np.asarray(x, float) / METHANE_SCALE
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(u), u, u**2]), y, rcond=None)
    return lambda values: methane_compact(values, coef)


def _mlp(xtr, ytr, xte, seed=config.SEED):
    sx, sy = StandardScaler(), StandardScaler()
    m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, random_state=seed)
    m.fit(sx.fit_transform(xtr[:, None]), sy.fit_transform(ytr[:, None]).ravel())
    return sy.inverse_transform(m.predict(sx.transform(xte[:, None]))[:, None]).ravel()


def surrogate_table(x, y, paper_pred, label, fit_fn=None):
    """Score the fixed compact equation and an optional diagnostic refit."""
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < FILL) & (np.abs(y) < FILL)
    idx = np.where(m)[0]
    if len(idx) < 500 or np.std(y[idx]) == 0 or np.std(x[idx]) == 0:
        return pd.DataFrame(), None
    cut = idx[int(0.7 * len(idx))]
    tr, te = idx[idx < cut], idx[idx >= cut]
    rows = []
    rows.append({"relation": label, "model": "current compact equation (test split)", **metrics(y[te], paper_pred[te])})
    rows.append({"relation": label, "model": "current compact equation (full series)", **metrics(y[idx], paper_pred[idx])})
    refit_pred = None
    if fit_fn is not None:
        predictor = fit_fn(x[tr], y[tr])
        refit_pred = predictor(x)
        rows.append({"relation": label, "model": "same compact form, re-fitted on train", **metrics(y[te], refit_pred[te])})
    rows.append({"relation": label, "model": "MLP baseline (64-64)", **metrics(y[te], _mlp(x[tr], y[tr], x[te]))})
    rows.append({"relation": label, "model": "constant (train mean)", **metrics(y[te], np.full(len(te), y[tr].mean()))})
    return pd.DataFrame(rows), (tr, te, refit_pred)


def lake_surrogates(df, grid=1):
    """Current one-variable lake equation on level-1 savedtke1 = kme1*cwat."""
    x = df[f"LAKE_KME[levlak=1,lndgrid={grid}]"].to_numpy(float) * CWAT
    out, extra = [], {}
    for j in (1,):
        y = df[f"LAKE_RHOW[levlak={j},lndgrid={grid}]"].to_numpy(float)
        t, info = surrogate_table(x, y, lake_compact(x), f"rho_w(level {j}) vs savedtke1")
        if t.empty:
            continue
        t.insert(1, "level", j)
        out.append(t)
        extra[j] = (x, y, info)
    return pd.concat(out, ignore_index=True), extra


def methane_surrogates(df, grid=1):
    """Current one-variable methane quadratic on level-1 concentration, in kelvin."""
    out, extra = [], {}
    for j in (1,):
        x = df[f"CONC_CH4_SAT[levgrnd={j},lndgrid={grid}]"].to_numpy(float)
        y = df[f"TSOI[levgrnd={j},lndgrid={grid}]"].to_numpy(float)
        t, info = surrogate_table(x, y, methane_compact(x), f"T_soisno(level {j}) vs conc_ch4_sat", _fit_methane)
        if t.empty:
            continue
        t.insert(1, "level", j)
        out.append(t)
        extra[j] = (x, y, info)
    return pd.concat(out, ignore_index=True), extra
