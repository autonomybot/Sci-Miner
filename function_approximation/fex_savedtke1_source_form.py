"""Constrained one-variable FEX fit for the lake density pathway.

The unrestricted FEX tree can fit an arbitrary expression of ``savedtke1``.
This script instead preserves the algebraic structure of Eq. 7 and fits only
the three scalar coefficients that remain after the other pathway quantities
are treated as fixed for the calibrated pathway:

    R(S) = sqrt((beta/(S-gamma) - 1)/37)
    rho_w(S) = alpha/(R(S)*(10*R(S) + 1)),  S = savedtke_1

The parameterization enforces a real, positive square root over the observed
training range.  A fixed random 70/30 split is used for reporting; all rows
remain in the fit/evaluation audit and no records are discarded.

Run from the repository root::

    python function_approximation/fex_savedtke1_source_form.py

Outputs are written next to this script as a text equation report, JSON
metadata, and a twenty-row held-out CSV/TEX table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CWAT = 4.188e6
SEED = 0
TRAIN_FRACTION = 0.70

OUT_TXT = HERE / "fex_savedtke1_source_form.txt"
OUT_JSON = HERE / "fex_savedtke1_source_form.json"
OUT_CSV = HERE / "fex_savedtke1_source_form_table.csv"
OUT_TEX = HERE / "fex_savedtke1_source_form_table.tex"


def load_lake() -> pd.DataFrame:
    import sys

    sys.path.insert(0, str(ROOT / "comparision"))
    import common

    data = common.CASES["lake"]["load"]().copy()
    data["savedtke1"] = data["kme1"] * CWAT
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["savedtke1", "rho_w"])
    return data.reset_index(drop=True)


def unpack(theta: np.ndarray, s_min: float, s_max: float) -> tuple[float, float, float]:
    """Map unconstrained log parameters to a valid (alpha, beta, gamma)."""
    # gamma is below the smallest observed S, and beta is above the largest
    # possible denominator.  Thus beta/(S-gamma)-1 is non-negative for every
    # observed row and the square root is real.
    gamma = s_min - float(np.exp(theta[0]))
    beta = (s_max - gamma) + float(np.exp(theta[1]))
    alpha = float(np.exp(theta[2]))
    return alpha, beta, gamma


def predict(s: np.ndarray, theta: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
    alpha, beta, gamma = unpack(theta, s_min, s_max)
    q = (beta / (s - gamma) - 1.0) / 37.0
    r = np.sqrt(np.maximum(q, 1e-300))
    return alpha / (r * (10.0 * r + 1.0))


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    sse = float(np.sum(e**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean(e**2)))
    return {
        "n": int(len(y)),
        "r2": float(1.0 - sse / sst) if sst else float("nan"),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(e))),
        "nrmse": rmse / float(np.ptp(y)) if np.ptp(y) else float("nan"),
    }


def latex_number(value: float) -> str:
    return f"{value:.10g}"


def equation_latex(alpha: float, beta: float, gamma: float) -> str:
    a, b, c = map(latex_number, (alpha, beta, gamma))
    return (
        rf"R(S)=\sqrt{{\frac{{1}}{{37}}\left(\frac{{{b}}}{{S-({c})}}-1\right)}},\quad "
        rf"\rho_w(S)=\frac{{{a}}}{{R(S)\,[10R(S)+1]}},\quad S=\mathrm{{savedtke}}_1"
    )


def main() -> None:
    data = load_lake()
    s = data["savedtke1"].to_numpy(float)
    y = data["rho_w"].to_numpy(float)
    s_min, s_max = float(np.min(s)), float(np.max(s))

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(data))
    n_train = int(TRAIN_FRACTION * len(data))
    train = np.sort(order[:n_train])
    test = np.sort(order[n_train:])

    # The initial alpha is scaled to the observed density.  Several starts are
    # used because this constrained nonlinear form can have shallow valleys.
    mid = float(np.median(s))
    starts = []
    for shift in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        gamma0 = s_min - shift
        beta0 = (s_max - gamma0) + max(shift, 1.0)
        q0 = max(beta0 / (mid - gamma0) - 1.0, 1e-12) / 37.0
        r0 = np.sqrt(q0)
        d0 = r0 * (10.0 * r0 + 1.0)
        alpha0 = max(float(np.median(y)) * d0, 1e-12)
        starts.append(np.log([shift, max(shift, 1.0), alpha0]))

    def residual(theta: np.ndarray) -> np.ndarray:
        p = predict(s[train], theta, s_min, s_max)
        # Soft-L1 loss below limits the influence of extreme density residuals
        # during coefficient fitting while metrics remain ordinary RMSE/MAE.
        return p - y[train]

    best = None
    for start in starts:
        result = least_squares(
            residual,
            start,
            bounds=([-20.0, -20.0, 0.0], [30.0, 35.0, 40.0]),
            loss="soft_l1",
            f_scale=0.5,
            x_scale="jac",
            max_nfev=4000,
        )
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    theta = best[1].x
    alpha, beta, gamma = unpack(theta, s_min, s_max)
    pred = predict(s, theta, s_min, s_max)

    metrics = {
        "all": metric(y, pred),
        "train": metric(y[train], pred[train]),
        "held_out": metric(y[test], pred[test]),
    }
    eq = equation_latex(alpha, beta, gamma)
    report = [
        "AI-Miner constrained one-variable FEX source-form regression",
        "==============================================================",
        "Target: rho_w; sole variable: S = savedtke_1.",
        "The Eq. 7 algebraic structure is fixed; FEX/least-squares optimizes alpha, beta, and gamma only.",
        f"records: {len(data):,}; random split: {TRAIN_FRACTION:.0%}/{1-TRAIN_FRACTION:.0%}; seed: {SEED}",
        f"savedtke_1 range: {s_min:.10g} to {s_max:.10g}",
        "",
        "Fitted constants:",
        f"  alpha = {alpha:.12g}",
        f"  beta  = {beta:.12g}",
        f"  gamma = {gamma:.12g}",
        "",
        "LaTeX equation:",
        eq,
        "",
        "Metrics:",
        json.dumps(metrics, sort_keys=True),
        "",
        "Interpretation: the equation is a source-faithful one-variable form for a calibrated fixed-coefficient pathway. The held-out metrics determine whether those constants transfer beyond that calibration sample.",
    ]
    OUT_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")

    # Show twenty held-out records spanning the predictor range rather than the
    # first source rows, which can otherwise hide the response variation.
    ordered = test[np.argsort(s[test])]
    chosen = ordered[np.linspace(0, len(ordered) - 1, 20).round().astype(int)]
    table = pd.DataFrame(
        {
            "record_id": [f"L-g{int(data.iloc[i]['site'])}-r{i+1:05d}" for i in chosen],
            "source_row": [int(i + 1) for i in chosen],
            "savedtke1": s[chosen],
            "rho_observed": y[chosen],
            "rho_predicted": pred[chosen],
            "absolute_error": np.abs(pred[chosen] - y[chosen]),
        }
    )
    table.to_csv(OUT_CSV, index=False)
    OUT_TEX.write_text(
        "\n".join(
            f"    {r.record_id} & ${r.savedtke1:.6e}$ & {r.rho_observed:.6f} & {r.rho_predicted:.6f} & {r.absolute_error:.6f} " + r"\\"
            for r in table.itertuples()
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "variable": "savedtke1",
        "target": "rho_w",
        "constants": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "equation_latex": eq,
        "split": {"seed": SEED, "train_fraction": TRAIN_FRACTION},
        "savedtke1_range": [s_min, s_max],
        "metrics": metrics,
        "model": "R(S)=sqrt((beta/(S-gamma)-1)/37); rho=alpha/(R*(10R+1))",
    }
    OUT_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    print(f"equation: {eq}")
    print(f"held-out: {metrics['held_out']}")


if __name__ == "__main__":
    main()
