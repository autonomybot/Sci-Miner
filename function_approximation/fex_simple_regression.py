"""Compact FEX-compatible surrogates for the two paper case studies.

The original depth-2 FEX tree is expressive but produces long equations.  This
script uses a deliberately restricted grammar consisting of standardized
variables, powers up to degree two for the lake pathway, and powers up to degree
four for the methane relation.  L1 selection is used for the multivariate lake
case so that inactive terms are removed.  The resulting expressions remain
symbolic, deterministic, and easy to typeset, but they are reported as
restricted FEX-compatible surrogates rather than the unrestricted archive tree.

Run from the repository root::

    python function_approximation/fex_simple_regression.py

Outputs are written beside the original-FEX reports and include one-line LaTeX
expressions, split metrics, and representative held-out rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_TXT = HERE / "fex_simple_equations.txt"
OUT_JSON = HERE / "fex_simple_metadata.json"
CWAT = 4.188e6
SEED = 0
TRAIN_FRACTION = 0.70

sys.path.insert(0, str(ROOT / "comparision"))
import common  # noqa: E402


SPECS = {
    "lake": {
        "key": "lake_pathway_simple",
        "inputs": ["tke1", "savedtke1", "drhodz1", "ri1", "ws", "ks"],
        "labels": [r"TKE_1", r"\mathrm{savedtke}_{1}", r"\partial_z\rho_w", r"Ri_1", r"w_s", r"k_s"],
        "target": "rho_w",
        "target_latex": r"\rho_w",
        "degree": 2,
        "alpha": 0.01,
        "model": "lasso",
    },
    "methane": {
        "key": "methane_concentration_simple",
        "inputs": ["conc_CH4_sat_col"],
        "labels": [r"\mathrm{conc}_{\mathrm{CH_4,sat,col}}"],
        "target": "t_soisno",
        "target_latex": r"t_{\mathrm{soisno}}",
        "degree": 4,
        "alpha": 0.0,
        "model": "ridge",
    },
}


def load_case(case: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = common.CASES[case]["load"]().copy()
    if case == "lake":
        data["savedtke1"] = data["kme1"] * CWAT
        x = data[SPECS[case]["inputs"]].to_numpy(float)
        y = data["rho_w"].to_numpy(float)
    else:
        data = data.rename(columns={"conc1": "conc_CH4_sat_col"})
        x = data[["conc_CH4_sat_col"]].to_numpy(float)
        y = data["t_soisno"].to_numpy(float)
    return data, x, y


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    sst = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "n": int(len(y)),
        "r2": float(1.0 - np.sum(err**2) / sst),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "nrmse": rmse / float(np.ptp(y)),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def fit_case(case: str) -> dict:
    spec = SPECS[case]
    data, x, y = load_case(case)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    cut = int(TRAIN_FRACTION * len(y))
    train, test = np.sort(order[:cut]), np.sort(order[cut:])

    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x)
    poly = PolynomialFeatures(degree=spec["degree"], include_bias=False)
    design = poly.fit_transform(z)
    if spec["model"] == "lasso":
        model = Lasso(alpha=spec["alpha"], max_iter=20000, tol=1e-7, selection="cyclic")
    else:
        model = Ridge(alpha=1e-8)
    model.fit(design[train], y[train])
    pred = model.predict(design)

    feature_names = poly.get_feature_names_out([f"u_{i+1}" for i in range(x.shape[1])])
    coefficients = np.asarray(model.coef_, dtype=float)
    active = np.abs(coefficients) > 1e-8
    terms = [(str(name), float(coef)) for name, coef in zip(feature_names, coefficients) if abs(coef) > 1e-8]

    # Chronological holdout: fit on the first 70% without refitting on the tail.
    chrono_cut = int(TRAIN_FRACTION * len(y))
    chrono_scaler = StandardScaler().fit(x[:chrono_cut])
    chrono_z = chrono_scaler.transform(x)
    chrono_poly = PolynomialFeatures(degree=spec["degree"], include_bias=False)
    chrono_design = chrono_poly.fit_transform(chrono_z)
    chrono_model = Lasso(alpha=spec["alpha"], max_iter=20000, tol=1e-7) if spec["model"] == "lasso" else Ridge(alpha=1e-8)
    chrono_model.fit(chrono_design[:chrono_cut], y[:chrono_cut])
    chrono_pred = chrono_model.predict(chrono_design[chrono_cut:])

    site_metrics = {}
    if case == "lake":
        for site in sorted(data["site"].unique()):
            idx = np.flatnonzero(data["site"].to_numpy() == site)
            site_metrics[f"site_{int(site)}"] = metric(y[idx], pred[idx])

    return {
        "case": case,
        "key": spec["key"],
        "inputs": spec["inputs"],
        "labels": spec["labels"],
        "target": spec["target_latex"],
        "degree": spec["degree"],
        "alpha": spec["alpha"],
        "model": spec["model"],
        "input_mean": scaler.mean_.tolist(),
        "input_scale": scaler.scale_.tolist(),
        "intercept": float(model.intercept_),
        "terms": terms,
        "active_terms": int(active.sum()),
        "metrics": {
            "all": metric(y, pred),
            "random_holdout": metric(y[test], pred[test]),
            "random_train": metric(y[train], pred[train]),
            "chronological_holdout": metric(y[chrono_cut:], chrono_pred),
            "sites": site_metrics,
        },
        "train_indices": train,
        "test_indices": test,
        "predictions": pred,
        "observations": y,
        "data": data,
    }


def fmt(value: float) -> str:
    return f"{value:.8g}"


def term_latex(name: str) -> str:
    text = name.replace(" ", "\\,")
    text = text.replace("^", "^{")
    if "^{" in text:
        text += "}"
    return text


def expression_latex(result: dict) -> str:
    parts = [fmt(result["intercept"])]
    for name, coefficient in result["terms"]:
        sign = "+" if coefficient >= 0 else "-"
        parts.append(f" {sign} {fmt(abs(coefficient))}\,{term_latex(name)}")
    return result["target"] + " = " + "".join(parts)


def write_rows(result: dict) -> None:
    case = result["case"]
    data = result["data"]
    y = result["observations"]
    pred = result["predictions"]
    test = result["test_indices"]
    rows = []
    for i in np.sort(test)[:10]:
        site = int(data.iloc[i]["site"])
        prefix = "L" if case == "lake" else "M"
        rows.append({
            "record_id": f"{prefix}-g{site}-r{i+1:05d}",
            "source_row": int(i + 1),
            "site": site,
            "observed": y[i],
            "predicted": pred[i],
            "absolute_error": abs(pred[i] - y[i]),
            "split": "held_out",
        })
    pd.DataFrame(rows).to_csv(HERE / f"{result['key']}_table.csv", index=False)


def main() -> None:
    results = [fit_case(case) for case in ("lake", "methane")]
    serializable = {}
    lines = [
        "AI-Miner restricted FEX-compatible compact regression",
        "=======================================================",
        f"seed={SEED}; random split={TRAIN_FRACTION:.0%}/{1-TRAIN_FRACTION:.0%};",
        "grammar: standardized variables and powers up to the case-specific degree; L1 selection for lake terms.",
        "",
    ]
    for result in results:
        expr = expression_latex(result)
        lines += [
            result["key"].upper(),
            "-" * len(result["key"]),
            f"degree={result['degree']}; model={result['model']}; alpha={result['alpha']}; active_terms={result['active_terms']}",
            "standardization: " + "; ".join(f"u_{i+1}=({label}-({fmt(result['input_mean'][i])}))/({fmt(result['input_scale'][i])})" for i, label in enumerate(result["labels"])),
            "one-line equation:",
            expr,
            "metrics: " + json.dumps(result["metrics"], sort_keys=True),
            "",
        ]
        serializable[result["key"]] = {
            k: v for k, v in result.items() if k not in {"train_indices", "test_indices", "predictions", "observations", "data"}
        }
        write_rows(result)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_JSON}")
    for result in results:
        print(result["key"], expression_latex(result))
        print("  random holdout", result["metrics"]["random_holdout"])
        print("  chronological holdout", result["metrics"]["chronological_holdout"])


if __name__ == "__main__":
    main()
