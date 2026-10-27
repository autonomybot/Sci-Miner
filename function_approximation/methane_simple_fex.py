"""Find a short one-variable methane-temperature surrogate.

The long unrestricted FEX tree is useful for provenance, but the paper also
needs a readable one-line expression.  This script searches a small, explicit
family of finite expressions using only saturated methane concentration:

* linear, quadratic, and cubic polynomials in a dimensionless concentration;
* a three-term logarithm/inverse form with a fitted positive shift.

The shift is optimized on the training split and the remaining coefficients
are solved linearly.  Random and chronological holdouts are reported so a
short equation is not selected solely because it memorizes autocorrelated
records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCALE = 1.0e-4  # mol m^-3; makes the logarithm dimensionless
SEED = 0
TRAIN_FRACTION = 0.70

OUT_TXT = HERE / "methane_simple_fex_equation.txt"
OUT_JSON = HERE / "methane_simple_fex_metadata.json"
OUT_CSV = HERE / "methane_simple_fex_table.csv"
OUT_TEX = HERE / "methane_simple_fex_table.tex"


def load_records() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "comparision"))
    import common

    frame = common.CASES["methane"]["load"]().copy()
    return frame.rename(columns={"conc1": "concentration", "t_soisno": "temperature"})


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    sse = float(np.sum(e**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean(e**2)))
    return {
        "n": int(len(y)),
        "r2": float(1.0 - sse / sst) if sst else float("nan"),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(e))),
    }


def poly_design(u: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([u**i for i in range(degree + 1)])


def fit_poly(u: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, callable]:
    coef, *_ = np.linalg.lstsq(poly_design(u, degree), y, rcond=None)
    return coef, lambda x: poly_design(x, degree) @ coef


def shifted_design(u: np.ndarray, shift: float) -> np.ndarray:
    v = u + shift
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([np.ones_like(v), np.log(v), 1.0 / v])


def fit_log_inverse(u: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, callable]:
    """Fit a + b*ln(u+s) + d/(u+s), optimizing only positive s."""
    # Search in log space so the fit remains stable across several decades.
    def solve(log_shift: float) -> tuple[float, np.ndarray]:
        shift = float(np.exp(log_shift))
        design = shifted_design(u, shift)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = design @ coef - y
        return float(np.mean(residual**2)), coef

    result = minimize_scalar(
        lambda z: solve(float(z))[0],
        bounds=(np.log(1e-6), np.log(1e3)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    shift = float(np.exp(result.x))
    _, coef = solve(result.x)
    return coef, shift, lambda x: shifted_design(x, shift) @ coef


def latex_num(value: float) -> str:
    return f"{value:.10g}"


def signed_term(value: float, term: str) -> str:
    sign = "+" if value >= 0 else "-"
    return f" {sign} {latex_num(abs(value))}\,{term}"


def equation_latex(kind: str, coef: np.ndarray, shift: float | None = None) -> str:
    if kind.startswith("poly"):
        degree = int(kind[-1])
        terms = [latex_num(coef[0])]
        for power in range(1, degree + 1):
            term = "u" if power == 1 else rf"u^{{{power}}}"
            terms.append(signed_term(coef[power], term))
        return r"t_{\mathrm{soisno}}[\mathrm{K}] = " + "".join(terms) + r",\quad u=\frac{C}{10^{-4}\,\mathrm{mol\,m^{-3}}}"
    assert shift is not None
    return (
        r"t_{\mathrm{soisno}}[\mathrm{K}] = "
        + latex_num(coef[0])
        + signed_term(coef[1], rf"\ln(u+{latex_num(shift)})")
        + signed_term(coef[2], rf"\frac{{1}}{{u+{latex_num(shift)}}}")
        + r",\quad u=\frac{C}{10^{-4}\,\mathrm{mol\,m^{-3}}}"
    )


def main() -> None:
    data = load_records()
    c = data["concentration"].to_numpy(float)
    y = data["temperature"].to_numpy(float)
    u = c / SCALE
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(data))
    n_train = int(TRAIN_FRACTION * len(data))
    train_random = np.sort(order[:n_train])
    test_random = np.sort(order[n_train:])
    train_chrono = np.arange(n_train)
    test_chrono = np.arange(n_train, len(data))

    candidates: list[dict] = []
    for degree in (1, 2, 3):
        coef, predict = fit_poly(u[train_random], y[train_random], degree)
        p_random = predict(u[test_random])
        p_chrono = predict(u[test_chrono])
        candidates.append({
            "kind": f"poly{degree}",
            "coef": coef.tolist(),
            "shift": None,
            "equation": equation_latex(f"poly{degree}", coef),
            "random_holdout": metrics(y[test_random], p_random),
            "chronological_holdout": metrics(y[test_chrono], p_chrono),
            "complexity": degree,
        })

    coef, shift, predict = fit_log_inverse(u[train_random], y[train_random])
    candidates.append({
        "kind": "log_inverse",
        "coef": coef.tolist(),
        "shift": shift,
        "equation": equation_latex("log_inverse", coef, shift),
        "random_holdout": metrics(y[test_random], predict(u[test_random])),
        "chronological_holdout": metrics(y[test_chrono], predict(u[test_chrono])),
        "complexity": 3,
    })

    # Select the shortest candidate with the strongest worst-case R2 across the
    # random and chronological holdouts.  This avoids choosing a polynomial
    # that looks good only because the random split preserves autocorrelation.
    for item in candidates:
        item["worst_case_r2"] = min(item["random_holdout"]["r2"], item["chronological_holdout"]["r2"])
    selected = max(
        candidates,
        key=lambda item: (item["worst_case_r2"], -item["complexity"], -item["random_holdout"]["rmse"]),
    )
    kind = selected["kind"]
    coef = np.asarray(selected["coef"], dtype=float)
    if kind == "log_inverse":
        shift = float(selected["shift"])
        predict = lambda x: shifted_design(x, shift) @ coef
    else:
        degree = int(kind[-1])
        predict = lambda x: poly_design(x, degree) @ coef
    p_all = predict(u)

    report = [
        "AI-Miner compact one-variable methane FEX-compatible search",
        "==============================================================",
        "Target: t_soisno [K]; sole predictor: C = conc_CH4_sat_col [mol m^-3].",
        f"records: {len(data):,}; random split: {TRAIN_FRACTION:.0%}/{1-TRAIN_FRACTION:.0%}; seed: {SEED}",
        "Candidate equations:",
    ]
    for item in candidates:
        report += [
            f"  {item['kind']}: {item['equation']}",
            f"    random holdout: {json.dumps(item['random_holdout'], sort_keys=True)}",
            f"    chronological holdout: {json.dumps(item['chronological_holdout'], sort_keys=True)}",
            f"    worst-case holdout R2: {item['worst_case_r2']:.8f}",
        ]
    report += [
        "",
        f"SELECTED EQUATION ({selected['kind']}):",
        selected["equation"],
        f"all-record metrics: {json.dumps(metrics(y, p_all), sort_keys=True)}",
        "The equation is a short empirical screening surrogate; independent-period validation is still required.",
    ]
    OUT_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")

    # Show twenty held-out records spanning the concentration range.
    ordered = test_random[np.argsort(c[test_random])]
    chosen = ordered[np.linspace(0, len(ordered) - 1, 20).round().astype(int)]
    table = pd.DataFrame({
        "record_id": [f"M-g1-r{i+1:05d}" for i in chosen],
        "source_row": chosen + 1,
        "concentration": c[chosen],
        "temperature_observed": y[chosen],
        "temperature_predicted": p_all[chosen],
        "absolute_error": np.abs(p_all[chosen] - y[chosen]),
    })
    table.to_csv(OUT_CSV, index=False)
    OUT_TEX.write_text(
        "\n".join(
            f"    {r.record_id} & ${r.concentration:.6e}$ & {r.temperature_observed:.6f} & {r.temperature_predicted:.6f} & {r.absolute_error:.6f} " + r"\\"
            for r in table.itertuples()
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "predictor": "conc_CH4_sat_col",
        "target": "t_soisno",
        "scale": SCALE,
        "selected": selected,
        "candidates": candidates,
        "all_metrics": metrics(y, p_all),
    }
    OUT_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    print(f"selected: {selected['equation']}")
    print(f"random holdout: {selected['random_holdout']}")
    print(f"chronological holdout: {selected['chronological_holdout']}")


if __name__ == "__main__":
    main()
