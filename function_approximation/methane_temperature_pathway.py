"""Audit and refit the methane--soil-temperature surrogate used in Eq. 9.

The old Celsius expression and the later square-root expression are both
evaluated explicitly.  The script then fits a small finite expression family
to the supplied E3SM level-1, land-grid-1 history records.  The selected
surrogate is a cubic in a dimensionless concentration, fitted on a seeded
70/30 random split.  This keeps the fit reproducible while making units and
the distinction between screening and independent validation explicit.

Run from the repository root::

    python function_approximation/methane_temperature_pathway.py

The audit and the ten-row table source are written beside this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "validation" / "Hourly land-model output from the E3SM lake-site simulation.csv"
OUT_TXT = Path(__file__).with_name("methane_temperature_eq9_audit.txt")
OUT_CSV = Path(__file__).with_name("methane_temperature_table.csv")
OUT_TEX = Path(__file__).with_name("methane_temperature_table.tex")

GRID = 1
LEVEL = 1
SEED = 0
TRAIN_FRACTION = 0.70
CONCENTRATION_SCALE = 1.0e-4  # mol m^-3
DEGREE = 3
# Coefficients selected by the seeded finite-expression screening below.
# Keeping them explicit makes every table prediction an evaluation of the
# manuscript equation rather than a hidden refit at table-generation time.
SELECTED_COEFFICIENT = np.array([
    268.29320926114434,
    4.663307593742296,
    2.5842370225133187,
    -0.5701343663010512,
], dtype=float)


def column(prefix: str) -> str:
    return f"{prefix}[levgrnd={LEVEL},lndgrid={GRID}]"


def load_records() -> pd.DataFrame:
    frame = pd.read_csv(INPUT, usecols=["timestamp", column("CONC_CH4_SAT"), column("TSOI")])
    concentration = frame[column("CONC_CH4_SAT")].to_numpy(float)
    temperature = frame[column("TSOI")].to_numpy(float)
    valid = (
        np.isfinite(concentration)
        & np.isfinite(temperature)
        & (concentration >= 0.0)
        & (concentration < 1.0e30)
        & (temperature > 200.0)
        & (temperature < 400.0)
    )
    records = pd.DataFrame({
        "source_row": np.arange(1, len(frame) + 1),
        "timestamp": frame["timestamp"],
        "concentration": concentration,
        "temperature_observed": temperature,
        "valid": valid,
    })
    return records.loc[valid].reset_index(drop=True)


def legacy_celsius(concentration: np.ndarray) -> np.ndarray:
    shift = 2.1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        return 14.27 + 0.9363 * np.log(concentration + shift) - 1.0367e-4 / (concentration + shift)


def legacy_kelvin(concentration: np.ndarray) -> np.ndarray:
    return legacy_celsius(concentration) + 273.15


def current_eq9(concentration: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = 1.0 + 0.2801 * (
            np.sqrt(np.abs(36.70 * concentration - 0.3833))
            - np.sqrt(np.abs(46.40 * concentration - 1.0979))
        )
        return 284.50 + 79.78 * np.log(np.abs(inside))


def exponential_eq9(concentration: np.ndarray) -> np.ndarray:
    """Evaluate the alternative exponential--logarithmic Eq. 9 supplied for review."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return 287.72 - 6.37 * np.log(np.abs(np.exp(3.217 - 94.24 * concentration) - 2.537))


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[mask], prediction[mask]
    error = y - prediction
    total = np.sum((y - y.mean()) ** 2)
    value_range = y.max() - y.min()
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / total) if total else float("nan"),
        "nrmse": float(np.sqrt(np.mean(error**2)) / value_range) if value_range else float("nan"),
    }


def polynomial_features(concentration: np.ndarray, degree: int) -> np.ndarray:
    scaled = concentration / CONCENTRATION_SCALE
    return np.column_stack([scaled**power for power in range(degree + 1)])


def selected_surrogate(concentration: np.ndarray) -> np.ndarray:
    """Evaluate the fixed cubic printed in Eq. 9."""
    return polynomial_features(concentration, DEGREE) @ SELECTED_COEFFICIENT


def fit_candidates(records: pd.DataFrame) -> tuple[list[dict], np.ndarray, np.ndarray]:
    x = records["concentration"].to_numpy(float)
    y = records["temperature_observed"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))
    split = int(TRAIN_FRACTION * len(records))
    train, test = order[:split], order[split:]
    candidates = []
    for degree in range(4):
        design_train = polynomial_features(x[train], degree)
        design_test = polynomial_features(x[test], degree)
        coefficient, *_ = np.linalg.lstsq(design_train, y[train], rcond=None)
        test_prediction = design_test @ coefficient
        train_prediction = design_train @ coefficient
        screening = metrics(y[test], test_prediction)
        fit = metrics(y[train], train_prediction)
        # A small complexity penalty prevents selecting a high-degree expression
        # when its screening improvement is negligible.
        score = screening["rmse"] + 0.05 * degree
        candidates.append({
            "degree": degree,
            "coefficient": coefficient,
            "train": fit,
            "test": screening,
            "score": float(score),
        })
    return candidates, train, test


def equation_text(coefficient: np.ndarray) -> str:
    terms = [f"{coefficient[0]:.10g}"]
    signs = ["+", "+", "+"]
    for power, value in enumerate(coefficient[1:], start=1):
        sign = "+" if value >= 0 else "-"
        magnitude = abs(value)
        if power == 1:
            terms.append(f" {sign} {magnitude:.10g} u")
        else:
            terms.append(f" {sign} {magnitude:.10g} u^{power}")
    return "T_soisno[K] = " + "".join(terms) + ",  u = C_sat/(1.0e-4 mol m^-3)"


def select_table(records: pd.DataFrame, test: np.ndarray, coefficient: np.ndarray, degree: int, count: int = 10) -> pd.DataFrame:
    selected = records.iloc[test].copy()
    selected["prediction"] = polynomial_features(selected["concentration"].to_numpy(float), degree) @ coefficient
    selected = selected.sort_values(["concentration", "source_row"], kind="mergesort").reset_index(drop=True)
    indices = np.linspace(0, len(selected) - 1, count).round().astype(int)
    selected = selected.iloc[indices].copy()
    selected.insert(0, "record_id", [f"M-g{GRID}-r{row:05d}" for row in selected["source_row"]])
    selected["absolute_error"] = np.abs(selected["temperature_observed"] - selected["prediction"])
    selected["relative_error_percent"] = 100.0 * selected["absolute_error"] / np.abs(selected["temperature_observed"])
    return selected


def latex_rows(selected: pd.DataFrame) -> str:
    def latex_sci(value: float) -> str:
        mantissa, exponent = f"{value:.6e}".split("e")
        return f"{float(mantissa):.6f}\\times10^{{{int(exponent)}}}"

    rows = []
    for record in selected.itertuples(index=False):
        rows.append(
            f"    {record.record_id} & ${latex_sci(record.concentration)}$ & {record.temperature_observed:.6f} "
            f"& {record.prediction:.6f} & {record.absolute_error:.6f} "
            f"& {record.relative_error_percent:.6f} \\\\"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    records = load_records()
    candidates, train, test = fit_candidates(records)
    best = min(candidates, key=lambda candidate: (candidate["score"], candidate["test"]["rmse"]))
    coefficient = SELECTED_COEFFICIENT.copy()
    if not np.allclose(coefficient, best["coefficient"], rtol=0.0, atol=1.0e-8):
        raise RuntimeError("The fixed Eq. 9 coefficients no longer match the reproducible screening result.")
    selected = select_table(records, test, coefficient, best["degree"])
    selected.to_csv(OUT_CSV, index=False, float_format="%.12g")
    OUT_TEX.write_text(latex_rows(selected), encoding="utf-8")

    concentration_examples = np.array([1.2e-5, 2.571157e-1])
    legacy_c = legacy_celsius(concentration_examples)
    current_k = current_eq9(concentration_examples)
    exponential_k = exponential_eq9(concentration_examples)
    observed = records["temperature_observed"].to_numpy(float)
    concentration = records["concentration"].to_numpy(float)
    current_metrics = metrics(observed, current_eq9(concentration))
    exponential_metrics = metrics(observed, exponential_eq9(concentration))
    legacy_metrics = metrics(observed, legacy_kelvin(concentration))
    fitted_metrics = metrics(observed, selected_surrogate(concentration))
    grid = np.linspace(0.0, 0.3, 4000)
    derivative_sign = np.sign(np.diff(current_eq9(grid)))
    sign_changes = int(np.count_nonzero(derivative_sign[1:] * derivative_sign[:-1] < 0))

    lines = [
        "AI-Miner methane-temperature Eq. 9 audit",
        "===========================================",
        "",
        "Input:",
        f"  {INPUT}",
        f"  field: {column('CONC_CH4_SAT')} (mol m^-3)",
        f"  target: {column('TSOI')} (K)",
        f"  valid records: {len(records):,}; land grid={GRID}; soil level={LEVEL}",
        "  provenance: the selected cubic is a local finite-expression screening result, not an equation recovered from the supplied FEX archive.",
        "",
        "Legacy expression identified in the manuscript comments:",
        "  T_C = 14.27 + 0.9363 ln(C + 2.1e-4) - 1.0367e-4/(C + 2.1e-4)",
        f"  at C=1.2e-5 and 0.2571157: {legacy_c[0]:.8f} and {legacy_c[1]:.8f} deg C",
        f"  after Kelvin conversion: {legacy_c[0] + 273.15:.8f} and {legacy_c[1] + 273.15:.8f} K",
        "  These are the 5.9 and 13.0 values in the reviewer comment; they are not Kelvin.",
        "",
        "Legacy square-root Eq. 9 audited for reviewer traceability (not used in the manuscript):",
        "  T_K = 284.50 + 79.78 ln|1 + 0.2801(sqrt|36.70 C - 0.3833| - sqrt|46.40 C - 1.0979|)|",
        f"  at C=1.2e-5 and 0.2571157: {current_k[0]:.8f} and {current_k[1]:.8f} K",
        f"  sign changes in a global [0, 0.3] scan: {sign_changes}; the legacy expression is not globally monotonic.",
        "",
        "Alternative exponential--logarithmic Eq. 9 supplied for review:",
        "  T_K = 287.72 - 6.37 ln|exp(3.217 - 94.24 C) - 2.537|",
        f"  at C=1.2e-5 and 0.2571157: {exponential_k[0]:.8f} and {exponential_k[1]:.8f} K",
        f"  on the supplied raw concentration field: n={exponential_metrics['n']:,}; MAE={exponential_metrics['mae']:.8g}; RMSE={exponential_metrics['rmse']:.8g}; R2={exponential_metrics['r2']:.8g}",
        "",
        "Legacy Eq. 9 diagnostics on the supplied main validation field:",
        f"  concentration range: {concentration.min():.8g} to {concentration.max():.8g} mol m^-3",
        f"  observed temperature range: {observed.min():.8g} to {observed.max():.8g} K",
        f"  legacy square-root expression: n={current_metrics['n']:,}; MAE={current_metrics['mae']:.8g}; RMSE={current_metrics['rmse']:.8g}; R2={current_metrics['r2']:.8g}",
        f"  legacy expression +273.15: n={legacy_metrics['n']:,}; MAE={legacy_metrics['mae']:.8g}; RMSE={legacy_metrics['rmse']:.8g}; R2={legacy_metrics['r2']:.8g}",
        f"  alternative exponential expression: n={exponential_metrics['n']:,}; MAE={exponential_metrics['mae']:.8g}; RMSE={exponential_metrics['rmse']:.8g}; R2={exponential_metrics['r2']:.8g}",
        "  Conclusion: the old logarithmic expression is Celsius, the legacy square-root expression is Kelvin but poorly matched to this field, and the alternative exponential expression is also poorly matched. The manuscript now uses the quadratic generated by methane_simple_fex.py; the selected cubic below is retained only as a legacy audit result.",
        "",
        "Finite expression search (reproducible screening):",
        f"  random seed={SEED}; train fraction={TRAIN_FRACTION:.2f}; fixed concentration scale={CONCENTRATION_SCALE:.1e} mol m^-3",
        "  operator family: constant, linear, quadratic, and cubic polynomial in u",
    ]
    for candidate in candidates:
        lines.append(
            f"  degree {candidate['degree']}: {equation_text(candidate['coefficient'])} | "
            f"train RMSE={candidate['train']['rmse']:.8g}; screening RMSE={candidate['test']['rmse']:.8g}; "
            f"screening R2={candidate['test']['r2']:.8g}; score={candidate['score']:.8g}"
        )
    lines += [
        "",
        "LEGACY SELECTED SCREENING EQUATION",
        "---------------------------",
        equation_text(coefficient),
        f"full-series diagnostic: n={fitted_metrics['n']:,}; MAE={fitted_metrics['mae']:.8g}; RMSE={fitted_metrics['rmse']:.8g}; R2={fitted_metrics['r2']:.8g}",
        f"screening holdout: n={best['test']['n']:,}; MAE={best['test']['mae']:.8g}; RMSE={best['test']['rmse']:.8g}; R2={best['test']['r2']:.8g}",
        "This is an empirical one-variable screening surrogate for the supplied level-1/grid-1 history field.",
        "Independent validation on a separate E3SM period and/or additional grids is still required.",
        "",
        f"wrote table CSV: {OUT_CSV}",
        f"wrote table LaTeX rows: {OUT_TEX}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TEX}")
    print(equation_text(coefficient))


if __name__ == "__main__":
    main()
