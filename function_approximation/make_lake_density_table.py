"""Create the ten-row lake-density table used in ``TAIS.tex``.

The published coefficients are evaluated with the level-1 E3SM ``LAKE_KME``
history field.  This is deliberate: the Fortran code defines
``savedtke1 = cwat * kme`` and the existing numerical coefficients match
``kme`` rather than the dimensional ``savedtke1`` output.  The table reports
the dimensional ``savedtke1`` value and applies the inverse conversion inside
the surrogate.  It is a record-level data extract, not the independent
validation performed later.

Run from the repository root::

    python function_approximation/make_lake_density_table.py

The script writes both a CSV audit file and a LaTeX row file beside this
script, so the manuscript table can be regenerated from the supplied CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "validation" / "Hourly lake-physics output from the E3SM lake-site simulation.csv"
OUT_CSV = Path(__file__).with_name("lake_density_savedtke1_table.csv")
OUT_TEX = Path(__file__).with_name("lake_density_savedtke1_table.tex")
CWAT = 4.188e6


def surrogate(kme: np.ndarray) -> np.ndarray:
    shift = 4.547e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1001.4774 + 0.14073 * np.log(kme + shift) - 6.399e-11 / (kme + shift)


def load_records() -> pd.DataFrame:
    columns = ["timestamp"]
    for grid in range(1, 5):
        columns.extend([
            f"LAKE_KME[levlak=1,lndgrid={grid}]",
            f"LAKE_RHOW[levlak=1,lndgrid={grid}]",
        ])
    raw = pd.read_csv(INPUT, usecols=columns)
    records: list[dict[str, object]] = []
    for grid in range(1, 5):
        kme_col = f"LAKE_KME[levlak=1,lndgrid={grid}]"
        rho_col = f"LAKE_RHOW[levlak=1,lndgrid={grid}]"
        kme = raw[kme_col].to_numpy(float)
        observed = raw[rho_col].to_numpy(float)
        predicted = surrogate(kme)
        valid = np.isfinite(kme) & (kme > 0.0) & np.isfinite(observed) & np.isfinite(predicted)
        for row, (timestamp, x, y, yhat) in enumerate(
            zip(raw["timestamp"], kme, observed, predicted), start=1
        ):
            if valid[row - 1]:
                absolute = abs(y - yhat)
                records.append({
                    "grid": grid,
                    "source_row": row,
                    "timestamp": timestamp,
                    "kme": x,
                    "savedtke1": CWAT * x,
                    "rho_observed": y,
                    "rho_predicted": yhat,
                    "absolute_error": absolute,
                    "relative_error_percent": 100.0 * absolute / abs(y),
                })
    return pd.DataFrame.from_records(records)


def select_rows(records: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    ordered = records.sort_values(["savedtke1", "grid", "source_row"], kind="mergesort").reset_index(drop=True)
    targets = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    selected = ordered.iloc[targets].copy()
    selected.insert(
        0,
        "record_id",
        [f"L-g{grid}-r{row:05d}" for grid, row in zip(selected["grid"], selected["source_row"])],
    )
    return selected


def latex_rows(selected: pd.DataFrame) -> str:
    def latex_sci(value: float) -> str:
        mantissa, exponent = f"{value:.6e}".split("e")
        exponent = int(exponent)
        return f"{float(mantissa):.6f}\\times10^{{{exponent}}}"

    rows = []
    for record in selected.itertuples(index=False):
        rows.append(
            f"    {record.record_id} & ${latex_sci(record.savedtke1)}$ & {record.rho_observed:.6f} "
            f"& {record.rho_predicted:.6f} & {record.absolute_error:.6f} "
            f"& {record.relative_error_percent:.6f} \\\\"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    selected = select_rows(load_records())
    selected.to_csv(OUT_CSV, index=False, float_format="%.12g")
    OUT_TEX.write_text(latex_rows(selected), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TEX}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
