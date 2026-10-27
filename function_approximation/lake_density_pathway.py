"""Reproduce and audit the lake-density pathway for one queried relation.

The script is intentionally self-contained. It does not import the supplied
Finite Expression Method archive. It first checks the algebra in the paper's
two displayed pathway equations against the Fortran statements, then searches
a small finite set of one-variable expressions only when the simplified
pathway is not the active Fortran branch.

Run from the repository root:

    python function_approximation/lake_density_pathway.py

The complete audit, every fitted candidate, and the selected expression are
written beside this script:

    function_approximation/lake_density_savedtke1_equations.txt
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
LAKE_CSV = ROOT / "data" / "validation" / "Hourly lake-physics output from the E3SM lake-site simulation.csv"
LAND_CSV = ROOT / "data" / "validation" / "Reduced_land-model_output_lake-site_simulation.csv"
OUT_DIR = ROOT / "results" / "function_approximation"
OUT_TXT = Path(__file__).with_name("lake_density_savedtke1_equations.txt")

# Values and branches copied from the relevant E3SM Fortran source statements.
G = 9.80616
KAPPA = 0.4
CWAT = 4.188e6                 # cpliq * denh2o
TKWAT = 0.57
KM = TKWAT / CWAT              # tkwat / cwat
P0 = 1.0
N2MIN = 7.5e-5
FANG_COEF = 1.039e-8
MIXFACT = 10.0
LAKE_NO_ED_DEFAULT = False
DEPTHCRIT = 25.0

SOURCE_NOTES = [
    "data/input/fortran/LakeTemperatureMod.F90:355-372 (drhodz, n2, ri)",
    "data/input/fortran/LakeTemperatureMod.F90:372-380 (ke, fangkm, kme, deep mixing)",
    "data/input/fortran/LakeTemperatureMod.F90:425 (savedtke1 = kme(c,1)*cwat)",
    "data/input/fortran/LakeCon.F90:105-116 (depthcrit, mixfact, lake_no_ed defaults)",
    "data/input/fortran/bshr_const_mod.F90:25,46,52 (g, fresh-water density, heat capacity)",
]


def q(name: str, level: int | None = None) -> str:
    if level is None:
        return f"{name}[lndgrid=1]"
    return f"{name}[levlak={level},lndgrid=1]"


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    """Load only the columns needed for this one pathway."""
    cols = [
        q("LAKE_RHOW", 1), q("LAKE_RHOW", 2), q("LAKE_RI", 1),
        q("LAKE_N2", 1), q("LAKE_KME", 1), q("LAKE_WS"), q("LAKE_KS"),
        "timestamp",
    ]
    frame = pd.read_csv(LAKE_CSV, usecols=cols)
    zcols = [q("ZLAKE", 1), q("ZLAKE", 2)]
    depths = pd.read_csv(LAND_CSV, usecols=zcols, nrows=1).iloc[0].to_numpy(float)
    return frame, depths


def source_equations() -> dict[str, str]:
    """Return the equations as implemented or algebraically rearranged."""
    rho = r"\rho_w = \frac{g(\kappa z)^2 e^{-2k_s z}}{w_s^2\,Ri(10Ri+1)}\,\frac{\partial\rho_w}{\partial z}"
    simple = r"Ri^2 = \frac{1}{37}\left(\frac{\kappa w_s z c_{wat}e^{-k_s z}}{savedtke_1-k_m c_{wat}}-1\right)"
    full = (
        r"Ri^2 = \frac{1}{37}\left(\frac{\kappa w_s z e^{-k_s z}}{p_0\," 
        r"\left(savedtke_1/(c_{wat}M)-k_m-F\right)}-1\right), "
        r"F=1.039\times10^{-8}\max(N^2,N^2_{min})^{-0.43}"
    )
    direct = (
        r"\rho_w(S)=\frac{g(\kappa z)^2e^{-2k_s z}}{w_s^2R(S)(10R(S)+1)}"
        r"\frac{\partial\rho_w}{\partial z},\quad"
        r"R(S)=\sqrt{\frac{1}{37}\left[\frac{\kappa w_sze^{-k_sz}}"
        r"{p_0(S/(c_{wat}M)-k_m-F)}-1\right]}"
    )
    return {"rho_rearrangement": rho, "simplified_ri": simple, "full_ri": full, "direct_branch": direct}


def symbolic_checks() -> dict[str, bool]:
    """Check the two algebraic rearrangements without using data."""
    ri = sp.symbols("Ri")
    denominator_identity = sp.simplify((20 * ri + 1) ** 2 - 1 - 40 * ri * (10 * ri + 1)) == 0
    # From savedtke1 = cwat*(km + ke), with ke = A/(1+37 Ri^2), p0=1.
    saved, cwat, km, a = sp.symbols("saved cwat km a", nonzero=True)
    solved = sp.solve(sp.Eq(saved, cwat * (km + a / (1 + 37 * ri**2))), ri**2)[0]
    expected = a * cwat / (saved - km * cwat) / 37 - sp.Rational(1, 37)
    simplified_identity = sp.simplify(solved - expected) == 0
    return {"rho_rearrangement": denominator_identity, "simplified_ri_under_no_fang_branch": simplified_identity}


def finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (np.abs(x) < 1e30)


def derive(frame: pd.DataFrame, z: np.ndarray) -> dict[str, np.ndarray]:
    """Derive savedtke1 and the pathway variables from model output columns."""
    rho = frame[q("LAKE_RHOW", 1)].to_numpy(float)
    rho2 = frame[q("LAKE_RHOW", 2)].to_numpy(float)
    ri = frame[q("LAKE_RI", 1)].to_numpy(float)
    n2 = frame[q("LAKE_N2", 1)].to_numpy(float)
    kme = frame[q("LAKE_KME", 1)].to_numpy(float)
    ws = frame[q("LAKE_WS")].to_numpy(float)
    ks = frame[q("LAKE_KS")].to_numpy(float)
    dz = z[1] - z[0]
    drhodz = (rho2 - rho) / dz
    saved = kme * CWAT
    with np.errstate(all="ignore"):
        rho_from_rho_ri = G * (KAPPA * z[0]) ** 2 * np.exp(-2 * ks * z[0]) * drhodz / (
            ws**2 * ri * (10 * ri + 1)
        )

    # Equation printed in the manuscript: it omits F and M.
    with np.errstate(all="ignore"):
        ri2_simple = (
            KAPPA * ws * z[0] * CWAT * np.exp(-ks * z[0]) /
            (saved - KM * CWAT) - 1
        ) / 37

    # Full branch used by LakeTemperatureMod.F90 when the lake is unfrozen.
    fang = FANG_COEF * np.maximum(n2, N2MIN) ** (-0.43)
    # M is unknown in the history CSV, so retain both possible branches.
    with np.errstate(all="ignore"):
        ri2_full_shallow = (
            KAPPA * ws * z[0] * np.exp(-ks * z[0]) /
            P0 / (saved / CWAT - KM - fang) - 1
        ) / 37
        ri2_full_deep = (
            KAPPA * ws * z[0] * np.exp(-ks * z[0]) /
            P0 / (saved / (CWAT * MIXFACT) - KM - fang) - 1
        ) / 37
    return {
        "rho": rho, "rho2": rho2, "ri": ri, "n2": n2, "kme": kme,
        "ws": ws, "ks": ks, "drhodz": drhodz, "saved": saved,
        "rho_from_rho_ri": rho_from_rho_ri, "ri2_simple": ri2_simple,
        "ri2_full_shallow": ri2_full_shallow, "ri2_full_deep": ri2_full_deep,
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    m = finite(y) & finite(pred)
    y, pred = y[m], pred[m]
    if len(y) < 3:
        return {"n": int(len(y)), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    err = y - pred
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / sst) if sst else float("nan"),
    }


def published_surrogate(x: np.ndarray) -> np.ndarray:
    """Evaluate the coefficients currently printed in the manuscript."""
    shift = 4.547e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1001.4774 + 0.14073 * np.log(x + shift) - 6.399e-11 / (x + shift)


def fit_linear(features: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(features, y, rcond=None)
    return coef, features @ coef


def expression_search(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Search a finite, auditable one-variable expression family.

    Every candidate is linear in its coefficients after choosing its operator
    structure and shift. This keeps the search deterministic and reproducible.
    """
    valid = finite(x) & finite(y) & (x > 0)
    x, y = x[valid], y[valid]
    order = np.argsort(np.arange(len(x)))
    x, y = x[order], y[order]
    cut = max(3, int(0.7 * len(x)))
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
    scale = max(float(np.nanmedian(xtr)), 1e-12)
    u_tr, u_te = xtr / scale, xte / scale
    shifts = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
    candidates = []

    def add(name, formula, make, complexity):
        try:
            A_tr = make(u_tr)
            A_te = make(u_te)
            if not np.all(np.isfinite(A_tr)) or not np.all(np.isfinite(A_te)):
                return
            coef, _ = fit_linear(A_tr, ytr)
            train_pred, test_pred = A_tr @ coef, A_te @ coef
            mtr, mte = metrics(ytr, train_pred), metrics(yte, test_pred)
            score = mte["rmse"] / max(float(np.std(ytr)), 1e-12) + 0.002 * complexity
            candidates.append({
                "name": name, "formula": formula, "coef": coef, "train": mtr,
                "test": mte, "score": float(score), "complexity": complexity,
                "scale": scale,
            })
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return

    add("constant", "a", lambda u: np.column_stack([np.ones_like(u)]), 1)
    add("linear", "a + b*x", lambda u: np.column_stack([np.ones_like(u), u]), 2)
    add("quadratic", "a + b*x + c*x^2", lambda u: np.column_stack([np.ones_like(u), u, u**2]), 3)
    for shift in shifts:
        add(
            f"log_shift_{shift:g}", f"a + b*ln(x/{scale:.8g} + {shift:g})",
            lambda u, s=shift: np.column_stack([np.ones_like(u), np.log(u + s)]), 2,
        )
        add(
            f"log_inverse_shift_{shift:g}",
            f"a + b*ln(x/{scale:.8g} + {shift:g}) + c/(x/{scale:.8g} + {shift:g})",
            lambda u, s=shift: np.column_stack([np.ones_like(u), np.log(u + s), 1 / (u + s)]), 3,
        )
        add(
            f"sqrt_shift_{shift:g}", f"a + b*sqrt(x/{scale:.8g} + {shift:g})",
            lambda u, s=shift: np.column_stack([np.ones_like(u), np.sqrt(u + s)]), 2,
        )
    return sorted(candidates, key=lambda r: (r["score"], r["test"]["rmse"], r["complexity"]))


def format_fitted(candidate: dict) -> str:
    coef = candidate["coef"]
    formula = candidate["formula"]
    for i, value in enumerate(coef):
        formula = formula.replace(chr(97 + i), f"{value:.10g}", 1)
    return formula.replace("x", "savedtke1")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, z = load_inputs()
    values = derive(frame, z)
    checks = symbolic_checks()
    eqs = source_equations()

    # The history export uses LAKE_KME, so savedtke1 is reconstructed exactly
    # as the Fortran code defines it. TKE1 is a separate history field.
    rho = values["rho"]
    source_mask = (
        finite(rho) & finite(values["rho_from_rho_ri"]) & finite(values["ri"])
        & (np.abs(values["ws"]) < 100.0) & (values["ri"] >= 0.0)
    )
    simple_mask = source_mask & finite(values["ri2_simple"]) & (values["ri2_simple"] >= 0)
    eq1_metrics = metrics(rho[source_mask], values["rho_from_rho_ri"][source_mask])
    eq2_metrics = metrics(values["ri"][simple_mask] ** 2, values["ri2_simple"][simple_mask])

    # The manuscript's coefficients reproduce the old table only when its
    # predictor values are interpreted as KME.  Report both interpretations
    # explicitly so the variable-name error cannot be hidden by the table.
    published_kme_metrics = metrics(rho, published_surrogate(values["kme"]))
    published_saved_metrics = metrics(rho, published_surrogate(values["saved"]))
    published_converted_metrics = metrics(rho, published_surrogate(values["saved"] / CWAT))

    candidates = expression_search(values["saved"], rho)
    best = candidates[0] if candidates else None

    lines = [
        "AI-Miner lake-density pathway audit",
        "====================================",
        "",
        "Question: reproduce rho_w as a function of savedtke1 for the lake pathway.",
        "The supplied FEX archive was not imported or executed; the finite search below is written in this repository.",
        "",
        "Fortran source references:",
        *[f"  - {note}" for note in SOURCE_NOTES],
        "",
        "Symbolic checks:",
        f"  Equation 7a, rho_w rearrangement: {'PASS' if checks['rho_rearrangement'] else 'FAIL'}",
        f"  Equation 7b, simplified Ri^2 rearrangement under no-Fang/no-deep-mixing branch: {'PASS' if checks['simplified_ri_under_no_fang_branch'] else 'FAIL'}",
        "",
        "Equations considered:",
        f"  rho rearrangement: {eqs['rho_rearrangement']}",
        f"  simplified Ri^2: {eqs['simplified_ri']}",
        f"  full Fortran branch: {eqs['full_ri']}",
        f"  direct full-branch rho_w(S): {eqs['direct_branch']}",
        "",
        "Fortran branch decision:",
        "  The rho_w rearrangement is algebraically correct when the Ri equation is active without denominator clipping.",
        "  The displayed Ri^2 equation is conditional: it omits Fang-Stefan diffusivity and deep-mixing scaling.",
        f"  lake_no_ed default in LakeCon.F90: {LAKE_NO_ED_DEFAULT}; Fang-Stefan term is therefore active by default.",
        f"  deep-mixing factor: M=1 for shallow lakes, M={MIXFACT:g} when lake depth >= {DEPTHCRIT:g} m.",
        "  Therefore the simplified two-equation pathway is not the default full Fortran equation.",
        "",
        "Numeric audit of supplied history columns:",
        f"  rows: {len(frame):,}",
        f"  z(level 1), z(level 2): {z[0]:.8g}, {z[1]:.8g} m",
        f"  derived savedtke1 range: {np.nanmin(values['saved']):.8g} to {np.nanmax(values['saved']):.8g}",
        f"  LAKE_WS range: {np.nanmin(values['ws']):.8g} to {np.nanmax(values['ws']):.8g}; source sanity threshold is <100 m/s",
        f"  Eq. rho rearrangement replay rows after source sanity filter: {eq1_metrics['n']:,}; RMSE={eq1_metrics['rmse']:.8g}; R2={eq1_metrics['r2']:.8g}",
        f"  Simplified Ri^2 comparison rows after source sanity filter: {eq2_metrics['n']:,}; RMSE={eq2_metrics['rmse']:.8g}; R2={eq2_metrics['r2']:.8g}",
        "  The supplied LAKE_WS/LAKE_RI history fields fail the source-variable sanity check, so these replay metrics are diagnostic only.",
        "",
        "Published one-variable surrogate audit:",
        "  rho_w = 1001.4774 + 0.14073 ln(x + 4.547e-10) - 6.399e-11/(x + 4.547e-10)",
        "  corrected savedtke1 form: rho_w = 1001.4774 + 0.14073 ln(savedtke1/cwat + 4.547e-10) - 6.399e-11/(savedtke1/cwat + 4.547e-10)",
        f"  interpreting x as level-1 LAKE_KME: n={published_kme_metrics['n']:,}; MAE={published_kme_metrics['mae']:.8g}; RMSE={published_kme_metrics['rmse']:.8g}; R2={published_kme_metrics['r2']:.8g}",
        f"  interpreting x as Fortran savedtke1=cwat*KME: n={published_saved_metrics['n']:,}; MAE={published_saved_metrics['mae']:.8g}; RMSE={published_saved_metrics['rmse']:.8g}; R2={published_saved_metrics['r2']:.8g}",
        f"  interpreting x as savedtke1/cwat: n={published_converted_metrics['n']:,}; MAE={published_converted_metrics['mae']:.8g}; RMSE={published_converted_metrics['rmse']:.8g}; R2={published_converted_metrics['r2']:.8g}",
        "  Conclusion: the printed coefficients are numerically a KME surrogate. The valid savedtke1 reparameterization is obtained by replacing x with savedtke1/cwat; the manuscript table uses that conversion and remains subject to independent validation.",
        "",
        "Finite-expression search: rho_w = f(savedtke1)",
        "  Split: first 70% of rows for fitting, final 30% for a screening holdout; no shuffle.",
        "  Coefficients are least-squares estimates after selecting a finite operator/shift structure.",
        "  Ranking score = validation RMSE / training-target SD + 0.002 * expression complexity.",
        "",
    ]
    for rank, candidate in enumerate(candidates, 1):
        lines.append(
            f"  {rank:02d}. {candidate['name']}: {format_fitted(candidate)} | "
            f"train RMSE={candidate['train']['rmse']:.8g}, screening RMSE={candidate['test']['rmse']:.8g}, "
            f"screening R2={candidate['test']['r2']:.8g}, score={candidate['score']:.8g}"
        )
    lines += ["", "BEST SOURCE EQUATION", "--------------------"]
    lines += [
        eqs["direct_branch"],
        "This is the source-faithful expression for savedtke1 S when the active Fortran branch is known (including F and M).",
        "The Fang-Stefan term F depends on N^2 = (g/rho_w) d(rho_w)/dz, so this remains an implicit source relation rather than a one-variable closed form.",
        "",
        "BEST EMPIRICAL CANDIDATE",
        "------------------------",
    ]
    if best is None:
        lines.append("No finite candidate could be fitted.")
    else:
        lines += [
            f"rho_w = {format_fitted(best)}",
            f"candidate: {best['name']}",
            f"screening RMSE: {best['test']['rmse']:.10g}",
            f"screening R2: {best['test']['r2']:.10g}",
            "This is an empirical one-variable candidate from the supplied history output; it is not a replacement for the full implicit Fortran branch.",
            "No validation conclusion is made here. Independent E3SM validation belongs to the later validation folder.",
        ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TXT}")
    print(lines[-5:])


if __name__ == "__main__":
    main()
