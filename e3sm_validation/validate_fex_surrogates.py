"""Validate the saved original-FEX equations against E3SM history output.

The equations and scaler values are read from
``function_approximation/fex_original_lake_methane_metadata.json``.  No
refitting is performed here.  The script reports the original random holdout,
chronological holdout, and lake site-wise diagnostics, then writes one figure
for each case (never a multi-panel figure).

Run from the repository root::

    python e3sm_validation/validate_fex_surrogates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e3sm_validation"
FIGURES = ROOT / "figures"
META_PATH = ROOT / "function_approximation" / "fex_original_lake_methane_metadata.json"
SEED = 0
CWAT = 4.188e6

sys.path.insert(0, str(ROOT / "function_approximation"))
import fex_original_regression as fex  # noqa: E402

sys.path.insert(0, str(ROOT / "comparision"))
import common  # noqa: E402


COLORS = {"lake": "#0072B2", "methane": "#D55E00"}


def load_data(case: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    data = common.CASES[case]["load"]().copy()
    if case == "lake":
        data["savedtke1"] = data["kme1"] * CWAT
        names = ["tke1", "savedtke1", "drhodz1", "ri1", "ws", "ks"]
        return data, data[names].to_numpy(float), data["rho_w"].to_numpy(float), names
    data = data.rename(columns={"conc1": "conc_CH4_sat_col"})
    names = ["conc_CH4_sat_col"]
    return data, data[names].to_numpy(float), data["t_soisno"].to_numpy(float), names


def predict(item: dict, X: np.ndarray) -> np.ndarray:
    params = {k: torch.tensor(v, dtype=torch.float64) for k, v in item["parameters"].items()}
    scale = (np.asarray(item["input_mean"], float) if "input_mean" in item else None)
    if scale is None:
        # Original regression metadata stores these under input_standardization;
        # the numeric arrays are reconstructed from the saved parameter vector
        # fields below when the legacy key is absent.
        raise KeyError("input scaler arrays missing from FEX metadata")
    z = (X - scale) / np.asarray(item["input_scale"], float)
    with torch.no_grad():
        normalized = fex.forward(params, torch.tensor(z, dtype=torch.float64), item["choices"])
    return float(item["target_mean"]) + float(item["target_scale"]) * normalized.cpu().numpy()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    error = p - y
    sst = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "n": int(len(y)),
        "r2": float(1.0 - np.sum(error**2) / sst) if sst else float("nan"),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "nrmse": rmse / float(np.ptp(y)) if np.ptp(y) else float("nan"),
        "median_abs_error": float(np.median(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def random_holdout(n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(SEED).permutation(n)
    cut = int(0.70 * n)
    return np.sort(order[:cut]), np.sort(order[cut:])


def chronological_holdout(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train, test = [], []
    for site in data["site"].unique():
        indices = np.flatnonzero(data["site"].to_numpy() == site)
        cut = int(0.70 * len(indices))
        train.extend(indices[:cut])
        test.extend(indices[cut:])
    return np.asarray(train, int), np.asarray(test, int)


def site_partitions(data: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    return [(f"site_{int(site)}", np.flatnonzero(data["site"].to_numpy() == site)) for site in sorted(data["site"].unique())]


def plot_case(case: str, y: np.ndarray, p: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.1), dpi=180)
    color = COLORS[case]
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(y[mask], p[mask], s=7, alpha=0.22, color=color, linewidths=0, rasterized=True)
    lo = float(min(np.min(y), np.min(p)))
    hi = float(max(np.max(y), np.max(p)))
    pad = 0.03 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    target = r"$\rho_w$ (kg m$^{-3}$)" if case == "lake" else r"Observed $t_{\mathrm{soisno}}$ (K)"
    predicted = r"Predicted $\rho_w$ (kg m$^{-3}$)" if case == "lake" else r"Predicted $t_{\mathrm{soisno}}$ (K)"
    ax.set_xlabel(target)
    ax.set_ylabel(predicted)
    ax.grid(True, color="#D9D9D9", lw=0.55, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Lake FEX pathway replay" if case == "lake" else "Methane FEX concentration replay", loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_residuals(case: str, y: np.ndarray, p: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.1), dpi=180)
    color = COLORS[case]
    residual = p - y
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(y[mask], residual[mask], s=7, alpha=0.22, color=color, linewidths=0, rasterized=True)
    ax.axhline(0.0, color="#222222", lw=1.0, ls="--")
    ax.set_xlabel(r"Observed target")
    ax.set_ylabel(r"Prediction residual (predicted $-$ observed)")
    ax.grid(True, color="#D9D9D9", lw=0.55, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Lake FEX residuals" if case == "lake" else "Methane FEX residuals", loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_absolute_error(case: str, y: np.ndarray, p: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.1), dpi=180)
    color = COLORS[case]
    error = np.abs(p - y)
    ax.hist(error, bins=45, color=color, alpha=0.88, edgecolor="white", linewidth=0.35)
    ax.axvline(np.median(error), color="#222222", lw=1.0, ls="--", label=f"median = {np.median(error):.3g}")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Records")
    ax.grid(True, axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Lake FEX absolute error" if case == "lake" else "Methane FEX absolute error", loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_time_series(case: str, data: pd.DataFrame, indices: np.ndarray, y: np.ndarray, p: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=180)
    color = COLORS[case]
    # For lake, plot the first chronological site to keep the figure readable;
    # the complete site-wise metrics remain in the CSV.
    site = sorted(data.iloc[indices]["site"].unique())[0]
    selected = indices[data.iloc[indices]["site"].to_numpy() == site]
    selected = selected[np.argsort(selected)]
    if len(selected) > 1500:
        selected = selected[:: max(1, len(selected) // 1500)]
    x = np.arange(len(selected))
    ax.plot(x, y[selected], color="#222222", lw=0.9, alpha=0.82, label="E3SM observed")
    ax.plot(x, p[selected], color=color, lw=1.0, alpha=0.9, label="FEX prediction")
    ax.set_xlabel("Chronological holdout record")
    ax.set_ylabel(r"$\rho_w$ (kg m$^{-3}$)" if case == "lake" else r"$t_{\mathrm{soisno}}$ (K)")
    ax.grid(True, color="#D9D9D9", lw=0.55, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.set_title("Lake FEX chronological replay" if case == "lake" else "Methane FEX chronological replay", loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    rows = []
    for case, key in (("lake", "lake_pathway"), ("methane", "methane_concentration")):
        data, X, y, names = load_data(case)
        item = metadata["cases"][key]
        # The saved metadata uses the machine-readable scaler arrays from the
        # regression script; fail loudly if a hand-edited file omits them.
        if "input_mean" not in item or "input_scale" not in item:
            raise KeyError(f"{key}: input scaler arrays are absent from metadata")
        p = predict(item, X)
        train_r, test_r = random_holdout(len(data))
        chrono_r, chrono_t = chronological_holdout(data)
        partitions = [("random_train", train_r), ("random_holdout", test_r), ("chronological_train", chrono_r), ("chronological_holdout", chrono_t)]
        if case == "lake":
            partitions.extend(site_partitions(data))
        for split, indices in partitions:
            m = metrics(y[indices], p[indices])
            rows.append({"case": case, "equation": key, "split": split, **m})
        # Figure uses the temporally held-out records; lake points are colored
        # by site and methane points by its single land grid.
        _, test = chronological_holdout(data)
        labels = data.iloc[test]["site"].to_numpy()
        plot_case(case, y[test], p[test], labels, FIGURES / f"Fig_fex_{case}_validation")
        plot_residuals(case, y[test], p[test], labels, FIGURES / f"Fig_fex_{case}_residuals")
        plot_absolute_error(case, y[test], p[test], FIGURES / f"Fig_fex_{case}_absolute_error")
        plot_time_series(case, data, test, y, p, FIGURES / f"Fig_fex_{case}_chronological_replay")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "fex_validation_metrics.csv", index=False)
    (OUT / "README.md").write_text(
        "# E3SM validation\n\n"
        "`fex_validation_metrics.csv` evaluates the fixed equations saved by "
        "`function_approximation/fex_original_regression.py`; no refitting is "
        "performed. The two `Fig_fex_*_validation` files are separate figures "
        "using the chronological 30% holdout.\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"wrote {OUT / 'fex_validation_metrics.csv'}")


if __name__ == "__main__":
    main()
