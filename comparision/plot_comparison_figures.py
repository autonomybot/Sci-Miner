"""Create separate publication figures from the completed comparison runs.

No models are refit here. The script reads ``comparision/results/*.csv`` and
writes one PNG/PDF per figure into the repository-level ``figures`` folder.
All figures are single-panel plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "comparision" / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["Pathway ln-form", "FEX (1-var)", "FEX", "SINDy (STLSQ)", "MLP"]
MODEL_COLORS = {
    "Pathway ln-form": "#7A7A7A",
    "FEX (1-var)": "#D55E00",
    "FEX": "#0072B2",
    "SINDy (STLSQ)": "#009E73",
    "MLP": "#CC79A7",
}
SPLIT_ORDER = ["random", "chrono", "blocked5", "site"]
SPLIT_LABELS = {"random": "Random", "chrono": "Chronological", "blocked5": "5 blocks", "site": "Leave-site-out"}


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", lw=0.55, alpha=0.75)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def model_ranking(summary: pd.DataFrame, case: str, feature_set: str, stem: str, title: str) -> None:
    d = summary[(summary.case == case) & (summary.feature_set == feature_set)].copy()
    d = d[d.model.isin(MODEL_ORDER)]
    splits = [s for s in SPLIT_ORDER if s in set(d.scheme)]
    models = [m for m in MODEL_ORDER if m in set(d.model)]
    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=180)
    x = np.arange(len(splits))
    width = 0.78 / max(len(models), 1)
    for j, model in enumerate(models):
        vals = []
        errs = []
        for split in splits:
            row = d[(d.scheme == split) & (d.model == model)]
            vals.append(float(row.R2.iloc[0]) if len(row) else np.nan)
            errs.append(float(row.R2_sd.iloc[0]) if len(row) and pd.notna(row.R2_sd.iloc[0]) else 0.0)
        ax.bar(x + (j - (len(models) - 1) / 2) * width, np.clip(vals, -1, 1), width * 0.92,
               yerr=np.minimum(np.nan_to_num(errs), 1.0), capsize=2, color=MODEL_COLORS[model],
               label=model, linewidth=0)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in splits])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Held-out $R^2$ (clipped at −1)")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    save(fig, stem)


def fex_robustness(summary: pd.DataFrame, case: str, feature_sets: list[str], stem: str, title: str) -> None:
    d = summary[(summary.case == case) & summary.model.isin(["FEX", "FEX (1-var)"])].copy()
    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=180)
    x = np.arange(len(SPLIT_ORDER))
    colors = ["#D55E00", "#0072B2", "#009E73"]
    width = 0.74 / len(feature_sets)
    for j, feature_set in enumerate(feature_sets):
        model = "FEX (1-var)" if feature_set in ("univariate", "univariate_kme1") else "FEX"
        vals = []
        for split in SPLIT_ORDER:
            row = d[(d.feature_set == feature_set) & (d.scheme == split) & (d.model == model)]
            vals.append(float(row.R2.iloc[0]) if len(row) else np.nan)
        label = {"univariate": "Univariate", "univariate_kme1": "savedtke / $k_{me}$", "pathway": "Full pathway", "pathway_noT": "Pathway without $T$-dependent fields"}.get(feature_set, feature_set)
        ax.bar(x + (j - (len(feature_sets) - 1) / 2) * width, np.clip(vals, -1, 1), width * 0.9, color=colors[j], label=label, linewidth=0)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("FEX held-out $R^2$ (clipped at −1)")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, stem)


def path_recovery(summary: pd.DataFrame, stem: str) -> None:
    d = summary.copy()
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=180)
    x = np.arange(len(d))
    colors = ["#0072B2", "#7A7A7A"]
    bars = ax.bar(x, d.connected_share * 100, color=colors[:len(d)], width=0.58, linewidth=0)
    ax.set_xticks(x, ["Sci-Miner\nequation chain", "Definition-use\nchain"][:len(d)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Connected benchmark queries (%)")
    ax.set_title("Path-recovery connectivity", loc="left", fontsize=10, weight="bold")
    for bar, value in zip(bars, d.connected_share * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    style_axis(ax)
    save(fig, stem)


def model_error(summary: pd.DataFrame, case: str, feature_set: str, stem: str, title: str) -> None:
    """Plot normalized RMSE without clipping the extreme stress-test cases."""
    d = summary[(summary.case == case) & (summary.feature_set == feature_set)].copy()
    d = d[d.model.isin(MODEL_ORDER)]
    splits = [s for s in SPLIT_ORDER if s in set(d.scheme)]
    models = [m for m in MODEL_ORDER if m in set(d.model)]
    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=180)
    x = np.arange(len(splits))
    width = 0.78 / max(len(models), 1)
    for j, model in enumerate(models):
        vals = []
        for split in splits:
            row = d[(d.scheme == split) & (d.model == model)]
            vals.append(float(row.NRMSE.iloc[0]) if len(row) else np.nan)
        ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width * 0.92,
               color=MODEL_COLORS[model], label=model, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in splits])
    ax.set_ylabel("Held-out normalized RMSE (log scale; lower is better)")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    save(fig, stem)


def fex_error_robustness(summary: pd.DataFrame, case: str, feature_sets: list[str], stem: str, title: str) -> None:
    """Compare FEX feature sets using normalized RMSE on a log scale."""
    d = summary[(summary.case == case) & summary.model.isin(["FEX", "FEX (1-var)"])].copy()
    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=180)
    x = np.arange(len(SPLIT_ORDER))
    colors = ["#D55E00", "#0072B2", "#009E73"]
    width = 0.74 / len(feature_sets)
    for j, feature_set in enumerate(feature_sets):
        model = "FEX (1-var)" if feature_set in ("univariate", "univariate_kme1") else "FEX"
        vals = []
        for split in SPLIT_ORDER:
            row = d[(d.feature_set == feature_set) & (d.scheme == split) & (d.model == model)]
            vals.append(float(row.NRMSE.iloc[0]) if len(row) else np.nan)
        label = {"univariate": "Univariate", "univariate_kme1": "savedtke / $k_{me}$",
                 "pathway": "Full pathway", "pathway_noT": "Pathway without $T$-dependent fields"}.get(feature_set, feature_set)
        ax.bar(x + (j - (len(feature_sets) - 1) / 2) * width, vals, width * 0.9,
               color=colors[j], label=label, linewidth=0)
    ax.set_yscale("log")
    ax.set_xticks(x, [SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_ylabel("FEX normalized RMSE (log scale; lower is better)")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, stem)


def main() -> None:
    summary = pd.read_csv(RESULTS / "regression_summary_all.csv")
    model_ranking(summary, "lake", "pathway", "Fig_comparison_lake_models", "Lake pathway model comparison")
    model_ranking(summary, "methane", "pathway_noT", "Fig_comparison_methane_models", "Methane pathway comparison without temperature-dependent fields")
    model_error(summary, "lake", "pathway", "Fig_comparison_lake_model_error", "Lake pathway error comparison")
    model_error(summary, "methane", "pathway_noT", "Fig_comparison_methane_model_error", "Methane pathway error comparison without temperature-dependent fields")
    fex_robustness(summary, "lake", ["univariate_kme1", "pathway"], "Fig_comparison_lake_fex_robustness", "Lake FEX robustness across feature sets")
    fex_robustness(summary, "methane", ["univariate", "pathway_noT", "pathway"], "Fig_comparison_methane_fex_robustness", "Methane FEX robustness across feature sets")
    fex_error_robustness(summary, "lake", ["univariate_kme1", "pathway"], "Fig_comparison_lake_fex_error_robustness", "Lake FEX error across feature sets")
    fex_error_robustness(summary, "methane", ["univariate", "pathway_noT", "pathway"], "Fig_comparison_methane_fex_error_robustness", "Methane FEX error across feature sets")
    path_recovery(pd.read_csv(RESULTS / "path_recovery_summary.csv"), "Fig_comparison_path_recovery_connectivity")
    print("wrote comparison figures to", FIGURES)


if __name__ == "__main__":
    main()
