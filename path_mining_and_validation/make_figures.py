"""Generate separate, publication-style single-panel figures.

The palette is based on the Okabe-Ito color-blind-safe colors commonly used in
Nature/Science-style figures. It is deliberately restrained: color identifies
the primary series or status, while labels and geometry carry the meaning.
"""

import json
import os
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import config

R = config.RESULT_DIR
FIG = config.FIG_DIR
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito / journal-style palette: blue, vermilion, bluish green, orange,
# reddish purple, sky blue, yellow, and black. The lighter fills are reserved
# for secondary categories and never used as primary text colors.
BLUE = "#0072B2"
ORANGE = "#E69F00"
AQUA = "#009E73"
VIOLET = "#CC79A7"
YELLOW = "#F0E442"
RED = "#D55E00"
SKY = "#56B4E9"
INK = "#222222"
MUTED = "#666666"
GRID = "#D9D9D9"
BLUE_L = "#A6CEE3"
GREY = "#BDBDBD"
PALE = "#F2F2F2"
STATUS_COL = {"located": BLUE, "partial": SKY, "unlocated": RED, "no_identifier": GREY, "file_missing": PALE}
STATUS_LAB = {"located": "Located (coverage ≥ 0.8)", "partial": "Partial (0.5–0.8)", "unlocated": "Not located (< 0.5)",
              "no_identifier": "No identifier", "file_missing": "File absent"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 600,
    "legend.frameon": False, "legend.handlelength": 1.4,
    "axes.titleweight": "normal",
})
MM = 1 / 25.4
W = 183 * MM
mpl.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})


def single_plot(plots, name, wide=False):
    fig, ax = plt.subplots(figsize=(14 if wide else 7.2, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(length=3, width=0.7, colors=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    plots.append((fig, name))
    return ax


def save(fig, name):
    if len(fig.axes) != 1:
        raise ValueError(f"{name} must contain exactly one axes")
    fig.tight_layout(pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def csv(name):
    return pd.read_csv(os.path.join(R, name))


def pct(ax):
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))


# =============================================================== Figure 1: graph -> Fortran provenance
def fig_provenance():
    edge = csv("graph_vs_fortran.csv")
    summ = csv("equation_provenance_summary.csv")
    eq = csv("equation_provenance.csv")
    var = csv("variable_provenance_summary.csv")
    vin = csv("variable_in_statement.csv")
    chance_cov = np.load(os.path.join(R, "chance_coverage.npy"))

    plots = []

    # ---- a: edge-level agreement with the Fortran sources
    ax = single_plot(plots, "Fig_edge_source_support")
    chk_eq = [c for c in edge.check if c.startswith(">=")][0]
    chk_id = "symbol is an identifier of the equation's file (identifier-like, file found)"
    sel = [("has", "file exists in Fortran folder", "root → file: file present"),
           ("encodes", "source file exists", "file → equation: file present"),
           ("encodes", chk_eq, "file → equation: identifiers found"),
           ("contains", "symbol is an identifier of its file (identifier-like symbols)", "file → symbol: in file"),
           ("hasVariable", chk_id, "variable → equation: in file"),
           ("hasConstant", chk_id, "constant → equation: in file"),
           ("hasVariable", "symbol occurs in the equation text (all)", "variable occurs in equation text"),
           ("hasConstant", "symbol occurs in the equation text (all)", "constant occurs in equation text")]
    ys = np.arange(len(sel))[::-1]
    for y, (rel, chk, lab) in zip(ys, sel):
        r = edge[(edge.relationship == rel) & (edge.check == chk)].iloc[0]
        ax.barh(y, r.rate, color=BLUE, height=0.62, edgecolor="none")
        if not np.isnan(r.chance):
            ax.plot([r.chance], [y], marker="D", ms=3.2, color=ORANGE, mec="white", mew=0.4, zorder=3)
        ax.text(min(r.rate + 0.02, 0.86), y, f"{r.rate * 100:.0f}%", va="center", fontsize=9, color="white" if r.rate > 0.9 else INK,
                ha="right" if r.rate > 0.9 else "left")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[2] for s in sel], fontsize=9)
    ax.set_xlim(0, 1.0)
    pct(ax)
    ax.set_xlabel("Share of edges passing source-support check")
    ax.legend(handles=[Patch(color=BLUE, label="e3sm.csv"), Line2D([], [], marker="D", ls="", color=ORANGE, ms=3.2, label="random-file baseline")],
              loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, columnspacing=1.0)

    # ---- b: equation status vs random-file baseline
    ax = single_plot(plots, "Fig_equation_localization")
    order = ["located", "partial", "unlocated", "no_identifier", "file_missing"]
    s = summ.set_index("status")
    bars = [("e3sm.csv\n(n = %s)" % f"{int(s.equations.sum()):,}", s.share), ("random-file\nbaseline", s.chance_share)]
    for yi, (lab, ser) in enumerate(bars[::-1]):
        left = 0
        for st in order:
            v = float(ser.get(st, 0))
            ax.barh(yi, v, left=left, color=STATUS_COL[st], height=0.55, edgecolor="white", linewidth=0.6)
            if v > 0.06:
                ax.text(left + v / 2, yi, f"{v * 100:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if st in ("located", "unlocated") else INK)
            left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels([b[0] for b in bars[::-1]])
    ax.set_xlim(0, 1)
    pct(ax)
    ax.set_xlabel("Share of equation–file records")
    ax.legend(handles=[Patch(color=STATUS_COL[k], label=STATUS_LAB[k]) for k in order[:3]], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1, handlelength=1)

    # ---- c: ECDF of identifier coverage
    ax = single_plot(plots, "Fig_identifier_coverage")
    cov = np.sort(eq.loc[eq.status != "file_missing", "coverage"].to_numpy())
    ch = np.sort(chance_cov)
    ax.step(cov, np.arange(1, len(cov) + 1) / len(cov), where="post", color=BLUE, lw=1.1, label="e3sm.csv")
    ax.step(ch, np.arange(1, len(ch) + 1) / len(ch), where="post", color=ORANGE, lw=1.1, label="random file")
    ax.axvline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("Identifier coverage of best statement")
    ax.set_ylabel("Cumulative share of equations")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55))

    # ---- d: traceability of the equation-richest files
    ax = single_plot(plots, "Fig_source_file_traceability")
    top = eq.groupby("file").size().sort_values(ascending=False).head(16).index
    stat = eq[eq.file.isin(top)].groupby(["file", "status"]).size().unstack(fill_value=0).reindex(top)
    left = np.zeros(len(top))
    for st in order:
        v = stat[st].to_numpy() if st in stat else np.zeros(len(top))
        ax.barh(np.arange(len(top))[::-1], v, left=left, color=STATUS_COL[st], height=0.72, edgecolor="none")
        left += v
    ax.set_yticks(np.arange(len(top))[::-1])
    ax.set_yticklabels([f.rsplit(".", 1)[0][:22] for f in top], fontsize=9)
    ax.set_xlabel("Equations in the graph")

    # ---- e: variable provenance
    ax = single_plot(plots, "Fig_variable_provenance")
    v = var.set_index("status")
    keys = ["declared", "used_only", "absent"]
    labels = {"declared": "Declared\n(line known)", "used_only": "Used,\nnot declared", "absent": "Not in file"}
    cols = {"declared": BLUE, "used_only": BLUE_L, "absent": ORANGE}
    top_share = max(float(v.loc[k, "share"]) for k in keys if k in v.index)
    for i, k in enumerate(keys):
        sh = float(v.loc[k, "share"]) if k in v.index else 0.0
        n = int(v.loc[k, "symbols"]) if k in v.index else 0
        ax.bar(i, sh, color=cols[k], width=0.62)
        ax.text(i, sh + 0.012, f"{sh * 100:.0f}%\n({n:,})", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([labels[k] for k in keys], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of file→symbol edges")
    ins = single_plot(plots, "Fig_symbols_in_matched_statement")
    for i, (k, c) in enumerate([("var", BLUE), ("const", AQUA)]):
        sub = vin[vin.kind == k]
        r = sub.in_statement.mean() if len(sub) else np.nan
        ins.bar(i, r, color=c, width=0.6)
        ins.text(i, r + 0.03, f"{r * 100:.0f}%", ha="center", fontsize=9)
    ins.set_xticks([0, 1])
    ins.set_xticklabels(["var.", "const."], fontsize=9)
    ins.set_ylim(0, 1.2)
    ins.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ins.set_ylabel("Share of symbols in matched statements")
    ins.spines["left"].set_visible(False)
    ins.set_title("variables found in matched line", fontsize=9, pad=2, loc="left")

    # ---- f: where in the source files do the equations come from
    ax = single_plot(plots, "Fig_equation_source_positions")
    loc = eq[eq.status == "located"].copy()
    nlines = {}
    for f in loc.file.unique():
        try:
            nlines[f] = sum(1 for _ in open(os.path.join(config.FORTRAN_DIR, f), "r", encoding="utf-8", errors="replace"))
        except OSError:
            nlines[f] = np.nan
    loc["pos"] = loc.start / loc.file.map(nlines)
    topf = loc.groupby("file").size().sort_values(ascending=False).head(24).index
    rng = np.random.default_rng(0)
    for i, f in enumerate(topf):
        pos = loc.loc[loc.file == f, "pos"].dropna().to_numpy()
        ax.scatter(pos, np.full_like(pos, len(topf) - 1 - i) + rng.uniform(-0.18, 0.18, len(pos)), s=2.2, color=BLUE, alpha=0.55, linewidths=0)
    ax.set_yticks(np.arange(len(topf))[::-1][::3])
    ax.set_yticklabels([f.rsplit(".", 1)[0][:16] for f in topf][::3], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Relative position in file")
    ax.set_title("Located equations, 24 richest files", fontsize=9, pad=3, loc="left")
    for fig, name in plots:
        save(fig, name)


# =============================================================== Figure 2: verification
def fig_verification():
    bench, naive = csv("benchmark_paths.csv"), csv("benchmark_naive.csv")
    best = bench[bench["rank"] == 1]
    rep = csv("numerical_replay.csv")
    z = np.load(os.path.join(R, "raw_numeric.npz"))
    cs = csv("case_studies.csv")
    cs = cs[(cs.graph == "e3sm.csv") & (cs.case == "Lake chain (Fortran)") & (cs.status == "path found")].iloc[0]
    lines = csv("case_path_code_lines.csv")
    lines = lines[lines.case == "e3sm.csv::Lake chain (Fortran)"].drop_duplicates("equation").set_index("equation")
    path = pd.read_pickle(os.path.join(R, "case_paths.pkl"))["paths"]["e3sm.csv::Lake chain (Fortran)"]
    n_pairs = config.N_RANDOM_PAIRS

    plots = []

    ax = single_plot(plots, "Fig_ranked_chain_length")
    vc = best.n_equations.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=BLUE, width=0.7)
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y + 0.6, str(y), ha="center", fontsize=9)
    ax.set_xlabel("Equations in the top-ranked chain |P|")
    ax.set_ylabel("Variable pairs")
    ax.set_title(f"{bench['query'].nunique()} of {n_pairs} random pairs connected", fontsize=9, loc="left", pad=3)

    ax = single_plot(plots, "Fig_path_hub_degree")
    data = [naive.max_degree.to_numpy(), best.max_degree.to_numpy()]
    vp = ax.violinplot([np.log10(d) for d in data], positions=[0, 1], widths=0.8, showextrema=False)
    for body, c in zip(vp["bodies"], [ORANGE, BLUE]):
        body.set_facecolor(c)
        body.set_edgecolor("none")
        body.set_alpha(0.85)
    for i, d in enumerate(data):
        q = np.percentile(np.log10(d), [25, 50, 75])
        ax.plot([i, i], [q[0], q[2]], color=INK, lw=1.0)
        ax.plot([i], [q[1]], marker="o", ms=3, color="white", mec=INK, mew=0.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hop-count\nshortest", "AI-Miner\nranked"])
    ax.set_ylabel("Largest node degree on path (log$_{10}$)")

    ax = single_plot(plots, "Fig_verification_components")
    comps = [("S_src", "S$_{src}$"), ("S_var", "S$_{var}$"), ("S_sym", "S$_{sym}$"), ("V", "V")]
    for i, (k, lab) in enumerate(comps):
        for df, c, off in [(naive, ORANGE, -0.19), (best, BLUE, 0.19)]:
            ax.boxplot([df[k].dropna().to_numpy()], positions=[i + off], widths=0.3, patch_artist=True, showfliers=False,
                       medianprops=dict(color="white", lw=1.0), whiskerprops=dict(color=c, lw=0.7),
                       capprops=dict(color=c, lw=0.7), boxprops=dict(facecolor=c, edgecolor=c))
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[1] for c in comps])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.legend(handles=[Patch(color=ORANGE, label="hop-count shortest"), Patch(color=BLUE, label="AI-Miner ranked")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0, handlelength=1)

    ax = single_plot(plots, "Fig_path_verification_status")
    cls = ["verified", "partially_verified", "rejected"]
    cc = {"verified": BLUE, "partially_verified": BLUE_L, "rejected": ORANGE}
    cl = {"verified": "Verified", "partially_verified": "Partial", "rejected": "Rejected"}
    rows = [("AI-Miner\nranked", best), ("Hop-count\nshortest", naive)]
    for yi, (lab, df) in enumerate(rows):
        left = 0
        for k in cls:
            v = (df.status == k).mean()
            ax.barh(len(rows) - 1 - yi, v, left=left, color=cc[k], height=0.55, edgecolor="white", linewidth=0.6)
            if v > 0.07:
                ax.text(left + v / 2, len(rows) - 1 - yi, f"{v * 100:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if k != "partially_verified" else INK)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 1)
    pct(ax)
    ax.set_xlabel("Share of selected chains")
    ax.legend(handles=[Patch(color=cc[k], label=cl[k]) for k in cls], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3,
              columnspacing=0.8, handlelength=1)

    ax = single_plot(plots, "Fig_lake_replay_scores")
    names = ["$\\rho_w$(T)", "N$^2$", "Ri", "k$_{me}$", "savedtke1"]
    for i, r in rep.iterrows():
        yi = len(rep) - 1 - i
        if r.replayable:
            ax.barh(yi, max(r.R2, 0), color=BLUE, height=0.6)
            ax.text(max(r.R2, 0) + 0.02, yi, f"{r.R2:.6f}", va="center", fontsize=9)
        else:
            ax.barh(yi, 1.0, color="none", edgecolor=MUTED, hatch="////", linewidth=0.0, height=0.6)
            ax.text(0.5, yi, "not replayable", ha="center", va="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_yticks(range(len(rep)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlim(0, 1.32)
    ax.set_xlabel("Replay R$^2$ vs E3SM output")

    ax = single_plot(plots, "Fig_density_replay")
    x, y = z["rho_obs"], z["rho_rep"]
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e30)
    ax.hexbin(x[m], y[m], gridsize=40, bins="log", mincnt=1, cmap=mpl.colors.LinearSegmentedColormap.from_list("b", ["#dbe8f8", BLUE]), linewidths=0)
    lo, hi = x[m].min(), x[m].max()
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("E3SM water density (kg m$^{-3}$)")
    ax.set_ylabel("Re-computed from Fortran")
    ax.text(0.04, 0.95, f"n = {m.sum():,}\nR$^2$ = {rep.R2.iloc[0]:.6f}", transform=ax.transAxes, va="top", fontsize=9)

    # One vertical pathway diagram keeps equations and source lines readable.
    ax = single_plot(plots, "Fig_lake_path_source_lines", wide=True)
    ax.figure.set_size_inches(10, 11)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ys = np.linspace(0.95, 0.15, len(path))
    ax.plot([0.5, 0.5], [ys[-1], ys[0]], color=MUTED, lw=1, zorder=1)
    for yy, node in zip(ys, path):
        if node.startswith("s:"):
            ax.text(0.5, yy, node[2:], ha="center", va="center", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=INK), zorder=4)
            continue
        stmt = node[2:]
        row = lines.loc[stmt] if stmt in lines.index else None
        col = BLUE if row is not None and row.coverage >= 0.8 else BLUE_L
        ax.text(0.5, yy, "\n".join(textwrap.wrap(stmt, 70)), ha="center", va="center",
                fontsize=9, family="monospace", color="white" if col == BLUE else INK,
                bbox=dict(boxstyle="round,pad=0.5", fc=col, ec="none"), zorder=4)
        if row is not None:
            label = f"{row.file} | L{int(row.start)}"
            if row.end != row.start:
                label += f"–{int(row.end)}"
            label += f" | identifier coverage {row.coverage:.2f}"
            ax.text(0.5, yy - 0.04, label, ha="center", va="top", fontsize=9,
                    bbox=dict(fc="white", ec="none", pad=1), zorder=4)
    summary = (f"Source support = {cs.S_src:.2f}; variable support = {cs.S_var:.2f}; "
               f"symbolic parsing = {cs.S_sym:.2f}; numerical support = {cs.S_num:.2f}; "
               f"aggregate V = {cs.V:.2f}. Status: {str(cs.status_class).replace('_', ' ')}. "
               f"Case-level replay available for {int(rep.replayable.sum())}/{len(rep)} steps; "
               "this is not an independent replay of the displayed candidate.")
    ax.text(0.03, 0.04, "\n".join(textwrap.wrap(summary, 112)), fontsize=9, va="top")
    for fig, name in plots:
        save(fig, name)


if __name__ == "__main__":
    fig_provenance()
    fig_verification()
