"""Collect the result CSVs into paper-ready tables (Markdown) -> RESULT_DIR/TABLES.md

    python make_tables.py
"""

import json
import os

import numpy as np
import pandas as pd

import config

R = config.RESULT_DIR


def md(df, floatfmt=None):
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.get(c, "{:.3f}").format(v))
            else:
                cells.append(str(v).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def rd(name):
    return pd.read_csv(os.path.join(R, name))


def main():
    parts = ["# Path mining and validation: result tables\n",
             "All numbers are produced by `run_all.py`; nothing here is typed by hand.\n"]

    gs = rd("graph_statistics.csv")
    parts += ["## Table 1. Science-graph size and cleaning\n", md(gs, {"e3sm.csv": "{:,.4g}", "neo4j export": "{:,.4g}"}), ""]

    ev = rd("graph_vs_fortran.csv")
    ev["rate (%)"] = ev.rate * 100
    ev["chance (%)"] = ev.chance * 100
    parts += ["## Table 2. Agreement of every e3sm.csv edge with the Fortran sources\n",
              "`chance` re-pairs each edge with a random Fortran file.\n",
              md(ev[["relationship", "check", "edges", "passed", "rate (%)", "chance (%)"]],
                 {"rate (%)": "{:.1f}", "chance (%)": "{:.1f}"}), ""]

    ep = rd("equation_provenance_summary.csv")
    ep["share (%)"] = ep.share * 100
    ep["chance (%)"] = ep.chance_share * 100
    extra = json.load(open(os.path.join(R, "equation_provenance_extra.json")))
    parts += ["## Table 3. Line-level provenance of every equation\n",
              md(ep[["status", "equations", "share (%)", "chance (%)"]], {"share (%)": "{:.1f}", "chance (%)": "{:.1f}"}),
              "",
              f"Among located equations: {extra['located_with_matching_assignment_target'] * 100:.1f}% have a matching assignment target, "
              f"{extra['located_with_>=1_numeric_literal_match'] * 100:.1f}% share at least one numeric literal, "
              f"median statement window = {extra['median_statement_window']:.0f}.\n"]

    vp = rd("variable_provenance_summary.csv")
    vp["share (%)"] = vp.share * 100
    vin = rd("variable_in_statement.csv")
    vk = vin.groupby("kind").in_statement.agg(["size", "mean"]).reset_index()
    vk["in matched line (%)"] = vk["mean"] * 100
    parts += ["## Table 4. Variable provenance\n", md(vp[["status", "symbols", "share (%)"]], {"share (%)": "{:.1f}"}), "",
              "Variables of located equations that occur inside the matched statement:\n",
              md(vk[["kind", "size", "in matched line (%)"]], {"in matched line (%)": "{:.1f}"}), ""]

    sy = rd("symbolic_check_summary.csv")
    parts += ["## Table 5. Symbolic consistency (random equation sample)\n", md(sy, {"rate": "{:.3f}"}), ""]

    pf = rd("per_file_provenance.csv").head(15)
    parts += ["## Table 6. Fifteen equation-richest files\n", md(pf, {"traced_share": "{:.2f}"}), ""]

    bs = json.load(open(os.path.join(R, "benchmark_summary.json")))
    parts += ["## Table 7. Path mining on random variable pairs (e3sm.csv)\n",
              md(pd.DataFrame({"metric": list(bs), "value": list(bs.values())}), {"value": "{:.3f}"}), ""]

    cs = rd("case_studies.csv")
    keep = [c for c in ["graph", "case", "source", "target", "status", "n_equations", "S_src", "S_var", "S_sym", "V", "score", "status_class"] if c in cs]
    parts += ["## Table 8. Paper case studies\n",
              "`endpoint missing` = the variable has no equation link in that graph; `no path` = both endpoints exist but are not connected.\n",
              md(cs[keep], {"n_equations": "{:.0f}"}), ""]

    nr = rd("numerical_replay.csv")
    parts += ["## Table 9. Numerical replay of the lake chain (Fortran formulas vs E3SM output)\n",
              md(nr[["step", "n", "MAE", "RMSE", "R2", "replayable", "note"]],
                 {"MAE": "{:.3g}", "RMSE": "{:.3g}", "R2": "{:.6f}", "n": "{:.0f}"}), ""]

    sl = rd("surrogates_lake.csv")
    sm = rd("surrogates_methane.csv")
    parts += ["## Table 10. Current compact surrogate relations on the held-out 30 % of the series\n", "Lake (current one-variable equation):\n",
              md(sl[["relation", "model", "n", "MAE", "RMSE", "NRMSE", "R2", "r"]], {"n": "{:.0f}"}), "", "Methane (current one-variable equation):\n",
              md(sm[["relation", "model", "n", "MAE", "RMSE", "NRMSE", "R2", "r"]], {"n": "{:.0f}"}), ""]

    pl = rd("data_plausibility.csv")
    parts += ["## Table 11. Plausibility of the lake-scalar columns in the supplied CSV\n", md(pl, {"min": "{:.4g}", "median": "{:.4g}", "max": "{:.4g}"}), ""]

    path = os.path.join(R, "TABLES.md")
    open(path, "w", encoding="utf-8").write("\n".join(parts))
    print("wrote", path)


if __name__ == "__main__":
    main()
