# Baseline comparison (answers reviewer point 6)

| Reviewer concern | File | What it does |
|---|---|---|
| Only two pathways, in one component | `common.py` (`CASES`) | Lake pathway pooled over 3 lake sites, methane pathway; each with a univariate and a multivariate feature set. Add a case by adding an entry. |
| No comparison with def-use chains / slicing | `path_recovery_comparison.py` | Parses all Fortran sources into an assignment-level def-use graph; computes shortest def-use chains and backward slices (global and file-scoped) for the case studies and the 138 benchmark queries; compares with Sci-Miner's chains (length, hub exposure, recall/precision against mined statements). |
| R² of 0.22 / 0.31, surrogates explain little | `regression_comparison.py` | Held-out scores under four splits: random 70/30 (the paper's), chronological 70/30, 5 contiguous time blocks, leave-one-site-out. The random split is optimistic because hourly data are autocorrelated. |
| NN baseline has no stated architecture, inputs, split or tuning | `models.py` (`MLP`, `MLP_GRID`) | tanh MLP, Adam, MSE, standardised inputs, early stopping; hidden sizes / lr / weight decay chosen on the last 20 % of the *training* data from an 8-point grid; same inputs and split as every other model. |
| Need multivariate FEX plus SINDy / PySR | `models.py` | `FEX` (multivariate, same operator budget as `FEX (1-var)`), `SINDy` (STLSQ over a candidate library), `PySR` (runs only if `pysr` and Julia are installed; otherwise skipped and reported as absent, never faked). |
| No domain-expert judgement | `expert_review.py` | Generates a blinded rating sheet (Sci-Miner chains and def-use chains mixed, method hidden) and scores returned sheets (share correct / novel, Cohen's kappa). **No ratings are included; they have to come from real experts.** |

## Run

```
python run_all.py            # full run (about an hour on CPU)
python run_all.py --quick    # smoke test
python make_report.py        # rebuild REPORT.md from results/*.csv
python plot_comparison_figures.py  # write separate publication figures to ../figures/
python expert_review.py make  # optional: create a blank sheet for real expert raters
```

Outputs are in `results/`: `regression_summary_all.csv`, `regression_raw_*.csv`,
`regression_expressions_*.json`, `path_recovery_*.csv`, and `REPORT.md`.

`plot_comparison_figures.py` reads the completed CSV summaries without refitting any model and writes
single-panel PNG/PDF figures to the repository-level `figures/` directory. The model-comparison figures
include both held-out R² (clipped at −1 for readability) and normalized RMSE on a log scale, so extreme
stress-test failures remain visible rather than being silently hidden by the R² clipping.

`run_all.py` rebuilds the Fortran definition-use index on every full run so cached
parser state cannot silently outlive a source update. The path-recovery step also
depends on the saved Sci-Miner tables under `results/path_mining_validation/`.

The release pipeline does not create a multipanel comparison figure. Use the single-panel files in
`figures/` for the manuscript. Expert-review sheets are templates only and are not evidence until
completed by independent domain experts; they are therefore not produced by `run_all.py`.

## Feature sets and leakage

* Variables that are algebraic functions of the target are excluded from the multivariate sets (`CH4_DIAG_TFACT`, `CH4_DIAG_SOURCE` contain
  `Q10^((T-Tb)/10)`; `LAKE_N2 = g/rho * drhodz` contains the target). They would make every model look good for the wrong reason.
* `CH4_DIAG_DIFFUS` and `CH4_DIAG_BASE_DECOMP` are in the pathway equation but are themselves temperature dependent in E3SM
  (correlation of D with T is 0.998). The set `pathway_noT` removes them to show how much of a multivariate fit is this implicit temperature information.
* The exported `TKE1` field has mean 284 K and r = 0.99 with `TLAKE` level 1: it behaves like a temperature, not like `savedtke1`.
  The set `univariate_kme1` uses `LAKE_KME` (level 1; `savedtke1 = kme*cwat`), the actual variable at the end of the mined chain.
  Check what the `TKE1` history field is before reporting the paper's lake surrogate.

## Method notes

* Def-use chains and slices are name-based over the whole model (no scoping or alias analysis), with the same generic index symbols
  removed as in Sci-Miner. This is the standard cheap baseline; a compiler-grade analysis would be more precise but is not available for the full E3SM tree here.
* SINDy is implemented here (STLSQ) because the `pysindy` release available needs numpy >= 2, which conflicts with the installed pandas/torch.
* The comparison-folder `models.FEX` is an independently trained baseline for the reviewer comparison. It is not the implementation that produced the fixed compact manuscript equations; the current compact equations come from `function_approximation/fex_savedtke1_source_form.py` and `function_approximation/methane_simple_fex.py`. The original-FEX archive-grammar outputs remain available as a separate comparison baseline.
