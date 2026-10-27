# Path mining and source validation

This folder audits the science graph against the supplied Fortran files and
ranks candidate equation pathways. It supports the paper's provenance and
path-scoring discussion.

## Inputs and outputs

- Graph: `data/output/e3sm.csv` (`EDGES_CSV` override).
- Source files: `data/input/fortran/` (`FORTRAN_DIR` override).
- Numerical inputs: `data/validation/` (`VALIDATION_DIR` override).
- Tables and raw records: `results/path_mining_validation/` (`RESULT_DIR` override).
- All figures: `figures/` (`FIG_DIR` override), as PDF and PNG.

Settings are centralized in `config.py`. The existing full runner also loads
the Neo4j export and numerical inputs; it is not a source-only audit command.
From this folder, `python run_all.py` regenerates experiment outputs,
`python make_tables.py` builds Markdown tables, and `python make_figures.py`
builds figures from saved outputs. These commands overwrite their outputs.

## What the checks establish

| Check | Evidence | Limitation |
| --- | --- | --- |
| File-level source support | Equation identifiers occur in a linked Fortran file | Does not identify an equivalent assignment |
| Statement localization | Identifier overlap within up to three consecutive statements, with source line ranges | Coverage alone does not prove algebraic equivalence |
| Assignment target and numeric matches | Additional evidence recorded for matched statements | These are diagnostics, not acceptance requirements |
| Variable support | Path symbols occur in adjacent equation source files | Does not resolve scope, imports, or aliases |
| Symbolic check | Both equation sides parse with SymPy | Does not verify dimensions or substitutions |
| Numerical replay | Selected formulas agree with available simulation outputs | Not implemented independently for every candidate path |

`equation_provenance.csv` contains individual equation/file records, source line
ranges, snippets, coverage, and target/numeric matches. Summary counts refer to
equation/file records, not necessarily distinct equations. Outputs were regenerated on 2026-10-27. The existing runner does not record
input fingerprints; preserve the input data alongside the outputs for auditing.

## Ranking and classification

`path_mining.py` inspects the first 10 weighted loop-free candidates and discards
those with more than 8 equations. Specificity is a mean over nodes. Continuity
checks graph `contains` links at internal equation hand-offs; it is a proxy for
source continuity, not a physical-validity test.

The score is `(1*V + 0.5*Spec + 0.5*Phys)/(1 + 0.15*L)`, where `L` counts
equations. Component scores are bounded by one; the combined score need not be.
All retained candidates are ranked, including rejected ones. A caller requiring
only verified results must explicitly filter the returned status.

Verification renormalizes weights over available checks. Verified status needs
an aggregate of at least 0.80 and every available component at least 0.50.
An aggregate of at least 0.50 otherwise means partially verified. Missing
numerical evidence is omitted, so verified status alone does not mean replay
passed. The runner supplies a shared case-level lake replay score to cases
labeled `Lake chain`; that score is not a replay of each candidate's equations.

The paper's FEX operator set, approximation penalty, and tree bound must be
confirmed against the implementation that produced its results. They are
separate from this folder's path-ranking parameters.

The figure script produces 14 separate single-panel figures, each in PDF and PNG
format, with no inset axes. Its colors use an Okabe-Ito color-blind-safe palette
with restrained journal-style typography and neutral grid structure. Older
combined figures are not regenerated.
