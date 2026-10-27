# E3SM validation

`validate_fex_surrogates.py` evaluates the fixed original-FEX-grammar equations
from `function_approximation/fex_original_lake_methane_metadata.json`. It does
not refit the equations. This is a legacy, source-grammar baseline and is
separate from the current compact equations reported in `TAIS.tex`, which are
validated by `function_approximation/fex_savedtke1_source_form.py` and
`function_approximation/methane_simple_fex.py`. The script writes split-wise metrics to
`results/e3sm_validation/fex_validation_metrics.csv` and separate lake and
methane figures to `figures/`: parity (`*_validation.*`), residuals
(`*_residuals.*`), absolute-error distributions (`*_absolute_error.*`), and
chronological replays (`*_chronological_replay.*`).

The chronological holdout is the more conservative diagnostic. Lake site 3 is
reported separately because its replay error shows that the compact pathway is
not an independently transferable site model.

Run from the repository root:

```text
python e3sm_validation/validate_fex_surrogates.py
```
