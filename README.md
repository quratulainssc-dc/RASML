# Researcher-ranking reproducibility files

This package contains the locked analysis files used for the manuscript and the later five-indicator logistic-regression (LR5) extension. It includes the analytical matrix, grouped fold assignments, frozen settings, analysis scripts, archived results, added LR5 results, and the final nine manuscript figures.

## Contents

- `revision_20260827/analysis_matrix.csv`: 1,184 records, 64 indicator columns, author identifier, and award-status label.
- `fixed_folds.csv`: identity-group identifiers and the archived repeat-1 outer-fold assignment. The duplicate copy under `revision_20260827/` is retained because the locked pipeline expects that path.
- `19_frozen_manifest.json`: master seed, validation design, model grids, software versions, and source hashes.
- `final_locked_analysis_pipeline.py`: complete locked analysis, including individual indicators, transparent composites, the arithmetic five-indicator panel, HGB5, four full 64-indicator models, inference, correlation families, TreeSHAP, prevalence analysis, and consensus rankings.
- `postprocess_locked_results.py`: the 36-setting composite sensitivity analysis, HGB64-HGB5 inference, and selection-frequency summaries.
- `run_lr5_and_added_analyses.py`: LR5 fitting plus the three added paired contrasts and ordering/top-119 comparisons.
- `make_locked_figures.py`: figure-generation code from the locked analysis.
- `make_locked_figures_no_title.py`: alternate Figure 7 generation without the internal title.
- `Scientometrics_Complete_Rerun_Results.xlsx`: complete original rerun workbook.
- `final_locked_results/`: archived CSV outputs required by the figure code and main analyses.
- `added_results/`: LR5 outputs, added paired contrasts, rank correlations, and shortlist-overlap results.
- `final_locked_figures/`: the final Figure 1 through Figure 9 used in the current manuscript.
- `lr5_input/`: self-contained inputs required by the LR5 extension.
- `manifest/computational_manifest.txt`: SHA-256 checksum for every distributed file.

## Environment

Python 3.12 was used for the locked run. Create a clean environment and install the pinned packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run order

### Complete locked analysis

From the repository root:

```bash
python final_locked_analysis_pipeline.py
python postprocess_locked_results.py
python make_locked_figures_no_title.py
```

The first command writes the main outputs to `final_locked_results/`. The second adds the composite-sensitivity and reduced-representation outputs. The figure script reads those outputs and writes the figures defined by the locked figure code.

### LR5 and added comparisons

```bash
python run_lr5_and_added_analyses.py \
  --input-dir lr5_input \
  --output-dir added_results \
  --jobs 6
```

This command repeats five-indicator selection inside inner validation, tunes LR5 over the frozen 12-configuration logistic-regression grid, computes held-out and consensus scores, and produces:

- LR5 fold-, repeat-, and consensus-level performance;
- LR5 selected configurations and inner-search results;
- HGB5 versus panel, LR5 versus panel, and LR5 versus HGB5 paired inference;
- Spearman, Kendall, top-119 overlap, Jaccard, and unique-awardee results for HGB5 versus panel, LR64 versus HGB64, and composite versus HGB64.

The added contrasts form a separate three-comparison family and do not replace the original ten primary pairwise comparisons.

## Fold files

The repository includes the archived repeat-1 outer assignment in `fixed_folds.csv`. Repeats 2-10 and all inner folds are generated deterministically by the scripts from master seed `20260819`; they were not stored as a separate `inner_fold_assignments.csv` file. This package therefore documents the actual reproducibility path rather than supplying a reconstructed fold file.

## Indicator definitions

The analytical matrix preserves the indicator column names used by the analysis. A separate authoritative indicator-definition file was not available in the locked materials, so none has been reconstructed or inferred here.

## Verification

`added_results/validation_report.json` records the validation checks for the LR5 extension. File integrity can be checked against `manifest/computational_manifest.txt`:

```bash
sha256sum -c manifest/computational_manifest.txt
```

## License

No software or data license has been assigned in this package. Add the license permitted by the data source and agreed by the authors before making the repository public.
