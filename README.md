# Researcher-ranking locked rerun

This folder contains the inputs, frozen settings, authoritative result files, and final figures used for the updated manuscript.

## Required files

- `revision_20260827/analysis_matrix.csv`: 1,184 records, class label, and 64 scientometric indicators.
- `fixed_folds.csv`: identity groups and the archived repeat-1 outer-fold assignment. The identical copy under `revision_20260827/` is retained for path compatibility.
- `19_frozen_manifest.json`: master seed, repeated-validation design, model grids, package versions, and source-file hashes.
- `final_locked_results/`: authoritative CSV inputs used by the final figure-generation script.
- `make_locked_figures.py`: generates Figures 1–9 from the locked inputs.
- `final_locked_figures/`: the nine final 600-dpi PNG figures used in the manuscript.

The source dataset and derived analytical matrix should be uploaded only if their access conditions permit redistribution. Without `analysis_matrix.csv`, Figure 2 cannot be regenerated.

## Run order

1. Create and activate a Python 3.12 environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the analysis scripts at the repository root and preserve the folder names shown here.
4. Run the analytical pipeline in numerical order: data preparation, individual-indicator analysis, composite analysis, machine-learning models, paired inference, TreeSHAP, prevalence analysis, and reduced-representation analysis.
5. Freeze the resulting files under `final_locked_results/`.
6. Run `python make_locked_figures.py`.

The figure script writes `Figure_1.png` through `Figure_9.png` to `final_locked_figures/`.

## Analytical script outputs

| Script stage | Main output |
|---|---|
| Data preparation | Audited analytical matrix, identity groups, and fixed folds |
| Individual indicators | Complete-sample indicator rankings and effect summaries |
| Composite analysis | Fold selections, sensitivity results, and composite scores |
| Machine-learning models | Held-out scores, tuned configurations, and performance summaries |
| Pairwise inference | Primary and secondary paired comparisons with multiplicity adjustment |
| TreeSHAP | Individual- and correlation-family attribution tables |
| Prevalence analysis | Metrics and composition-resampling intervals at 5%–50% prevalence |
| Reduced representations | Five-indicator panel, HGB5 selections, and HGB64–HGB5 inference |
| Figure generation | Nine publication figures in `final_locked_figures/` |

## Reproducibility checks

Before running the analysis, verify the SHA-256 hashes of `analysis_matrix.csv` and `fixed_folds.csv` against `source_hashes` in `19_frozen_manifest.json`. Do not overwrite the locked results after manuscript values and figures have been generated.
