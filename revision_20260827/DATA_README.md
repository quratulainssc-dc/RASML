# Analytical data

`analysis_matrix.csv` contains 1,184 analytical records. It has one author-name field, 64 numerical scientometric indicators, and the binary `Class` field used as the historical-recognition criterion.

`fixed_folds.csv` contains the record identifier, class label, identity-group identifier, and archived repeat-1 outer-fold assignment. Grouped validation keeps records belonging to the same identity in one fold.

The analysis code generates later outer assignments and all inner assignments deterministically from the frozen master seed. No standalone inner-fold file was archived.
