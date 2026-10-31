# Changelog

Versions refer to the Zenodo dataset; the matching code is tagged in this repository.

## Dataset 3.0 — code `v3.0.0`

Dataset: https://doi.org/10.5281/zenodo.23837889

- **QAFImt 2.0** replaces QAFImt 1.0: the position-level mutation tolerance is estimated with XGBoost (30 training proteins, trained once and applied to every protein) instead of linear regression. Column `QAFImt2.0`.
- **QAFIsplit 2.0 and 3.0** recomputed with QAFImt 2.0 (QAFIsplit 3.0: fixed top-30 correlation-weighted fusion).
- **QAFIClass1/2** recomputed with QAFImt 2.0 as input feature (same models and training set; leave-one-protein-out for proteins in the training set).
- **QAFIMeta v1–v4** recomputed; v1–v3 rescaled (z-score) to the new QAFIsplit 3.0 distribution.
- Every protein with input features now has QAFImt 2.0, QAFIsplit 2.0/3.0 and QAFIMeta v4: coverage 91.5% of proteins / 88.0% of substitutions for QAFImt 2.0 and QAFIsplit 2.0/3.0 (version 2.0: 58.4–80.6% of substitutions); 92.3% of substitutions for QAFIMeta v1–v4.
- `qafi1`, `qafi2` and QAFIsplit 1.0 unchanged from 2.0 (QAFIsplit 1.0 is still based on QAFImt 1.0).
- Validation: ClinVar/Humsavar (QAFIMeta AUC 0.949, QAFIClass AUC 0.927), Beltran et al. DMS benchmark (387 proteins, median r = 0.470), CAGI7.

## Dataset 2.0 — code `v2.0.0`

Dataset: https://doi.org/10.5281/zenodo.23784508

- QAFI predictors based on ensembles of **30 PSPs** (60 in 1.x).
- QAFIsplit 3.0 updated and extended to 2,558 additional proteins.
- QAFIMeta v1–v3 rescaled (z-score) to the QAFIsplit 3.0 distribution; QAFIMeta v4 recomputed.
- **One row per amino-acid substitution** (217,408,803 rows). Version 1.x listed one row per underlying genomic variant (564,287,628 rows), so several rows could correspond to the same substitution.
- Column `QAFImt` renamed `QAFImt1.0`.
- Coverage statistics and validation files (ClinVar/Humsavar, Beltran et al. DMS benchmark, CAGI7) added to the release.

## Dataset 1.1 — code `v1.0.0`

Dataset: https://doi.org/10.5281/zenodo.23740768

- Adds proteome-wide coverage statistics. Prediction data unchanged from 1.0.

## Dataset 1.0 — code `v1.0.0`

Dataset: https://doi.org/10.5281/zenodo.21132349

- First release: 20,521 proteins, QAFI family based on 60 PSPs, external predictors (AlphaMissense, popEVE, EVE, CPT-1, REVEL, BayesDel).