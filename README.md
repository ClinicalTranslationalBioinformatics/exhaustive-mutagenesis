# Human Proteome Exhaustive Mutagenesis

## Overview

This repository contains the computational pipeline used to generate a proteome-wide exhaustive missense variant effect prediction resource for the human reference proteome.

The dataset covers all proteins in the UniProt human reference proteome (Proteome ID: **UP000005640**) and includes all 19 possible amino acid substitutions at every protein position: **217,408,803 amino-acid substitutions across 20,521 proteins** (one row per substitution, versions ≥ 2.0).

Predictions were generated using the QAFI predictor family together with several widely used external variant effect predictors.

QAFI training datasets and intermediate data files are not included in this repository due to size constraints and are obtained from the original QAFI framework.

The complete prediction resource is freely available on [Zenodo](https://doi.org/10.5281/zenodo.23837889) (this link always resolves to the latest version).

The QAFI methodology is described in the [QAFI publication](https://www.mdpi.com/1422-0067/27/19/8777).

---

## Dataset versions

| Dataset version | Code (GitHub tag) | QAFI ensembles | QAFImt | Notes |
|---|---|---|---|---|
| 1.0 / 1.1 | `v1.0.0` | 60 PSPs | QAFImt (linear regression) | One row per genomic variant (564,287,628 rows). 1.1 adds coverage statistics. Superseded. |
| 2.0 | `v2.0.0` | 30 PSPs | QAFImt 1.0 (linear regression) | One row per amino-acid substitution. QAFImt-based predictors available for a subset of proteins. |
| **3.0 (recommended)** | `v3.0.0` | 30 PSPs | **QAFImt 2.0 (XGBoost)** | QAFImt, QAFIsplit 2.0/3.0, QAFIClass and QAFIMeta extended to nearly the whole proteome. |

DOIs: v1.0 [10.5281/zenodo.21132349](https://doi.org/10.5281/zenodo.21132349) · v1.1 [10.5281/zenodo.23740768](https://doi.org/10.5281/zenodo.23740768) · v2.0 [10.5281/zenodo.23784508](https://doi.org/10.5281/zenodo.23784508) · v3.0 [10.5281/zenodo.23837889](https://doi.org/10.5281/zenodo.23837889)

See [CHANGELOG.md](CHANGELOG.md) for a detailed description of the changes between versions.

### Coverage (version 3.0)

| Predictor | % proteins | % substitutions |
|---|---|---|
| AlphaMissense | 99.2 | 97.8 |
| CPT-1 | 89.5 | 90.5 |
| popEVE / EVE / REVEL / BayesDel | 84–88 | 23–24 (single-nucleotide-accessible substitutions only) |
| qafi2, QAFIClass1/2, QAFIMeta v1–v4 | 91.5 | 92.3 |
| QAFImt 2.0, QAFIsplit 2.0, QAFIsplit 3.0 | 91.5 | 88.0 |
| qafi1, QAFIsplit 1.0 | 74.5 | 58.4 |

Coverage depends on the availability of AlphaFold structural models, multiple sequence alignments and other required input features. For long proteins, structure-based features (and therefore QAFI scores) may be available only for the modelled region. Per-protein coverage is distributed with each Zenodo version.

---

## Dataset

The released dataset is organized as one directory per protein:

```text
exhaustive_mutagenesis_data/
└── <UniProtID>/
    └── <UniProtID>_variants.tsv
```

Each file contains one row per possible amino-acid substitution of the protein. Main columns (version 3.0):

| Column(s) | Description |
|---|---|
| `uniprot`, `GeneName`, `GeneID` | Protein and gene identifiers |
| `chr`, `hg19_pos` | Genomic coordinates (hg19) of the first mapped genomic variant producing the substitution, when one exists |
| `PositionAA`, `RefAA`, `AltAA`, `VariantID` | Protein position, reference and alternative amino acid |
| `AM_pathogenicity`, `AM_class`, `popEVE`, `EVE`, `CPT1`, `REVEL`, `BayesDel` | External predictors |
| `qafi1`, `qafi2` | QAFI 1.0 and 2.0 |
| `QAFImt2.0` | QAFImt 2.0: position-level mutation tolerance (`QAFImt1.0` in version 2.0) |
| `qafisplit1`, `qafisplit2`, `qafisplit3` | QAFIsplit 1.0, 2.0 and 3.0 (2.0 and 3.0 computed with QAFImt 2.0 in version 3.0) |
| `QAFIClass1`, `QAFIClass2` | QAFIClass 1.0 and 2.0 (pathogenicity classifiers; 0 = pathogenic, 1 = benign) |
| `QAFIMeta_v1` … `QAFIMeta_v4` | QAFIMeta 2.0 metapredictors |

All QAFI scores follow the convention *higher = more benign / closer to wild type*. Scores are rounded to three decimals. Where several genomic variants produce the same amino-acid substitution, genomic coordinates and genomic-level scores (REVEL, BayesDel) correspond to the first mapped genomic variant.

---

## Repository structure

```text
exhaustive-mutagenesis/
├── pipeline/                         # build exhaustive variant tables and merge external predictors
│   ├── build_protein_variants.py
│   └── merge_external_predictors.py
│
├── qafi_pipeline/                    # QAFI family
│   ├── run_qafi_family.py
│   ├── merge_qafi_family.py
│   ├── QAFI_family.py
│   ├── run_qafi_class.py             # QAFIClass (dataset v1)
│   ├── run_qafi_meta.py              # QAFIMeta (dataset v1)
│   ├── recalc_qafi_meta.py           # QAFIMeta v1–v4 with rescaling to QAFIsplit 3.0 (v2, v3)
│   ├── recalc_qafi_class.py          # QAFIClass1/2 from a given QAFImt column (v3)
│   ├── add_qafimt2.py                # add QAFImt 2.0 and QAFIsplit 2.0/3.0 to per-protein tables (v3)
│   └── qafimt2/                      # QAFImt 2.0 model (v3)
│       ├── qafimt2.py
│       ├── run_qafimt2.py
│       └── validate.py
│
├── analysis/                         # release building and quality control
│   ├── make_missing_updated.py       # one table per protein, one row per substitution (v3)
│   ├── build_zenodo_release.py       # build a Zenodo release (--version v2 | v3)
│   ├── compute_proteome_coverage.py
│   ├── add_release_scores.py         # add release scores to an external list of variants
│   ├── build_manifest.py
│   ├── audit_release.py
│   ├── format_and_round_columns.py
│   └── reorder_columns.py
│
├── validation/
│   ├── validate_clinvar.py           # ClinVar/Humsavar AUC-ROC
│   ├── validate_beltran.py           # Beltran et al. DMS benchmark (observed vs predicted)
│   └── cagi_scatter.py               # CAGI7 (ARSA, ATP7B, LPL, TSC2)
│
└── scripts_deprecated/
```

---

## Computational pipeline

1. Generate all possible missense variants for every protein in the human reference proteome (`pipeline/`).
2. Integrate predictions from external variant effect predictors (`pipeline/`).
3. Compute the QAFI family predictions (`qafi_pipeline/`).
4. Compute QAFImt 2.0 and QAFIsplit 2.0/3.0 on it (`qafi_pipeline/qafimt2/`, `add_qafimt2.py`), recompute QAFIClass (`recalc_qafi_class.py`) and QAFIMeta (`recalc_qafi_meta.py`).
5. Build the release: one row per substitution, harmonised columns, rounding (`analysis/build_zenodo_release.py`).
6. Compute coverage statistics and run the validations (`analysis/`, `validation/`).

### Validation

Each Zenodo version (≥ 2.0) includes:

- **ClinVar/Humsavar**: AUC-ROC per predictor. These labels were used to train QAFIClass; QAFIClass scores are leave-one-protein-out predictions.
- **Beltran et al. DMS benchmark**: per-protein Pearson correlation between observed fitness and predicted QAFIsplit 3.0.
- **CAGI7** (ARSA, ATP7B, LPL, TSC2): comparison of QAFIMeta with the predictions submitted to CAGI7.

---

## Requirements

Python ≥ 3.9 with:

- pandas (< 3.0)
- numpy
- scikit-learn
- scipy
- matplotlib
- xgboost (QAFImt 2.0 was developed with xgboost 2.1.4)
- tqdm

Additional dependencies may be required for specific components of the QAFI pipeline.

---

## External prediction methods

This resource incorporates predictions generated by the following external methods:

- **AlphaMissense**: Cheng et al., *Science* (2023).
- **EVE**: Frazer et al., *Nature* (2021).
- **popEVE**: Orenbuch et al., *Nature* (2025).
- **CPT-1**: Jagota et al., *Genome Biology* (2021).
- **REVEL**: Ioannidis et al., *American Journal of Human Genetics* (2016).
- **BayesDel**: Feng et al., *Human Mutation* (2016).

Users of this dataset should cite the original publications of these methods in addition to this dataset.

---

## Citation

If you use this resource, please cite:

**Dataset**

Human Proteome Exhaustive Mutagenesis. Zenodo. https://doi.org/10.5281/zenodo.23837889 (please cite the specific version you used).

**Method**

Ye, S.; Özkan, S.; Díaz-Vásquez, A.; Padilla, N.; de la Cruz, X. Toward Advanced Quantitative Prediction of Missense Variant Effects. *Int. J. Mol. Sci.* **2026**, *27*, 8777. https://doi.org/10.3390/ijms27198777

---

## Funding

The authors acknowledge support from the Ministerio de Ciencia, Innovación y Universidades de España through grant PID2022-142753OB-I00, co-funded by the European Regional Development Fund (FEDER).

---

## License

The source code is distributed under the MIT License.

The released prediction dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.