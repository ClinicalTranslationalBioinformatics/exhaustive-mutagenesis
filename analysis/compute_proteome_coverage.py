"""
Compute coverage statistics across the whole proteome.
Reads the best available file (8preds > 7preds > 6preds) per protein.

Usage:
    python proteome_stats.py --proteins-dir ~/Desktop/human_proteome_2026/proteins
    python proteome_stats.py --proteins-dir ~/Desktop/human_proteome_2026/proteins --out stats.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

PREDICTOR_COLS = [
    "am_pathogenicity", "popEVE", "EVE", "CPT1", "REVEL", "BayesDel",
    "qafi1", "qafi2", "qafisplit1", "qafisplit1_median",
    "qafisplit2", "QAFImt", "qafisplit2_residual", "qafisplit3",
    "QAFIClass1", "QAFIClass2",
    "QAFIMeta_v1", "QAFIMeta_v2", "QAFIMeta_v3", "QAFIMeta_v4",
]


def best_file(protein_dir: Path) -> Path | None:
    for suffix in ("*_8preds.tsv", "*_7preds.tsv", "*_6preds.tsv"):
        found = sorted(protein_dir.glob(suffix))
        if found:
            return found[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins-dir", required=True)
    parser.add_argument("--out", default="proteome_stats.csv", help="Output CSV for per-protein summary.")
    args = parser.parse_args()

    proteins_dir = Path(args.proteins_dir)
    folders = sorted(p for p in proteins_dir.iterdir() if p.is_dir())
    print(f"Found {len(folders)} protein folders\n")

    # Per-protein stats
    rows = []
    col_variant_counts = defaultdict(int)   # total non-NaN variants per column
    col_protein_counts = defaultdict(int)   # proteins with ≥1 non-NaN value per column
    total_variants = 0
    no_file = []

    for folder in folders:
        uniprot = folder.name
        path = best_file(folder)
        if path is None:
            no_file.append(uniprot)
            continue

        df = pd.read_csv(path, sep="\t", low_memory=False)
        n_variants = len(df)
        total_variants += n_variants

        row = {"uniprot": uniprot, "source_file": path.name, "n_variants": n_variants}
        n_cols_present = 0
        for col in PREDICTOR_COLS:
            if col in df.columns:
                n_nonnull = df[col].notna().sum()
                frac = round(n_nonnull / n_variants, 3) if n_variants > 0 else 0
                row[f"{col}_coverage"] = frac
                col_variant_counts[col] += n_nonnull
                if n_nonnull > 0:
                    col_protein_counts[col] += 1
                    n_cols_present += 1
            else:
                row[f"{col}_coverage"] = None

        row["n_predictors_present"] = n_cols_present
        rows.append(row)

    per_protein_df = pd.DataFrame(rows)
    per_protein_df.to_csv(args.out, index=False)

    n_proteins = len(rows)

    # ------------------------------------------------------------------ #
    #  SUMMARY REPORT
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print(f"PROTEOME COVERAGE SUMMARY")
    print(f"  Proteins with data : {n_proteins:,}")
    print(f"  Proteins no file   : {len(no_file):,}")
    print(f"  Total variants     : {total_variants:,}")
    print("=" * 60)

    print("\n── Per-predictor coverage ──────────────────────────────────")
    print(f"{'Predictor':<30} {'Proteins w/ data':>17} {'% proteins':>11} {'% variants':>11}")
    print("-" * 62)
    for col in PREDICTOR_COLS:
        p_count = col_protein_counts[col]
        v_count = col_variant_counts[col]
        pct_p = 100 * p_count / n_proteins if n_proteins else 0
        pct_v = 100 * v_count / total_variants if total_variants else 0
        print(f"{col:<30} {p_count:>17,} {pct_p:>10.1f}% {pct_v:>10.1f}%")

    print("\n── Proteins by number of predictors available ──────────────")
    counts = per_protein_df["n_predictors_present"].value_counts().sort_index(ascending=False)
    for n_preds, count in counts.items():
        bar = "█" * int(40 * count / n_proteins)
        print(f"  {n_preds:>2} predictors: {count:>6,} proteins  {bar}")

    print("\n── Proteins with ALL predictors ────────────────────────────")
    all_present = (per_protein_df["n_predictors_present"] == len(PREDICTOR_COLS)).sum()
    print(f"  {all_present:,} / {n_proteins:,} ({100*all_present/n_proteins:.1f}%)")

    print("\n── Proteins with ZERO predictors ───────────────────────────")
    none_present = (per_protein_df["n_predictors_present"] == 0).sum()
    print(f"  {none_present:,} / {n_proteins:,} ({100*none_present/n_proteins:.1f}%)")

    print(f"\nPer-protein detail saved to: {args.out}")


if __name__ == "__main__":
    main()
