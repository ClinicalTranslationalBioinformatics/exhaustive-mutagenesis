import os
from pathlib import Path

import numpy as np
import pandas as pd

# === Config ===
proteins_dir = "/home/frog/Desktop/human_proteome_2026/proteins"

log_path = "/home/frog/Desktop/human_proteome_2026/qafi_meta/skipped_proteins.txt"


# Core functions
def z_to_target(values, target_mean, target_std):
    p_mean, p_std = values.mean(), values.std()
    if p_std == 0 or np.isnan(p_std):
        return values
    return ((values - p_mean) / p_std) * target_std + target_mean


def compute_meta(df, anchor_col, skip_cpt):
    target_mean = df[anchor_col].mean()
    target_std = df[anchor_col].std()

    scaled_cols = []

    if "am_pathogenicity" in df.columns:
        am = z_to_target(-df["am_pathogenicity"], target_mean, target_std)
        if "am_class" in df.columns:
            am[df["am_class"] == "ambiguous"] = np.nan
        df["_am"] = am
        scaled_cols.append("_am")

    if "popEVE" in df.columns:
        df["_popEVE"] = z_to_target(df["popEVE"], target_mean, target_std)
        scaled_cols.append("_popEVE")

    if not skip_cpt and "CPT1" in df.columns:
        df["_CPT1"] = z_to_target(-df["CPT1"], target_mean, target_std)
        scaled_cols.append("_CPT1")

    if "REVEL" in df.columns:
        df["_REVEL"] = z_to_target(-df["REVEL"], target_mean, target_std)
        scaled_cols.append("_REVEL")

    if "BayesDel" in df.columns:
        df["_BayesDel"] = z_to_target(-df["BayesDel"], target_mean, target_std)
        scaled_cols.append("_BayesDel")

    return df[[anchor_col] + scaled_cols].median(axis=1, skipna=True).round(3)


def run_protein(uniprot, protein_dir):
    files7 = sorted(protein_dir.glob("*_7preds.tsv"))
    files6 = sorted(protein_dir.glob("*_6preds.tsv")) + sorted(
        protein_dir.glob("*_5preds.tsv")
    )

    if not files6:
        raise FileNotFoundError("No 6preds found")

    df6 = pd.read_csv(files6[0], sep="\t")

    if files7:
        df7 = pd.read_csv(files7[0], sep="\t")
        extra = [
            c
            for c in [
                "am_pathogenicity",
                "am_class",
                "popEVE",
                "CPT1",
                "REVEL",
                "BayesDel",
                "qafi2",
                "qafisplit3",
            ]
            if c in df6.columns and c not in df7.columns
        ]

        df = df7.merge(df6[["VariantID"] + extra], on="VariantID", how="left")
    else:
        df = df6.copy()
        df["QAFIClass1"] = np.nan
        df["QAFIClass2"] = np.nan

    skip_cpt = "CPT1" not in df.columns or df["CPT1"].isna().all()

    df["QAFIMeta_v1"] = compute_meta(df.copy(), "QAFIClass1", skip_cpt)
    df["QAFIMeta_v2"] = compute_meta(df.copy(), "QAFIClass2", skip_cpt)

    df["QAFIMeta_v3"] = (
        compute_meta(df.copy(), "qafi2", skip_cpt)
        if "qafi2" in df.columns and not df["qafi2"].isna().all()
        else np.nan
    )

    df["QAFIMeta_v4"] = (
        compute_meta(df.copy(), "qafisplit3", skip_cpt)
        if "qafisplit3" in df.columns and not df["qafisplit3"].isna().all()
        else np.nan
    )

    df = df[[c for c in df.columns if not c.startswith("_")]]

    out = protein_dir / f"{uniprot}_8preds.tsv"
    df.to_csv(out, sep="\t", index=False)

    return out


# Run all proteins
for uniprot in os.listdir(proteins_dir):
    protein_dir = Path(proteins_dir) / uniprot

    if not protein_dir.is_dir():
        continue

    if not list(protein_dir.glob("*_6preds.tsv")) and not list(
        protein_dir.glob("*_5preds.tsv")
    ):
        print(f"Missing prediction files for {uniprot}, skipping.")
        with open(log_path, "a") as f:
            f.write(f"{uniprot}: missing 6preds/5preds\n")
        continue

    try:
        print(f"Running QAFIMeta for {uniprot}...")
        out = run_protein(uniprot, protein_dir)
        print(f"Finished {uniprot}. Saved: {out.name}")
    except Exception as e:
        print(f"Failed {uniprot}: {e}")
        with open(log_path, "a") as f:
            f.write(f"{uniprot}: {e}\n")

print("Done.")
