import os
from pathlib import Path

import pandas as pd

from pipeline import load_training_data, predict_for_protein

# === Config ===
proteins_dir = "/home/frog/Desktop/human_proteome_2026/proteins"
train_csv = "data/predictions_clinhum_qafi_insilico.csv"

log_path = "/home/frog/Desktop/human_proteome_2026/qafi_class/skipped_proteins.txt"


def load_protein_input(protein_dir: Path) -> pd.DataFrame:
    """Merge *_final2.csv with qafisplit1_median from *_6preds.tsv."""
    final = pd.read_csv(next(protein_dir.glob("*_final2.csv")))

    if "qafisplit1_median" not in final.columns:
        preds_files = sorted(protein_dir.glob("*_6preds.tsv"))
        if preds_files:
            preds = pd.read_csv(preds_files[0], sep="\t")
            if "qafisplit1_median" in preds.columns:
                preds = preds.rename(columns={"VariantID": "variant"})
                final = final.merge(
                    preds[["variant", "qafisplit1_median"]],
                    on="variant",
                    how="left",
                )

    if "qafisplit1_median" not in final.columns:
        final["qafisplit1_median"] = float("nan")

    return final


def run_protein(uniprot: str, protein_dir: Path, train_df: pd.DataFrame):
    test_df = load_protein_input(protein_dir)

    preds = predict_for_protein(
        train_df=train_df,
        test_df=test_df,
        uniprot=uniprot,
        verbose=False,
    )

    preds_files = sorted(protein_dir.glob("*_6preds.tsv"))
    if preds_files:
        base = pd.read_csv(preds_files[0], sep="\t")
        base = base.merge(
            preds[["variant", "QAFIClass1", "QAFIClass2"]].rename(
                columns={"variant": "VariantID"}
            ),
            on="VariantID",
            how="left",
        )
    else:
        base = preds[["uniprot", "variant", "QAFIClass1", "QAFIClass2"]]

    out_path = protein_dir / f"{uniprot}_7preds.tsv"
    base.to_csv(out_path, sep="\t", index=False)

    return out_path


# === Load training data once ===
train_df = load_training_data(train_csv)

# === Run all proteins ===
for uniprot in os.listdir(proteins_dir):
    protein_dir = Path(proteins_dir) / uniprot

    if not protein_dir.is_dir():
        continue

    if not list(protein_dir.glob("*_6preds.tsv")):
        print(f"⚠️ Missing 6preds for {uniprot}, skipping.")
        with open(log_path, "a") as f:
            f.write(f"{uniprot}: missing 6preds\n")
        continue

    try:
        print(f"🔁 Running QAFIClass for {uniprot}...")
        out = run_protein(uniprot, protein_dir, train_df)
        print(f"✅ Finished {uniprot}. Saved: {out.name}")
    except Exception as e:
        print(f"⚠️ Failed {uniprot}: {e}")
        with open(log_path, "a") as f:
            f.write(f"{uniprot}: {e}\n")

print("Done.")
