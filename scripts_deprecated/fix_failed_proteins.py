import pandas as pd, shutil
from pathlib import Path

proteins_dir = Path("~/Desktop/human_proteome_2026/proteins").expanduser()
zero = pd.read_csv("proteome_stats.csv")
zero = zero[zero["n_predictors_present"] == 0]["uniprot"].tolist()

for uniprot in zero:
    folder = proteins_dir / uniprot
    # Delete empty 8preds
    for f in folder.glob("*_8preds.tsv"): f.unlink()
    # Copy best available
    src = next((folder / f for f in [f"{uniprot}_7preds.tsv", f"{uniprot}_variants_6preds.tsv"] 
                if (folder / f).exists()), None)
    if src:
        shutil.copy(src, folder / f"{uniprot}_8preds.tsv")
        print(f"[done] {uniprot} ← {src.name}")
    else:
        print(f"[MISSING] {uniprot} — no 7 or 6preds found")
