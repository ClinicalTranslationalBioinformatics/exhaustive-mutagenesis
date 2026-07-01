import os
import pandas as pd

release_dir = "/home/frog/Desktop/human_proteome_2026/human_proteome_2026_release"

rows = []

for protein in sorted(os.listdir(release_dir)):

    pdir = os.path.join(release_dir, protein)

    if not os.path.isdir(pdir):
        continue

    files = os.listdir(pdir)

    if len(files) != 1:
        continue

    fname = files[0]
    fpath = os.path.join(pdir, fname)

    try:
        df = pd.read_csv(fpath, sep="\t")
    except Exception:
        continue

    n_variants = len(df)

    predictors = [
        c for c in df.columns
        if not c.endswith("_sd")
        and c not in [
            "UniProtID","GeneName","GeneID","chr","hg19_pos",
            "PositionAA","RefAA","AltAA","VariantID"
        ]
        and df[c].notna().any()
    ]

    has_sd = any(c.endswith("_sd") for c in df.columns)

    rows.append({
        "protein_id": protein,
        "file_name": fname,
        "n_variants": n_variants,
        "n_predictors": len(predictors),
        "has_sd": has_sd,
        "file_size_mb": round(os.path.getsize(fpath)/(1024**2), 2)
    })

manifest = pd.DataFrame(rows)

manifest.to_csv("manifest.tsv", sep="\t", index=False)

print(manifest.head())
print()
print(f"Proteins: {len(manifest):,}")
print(f"Variants: {manifest['n_variants'].sum():,}")
