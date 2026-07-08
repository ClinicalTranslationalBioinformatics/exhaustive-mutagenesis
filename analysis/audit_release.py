import pandas as pd
import os

release = "/home/frog/Desktop/human_proteome_2026/human_proteome_2026_release"

predictors = [
    "AM_pathogenicity",
    "popEVE",
    "EVE",
    "CPT1",
    "REVEL",
    "BayesDel",
    "qafi1",
    "qafi2",
    "qafisplit1",
    "qafisplit2",
    "qafisplit3",
    "QAFImt",
    "QAFIClass1",
    "QAFIClass2",
    "QAFIMeta_v1",
    "QAFIMeta_v2",
    "QAFIMeta_v3",
    "QAFIMeta_v4",
]

empty = []

for protein in os.listdir(release):
    pdir = os.path.join(release, protein)

    if not os.path.isdir(pdir):
        continue

    files = os.listdir(pdir)

    if len(files) != 1:
        continue

    f = os.path.join(pdir, files[0])

    try:
        df = pd.read_csv(f, sep="\t")
    except:
        continue

    valid = 0

    for col in predictors:
        if col in df.columns and df[col].notna().any():
            valid += 1

    if valid == 0:
        empty.append(protein)

print("Proteins with 0 predictors:", len(empty))

with open("proteins_zero_predictors.txt", "w") as out:
    for p in empty:
        out.write(p + "\n")
