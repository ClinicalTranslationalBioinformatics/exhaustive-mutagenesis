import pandas as pd
import os

revel_file = "/home/frog/Desktop/human_proteome_2026/revel_bayesdel_readytomerge.csv"
protein_dir = "/home/frog/Desktop/human_proteome_2026/proteins"

revel = pd.read_csv(revel_file, dtype=str)
revel["Protein_position"] = revel["Protein_position"].astype(int)

revel_symbol_set = set(revel["SYMBOL"])

for subfolder in os.listdir(protein_dir):

    folder_path = os.path.join(protein_dir, subfolder)
    if not os.path.isdir(folder_path):
        continue

    protein_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    print(f"Processing {protein_path}")

    protein_df = pd.read_csv(protein_path, sep="\t", dtype=str)
    protein_df["Position"] = protein_df["Position"].astype(int)

    merged = protein_df.merge(
        revel,
        left_on=["RefAA", "AltAA", "GeneName", "Position"],
        right_on=["aaref", "aaalt", "SYMBOL", "Protein_position"],
        how="left"
    )

    if "BayesDel_nsfp33a_noAF" in merged.columns:
        merged = merged.rename(columns={"BayesDel_nsfp33a_noAF": "BayesDel"})

    cols_to_keep = [
    "UniProtID", "GeneName", "GeneID", "chr", "hg19_pos", "Position",
    "RefAA", "AltAA", "VariantID", "am_pathogenicity", "am_class",
    "popEVE", "EVE", "CPT1", "REVEL", "BayesDel"]

    for c in cols_to_keep:
       if c not in merged.columns:
        merged[c] = pd.NA

    merged = merged[cols_to_keep]
    merged = merged.rename(columns={"Position": "PositionAA"})

    output_path = os.path.join(folder_path, os.path.splitext(os.path.basename(protein_path))[0] + "_5preds.tsv")
    merged.to_csv(output_path, sep="\t", index=False)

print("Done.")
