import os
import pandas as pd

proteins_base = "/home/frog/Desktop/human_proteome_2026/proteins"
qafi_base = "/home/frog/Desktop/human_proteome_2026/qafi_computation/output"
models = ["qafi1", "qafi2", "qafisplit1", "qafisplit2", "qafisplit3"]
common_cols = ["uniprot", "variant", "pos", "first", "second"]
merge_keys = ["UniProtID", "VariantID", "PositionAA", "RefAA", "AltAA"]

for uniprot_id in os.listdir(proteins_base):
    protein_dir = os.path.join(proteins_base, uniprot_id)
    if not os.path.isdir(protein_dir):
        continue

    # Load 5preds
    tsv_files = [f for f in os.listdir(protein_dir) if f.endswith("_variants_5preds.tsv")]
    if not tsv_files:
        continue
    fivepreds = pd.read_csv(os.path.join(protein_dir, tsv_files[0]), sep="\t")

    # Load QAFI
    qafi_dir = os.path.join(qafi_base, uniprot_id)
    if not os.path.isdir(qafi_dir):
        combined_qafi = None

    combined_qafi = None
    for model_name in models:
        model_csv = os.path.join(qafi_dir, model_name, f"{model_name}.csv")
        if not os.path.isfile(model_csv):
            continue
        df = pd.read_csv(model_csv)
        model_cols = [c for c in df.columns if c not in common_cols]
        df_model = df[common_cols + model_cols]
        combined_qafi = df_model if combined_qafi is None else pd.merge(combined_qafi, df_model, on=common_cols, how="outer")

    if combined_qafi is not None:
        combined_qafi = combined_qafi.rename(columns={
            "uniprot": "UniProtID", "variant": "VariantID",
            "pos": "PositionAA", "first": "RefAA", "second": "AltAA"
        })
        fivepreds = fivepreds.merge(combined_qafi, on=merge_keys, how="left")

    out_path = os.path.join(protein_dir, f"{uniprot_id}_variants_6preds.tsv")
    fivepreds.to_csv(out_path, sep="\t", index=False)
    print(f"✅ Merged {uniprot_id}")

print("🎉 All done")