import os
import pandas as pd
import shutil
import QAFI_family

# === Config ===
models = ["qafi1", "qafi2", "qafisplit1", "qafisplit2", "qafisplit3"]  # all QAFI models
path_base = "/home/frog/Desktop/human_proteome_2026/proteins"       # proteins folder

# === QAFI data paths ===
path_db = "data/Dataset_60proteins_features.csv"
path_protein_set = "data/QAFI_build_proteins.csv"

# === Load QAFI DBs ===
DB = pd.read_csv(path_db)
df_build_proteins = pd.read_csv(path_protein_set)

log_path = "/home/frog/Desktop/human_proteome_2026/qafi_computation/skipped_proteins.txt"

# === Loop over all proteins ===
for uniprot_id in os.listdir(path_base):
    protein_folder = os.path.join(path_base, uniprot_id)
    if not os.path.isdir(protein_folder):
        with open(log_path, "a") as log_file:
            log_file.write(f"{uniprot_id}: no folder found\n")
        continue

    df_test_path = os.path.join(protein_folder, f"{uniprot_id}_final2.csv")
    if not os.path.isfile(df_test_path):
        print(f"⚠️ Missing final2.csv for {uniprot_id}, skipping.")
        with open(log_path, "a") as log_file:
            log_file.write(f"{uniprot_id}: missing final2.csv\n")
        continue

    df_test_base = pd.read_csv(df_test_path)
    df_test_base["protein_pos"] = df_test_base["pos"]

    successful_models = []
    # === Loop over all models for this protein ===
    for model_name in models:
        model_output_dir = os.path.join("output", uniprot_id, model_name)
        try:
            print(f"🔁 Running model {model_name} for {uniprot_id}...")
            final_save_path = QAFI_family.run_qafi_model(
                model_name=model_name,
                uniprot_id=uniprot_id,
                DB=DB,
                df_build_proteins=df_build_proteins,
                df_test_base=df_test_base,
                path_base="output"
            )
            successful_models.append(model_name)
            print(f"✅ Finished {model_name} for {uniprot_id}. Saved: {final_save_path}")
        except Exception as e:
            print(f"⚠️ Skipped {model_name} for {uniprot_id}. Reason: {e}")
            if os.path.exists(model_output_dir):
                shutil.rmtree(model_output_dir)  # delete failed model folder

    # After all models
    if not successful_models:  # no models succeeded
        prot_folder = os.path.join("output", uniprot_id)
        if os.path.exists(prot_folder):
            shutil.rmtree(prot_folder)  # delete protein folder entirely

print("Done.")
