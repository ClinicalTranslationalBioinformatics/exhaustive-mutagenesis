import os
import pandas as pd
import zipfile

base_dir = "/home/frog/Desktop/human_proteome_2026"
proteins_dir = os.path.join(base_dir, "proteins")
os.makedirs(proteins_dir, exist_ok=True)
proteins_df = pd.read_csv("proteins_df.csv")


start_idx = 0
end_idx   = 20659
batch_df = proteins_df.iloc[start_idx:end_idx]
print("Start protein:", batch_df.iloc[0]['UniProtID'], batch_df.iloc[0]['GeneName'])
print("End protein:", batch_df.iloc[-1]['UniProtID'], batch_df.iloc[-1]['GeneName'])

N = batch_df.shape[0]

for i, prot in batch_df.iterrows():
    uniprot_id = prot['UniProtID']
    gene_name = prot['GeneName']
    gene_id   = prot['GeneID']
    sequence  = prot['Sequence']

    print(f"Processing protein {i+1}/{N}: {uniprot_id} ({gene_name})")

    # Create per-protein folder
    prot_folder = os.path.join(proteins_dir, uniprot_id)
    os.makedirs(prot_folder, exist_ok=True)

    # Generate variants table (all possible AA changes)
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    variant_rows = []
    for pos, ref_aa in enumerate(sequence, start=1):
        alt_aas = [aa for aa in aa_list if aa != ref_aa]
        for alt_aa in alt_aas:
            variant_id = f"{ref_aa}{pos}{alt_aa}"
            variant_rows.append({
                'UniProtID': uniprot_id,
                'GeneName': gene_name,
                'GeneID': gene_id,
                'Position': pos,
                'RefAA': ref_aa,
                'AltAA': alt_aa,
                'VariantID': variant_id
            })
    protein_variants_df = pd.DataFrame(variant_rows)

    # 1. Merge AlphaMissense
    alphamissense_df = pd.read_csv("AlphaMissense_aa_substitutions.tsv.gz", sep="\t", comment="#", compression="gzip")
    protein_variants_df = protein_variants_df.merge(
        alphamissense_df[['uniprot_id', 'protein_variant', 'am_pathogenicity', 'am_class']],
        left_on=['UniProtID', 'VariantID'],
        right_on=['uniprot_id', 'protein_variant'],
        how='left'
    ).drop(columns=['uniprot_id','protein_variant'])

    # 2. Merge popEVE
    pop_cols = ['gene','mutant','popEVE','EVE']
    with zipfile.ZipFile("grch38_popEVE_ukbb_20250715.zip",'r') as z:
        for file in z.namelist():
            if file.endswith(".tsv"):
                with z.open(file) as f:
                    df = pd.read_csv(f, sep="\t")
                    df_prot = df[df['gene'] == gene_name][pop_cols]

                    if df_prot.empty:
                        continue

                    protein_variants_df = protein_variants_df.merge(
                        df_prot,
                        left_on=['GeneName','VariantID'],
                        right_on=['gene','mutant'],
                        how='left'
                    ).drop(columns=['gene','mutant'])

    # 3. Merge CPT1
    protein_variants_df['CPT1'] = pd.NA
    cpt1_zips = ["CPT1_score_EVE_set.zip",
                 "CPT1_score_no_EVE_set_1.zip",
                 "CPT1_score_no_EVE_set_2.zip"]
    for zip_path in cpt1_zips:
        with zipfile.ZipFile(zip_path,'r') as z:
            for file in z.namelist():
                if not file.endswith(".csv.gz"):
                    continue
                file_gene_id = os.path.basename(file).replace(".csv.gz","")
                if file_gene_id != gene_id:
                    continue
                with z.open(file) as f:
                    cpt1_df = pd.read_csv(f, compression='gzip')
                mask = protein_variants_df['GeneID'] == file_gene_id
                merged = protein_variants_df.loc[mask,['VariantID']].merge(
                    cpt1_df,
                    left_on='VariantID',
                    right_on='mutant',
                    how='left'
                )
                protein_variants_df.loc[mask,'CPT1'] = merged['CPT1_score'].values

    # Save per-protein table
    protein_variants_df.to_csv(os.path.join(prot_folder,f"{uniprot_id}_variants.tsv"), sep="\t", index=False)
