from pathlib import Path
import pandas as pd

PRED_ORDER = [
    "am_pathogenicity", "am_class", "popEVE", "EVE", "CPT1", "REVEL", "BayesDel",
    "qafi1", "qafi2", "qafi3",
    "qafisplit1", "qafisplit1_median",
    "qafisplit2", "qafisplit1_median", "qafisplit2_residual",
    "qafisplit3", 
    "QAFImt",
    "QAFIClass1", "QAFIClass2",
    "QAFIMeta_v1", "QAFIMeta_v2", "QAFIMeta_v3", "QAFIMeta_v4",
]

def reorder_cols(path: Path) -> None:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    
    meta_cols = ["UniProtID", "GeneName", "GeneID", "chr", "hg19_pos",
                 "PositionAA", "RefAA", "AltAA", "VariantID"]
    
    ordered = [c for c in meta_cols if c in df.columns]
    
    # Add each predictor + its _sd if exists, in preferred order
    seen = set(ordered)
    for col in PRED_ORDER:
        if col in df.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
            if f"{col}_sd" in df.columns:
                ordered.append(f"{col}_sd")
                seen.add(f"{col}_sd")
    
    # Append anything remaining not yet included
    ordered += [c for c in df.columns if c not in seen]
    
    df[ordered].to_csv(path, sep="\t", index=False)
    print(f"[done] {path.name}")

# Test
# reorder_cols(Path("~/Desktop/human_proteome_2026/proteins/Q15858/Q15858_8preds_with_sd.tsv").expanduser())

# All
proteins_dir = Path("~/Desktop/human_proteome_2026/proteins").expanduser()
for f in proteins_dir.glob("*/*with_sd.tsv"):
    reorder_cols(f)