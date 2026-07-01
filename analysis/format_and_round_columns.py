from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

PREDICTOR_COLS = [
    "am_pathogenicity", "popEVE", "EVE", "CPT1", "REVEL", "BayesDel",
    "qafi1", "qafi2", "qafisplit1", "qafisplit1_median",
    "qafisplit2", "QAFImt", "qafisplit2_residual", "qafisplit3",
    "QAFIClass1", "QAFIClass2",
    "QAFIMeta_v1", "QAFIMeta_v2", "QAFIMeta_v3", "QAFIMeta_v4",
]


def get_8preds_file(protein_dir: Path) -> Path:
    files = sorted(protein_dir.glob("*_8preds.tsv"))

    if len(files) != 1:
        raise ValueError(f"Expected exactly 1 *_8preds.tsv file, found {len(files)}")

    return files[0]


def fix_protein(protein_dir: Path) -> None:
    path = get_8preds_file(protein_dir)

    df = pd.read_csv(path, sep="\t")

    # Rename column
    df = df.rename(columns={"qafisplit2_median": "QAFImt"})

    # Round predictors
    cols_to_round = [c for c in PREDICTOR_COLS if c in df.columns]
    df[cols_to_round] = df[cols_to_round].round(3)

    # Overwrite
    df.to_csv(path, sep="\t", index=False)

    print(f"[done] {protein_dir.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins-dir", required=True)
    parser.add_argument("--uniprot", help="Single protein for testing.")
    args = parser.parse_args()

    proteins_dir = Path(args.proteins_dir)

    folders = (
        [proteins_dir / args.uniprot]
        if args.uniprot
        else sorted(p for p in proteins_dir.iterdir() if p.is_dir())
    )

    print(f"Processing {len(folders)} proteins...")

    failed = []

    log_path = Path("failed_proteins.log")
    log_path.write_text(f"Run {datetime.now()}\n\n")

    for folder in folders:
        try:
            fix_protein(folder)
        except Exception as e:
            msg = f"{folder.name}: {e}"
            print(f"[FAILED] {msg}")
            failed.append(folder.name)

            with open(log_path, "a") as f:
                f.write(msg + "\n")

    print(f"\nDone. {len(folders) - len(failed)}/{len(folders)} succeeded.")

    if failed:
        print(f"Failed: {failed[:20]}{'...' if len(failed) > 20 else ''}")
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
