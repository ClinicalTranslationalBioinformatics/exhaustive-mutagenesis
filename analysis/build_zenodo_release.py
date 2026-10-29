"""
Construye la carpeta de una versión de Zenodo a partir del release de trabajo,
con el mismo formato que la v1:

    <out-dir>/exhaustive_mutagenesis_data/<ID>/<ID>_variants.tsv

Por cada proteína:
  - usa <ID>_variants_updated.tsv si existe y no está vacío; si no, <ID>_variants.tsv
  - renombra qafisplit2_median y QAFImt -> QAFImt1.0
  - redondea los predictores a 3 decimales (como en la v1)
  - deja UNA fila por sustitución de aminoácido (PositionAA, RefAA, AltAA), quedándose con la
    primera, que es la misma regla con la que se generaron los *_variants_updated.tsv
  - el resto de columnas se copian tal cual
No modifica el release de trabajo.

Uso:
  # Prueba con una proteína
  python build_zenodo_release.py --release-dir ~/Desktop/human_proteome_2026/human_proteome_2026_release \
      --out-dir ~/Desktop/human_proteome_2026/zenodo_v2.0 --uniprot Q14137
  # Todas
  python build_zenodo_release.py --release-dir ... --out-dir ~/Desktop/human_proteome_2026/zenodo_v2.0
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

RENAME = {"qafisplit2_median": "QAFImt1.0", "QAFImt": "QAFImt1.0"}
ROUND_COLS = [
    "AM_pathogenicity", "am_pathogenicity", "popEVE", "EVE", "CPT1", "REVEL", "BayesDel",
    "qafi1", "qafi2", "QAFImt", "QAFImt1.0", "qafisplit1", "qafisplit1_median", "qafisplit2", "qafisplit2_residual",
    "qafisplit3", "QAFIClass1", "QAFIClass2", "QAFIMeta_v1", "QAFIMeta_v2", "QAFIMeta_v3", "QAFIMeta_v4",
    "QAFImt2", "qafisplit2_mt2", "qafisplit3_mt2",
]


def fmt3(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").round(3)
    # mismo aspecto que pandas.round(3) + to_csv en la v1 (0.5, 1.0, 0.123); vacío si NaN
    return x.map(lambda v: "" if pd.isna(v) else repr(float(v))).where(x.notna() | (s == ""), s)


def build(folder: Path, out_root: Path, overwrite: bool):
    u = folder.name
    upd, var = folder / f"{u}_variants_updated.tsv", folder / f"{u}_variants.tsv"
    upd_ok = upd.exists() and sum(1 for _ in open(upd)) > 1     # un _updated vacío (solo cabecera) no vale
    src, kind = (upd, "updated") if upd_ok else ((var, "variants") if var.exists() else (None, None))
    if src is None:
        return "no_file"
    dst = out_root / "exhaustive_mutagenesis_data" / u / f"{u}_variants.tsv"
    if dst.exists() and not overwrite:
        return "already_done"
    df = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False).rename(columns=RENAME)
    n0 = len(df)
    df = df.drop_duplicates(["PositionAA", "RefAA", "AltAA"], keep="first")
    for c in ROUND_COLS:
        if c in df.columns:
            df[c] = fmt3(df[c])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(dst)  # escritura atómica: un corte nunca deja un archivo a medias
    return f"from_{kind}" + ("_dedup" if len(df) < n0 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--uniprot", nargs="*")
    ap.add_argument("--uniprot-file", help="Archivo con un UniProt por línea (alternativa a --uniprot).")
    ap.add_argument("--overwrite", action="store_true", help="Rehacer proteínas ya construidas.")
    a = ap.parse_args()
    rel, out = Path(a.release_dir).expanduser(), Path(a.out_dir).expanduser()
    ids = a.uniprot or ([l.strip() for l in open(Path(a.uniprot_file).expanduser()) if l.strip()] if a.uniprot_file else None)
    folders = [rel / x for x in ids] if ids else sorted(p for p in rel.iterdir() if p.is_dir())

    c, failed = Counter(), []
    for i, f in enumerate(folders, 1):
        try:
            c[build(f, out, a.overwrite)] += 1
        except Exception as e:
            c["failed"] += 1
            failed.append(f"{f.name}: {e}")
        if i % 2000 == 0:
            print(f"[{i}/{len(folders)}] {dict(c)}", flush=True)
    print(f"\nHecho: {dict(c)}")
    print(f"Carpeta: {out / 'exhaustive_mutagenesis_data'}")
    for x in failed[:20]:
        print("[FAILED]", x)


if __name__ == "__main__":
    main()