"""
Crea <ID>_variants_updated.tsv para las proteínas que no lo tienen (o lo tienen vacío),
a partir de <ID>_variants.tsv, con UNA fila por sustitución (PositionAA, RefAA, AltAA),
quedándose con la primera (misma regla que el resto de _updated).
Renombra qafisplit2_median -> QAFImt (las 4 proteínas con el nombre antiguo).

Así todas las proteínas tienen _updated y los pasos de la v3 (add_qafimt2, recalc_class,
recalc_meta) las procesan todas.

Uso (sobre la COPIA de trabajo de la v3):
  python make_missing_updated.py --release-dir ~/Desktop/human_proteome_2026/release_v3
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True)
    a = ap.parse_args()
    rel = Path(a.release_dir).expanduser()

    c = Counter()
    for d in sorted(p for p in rel.iterdir() if p.is_dir()):
        u = d.name
        upd, var = d / f"{u}_variants_updated.tsv", d / f"{u}_variants.tsv"
        if upd.exists() and sum(1 for _ in open(upd)) > 1:
            c["ya_tenia_updated"] += 1
            continue
        if not var.exists():
            c["sin_variants"] += 1
            continue
        df = pd.read_csv(var, sep="\t", dtype=str, keep_default_na=False)
        df = df.rename(columns={"qafisplit2_median": "QAFImt"})
        n0 = len(df)
        df = df.drop_duplicates(["PositionAA", "RefAA", "AltAA"], keep="first")
        tmp = upd.with_suffix(".tmp")
        df.to_csv(tmp, sep="\t", index=False)
        tmp.replace(upd)
        c["creado" + ("_dedup" if len(df) < n0 else "")] += 1
    print(dict(c))


if __name__ == "__main__":
    main()