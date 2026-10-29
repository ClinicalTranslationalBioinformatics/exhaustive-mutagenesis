"""
Añade columnas del release (por defecto QAFIMeta_v4, QAFIClass1, QAFIClass2) a un
archivo de variantes externo, conservando el archivo original tal cual (mismas filas,
mismas columnas, mismo orden) y añadiendo las columnas nuevas al final.

Emparejamiento: uniprot + variant (p.ej. P20062 + R259P) contra
RefAA+PositionAA+AltAA del *_variants_updated.tsv de esa proteína
(o del *_variants.tsv si la proteína no tiene _updated).

Uso:
  python add_release_scores.py \
      --variants prospective_universe_39207_variants_for_aitana.csv \
      --release-dir ~/Desktop/human_proteome_2026/human_proteome_2026_release \
      --out prospective_universe_39207_variants_with_QAFI.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_COLS = ["QAFIMeta_v4", "QAFIClass1", "QAFIClass2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="CSV con columnas uniprot y variant")
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", nargs="+", default=DEFAULT_COLS, help="Columnas del release a añadir")
    ap.add_argument("--status-out", help="CSV opcional con uniprot, variant y el motivo de cada fila")
    a = ap.parse_args()

    rel = Path(a.release_dir).expanduser()
    var = pd.read_csv(a.variants, dtype=str, keep_default_na=False)
    for c in ["uniprot", "variant"]:
        if c not in var.columns:
            raise SystemExit(f"Falta la columna '{c}' en {a.variants}")
    clash = [c for c in a.cols if c in var.columns]
    if clash:
        raise SystemExit(f"El archivo ya tiene estas columnas: {clash}")

    key = var["variant"].str.strip()
    found = {c: pd.Series("", index=var.index) for c in a.cols}
    status = pd.Series("", index=var.index)

    for uni, idx in var.groupby(var["uniprot"].str.strip()).groups.items():
        f = rel / uni / f"{uni}_variants_updated.tsv"
        ok_label = "ok"
        if not f.exists():
            f = rel / uni / f"{uni}_variants.tsv"          # sin _updated: usar el archivo de la v1
            ok_label = "ok_from_variants_tsv"
        if not f.exists():
            status[idx] = "protein_not_in_release"
            continue
        head = pd.read_csv(f, sep="\t", nrows=0).columns
        use = ["RefAA", "PositionAA", "AltAA"] + [c for c in a.cols if c in head]
        d = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, usecols=use)
        d.index = d["RefAA"].str.strip() + d["PositionAA"].str.strip() + d["AltAA"].str.strip()
        d = d[~d.index.duplicated()]
        k = key[idx]
        hit = k.isin(d.index)
        status[idx[~hit.values]] = "variant_not_in_release"
        status[idx[hit.values]] = ok_label
        for c in a.cols:
            if c in d.columns:
                found[c][idx[hit.values]] = d.loc[k[hit].values, c].values

    out = var.copy()
    for c in a.cols:
        out[c] = found[c].values
    out.to_csv(a.out, index=False)
    if a.status_out:
        pd.DataFrame({"uniprot": var["uniprot"], "variant": var["variant"], "status": status}).to_csv(a.status_out, index=False)

    n = len(out)
    st = Counter(status)
    print(f"Variantes: {n:,}")
    for s in ["ok", "ok_from_variants_tsv", "variant_not_in_release", "protein_not_in_release"]:
        print(f"  {s:<24} {st.get(s, 0):>8,}  ({st.get(s, 0) / n * 100:.1f}%)")
    for c in a.cols:
        nv = (out[c] != "").sum()
        print(f"  con valor de {c:<12} {nv:>8,}  ({nv / n * 100:.1f}%)")
    miss = sorted(set(out.loc[status == "protein_not_in_release", "uniprot"]))
    if miss:
        print(f"\nProteínas sin _updated en el release ({len(miss)}), primeras 15: {miss[:15]}")
    print(f"\nGuardado en {a.out}")


if __name__ == "__main__":
    main()