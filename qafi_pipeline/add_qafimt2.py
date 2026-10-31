"""
Añade las columnas de QAFImt 2.0 (salida de run_qafimt2.py de Shaopei) a los
*_variants_updated.tsv del release:  QAFImt2, qafisplit2_mt2, qafisplit3_mt2

- Empareja por variante (RefAA+PositionAA+AltAA del release  <->  'variant' del qafimt2.csv).
- Si las columnas ya existen, las sobrescribe. El resto del archivo no se toca.

Uso:
  # Prueba (no escribe, solo informa del emparejamiento):
  python add_qafimt2.py --release-dir ~/Desktop/human_proteome_2026/human_proteome_2026_release \
                        --qafimt2-dir ~/Desktop/human_proteome_2026/qafimt2_output
  # Aplicar:
  python add_qafimt2.py ... --inplace
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

NEW_COLS = ["QAFImt2", "qafisplit2_mt2", "qafisplit3_mt2"]


def run_protein(folder: Path, mt2_dir: Path, inplace: bool):
    u = folder.name
    upd = folder / f"{u}_variants_updated.tsv"
    if not upd.exists():
        return "skip_no_updated", None
    src = mt2_dir / f"{u}_qafimt2.csv"
    if not src.exists():
        return "no_qafimt2", None

    rel = pd.read_csv(upd, sep="\t", dtype=str, keep_default_na=False)
    mt2 = pd.read_csv(src, dtype=str, keep_default_na=False)
    mt2["_key"] = mt2["variant"].str.strip().str.split("_").str[-1]
    mt2 = mt2.drop_duplicates("_key").set_index("_key")

    key = rel["RefAA"].str.strip() + rel["PositionAA"].str.strip() + rel["AltAA"].str.strip()
    for c in NEW_COLS:
        rel[c] = key.map(mt2[c]).fillna("").values
    matched = int((rel["QAFImt2"] != "").sum())

    if inplace:
        rel.to_csv(upd, sep="\t", index=False)
    return "done", (matched, len(rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--qafimt2-dir", required=True)
    ap.add_argument("--inplace", action="store_true")
    a = ap.parse_args()
    root = Path(a.release_dir).expanduser()
    mt2_dir = Path(a.qafimt2_dir).expanduser()

    counts, low, failed = Counter(), [], []
    tot_m = tot_n = 0
    for f in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            st, res = run_protein(f, mt2_dir, a.inplace)
        except Exception as e:
            st, res = "fail", None
            failed.append(f"{f.name}: {e}")
        counts[st] += 1
        if st == "done":
            m, n = res
            tot_m += m; tot_n += n
            if n and m / n < 0.95:
                low.append((f.name, m, n))

    print(f"{'APLICADO' if a.inplace else 'PRUEBA (no se ha escrito nada)'}: {dict(counts)}")
    if tot_n:
        print(f"Variantes con QAFImt2: {tot_m:,}/{tot_n:,} ({tot_m / tot_n * 100:.2f}%)")
    if low:
        print(f"\n{len(low)} proteínas con <95% de variantes emparejadas (primeras 15):")
        for x in low[:15]:
            print("  ", x)
    for x in failed[:20]:
        print("[FAILED]", x)
    if counts.get("no_qafimt2"):
        print(f"\nOjo: {counts['no_qafimt2']} proteínas del release sin archivo en {mt2_dir} "
              f"(sin final2 o fallidas: mira failed.txt)")


if __name__ == "__main__":
    main()