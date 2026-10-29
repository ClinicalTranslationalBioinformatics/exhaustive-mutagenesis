"""
Recalcula QAFIMeta 2.0 v1-v4 en los *_variants_updated.tsv del release.

Según "CAGI MODEL SELECT" + QAFIs_summary.xlsx:
  - v1: ancla QAFIClass1 | v2: ancla QAFIClass2 | v3: ancla qafi2 | v4: ancla qafisplit3
  - Externos (AM, popEVE, CPT1, REVEL, BayesDel) reescalados con z-score a la
    distribución del ancla (AM, CPT1, REVEL, BayesDel con signo invertido;
    AM 'ambiguous' -> NaN). Meta = mediana de todos los disponibles; si no hay
    externos, queda el ancla.
  - Rescale final de la salida (z-score) a la distribución de qafisplit3, por
    proteína, para las versiones en --rescale (por defecto v1, v2, v3).

Uso:
    # Prueba con una proteína (crea *_variants_updated_test.tsv, no toca el original)
    python recalc_meta.py --release-dir ~/Desktop/human_proteome_2026/human_proteome_2026_release --uniprot Q14137 --versions v3

    # Todas, sobrescribiendo
    python recalc_meta.py --release-dir ... --versions v3 --inplace

    # Con QAFIsplit 3.0 nuevo (QAFImt 2.0) como ancla de v4 y referencia del rescale
    python recalc_meta.py --release-dir ... --split3-col qafisplit3_mt2 --inplace
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANCHORS = {"v1": "QAFIClass1", "v2": "QAFIClass2", "v3": "qafi2", "v4": "qafisplit3"}
NUMERIC = ["QAFIClass1", "QAFIClass2", "qafi2", "qafisplit3",
           "am_pathogenicity", "popEVE", "CPT1", "REVEL", "BayesDel"]
RESCALE_TARGET = "qafisplit3"


def z_to_target(values, t_mean, t_std):
    p_mean, p_std = values.mean(), values.std()
    if (pd.isna(p_mean) or pd.isna(p_std) or p_std == 0
            or pd.isna(t_mean) or pd.isna(t_std) or t_std == 0):
        return pd.Series(np.nan, index=values.index)
    return ((values - p_mean) / p_std) * t_std + t_mean


def compute_meta(df, anchor, skip_cpt):
    if anchor not in df.columns or df[anchor].isna().all():
        return pd.Series(np.nan, index=df.index)
    a = df[anchor]
    t_mean, t_std = a.mean(), a.std()
    parts = {"anchor": a}

    if "am_pathogenicity" in df.columns:
        am = z_to_target(-df["am_pathogenicity"], t_mean, t_std)
        if "am_class" in df.columns:
            am[df["am_class"].astype(str).str.lower() == "ambiguous"] = np.nan
        parts["am"] = am
    if "popEVE" in df.columns:
        parts["popeve"] = z_to_target(df["popEVE"], t_mean, t_std)
    if not skip_cpt and "CPT1" in df.columns:
        parts["cpt"] = z_to_target(-df["CPT1"], t_mean, t_std)
    if "REVEL" in df.columns:
        parts["revel"] = z_to_target(-df["REVEL"], t_mean, t_std)
    if "BayesDel" in df.columns:
        parts["bayes"] = z_to_target(-df["BayesDel"], t_mean, t_std)

    return pd.DataFrame(parts).median(axis=1, skipna=True)


def fmt(s):
    return s.map(lambda x: "" if pd.isna(x) else f"{x:.3f}")


def run_protein(folder: Path, versions, rescale, inplace, split3="qafisplit3"):
    uniprot = folder.name
    src = folder / f"{uniprot}_variants_updated.tsv"
    if not src.exists():
        return "skip", f"[skip] {uniprot}: no hay _variants_updated.tsv"

    # Todo como texto para reescribir el resto de columnas exactamente igual
    df = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
    work = df.rename(columns={"AM_pathogenicity": "am_pathogenicity", "AM_class": "am_class"}).copy()
    anchors = dict(ANCHORS, v4=split3)
    for c in NUMERIC + [split3]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    skip_cpt = "CPT1" not in work.columns or work["CPT1"].isna().all()

    s3 = work[split3] if split3 in work.columns else pd.Series(np.nan, index=work.index)
    msgs = []
    for v in versions:
        meta = compute_meta(work, anchors[v], skip_cpt)
        if v in rescale and meta.notna().any():
            meta = z_to_target(meta, s3.mean(), s3.std())
        col = f"QAFIMeta_{v}"
        old = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        new = meta.round(3)
        changed = int(((new - old).abs() > 1e-9).sum() + (new.isna() != old.isna()).sum())
        df[col] = fmt(new)
        rng = f"[{new.min():.3f}, {new.max():.3f}]" if new.notna().any() else "todo NaN"
        msgs.append(f"{v}: {changed} cambiados, rango {rng}")

    out = src if inplace else folder / f"{uniprot}_variants_updated_test.tsv"
    df.to_csv(out, sep="\t", index=False)
    s3_rng = f"[{s3.min():.3f}, {s3.max():.3f}]" if s3.notna().any() else "NaN"
    return "done", f"[done] {uniprot} ({split3} {s3_rng}) | " + " | ".join(msgs) + f" -> {out.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--uniprot", help="Solo una proteína (prueba).")
    ap.add_argument("--versions", default="v1,v2,v3,v4", help="Qué versiones recalcular, p.ej. v3 o v3,v4.")
    ap.add_argument("--rescale", default="v1,v2,v3", help="Versiones con rescale final a qafisplit3.")
    ap.add_argument("--inplace", action="store_true", help="Sobrescribe *_variants_updated.tsv.")
    ap.add_argument("--split3-col", default="qafisplit3",
                    help="Columna de QAFIsplit 3.0: ancla de v4 y referencia del rescale (p.ej. qafisplit3_mt2).")
    args = ap.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    rescale = {v.strip() for v in args.rescale.split(",") if v.strip()}
    bad = [v for v in versions if v not in ANCHORS]
    if bad:
        raise SystemExit(f"Versiones no válidas: {bad}")

    root = Path(args.release_dir).expanduser()
    folders = [root / args.uniprot] if args.uniprot else sorted(p for p in root.iterdir() if p.is_dir())
    verbose = bool(args.uniprot)

    counts = {"done": 0, "skip": 0}
    failed = []
    for f in folders:
        try:
            status, msg = run_protein(f, versions, rescale, args.inplace, args.split3_col)
            counts[status] += 1
            if verbose or status == "done" and counts["done"] % 1000 == 0:
                print(msg)
        except Exception as e:
            print(f"[FAILED] {f.name}: {e}")
            failed.append(f.name)

    print(f"\nHecho. {counts['done']} actualizadas, {counts['skip']} sin _updated, {len(failed)} fallidas.")
    if failed:
        print(f"Fallidas: {failed}")


if __name__ == "__main__":
    main()