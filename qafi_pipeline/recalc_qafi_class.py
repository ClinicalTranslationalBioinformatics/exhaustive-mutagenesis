"""
Recalcula QAFIClass1 / QAFIClass2 en los *_variants_updated.tsv del release.

Misma lógica que el pipeline original (pipeline.py + config.py):
  features de *_final2.csv  +  qafisplit1_median (= QAFImt) tomado del release
  -> zero low pLDDT -> fill NaN con mediana del train -> MinMax (fit en train)
  -> XGBoost x2 -> predict_proba[:,1] -> QAFIClass1 / QAFIClass2
LOPO: si la proteína está en el training, se reentrena sin ella. Si no está,
se usa un único modelo global (idéntico al que entrenaría el pipeline original,
pero entrenado una sola vez -> mucho más rápido).

Uso:
  # 1) PRUEBA de reproducibilidad con el QAFImt actual (no escribe nada):
  python recalc_class.py --release-dir ~/Desktop/human_proteome_2026/human_proteome_2026_release \
      --final2-dirs ~/Desktop/human_proteome_2026/proteome_final2 ~/Desktop/human_proteome_2026/proteome_final2_part2 \
      --train-csv ~/Desktop/QAFI_meta_class/QAFI_meta_class/predictions_clinhum_qafi_insilico.csv \
      --mt-col QAFImt --uniprot Q14137 P49815 --test

  # 2) Cuando llegue QAFImt 2.0 (columna QAFImt_2.0 en el release y training actualizado):
  python recalc_class.py ... --mt-col QAFImt_2.0 --inplace
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------- config (copiado de config.py)
LABEL_COLUMN_RAW = "ClinicalSignificanceBinary"
LABEL_MAP = {"LB/B": 1, "LP/P": 0}
MT_FEATURE = "qafisplit1_median"
FEATURES = [
    "Blosum62", "PSSM", "Shannon's entropy", "Shannon's entropy of seq. neighbours", "pLDDT", "pLDDT bin",
    "colasi", "fraction cons. 3D neighbor", "fanc", "fbnc", "M.J. potential", "access.dependent vol.", "neco",
    "PSSM_pdff", "Shannon's entropy_pdff", "Shannon's entropy of seq. neighbours_pdff", "pLDDT_pdff",
    "colasi_pdff", "fraction cons. 3D neighbor_pdff", "fanc_pdff", "fbnc_pdff", "e_native_pdff",
    "neco2", "neco3", "neco12", "lpar", "NCE-NR", MT_FEATURE,
]
ZERO_LOW_PLDDT = [
    "colasi", "fraction cons. 3D neighbor", "fanc", "fbnc", "colasi_pdff", "fraction cons. 3D neighbor_pdff",
    "fanc_pdff", "e_native_pdff", "NCE-NR", "fbnc_pdff", "M.J. potential", "access.dependent vol.",
    "neco", "neco2", "neco3", "neco12",
]
PLDDT_BIN, PLDDT_LOW = "pLDDT bin", 0
RANDOM_STATE = 42


def build_models():
    return {
        "QAFIClass1": xgb.XGBClassifier(eval_metric="logloss", reg_alpha=10, reg_lambda=10, max_depth=4,
                                        learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                                        n_estimators=200, random_state=RANDOM_STATE),
        "QAFIClass2": xgb.XGBClassifier(eval_metric="logloss", max_depth=5, n_estimators=400,
                                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                        reg_alpha=8, reg_lambda=8, gamma=2, random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------- modelo
def zero_low(df):
    df = df.copy()
    df.loc[df[PLDDT_BIN] == PLDDT_LOW, ZERO_LOW_PLDDT] = 0
    return df


class Fitted:
    """Modelos + scaler + medianas entrenados sobre un training concreto."""

    def __init__(self, train: pd.DataFrame):
        train = zero_low(train)
        self.medians = train[FEATURES].median()
        X = train[FEATURES].fillna(self.medians)
        self.scaler = MinMaxScaler().fit(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=FEATURES)
        y = train[LABEL_COLUMN_RAW].map(LABEL_MAP).values
        self.models = build_models()
        for m in self.models.values():
            m.fit(Xs, y)

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        test = zero_low(test)
        X = test[FEATURES].fillna(self.medians)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=FEATURES)
        return pd.DataFrame({k: m.predict_proba(Xs)[:, 1] for k, m in self.models.items()}, index=test.index)


# ---------------------------------------------------------------- I/O
def var_key(s):
    m = re.search(r"([A-Z])(\d+)([A-Z])$", str(s).strip())
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def find_final2(final2_dirs, uniprot):
    for d in final2_dirs:
        hits = list((d / uniprot).glob("*_final2.csv"))
        if hits:
            return hits[0]
    return None


def run_protein(uniprot, release_dir, final2_dirs, train, global_fit, mt_col, inplace, test):
    src = release_dir / uniprot / f"{uniprot}_variants_updated.tsv"
    if not src.exists():
        return "skip", f"[skip] {uniprot}: no hay _variants_updated.tsv"
    f2 = find_final2(final2_dirs, uniprot)
    if f2 is None:
        return "nofinal2", f"[sin final2] {uniprot}"

    rel = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
    if mt_col not in rel.columns:
        return "fail", f"[FAILED] {uniprot}: no existe la columna {mt_col} en el release"
    rel_key = rel["RefAA"].str.strip() + rel["PositionAA"].str.strip() + rel["AltAA"].str.strip()

    feats = pd.read_csv(f2)
    vcol = "variant" if "variant" in feats.columns else ("VariantID" if "VariantID" in feats.columns else feats.columns[0])
    feats["_key"] = feats[vcol].map(var_key)
    feats = feats.drop_duplicates("_key").set_index("_key")
    missing = [c for c in FEATURES if c != MT_FEATURE and c not in feats.columns]
    if missing:
        return "fail", f"[FAILED] {uniprot}: faltan features en final2: {missing[:5]}"

    test_df = feats.reindex(rel_key.values)[[c for c in FEATURES if c != MT_FEATURE]].reset_index(drop=True)
    test_df[MT_FEATURE] = pd.to_numeric(rel[mt_col], errors="coerce").values
    matched = test_df["pLDDT"].notna().sum()

    # LOPO: si la proteína está en el training, reentrenar sin ella
    in_train = (train["uniprot"] == uniprot).sum()
    fit = Fitted(train[train["uniprot"] != uniprot].reset_index(drop=True)) if in_train else global_fit
    pred = fit.predict(test_df)
    pred[test_df["pLDDT"].isna().values] = np.nan  # variantes sin features -> NaN

    msg = f"{uniprot}: {matched}/{len(rel)} con features{' | LOPO ' + str(in_train) if in_train else ''}"
    if test:
        for c in ["QAFIClass1", "QAFIClass2"]:
            old = pd.to_numeric(rel.get(c), errors="coerce")
            m = old.notna() & pred[c].notna()
            if m.sum() > 2:
                r = np.corrcoef(old[m], pred[c][m])[0, 1]
                msg += f" | {c}: r={r:.4f} maxdiff={(old[m] - pred[c][m]).abs().max():.4f}"
        return "done", "[test] " + msg

    for c in ["QAFIClass1", "QAFIClass2"]:
        rel[c] = pred[c].map(lambda x: "" if pd.isna(x) else repr(float(x)))
    out = src if inplace else src.with_name(f"{uniprot}_variants_updated_classtest.tsv")
    rel.to_csv(out, sep="\t", index=False)
    return "done", f"[done] {msg} -> {out.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--final2-dirs", nargs="+", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--mt-col", default="QAFImt", help="Columna del release usada como qafisplit1_median.")
    ap.add_argument("--uniprot", nargs="*", help="Solo estas proteínas.")
    ap.add_argument("--test", action="store_true", help="No escribe; compara con QAFIClass actuales.")
    ap.add_argument("--inplace", action="store_true", help="Sobrescribe QAFIClass1/2 en *_variants_updated.tsv.")
    args = ap.parse_args()

    release_dir = Path(args.release_dir).expanduser()
    final2_dirs = [Path(d).expanduser() for d in args.final2_dirs]

    t0 = time.time()
    train = pd.read_csv(Path(args.train_csv).expanduser(), low_memory=False)
    train = train[train[LABEL_COLUMN_RAW].isin(LABEL_MAP)].reset_index(drop=True)
    print(f"Training: {len(train)} variantes, {train['uniprot'].nunique()} proteínas")
    global_fit = Fitted(train)
    print(f"Modelo global entrenado en {time.time() - t0:.0f}s")

    prots = args.uniprot or sorted(p.name for p in release_dir.iterdir() if p.is_dir())
    counts, failed = {}, []
    for i, u in enumerate(prots, 1):
        try:
            st, msg = run_protein(u, release_dir, final2_dirs, train, global_fit, args.mt_col, args.inplace, args.test)
        except Exception as e:
            st, msg = "fail", f"[FAILED] {u}: {e}"
        counts[st] = counts.get(st, 0) + 1
        if st == "fail":
            failed.append(u)
        if args.uniprot or st == "fail" or i % 500 == 0:
            print(msg if args.uniprot or st == "fail" else f"[{i}/{len(prots)}] {msg}")

    print(f"\nHecho en {(time.time() - t0) / 60:.1f} min: {counts}")
    if failed:
        print(f"Fallidas: {failed[:50]}")


if __name__ == "__main__":
    main()