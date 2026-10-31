"""
QAFImt 2.0 — mt computed with XGBoost — and the two models built on it.

  QAFImt2         mt from an XGBoost position model (QAFImt 1.0 used linear regression)
  qafisplit2_mt2  QAFIsplit 2.0 computed with QAFImt2
  qafisplit3_mt2  QAFIsplit 3.0 (fixed top-30 fusion) computed with QAFImt2

Everything except the mt estimator is the published pipeline
(QAFI_family.qafisplit2_train_and_predict + merge):

for each of the 30 training proteins p in QAFI_build_proteins.csv['qafisplit2']
  step 1  mt_p     XGBoost on p's positions (10 position features -> median fitness
                   of the variants at the position), predicted per test position
  step 2  resid_p  XGBoost on p's variants (28 features + mt_p -> fitness - mt_p),
                   predicted per test variant
          final_p  = mt_p + resid_p
then
  QAFImt2        = median over p of mt_p
  qafisplit2_mt2 = median over p of final_p
  qafisplit3_mt2 = sum over p of weight_p * final_p   (fixed top-30 weights)

The 60 models are trained ONCE (`train()`), then applied to any number of
proteins (`predict()`); training is deterministic, so this gives exactly what
retraining for every protein would.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / "data"
DB_PATH = DATA / "Dataset_60proteins_features.csv.gz"
BUILD_PATH = DATA / "QAFI_build_proteins.csv"
WEIGHTS_PATH = DATA / "weight(qafisplit2)_top30_weights.csv"

TARGET = "score_log_normalized"
POSITION_FEATURES = [
    "PSSM", "Shannon's entropy", "Shannon's entropy of seq. neighbours",
    "pLDDT", "pLDDT bin", "colasi", "fraction cons. 3D neighbor",
    "fanc", "fbnc", "NCE-NR",
]
FEATURES = [
    "Blosum62", "PSSM", "Shannon's entropy", "Shannon's entropy of seq. neighbours",
    "pLDDT", "pLDDT bin", "colasi", "fraction cons. 3D neighbor", "fanc", "fbnc",
    "M.J. potential", "access.dependent vol.", "neco",
    "PSSM_pdff", "Shannon's entropy_pdff", "Shannon's entropy of seq. neighbours_pdff",
    "pLDDT_pdff", "colasi_pdff", "fraction cons. 3D neighbor_pdff", "fanc_pdff",
    "fbnc_pdff", "e_native_pdff", "neco2", "neco3", "neco12", "lpar", "NCE-NR", "laar",
]
REQUIRED = ["uniprot", "variant", "pos"] + FEATURES
OUTPUT_COLUMNS = ["QAFImt2", "qafisplit2_mt2", "qafisplit3_mt2"]

# the step-2 settings of QAFI_family.py, used for both stages
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=3,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  gamma=0, reg_alpha=0.5, reg_lambda=1.0,
                  objective="reg:squarederror", random_state=12234,
                  verbosity=0, tree_method="hist")


def train(verbose=True):
    """Fit the 30 (mt, residual) model pairs. Returns what predict() needs."""
    db = pd.read_csv(DB_PATH, low_memory=False)
    db[TARGET] = pd.to_numeric(db[TARGET], errors="coerce").round(3)
    train_proteins = pd.read_csv(BUILD_PATH)["qafisplit2"].dropna().tolist()
    train_proteins = [p for p in train_proteins if p in set(db["protein"].unique())]

    models = []
    for j, prot in enumerate(train_proteins):
        d = db[db["protein"] == prot].copy()
        d["protein_pos"] = d["protein"] + "_" + d["pos"].astype(str)
        lv = d.groupby("protein_pos", as_index=False).agg(
            {**{f: "first" for f in POSITION_FEATURES}, "pos": "first", TARGET: "median"})
        m_pos = XGBRegressor(**XGB_PARAMS).fit(lv[POSITION_FEATURES].to_numpy(float),
                                               lv[TARGET].to_numpy(float))
        lv["median_pos_predicted"] = m_pos.predict(lv[POSITION_FEATURES].to_numpy(float))
        d = d.merge(lv[["protein_pos", "median_pos_predicted"]], on="protein_pos", how="left")
        d["residual"] = d[TARGET] - d["median_pos_predicted"]
        m_res = XGBRegressor(**XGB_PARAMS).fit(d[FEATURES + ["median_pos_predicted"]].values,
                                               d["residual"].values)
        models.append((prot, m_pos, m_res))
        if verbose:
            print(f"  trained [{j + 1:2d}/{len(train_proteins)}] {prot}", flush=True)

    w = pd.read_csv(WEIGHTS_PATH).set_index("model")["weight"]
    w = w[[p for p in w.index if p in train_proteins]]
    w = w / w.sum()
    weights = np.array([w.get(p, 0.0) for p, _, _ in models])
    return {"models": models, "weights": weights}


def read_features(path):
    """One protein's feature CSV (e.g. Q96BM1_final2.csv). Extra columns are ignored."""
    head = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED if c not in head]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = pd.read_csv(path, low_memory=False, usecols=REQUIRED)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df = df.dropna(subset=["pos"]).drop_duplicates(["uniprot", "variant"]).reset_index(drop=True)
    df["pos"] = df["pos"].astype(int)
    df["protein_pos"] = df["uniprot"].astype(str) + "_" + df["pos"].astype(str)
    return df


def predict(trained, test):
    """QAFImt2, qafisplit2_mt2, qafisplit3_mt2 for every variant of `test`."""
    pos_level = test.groupby("protein_pos", as_index=False).first()
    Xpos = pos_level[POSITION_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pos_ok = np.isfinite(Xpos).all(axis=1)
    idx = pd.Series(np.arange(len(pos_level)), index=pos_level.protein_pos)
    var_to_pos = idx.loc[test.protein_pos].to_numpy()
    Xfeat = test[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xfeat[~np.isfinite(Xfeat)] = np.nan

    n = len(trained["models"])
    mt_all = np.full((len(test), n), np.nan)
    final_all = np.full((len(test), n), np.nan)
    for j, (_, m_pos, m_res) in enumerate(trained["models"]):
        mt_pos = np.full(len(pos_level), np.nan)
        if pos_ok.any():
            mt_pos[pos_ok] = m_pos.predict(Xpos[pos_ok])
        mt_var = mt_pos[var_to_pos]
        resid = m_res.predict(np.column_stack([Xfeat, mt_var]))
        mt_all[:, j] = mt_var
        final_all[:, j] = mt_var + resid

    out = test[["uniprot", "variant", "pos"]].copy()
    with np.errstate(all="ignore"):
        out["QAFImt2"] = np.nanmedian(mt_all, axis=1)
        out["qafisplit2_mt2"] = np.nanmedian(final_all, axis=1)
    out["qafisplit3_mt2"] = final_all @ trained["weights"]
    return out
