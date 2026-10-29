"""
Validación de QAFIClass (y opcionalmente otros predictores) con variantes de ClinVar/Humsavar.

- Etiquetas: archivo de variantes etiquetadas (columnas UniprotID, Variant, BinaryClinicalSignificance = P/B).
- Scores: los del release de una versión de Zenodo (<data-dir>/<ID>/<ID>_variants.tsv).
- Salida:
    <prefix>.csv  -> AUC-ROC por predictor (n, n patogénicas, n benignas)
    <prefix>.pdf  -> curvas ROC + histogramas patogénicas vs benignas

Convención de signo: en la familia QAFI y popEVE, valor alto = benigna;
en AlphaMissense, CPT1, REVEL y BayesDel, valor alto = patogénica.
El AUC se calcula siempre como capacidad de identificar PATOGÉNICAS (0.5 = azar, 1 = perfecto).

Uso:
  python validate_clinvar.py \
      --labels ~/Desktop/human_proteome_2026/merged_variants_clinhum_with_QAFI.csv \
      --data-dir ~/Desktop/human_proteome_2026/zenodo_v2.0/exhaustive_mutagenesis_data \
      --out-prefix ~/Desktop/human_proteome_2026/zenodo_v2.0/exhaustive_mutagenesis_validation_clinvar_v2.0
  # añadir el resto de predictores como referencia:  --all-predictors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve

MAIN = ["QAFIClass1", "QAFIClass2"]
HIGH_IS_BENIGN = ["qafi1", "qafi2", "QAFImt1.0", "QAFImt", "qafisplit1", "qafisplit2", "qafisplit3",
                  "QAFIClass1", "QAFIClass2", "QAFIMeta_v1", "QAFIMeta_v2", "QAFIMeta_v3", "QAFIMeta_v4",
                  "QAFImt2.0", "QAFImt2", "qafisplit2_mt2", "qafisplit3_mt2", "popEVE"]
HIGH_IS_PATHOGENIC = ["AM_pathogenicity", "CPT1", "REVEL", "BayesDel"]


def load_scores(data_dir: Path, proteins, cols_wanted):
    parts = []
    for u in proteins:
        f = data_dir / u / f"{u}_variants.tsv"
        if not f.exists():
            continue
        head = pd.read_csv(f, sep="\t", nrows=0).columns
        cols = [c for c in cols_wanted if c in head]
        d = pd.read_csv(f, sep="\t", usecols=["RefAA", "PositionAA", "AltAA"] + cols, low_memory=False)
        d["UniprotID"] = u
        d["Variant"] = d["RefAA"].astype(str) + d["PositionAA"].astype(str) + d["AltAA"].astype(str)
        parts.append(d.drop(columns=["RefAA", "PositionAA", "AltAA"]).drop_duplicates("Variant"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--all-predictors", action="store_true")
    a = ap.parse_args()

    lab = pd.read_csv(Path(a.labels).expanduser(), low_memory=False,
                      usecols=["UniprotID", "Variant", "BinaryClinicalSignificance"])
    lab = lab[lab["BinaryClinicalSignificance"].isin(["P", "B"])].drop_duplicates(["UniprotID", "Variant"])
    lab["y"] = (lab["BinaryClinicalSignificance"] == "P").astype(int)   # 1 = patogénica
    print(f"Variantes etiquetadas: {len(lab):,} (P={lab.y.sum():,}, B={(1 - lab.y).sum():,}) "
          f"en {lab.UniprotID.nunique():,} proteínas")

    cols = MAIN + ([c for c in HIGH_IS_BENIGN + HIGH_IS_PATHOGENIC if c not in MAIN] if a.all_predictors else [])
    sc = load_scores(Path(a.data_dir).expanduser(), sorted(lab.UniprotID.unique()), cols)
    m = lab.merge(sc, on=["UniprotID", "Variant"], how="left")
    print(f"Emparejadas con el release: {m[MAIN[0]].notna().sum():,} con {MAIN[0]}")

    rows, curves = [], []
    for c in [c for c in cols if c in m.columns]:
        s = m[[c, "y"]].dropna()
        if s.y.nunique() < 2 or len(s) < 20:
            continue
        score = -s[c] if c in HIGH_IS_BENIGN else s[c]      # orientar: alto = patogénica
        auc = roc_auc_score(s.y, score)
        rows.append(dict(predictor=c, n=len(s), n_pathogenic=int(s.y.sum()), n_benign=int((1 - s.y).sum()),
                         coverage=round(len(s) / len(m), 3), AUC=round(auc, 3)))
        curves.append((c, *roc_curve(s.y, score)[:2], auc, s))

    t = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    out = Path(a.out_prefix).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(f"{out}.csv", index=False)
    print("\n" + t.to_string(index=False))

    with PdfPages(f"{out}.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for c, fpr, tpr, auc, _ in curves:
            ax.plot(fpr, tpr, lw=2 if c in MAIN else 0.8, alpha=1 if c in MAIN else 0.6, label=f"{c} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.7)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC – ClinVar/Humsavar (pathogenic vs benign)")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for c, *_, auc, s in curves:
            if c not in MAIN:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = np.linspace(s[c].min(), s[c].max(), 50)
            ax.hist(s.loc[s.y == 0, c], bins=bins, alpha=0.6, density=True, label=f"Benign (n={int((1 - s.y).sum()):,})")
            ax.hist(s.loc[s.y == 1, c], bins=bins, alpha=0.6, density=True, label=f"Pathogenic (n={int(s.y.sum()):,})")
            ax.set_xlabel(c); ax.set_ylabel("Density"); ax.set_title(f"{c} – AUC={auc:.3f}")
            ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"\nGuardado: {out}.csv y {out}.pdf")


if __name__ == "__main__":
    main()