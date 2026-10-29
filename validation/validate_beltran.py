"""
Validación con el benchmark de Beltrán (DMS de dominios humanos): observado vs predicho, por proteína.

  Obs           normalized_fitness            (beltranbench.csv)
  Pred Shaopei  qafisplit2_fusion30           (beltranbench.csv, QAFIsplit 3.0 de Shaopei)
  Pred Aitana   qafisplit3                    (release de la versión de Zenodo; en la v3: qafisplit3_mt2)

Salida:
  <prefix>.csv  una fila por proteína: n, Pearson obs-Shaopei, obs-Aitana, Shaopei-Aitana
  <prefix>.pdf  página resumen + scatter plots por proteína (obs vs Shaopei, obs vs Aitana)

Uso:
  python validate_beltran.py \
      --beltran ~/Desktop/human_proteome_2026/reproduce_Shaopei_plots/Aitana_code/data/beltranbench.csv \
      --data-dir ~/Desktop/human_proteome_2026/zenodo_v2.0/exhaustive_mutagenesis_data \
      --out-prefix ~/Desktop/human_proteome_2026/zenodo_v2.0/exhaustive_mutagenesis_validation_beltran_v2.0
  # v3:  --pred-col qafisplit3_mt2
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


def pearson(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beltran", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--obs-col", default="normalized_fitness")
    ap.add_argument("--shaopei-col", default="qafisplit2_fusion30")
    ap.add_argument("--pred-col", default="qafisplit3", help="Columna del release (v3: qafisplit3_mt2)")
    ap.add_argument("--per-page", type=int, default=4, help="Proteínas por página en el PDF")
    a = ap.parse_args()

    b = pd.read_csv(Path(a.beltran).expanduser(), usecols=["uniprot", "variant", a.obs_col, a.shaopei_col],
                    low_memory=False).drop_duplicates(["uniprot", "variant"])
    data = Path(a.data_dir).expanduser()
    print(f"Beltrán: {len(b):,} variantes en {b.uniprot.nunique()} proteínas")

    parts = []
    for u in sorted(b.uniprot.unique()):
        f = data / u / f"{u}_variants.tsv"
        if not f.exists():
            continue
        if a.pred_col not in pd.read_csv(f, sep="\t", nrows=0).columns:
            continue
        d = pd.read_csv(f, sep="\t", usecols=["RefAA", "PositionAA", "AltAA", a.pred_col], low_memory=False)
        d["uniprot"] = u
        d["variant"] = d.RefAA.astype(str) + d.PositionAA.astype(str) + d.AltAA.astype(str)
        parts.append(d[["uniprot", "variant", a.pred_col]].drop_duplicates("variant"))
    rel = pd.concat(parts, ignore_index=True).rename(columns={a.pred_col: "pred_aitana"})
    m = b.merge(rel, on=["uniprot", "variant"], how="left").rename(
        columns={a.obs_col: "obs", a.shaopei_col: "pred_shaopei"})

    rows = []
    for u, g in m.groupby("uniprot"):
        r_s, n_s = pearson(g.obs, g.pred_shaopei)
        r_a, n_a = pearson(g.obs, g.pred_aitana)
        r_sa, _ = pearson(g.pred_shaopei, g.pred_aitana)
        rows.append(dict(protein=u, n_variants=len(g), n_shaopei=n_s, n_aitana=n_a,
                         pearson_obs_vs_pred_shaopei=r_s, pearson_obs_vs_pred_aitana=r_a,
                         pearson_shaopei_vs_aitana=r_sa))
    t = pd.DataFrame(rows)
    out = Path(a.out_prefix).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t.round(3).to_csv(f"{out}.csv", index=False)

    ok = t.dropna(subset=["pearson_obs_vs_pred_aitana"])
    print(f"Proteínas con predicción tuya ({a.pred_col}): {len(ok)} de {len(t)}")
    for c in ["pearson_obs_vs_pred_shaopei", "pearson_obs_vs_pred_aitana", "pearson_shaopei_vs_aitana"]:
        print(f"  {c:<30} mediana {t[c].median():.3f}  (n proteínas={t[c].notna().sum()})")

    with PdfPages(f"{out}.pdf") as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        ax[0].scatter(ok.pearson_obs_vs_pred_shaopei, ok.pearson_obs_vs_pred_aitana, s=10, alpha=.7)
        lim = [np.nanmin(ok[["pearson_obs_vs_pred_shaopei", "pearson_obs_vs_pred_aitana"]].values) - .05, 1]
        ax[0].plot(lim, lim, ls="--", color="grey", lw=.7)
        ax[0].set_xlabel("Pearson obs vs pred (Shaopei)"); ax[0].set_ylabel(f"Pearson obs vs pred (Aitana, {a.pred_col})")
        ax[0].set_title("Per-protein correlation with observed fitness")
        ax[1].hist(t.pearson_shaopei_vs_aitana.dropna(), bins=40)
        ax[1].set_xlabel("Pearson Shaopei vs Aitana (per protein)"); ax[1].set_ylabel("Proteins")
        ax[1].set_title("Agreement between predictions")
        fig.suptitle(f"Beltrán benchmark – {len(ok)} proteins"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        prots = list(ok.protein)
        for i in range(0, len(prots), a.per_page):
            chunk = prots[i:i + a.per_page]
            fig, axes = plt.subplots(len(chunk), 2, figsize=(8, 2.6 * len(chunk)), squeeze=False)
            for k, u in enumerate(chunk):
                g, row = m[m.uniprot == u], t[t.protein == u].iloc[0]
                for j, (col, who, r) in enumerate([("pred_shaopei", "Shaopei", row.pearson_obs_vs_pred_shaopei),
                                                   ("pred_aitana", "Aitana", row.pearson_obs_vs_pred_aitana)]):
                    axx = axes[k, j]
                    axx.scatter(g[col], g.obs, s=3, alpha=.4, rasterized=True)
                    axx.set_title(f"{u} – {who}  r={r:.2f}", fontsize=8)
                    axx.set_xlabel(f"pred ({who})", fontsize=7); axx.set_ylabel("obs", fontsize=7)
                    axx.tick_params(labelsize=6)
            fig.tight_layout(); pdf.savefig(fig, dpi=150); plt.close(fig)

    print(f"\nGuardado: {out}.csv y {out}.pdf")


if __name__ == "__main__":
    main()