"""
Validación QAFIMeta con CAGI7: compara los valores del release (nuevos) con los
que se enviaron a CAGI (all_output/<prot>/<prot>_allmethod_scaled.csv).

Para cada proteína (ARSA, ATP7B, LPL, TSC2) genera:
  - <PROT>_scatter.png : QAFIMeta v1..v4 nuevas vs QAFIsplit 3.0 y vs la versión enviada a CAGI
  - cagi_correlations.csv : Pearson / Spearman / n de cada comparación

Uso:
    python cagi_scatter.py \
        --release-dir ~/Desktop/human_proteome_2026/human_proteome_2026_release \
        --cagi-dir ~/Desktop/CAGI/all_output \
        --out-dir ~/Desktop/CAGI/validation_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROTEINS = {"arsa": "P15289", "atp7b": "P35670", "lpl": "P06858", "tsc2": "P49815"}
THREE2ONE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
             "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
             "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}

# columna nueva (release)  ->  columna enviada a CAGI (allmethod_scaled)
SUBMITTED = {
    "QAFIMeta_v1": "QAFIMeta2.0 v1 (scaled)",
    "QAFIMeta_v2": None,
    "QAFIMeta_v3": "QAFIMeta2_v3",
    "QAFIMeta_v4": "QAFIMeta2_v4",
}
REF_COL = "QAFIsplit 3.0"  # en CAGI; en el release es qafisplit3


def aa1(x):
    x = str(x).strip()
    return THREE2ONE.get(x.capitalize(), x)


def load_release(release_dir: Path, uniprot: str) -> pd.DataFrame:
    f = release_dir / uniprot / f"{uniprot}_variants_updated.tsv"
    if not f.exists():
        f = release_dir / uniprot / f"{uniprot}_variants.tsv"
    d = pd.read_csv(f, sep="\t")
    d["key"] = d["RefAA"].map(aa1) + d["PositionAA"].astype(int).astype(str) + d["AltAA"].map(aa1)
    return d


def load_cagi(cagi_dir: Path, prot: str) -> pd.DataFrame:
    d = pd.read_csv(cagi_dir / prot / f"{prot}_allmethod_scaled.csv")
    if "uniprot_variant" in d.columns:
        d["key"] = d["uniprot_variant"].astype(str).str.split("_").str[-1]
    else:
        # Sin columna uniprot_variant: se usa la primera columna (variant / aa_substitution)
        cands = ["variant", "Variant", "hgvs_pro", "aa_substitution"]
        col = next((c for c in cands if c in d.columns), d.columns[0])
        print(f"   {prot.upper()}: sin 'uniprot_variant', emparejando por '{col}'")
        d["key"] = d[col].astype(str).map(to_key1)
        n0 = len(d)
        d = d[d["key"].notna()].copy()
        print(f"   {prot.upper()}: {n0 - len(d)} sinónimas/stop descartadas (no son missense)")
    return d


def to_key1(s: str):
    """'p.Ala1002Arg' / 'Ala1002Arg' / 'A1002R' -> 'A1002R'. Sinónimas (=) y stop (Ter/*) -> None."""
    import re
    s = s.replace("p.", "").strip()
    m = re.fullmatch(r"([A-Za-z]{3})(\d+)([A-Za-z]{3})", s)
    if m:
        ref, pos, alt = m.groups()
        ref, alt = THREE2ONE.get(ref.capitalize()), THREE2ONE.get(alt.capitalize())
        return f"{ref}{pos}{alt}" if ref and alt else None
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", s)
    return s if m else None


def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return float("nan"), float("nan"), int(m.sum())
    return pearsonr(x[m], y[m])[0], spearmanr(x[m], y[m])[0], int(m.sum())


def scatter(ax, x, y, xl, yl):
    m = x.notna() & y.notna()
    ax.scatter(x[m], y[m], s=4, alpha=0.4, color="#3b6ea5", edgecolors="none")
    r, rho, n = corr(x, y)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(f"r={r:.2f}  ρ={rho:.2f}  n={n}", fontsize=8)
    ax.tick_params(labelsize=7)
    if m.any():
        lo, hi = min(x[m].min(), y[m].min()), max(x[m].max(), y[m].max())
        ax.plot([lo, hi], [lo, hi], lw=0.6, color="grey", ls="--")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--cagi-dir", required=True, help="Carpeta all_output de CAGI")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rel = Path(args.release_dir).expanduser()
    cagi = Path(args.cagi_dir).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for prot, uni in PROTEINS.items():
        try:
            r = load_release(rel, uni)
            c = load_cagi(cagi, prot)
        except Exception as e:
            print(f"[FAILED] {prot.upper()}: {e}")
            continue

        m = c.merge(r, on="key", how="inner")
        print(f"{prot.upper()}: {len(c)} variantes CAGI, {len(m)} emparejadas con el release")
        if len(m) < 0.9 * len(c):
            print(f"   ¡Ojo! Solo {len(m)/len(c)*100:.0f}% emparejadas. Ejemplos CAGI: {c['key'].head(3).tolist()} "
                  f"| release: {r['key'].head(3).tolist()}")
        if m.empty:
            print("   ¡0 emparejadas! Revisa formato de RefAA/PositionAA/AltAA.")
            continue

        metas = [v for v in SUBMITTED if v in m.columns and m[v].notna().any()]
        fig, axes = plt.subplots(2, len(metas), figsize=(3.2 * len(metas), 6.4), squeeze=False)
        for i, v in enumerate(metas):
            # Fila 1: nueva vs QAFIsplit 3.0 (release)
            scatter(axes[0, i], m["qafisplit3"], m[v], "qafisplit3 (release)", f"{v} (nueva)")
            r1, s1, n1 = corr(m["qafisplit3"], m[v])
            rows.append(dict(protein=prot.upper(), new=v, vs="qafisplit3", pearson=r1, spearman=s1, n=n1))

            # Fila 2: nueva vs enviada a CAGI
            sub = SUBMITTED[v]
            if sub and sub in m.columns:
                y_old = pd.to_numeric(m[sub], errors="coerce")
                scatter(axes[1, i], y_old, m[v], f"{sub} (CAGI)", f"{v} (nueva)")
                r2, s2, n2 = corr(y_old, m[v])
                rows.append(dict(protein=prot.upper(), new=v, vs=f"CAGI {sub}", pearson=r2, spearman=s2, n=n2))
            else:
                axes[1, i].axis("off")
                axes[1, i].text(0.5, 0.5, "no enviada a CAGI", ha="center", va="center", fontsize=8)

        fig.suptitle(f"{prot.upper()} ({uni}) — QAFIMeta nuevas", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{prot.upper()}_scatter.png", dpi=200)
        plt.close(fig)

    t = pd.DataFrame(rows)
    t.to_csv(out / "cagi_correlations.csv", index=False, float_format="%.3f")
    print("\n", t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nFiguras y tabla en {out}")


if __name__ == "__main__":
    main()