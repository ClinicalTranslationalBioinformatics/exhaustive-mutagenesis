"""
Compare your QAFImt 2.0 output on the validation proteins with ours.

    python run_qafimt2.py --input-dir ../validation/input --output-dir ../validation/mine
    python validate.py    --mine ../validation/mine

For each protein and each column (QAFImt2, qafisplit2_mt2, qafisplit3_mt2) it
reports the Pearson r between yours and ours, the largest absolute difference and
the share of variants that agree to 1e-6.

PASS = r >= 0.9999 for every protein and column.

Why not "identical": the models are XGBoost, and on a different CPU or XGBoost
build a few variants that sit right on a split threshold can go the other way.
On our machine the result is identical with 1, 2, 4 or 8 threads; between
machines expect most variants to match to 1e-6 and a few to differ slightly.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECTED = HERE.parent / "validation" / "expected"
COLS = ["QAFImt2", "qafisplit2_mt2", "qafisplit3_mt2"]
R_MIN = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True, help="folder with your *_qafimt2.csv outputs")
    a = ap.parse_args()
    mine = Path(a.mine)

    rows, ok_all = [], True
    print(f"{'protein':<9}{'column':<17}{'n':>7}{'r':>10}{'max |diff|':>12}{'agree':>9}")
    for f in sorted(EXPECTED.glob("*_qafimt2.csv")):
        theirs = mine / f.name
        if not theirs.exists():
            print(f"{f.name.split('_')[0]:<9}missing: {theirs}")
            ok_all = False
            continue
        m = pd.read_csv(f).merge(pd.read_csv(theirs), on=["uniprot", "variant"],
                                 suffixes=("_ours", "_yours"))
        for c in COLS:
            s = m[[f"{c}_ours", f"{c}_yours"]].dropna()
            x, y = s.iloc[:, 0].to_numpy(), s.iloc[:, 1].to_numpy()
            r = np.corrcoef(x, y)[0, 1] if len(s) > 2 else np.nan
            d = np.abs(x - y)
            ok = bool(r >= R_MIN)
            ok_all &= ok
            rows.append({"protein": f.name.split("_")[0], "column": c, "n": len(s), "r": r,
                         "max_abs_diff": d.max(), "share_agree_1e-6": (d < 1e-6).mean(),
                         "pass": ok})
            print(f"{rows[-1]['protein']:<9}{c:<17}{len(s):>7,}{r:>10.6f}{d.max():>12.1e}"
                  f"{100 * (d < 1e-6).mean():>8.1f}%")
    pd.DataFrame(rows).to_csv(mine / "validation_report.csv", index=False)
    print(f"\n{'PASS' if ok_all else 'FAIL'} (r >= {R_MIN} for every protein and column)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
