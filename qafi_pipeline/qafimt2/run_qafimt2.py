"""
Run QAFImt 2.0 on the proteome.

Trains the 60 models once, then writes one output file per protein.

    # every protein under one or more folders laid out as <ID>/<ID>_final2.csv
    python run_qafimt2.py --input-dir proteome_final2 --input-dir proteome_final2_part2 \
                          --output-dir qafimt2_output

    # a single file
    python run_qafimt2.py --input Q96BM1/Q96BM1_final2.csv --output-dir qafimt2_output

Output: <output-dir>/<ID>_qafimt2.csv with columns
    uniprot, variant, pos, QAFImt2, qafisplit2_mt2, qafisplit3_mt2

Proteins whose output already exists are skipped, so a stopped run can simply be
restarted. Proteins that fail are listed in <output-dir>/failed.txt with the reason.
"""
import argparse
import sys
import time
from pathlib import Path

import qafimt2 as Q


def find_inputs(dirs, files):
    paths = [Path(f) for f in files]
    for d in dirs:
        paths += sorted(Path(d).glob("*/*_final2.csv"))
    return paths


def main():
    ap = argparse.ArgumentParser(description="QAFImt 2.0 (mt by XGBoost) + qafisplit2/3 on it")
    ap.add_argument("--input-dir", action="append", default=[],
                    help="folder of <ID>/<ID>_final2.csv files (repeatable)")
    ap.add_argument("--input", action="append", default=[], help="a single feature CSV (repeatable)")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    inputs = find_inputs(a.input_dir, a.input)
    if not inputs:
        sys.exit("no input files found")
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in inputs
            if not (out_dir / f"{p.name.replace('_final2.csv', '').replace('.csv', '')}_qafimt2.csv").exists()]
    print(f"{len(inputs):,} input files, {len(inputs) - len(todo):,} already done, {len(todo):,} to run")
    if not todo:
        return

    print("training the 60 models (once) ...")
    t0 = time.time()
    trained = Q.train()
    print(f"trained in {time.time() - t0:.0f} s\n")

    failed = open(out_dir / "failed.txt", "a")
    for i, path in enumerate(todo, 1):
        name = path.name.replace("_final2.csv", "").replace(".csv", "")
        try:
            res = Q.predict(trained, Q.read_features(path))
            res.to_csv(out_dir / f"{name}_qafimt2.csv", index=False)
            n_nan = int(res["QAFImt2"].isna().sum())
            note = f"  ({n_nan:,} variants NaN: missing/non-finite position features)" if n_nan else ""
            print(f"[{i:,}/{len(todo):,}] {name}: {len(res):,} variants{note}", flush=True)
        except Exception as e:                                   # keep going, log it
            failed.write(f"{name}\t{e}\n"); failed.flush()
            print(f"[{i:,}/{len(todo):,}] {name}: FAILED - {e}", flush=True)
    failed.close()
    print("\ndone")


if __name__ == "__main__":
    main()
