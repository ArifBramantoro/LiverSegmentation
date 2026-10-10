"""Find repeated scans inside the MedSeg collection and remove their effect on the detector evaluation.

Two MedSeg volumes are treated as the same scan when they have the same number of slices and pixel spacing
and identical scan signatures (the same best-matching LiTS volume with the same correlation in
data/overlap_scores.csv, written by overlap_check.py). For each group the lowest case number is kept.

Detector evaluation (the trained models are unchanged):
  * held-out protocol: MedSeg test cases with a copy in the training or validation partition are dropped;
  * leave-one-source-out with MedSeg unseen: each scan is counted once (the kept copy).
The original files are kept as test_predictions_all.csv and detector_metrics_all.csv.

Run:  python medseg_duplicates.py --overlap data/overlap_scores.csv --manifest data/manifest.csv --splits splits.csv --runs runs
"""
import argparse, json, os, shutil
import pandas as pd
from metrics import det_summary


def find_groups(overlap, manifest):
    o = overlap[overlap.source == "MedSeg"].copy()
    o["case_id"] = o.case.map(lambda n: f"MedSeg:{int(n):03d}")
    m = manifest[manifest.source == "MedSeg"].groupby("case_id").agg(slices=("z", "size"), sy=("spacing_y", "first"), sx=("spacing_x", "first"))
    o = o.join(m, on="case_id")
    o["key"] = list(zip(o.best_lits, o.corr24.round(4), o.slices, o.sy.round(4), o.sx.round(4)))
    groups = [sorted(g.case_id) for _, g in o.groupby("key") if len(g) > 1]
    return sorted(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", default="data/overlap_scores.csv"); ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--splits", default="splits.csv"); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest); n_medseg = man[man.source == "MedSeg"].case_id.nunique()
    groups = find_groups(pd.read_csv(a.overlap), man)
    split = pd.read_csv(a.splits).set_index("case_id").split.to_dict()
    rows = [dict(group=i + 1, case_id=c, kept=(c == g[0]), split=split.get(c)) for i, g in enumerate(groups) for c in g]
    D = pd.DataFrame(rows, columns=["group", "case_id", "kept", "split"]); D.to_csv(os.path.join(a.out, "medseg_duplicates.csv"), index=False)
    second = sorted(D[~D.kept].case_id)
    leak_test = sorted(c for g in groups for c in g if split.get(c) == "test" and any(split.get(o) in ("train", "val") for o in g if o != c))
    leak_val = sorted(c for g in groups for c in g if split.get(c) == "val" and any(split.get(o) == "train" for o in g if o != c))

    def redo(run, drop):
        d = os.path.join(a.runs, run); p = os.path.join(d, "test_predictions.csv")
        if not os.path.exists(p): return None
        allp = os.path.join(d, "test_predictions_all.csv")
        if not os.path.exists(allp): shutil.copy(p, allp); shutil.copy(os.path.join(d, "detector_metrics.csv"), os.path.join(d, "detector_metrics_all.csv"))
        te = pd.read_csv(allp); keep = te[~te.case_id.isin(drop)]
        t = json.load(open(os.path.join(d, "det_log.json")))["youden_threshold"]
        res = [dict(source=s, rule=r, **det_summary(g, tt)) for s, g in keep.groupby("source") for r, tt in [("0.5", 0.5), ("youden_val", t)]]
        keep.to_csv(p, index=False); pd.DataFrame(res).to_csv(os.path.join(d, "detector_metrics.csv"), index=False)
        return len(te) - len(keep)

    n_h = redo("det_heldout", leak_test); n_l = redo("det_loso_MedSeg", second)
    msg = (f"MedSeg repeated scans: {len(groups)} groups {[g for g in groups]}; second copies: {len(second)} {second}; "
           f"unique MedSeg scans: {n_medseg - len(second)} of {n_medseg}; "
           f"held-out test cases with a copy in training or validation (dropped): {len(leak_test)} {leak_test} ({n_h} slices); "
           f"validation cases with a copy in training: {len(leak_val)} {leak_val}; "
           f"leave-one-source-out MedSeg slices dropped (second copies): {n_l}")
    print(msg); open(os.path.join(a.out, "medseg_duplicates.txt"), "w").write(msg + "\n")


if __name__ == "__main__":
    main()
