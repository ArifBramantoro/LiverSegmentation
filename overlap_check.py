"""Re-run only the scan-matching checks (MSD Task03 and MedSeg against LiTS) and record the scores.

Writes data/msd_lits_overlap.txt, data/medseg_lits_overlap.txt and data/overlap_scores.csv.
Run:  python overlap_check.py --lits D:\\mnt\\Liver_Segmentation_2\\LITS17 --msd D:\\mnt\\Liver_Segmentation_2\\Task03_Liver
                              --medseg D:\\mnt\\Liver_Segmentation_2\\medseg
"""
import argparse, os, time
import numpy as np
import pandas as pd
from prepare_data import nii, num, raw_variants, raw_shape, medseg_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lits", required=True); ap.add_argument("--msd", required=True); ap.add_argument("--medseg", required=True)
    ap.add_argument("--out", default="data"); ap.add_argument("--thr", type=float, default=0.99)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    lits = {num(f): f for f in nii(os.path.join(a.lits, "volume-*.nii*"))}
    keys = sorted(lits)
    print(f"LiTS volumes: {len(keys)}; computing signatures ...", flush=True)
    S = np.stack([raw_variants(lits[k])[0] for k in keys])
    shapes = {k: raw_shape(lits[k]) for k in keys}

    def score(path):
        V = raw_variants(path); R = V @ S.T
        j = int(R.max(0).argmax()); k = keys[j]; vi = int(R[:, j].argmax())
        same = raw_shape(path) == shapes[k]
        c48 = float(raw_variants(path, 48)[vi] @ raw_variants(lits[k], 48)[0]) if same else np.nan
        match = bool(R[vi, j] > a.thr and same and c48 > a.thr)
        return dict(best_lits=k, corr24=round(float(R[vi, j]), 4), same_dims=same, corr48=round(c48, 4), match=match)

    rows = []
    msd_all = sorted(nii(os.path.join(a.msd, "imagesTr", "liver_*.nii*")), key=num)
    uniq = {}
    for f in sorted(msd_all, key=lambda f: (num(f), not f.lower().endswith(".nii.gz"))):
        uniq.setdefault(num(f), f)                     # one file per case; the distributed .nii.gz copy if present
    for i, n in enumerate(sorted(uniq)):
        r = dict(source="MSD", case=n, file=os.path.basename(uniq[n]), **score(uniq[n])); rows.append(r)
        print(f"MSD {i + 1}/{len(uniq)} liver_{n}: best LiTS {r['best_lits']}, corr {r['corr24']}/{r['corr48']}, match {r['match']}", flush=True)
    imgs, _ = medseg_pairs(a.medseg)
    for i, f in enumerate(sorted(imgs, key=num)):
        r = dict(source="MedSeg", case=num(f), file=os.path.basename(f), **score(f)); rows.append(r)
        print(f"MedSeg {i + 1}/{len(imgs)} {r['file']}: best LiTS {r['best_lits']}, corr {r['corr24']}/{r['corr48']}, match {r['match']}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "overlap_scores.csv"), index=False)

    m = df[df.source == "MSD"]; mm = m[m.match]
    same_num = int((mm.case == mm.best_lits).sum()); unmatched = sorted(m[~m.match].case)
    msg = (f"MSD Task03 training files: {len(msd_all)}; unique cases: {len(m)}; matching a LiTS training volume "
           f"(any orientation): {len(mm)}; of these, same case number: {same_num}; unmatched MSD case numbers: {unmatched}; "
           f"lowest corr24 among matches: {mm.corr24.min() if len(mm) else 'n/a'}")
    print(msg); open(os.path.join(a.out, "msd_lits_overlap.txt"), "w").write(msg + "\n")
    g = df[df.source == "MedSeg"]; gm = g[g.match]
    msg2 = (f"MedSeg volumes: {len(g)}; matching a LiTS volume (excluded): {len(gm)} "
            + str({int(r.case): int(r.best_lits) for r in gm.itertuples()})
            + f"; highest corr24 of any MedSeg volume with any LiTS volume: {g.corr24.max()}")
    print(msg2); open(os.path.join(a.out, "medseg_lits_overlap.txt"), "w").write(msg2 + "\n")
    if len(gm):
        print("WARNING: some MedSeg volumes now match LiTS. Stop here and tell Claude before doing anything else.")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Please upload data/msd_lits_overlap.txt, "
          "data/medseg_lits_overlap.txt and data/overlap_scores.csv")


if __name__ == "__main__":
    main()
