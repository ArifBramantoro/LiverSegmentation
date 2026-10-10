"""Patient-level split: every slice of a case goes to exactly one partition (train/val/test).
LiTS and MSD Task03 share patients, so MSD is not used as a separate source."""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", default="splits.csv")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--fractions", default="0.7,0.1,0.2")
a = ap.parse_args()
fr = [float(x) for x in a.fractions.split(",")]
m = pd.read_csv(a.manifest); rng = np.random.default_rng(a.seed); rows = []
for src, g in m.groupby("source"):
    cases = np.array(sorted(g.case_id.unique())); rng.shuffle(cases)
    n = len(cases); n_tr = int(round(fr[0] * n)); n_va = max(1, int(round(fr[1] * n)))
    for i, c in enumerate(cases):
        rows.append(dict(case_id=c, source=src, split="train" if i < n_tr else "val" if i < n_tr + n_va else "test"))
s = pd.DataFrame(rows); s.to_csv(a.out, index=False)
print(s.groupby(["source", "split"]).size().unstack(fill_value=0))
