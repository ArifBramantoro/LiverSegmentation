"""Turn per-slice results into the tables and figures of the revised manuscript.
Outputs (in --out): seg_summary.csv, seg_table.tex, volumetric.csv, runtime.csv, fig_dice_hist.tif,
end_to_end.csv (if --det-pred and --unet-dir are given).
Run: python analyse.py --results runs/refine/per_slice_results.csv --out results [--unet-dir runs/unet --det-pred runs/det_heldout/test_predictions.csv --det-threshold 0.5]"""
import argparse, os, json, numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from stats_utils import holm, wilcoxon_paired, cluster_bootstrap_median_diff, med_iqr
from metrics import dice as dice_fn
from data_utils import key2file

ORDER = ["unet", "init_only", "curvature_only", "texture", "edge", "edge_texture", "region", "region_texture",
         "legacy_curvature", "legacy_edge", "legacy_region"]
LABEL = {"unet": "U-Net baseline", "init_only": "Eroded initialisation (no evolution)", "curvature_only": "Curvature flow only",
         "texture": "GLCM texture force", "edge": "Geodesic edge", "edge_texture": "Geodesic edge + GLCM",
         "region": "Chan--Vese region", "region_texture": "Chan--Vese region + GLCM",
         "legacy_curvature": "Original rule: texture (``curvature-only'')", "legacy_edge": "Original rule: edge",
         "legacy_region": "Original rule: region"}

ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--unet-dir"); ap.add_argument("--det-pred"); ap.add_argument("--det-threshold", type=float, default=0.5)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
df = pd.read_csv(a.results); methods = [m for m in ORDER if m in df.method.unique()]
case = df.groupby(["method", "case_id"])[["dice", "iou", "hd95_mm", "assd_mm"]].mean().reset_index()

rows, p_case, p_slice = [], [], []
for m in methods:
    s = df[df.method == m]; r = dict(method=m, label=LABEL[m], n_slices=len(s), n_cases=s.case_id.nunique(),
                                      empty_predictions=int(s.empty_pred.sum()))
    for c in ["dice", "iou", "hd95_mm", "assd_mm"]:
        md, q1, q3 = med_iqr(s[c]); r.update({f"{c}_median": md, f"{c}_q1": q1, f"{c}_q3": q3,
                                             f"{c}_mean": s[c].mean(), f"{c}_sd": s[c].std()})
    if m != "unet":
        est, lo, hi = cluster_bootstrap_median_diff(df, "dice", m, "unet"); r.update(d_dice=est, d_dice_lo=lo, d_dice_hi=hi)
        cu = case[case.method == "unet"].set_index("case_id"); cm = case[case.method == m].set_index("case_id")
        pc, rc = wilcoxon_paired(cm.loc[cu.index, "dice"], cu["dice"]); p_case.append(pc); r["rbc_case"] = rc
        su = df[df.method == "unet"].set_index("slice_key"); sm = s.set_index("slice_key")
        ps, rs = wilcoxon_paired(sm.loc[su.index, "dice"], su["dice"]); p_slice.append(ps); r["rbc_slice"] = rs
    rows.append(r)
S = pd.DataFrame(rows)
# Holm-Bonferroni within two families: corrected configurations, and diagnostic re-runs of the original rule
others = S[S.method != "unet"].copy(); others["p_case"] = p_case; others["p_slice"] = p_slice
for fam in (others.method.str.startswith("legacy"), ~others.method.str.startswith("legacy")):
    idx = others.index[fam]
    if len(idx):
        S.loc[idx, "p_case_holm"] = holm(others.loc[idx, "p_case"]); S.loc[idx, "p_slice_holm"] = holm(others.loc[idx, "p_slice"])
S["p_case_raw"] = others["p_case"]; S["p_slice_raw"] = others["p_slice"]
S.to_csv(os.path.join(a.out, "seg_summary.csv"), index=False)

def fp(p): return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
with open(os.path.join(a.out, "seg_table.tex"), "w") as f:
    for _, r in S.iterrows():
        d = "--" if r.method == "unet" else f"{r.d_dice:+.4f} [{r.d_dice_lo:+.4f}, {r.d_dice_hi:+.4f}]"
        f.write(f"{r.label} & {r.dice_median:.4f} [{r.dice_q1:.4f}--{r.dice_q3:.4f}] & {r.dice_mean:.4f} $\\pm$ {r.dice_sd:.4f} & "
                f"{r.iou_median:.4f} [{r.iou_q1:.4f}--{r.iou_q3:.4f}] & {r.hd95_mm_median:.2f} [{r.hd95_mm_q1:.2f}--{r.hd95_mm_q3:.2f}] & "
                f"{r.assd_mm_median:.2f} [{r.assd_mm_q1:.2f}--{r.assd_mm_q3:.2f}] & {d} & {fp(r.get('p_case_holm'))} \\\\\n")

# per-case volumetric Dice under oracle slice gating (liver-containing slices only)
if {"inter_px", "pred_px"}.issubset(df.columns):
    v = df.groupby(["method", "case_id"]).agg(i=("inter_px", "sum"), p=("pred_px", "sum"), g=("gt_px", "sum")).reset_index()
    v["vol_dice"] = 2 * v.i / (v.p + v.g); v.to_csv(os.path.join(a.out, "volumetric.csv"), index=False)

# runtime
rt = df.groupby("method").seconds.describe(percentiles=[0.5])[["mean", "50%", "std"]]; rt.to_csv(os.path.join(a.out, "runtime.csv"))

# histogram of per-slice Dice
show = [m for m in ["unet", "init_only", "curvature_only", "texture", "edge", "edge_texture", "region", "region_texture"] if m in methods]
fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.8), sharex=True, sharey=True)
for ax, m in zip(axes.ravel(), show):
    v = df[df.method == m].dice; ax.hist(v, bins=np.linspace(0, 1, 41), color="0.35")
    ax.axvline(v.median(), color="#eb6834", lw=1, ls="--"); ax.text(0.03, 0.9, f"median {v.median():.3f}", transform=ax.transAxes, fontsize=5.5)
    ax.set_title(LABEL[m].replace("--", "-").replace(" (no evolution)", ""), fontsize=7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=6)
for ax in axes[-1]: ax.set_xlabel("Per-slice Dice", fontsize=7)
for ax in axes[:, 0]: ax.set_ylabel("Slices", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_dice_hist.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})

# end-to-end: detector gating -> U-Net -> per-case volumetric Dice over all test slices
if a.unet_dir and a.det_pred:
    T = pd.read_csv(os.path.join(a.unet_dir, "test_table_all_slices.csv"))
    D = pd.read_csv(a.det_pred)[["slice_key", "prob"]]; T = T.merge(D, on="slice_key", how="left")
    out = []
    for c, g in T.groupby("case_id"):
        P, G, Pg, Po = [], [], [], []
        for _, r in g.iterrows():
            gt = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 0; u = np.load(os.path.join(a.unet_dir, "pred", key2file(r.slice_key))).astype(bool)
            G.append(gt); P.append(u); Pg.append(u if r.prob >= a.det_threshold else np.zeros_like(u)); Po.append(u if r.has_liver else np.zeros_like(u))
        missed = int(((g.prob < a.det_threshold) & (g.has_liver == 1)).sum())
        out.append(dict(case_id=c, source=g.source.iloc[0], slices=len(g), liver_slices=int(g.has_liver.sum()), missed_liver_slices=missed,
                        vol_dice_no_gating=dice_fn(np.stack(P), np.stack(G)), vol_dice_detector_gating=dice_fn(np.stack(Pg), np.stack(G)),
                        vol_dice_oracle_gating=dice_fn(np.stack(Po), np.stack(G))))
    pd.DataFrame(out).to_csv(os.path.join(a.out, "end_to_end.csv"), index=False)
print(S[["label", "n_slices", "n_cases", "dice_median", "dice_mean", "hd95_mm_median", "p_case_holm"]].to_string())
