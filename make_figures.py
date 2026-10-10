"""Figures for the revised manuscript, written as 300 dpi LZW TIFF (PLOS requirement).
  python make_figures.py static                       -> Fig 1 (pipeline), Fig 2 (U-Net), S1 Fig (solver verification)
  python make_figures.py qualitative --unet-dir runs/unet --results runs/refine/per_slice_results.csv
  python make_figures.py detector --det-dirs runs/det_heldout runs/det_loso_CHAOS ...
"""
import argparse, os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
OUT = "figures"; os.makedirs(OUT, exist_ok=True)
save = lambda fig, n: (fig.savefig(os.path.join(OUT, n), dpi=300, pil_kwargs={"compression": "tiff_lzw"}),
                       fig.savefig(os.path.join(OUT, n.replace(".tif", ".png")), dpi=150), plt.close(fig))


def box(ax, x, y, w, h, txt, fc="white", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 2.1)); ax.set_xlim(0, 15); ax.set_ylim(-0.1, 4.1); ax.axis("off")
    steps = ["CT volume\n(NIfTI or\nDICOM)", "Reorient,\nHU window\n[-100, 400],\nresize 256", "ResNet50\nslice\ndetector",
             "U-Net\nsegmentation", "Disc erosion\n(r = 2 px),\nsigned\ndistance", "Level set\nrefinement\n(4 force\nfamilies)",
             "Restack\nslices,\nper-case 3D\nmetrics"]
    xs = np.linspace(0.1, 13.1, len(steps)); w = 1.75
    for i, (x, t) in enumerate(zip(xs, steps)):
        box(ax, x, 1.5, w, 1.7, t, fc="0.92" if i in (2, 3) else "white", fs=6.0)
        if i: arrow(ax, xs[i - 1] + w, 2.35, x, 2.35)
    box(ax, xs[5] - 0.25, 0.0, 2.25, 1.05, "GLCM contrast and\nhomogeneity maps\n(w = 9, 4 angles)", fs=6.0)
    arrow(ax, xs[5] + w / 2, 1.05, xs[5] + w / 2, 1.5)
    ax.text(xs[2] + w / 2, 3.4, "non-liver slices\nreceive an empty mask", ha="center", fontsize=6, style="italic")
    ax.text(xs[4] + w / 2, 3.4, "U-Net mask dilated (r = 12 px)\ndefines the constraint band", ha="center", fontsize=6, style="italic")
    save(fig, "Fig1.tif")


def fig_unet():
    """Architecture drawn from the study code (common.UNet); 7,701,825 trainable parameters."""
    fig, ax = plt.subplots(figsize=(7.5, 3.3)); ax.set_xlim(0, 20); ax.set_ylim(0, 9); ax.axis("off")
    ch = [64, 128, 256]; sz = [256, 128, 64]; ys = [7.2, 5.2, 3.2]; yb = 1.3
    wid = lambda c: 0.22 + 0.16 * np.log2(c / 32)
    hei = lambda s: 0.5 + 1.1 * (s / 256)
    def rect(x, y, c, s, fc, label=True):
        w, h = wid(c), hei(s); ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, fc=fc, ec="black", lw=0.7))
        if label: ax.text(x + w / 2, y + h / 2 + 0.08, str(c), ha="center", va="bottom", fontsize=6)
        return x + w
    # input
    ax.add_patch(plt.Rectangle((0.3, ys[0] - hei(256) / 2), 0.12, hei(256), fc="0.8", ec="black", lw=0.7))
    ax.text(0.36, ys[0] + hei(256) / 2 + 0.08, "1", ha="center", va="bottom", fontsize=6)
    ax.text(0.36, ys[0] - hei(256) / 2 - 0.15, "input", ha="center", va="top", fontsize=6)
    xe = [1.0, 2.6, 4.2]; xd = [17.2, 15.6, 14.0]; ends = []
    for i in range(3):
        a1 = rect(xe[i], ys[i], ch[i], sz[i], "0.85"); a2 = rect(a1 + 0.3, ys[i], ch[i], sz[i], "0.85"); ends.append(a2)
        arrow(ax, (0.42 if i == 0 else a1), ys[i], (xe[0] if i == 0 else a1 + 0.3), ys[i]) if i == 0 else arrow(ax, a1, ys[i], a1 + 0.3, ys[i])
        if i: arrow(ax, a1, ys[i], a1 + 0.3, ys[i])
        ax.text(xe[i] + wid(ch[i]) / 2, ys[i] - hei(sz[i]) / 2 - 0.1, f"{sz[i]}$^2$", ha="center", va="top", fontsize=5.5)
        d1 = rect(xd[i], ys[i], ch[i], sz[i], "white"); d2 = rect(d1 + 0.3, ys[i], ch[i], sz[i], "white"); arrow(ax, d1, ys[i], d1 + 0.3, ys[i])
        arrow(ax, a2, ys[i], xd[i], ys[i]); ax.text((a2 + xd[i]) / 2, ys[i] + 0.12, "copy and concatenate", ha="center", fontsize=5.5, color="0.35")
        # down / up arrows
        nxt_y = ys[i + 1] if i < 2 else yb; nxt_x = xe[i + 1] if i < 2 else 6.0
        arrow(ax, a2 - 0.1, ys[i] - hei(sz[i]) / 2, nxt_x + 0.1, nxt_y + hei(sz[i] // 2) / 2)
    arrow(ax, 0.42, ys[0], xe[0], ys[0])
    b1 = rect(6.0, yb, 512, 32, "0.6"); b2 = rect(b1 + 0.3, yb, 512, 32, "0.6"); arrow(ax, b1, yb, b1 + 0.3, yb)
    ax.text(6.0 + wid(512) / 2, yb - hei(32) / 2 - 0.1, "32$^2$", ha="center", va="top", fontsize=5.5)
    ups = [(b2 - 0.1, yb + hei(32) / 2, xd[2] + 0.1, ys[2] - hei(64) / 2)]
    for i in (2, 1):
        ups.append((xd[i] + 2 * wid(ch[i]) + 0.2, ys[i] + hei(sz[i]) / 2, xd[i - 1] + 0.1, ys[i - 1] - hei(sz[i - 1]) / 2))
    for u in ups: arrow(ax, *u)
    xo = xd[0] + 2 * wid(64) + 0.3 + 0.35
    ax.add_patch(plt.Rectangle((xo, ys[0] - hei(256) / 2), 0.12, hei(256), fc="0.8", ec="black", lw=0.7))
    arrow(ax, xo - 0.35, ys[0], xo, ys[0]); ax.text(xo + 0.06, ys[0] + hei(256) / 2 + 0.08, "1", ha="center", va="bottom", fontsize=6)
    ax.text(xo + 0.06, ys[0] - hei(256) / 2 - 0.15, "sigmoid\nmask", ha="center", va="top", fontsize=6)
    leg = ["Grey: encoder; white: decoder; dark grey: bottleneck. Numbers above blocks: channels; left: spatial size.",
           "Each block: 3 x 3 convolution, batch normalisation, ReLU. Down: 2 x 2 max pooling. Up: 2 x 2 transposed convolution.",
           "Output: 1 x 1 convolution and sigmoid. Trainable parameters: 7,701,825."]
    ax.set_ylim(0.4, 8.2)
    save(fig, "Fig2.tif")


def fig_s1_verification():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from level_set import refine, legacy_refine, curvature, sdf
    from scipy.ndimage import gaussian_filter
    N = 256; yy, xx = np.mgrid[:N, :N]; disc = lambda r: ((yy - 128) ** 2 + (xx - 128) ** 2) <= r * r
    # (A) curvature flow radius vs time: corrected vs analytic vs original rule
    r0, g, dt = 40.0, 1.0, 0.2; its = np.arange(0, 401, 50)
    r_c = [np.sqrt(refine(np.zeros((N, N)), disc(r0), "curvature_only", dict(erosion_radius=0, gamma=g, dt=dt, iterations=int(n) if n else 1, tol=0))[0].sum() / np.pi) if n else r0 for n in its]
    phi = sdf(disc(r0)); r_l = [r0]
    for k in range(1, its[-1] + 1):
        phi = phi + dt * (-g * curvature(phi))
        if k % 50 == 0: r_l.append(np.sqrt((phi < 0).sum() / np.pi))
    t = its * dt
    # (B) original edge rule applied to a perfect mask on a synthetic disc image
    rng = np.random.default_rng(0); img = np.clip(gaussian_filter(np.where(disc(40), 0.6, 0.35) + 0.03 * rng.standard_normal((N, N)), 1), 0, 1)
    leg = legacy_refine(img, disc(40), "edge")
    under = disc(34); new_edge = refine(img, under, "edge")[0]
    fig, axs = plt.subplots(1, 4, figsize=(7.5, 2.2), gridspec_kw=dict(width_ratios=[1.15, 0.9, 0.9, 1.15]))
    axs[0].plot(t, np.sqrt(np.maximum(r0 ** 2 - 2 * g * t, 0)), "k-", lw=1, label="analytic, $r^2=r_0^2-2\\gamma t$")
    axs[0].plot(t, r_c, "ko", ms=3, label="corrected solver"); axs[0].plot(t, r_l, "k^", ms=3, mfc="white", label="original rule")
    axs[0].set_xlabel("time t"); axs[0].set_ylabel("disc radius (px)"); axs[0].legend(fontsize=5.5, frameon=False); axs[0].set_title("A  Curvature flow", loc="left", fontsize=7)
    for ax, m, ttl in [(axs[1], leg, "B  Original edge rule"), (axs[2], new_edge, "C  Corrected edge model")]:
        ax.imshow(img[60:196, 60:196], cmap="gray", vmin=0, vmax=1); ax.contour(disc(40)[60:196, 60:196], [0.5], colors="w", linewidths=0.8)
        ax.contour(m[60:196, 60:196], [0.5], colors="k", linewidths=1.0, linestyles="--"); ax.axis("off"); ax.set_title(ttl, loc="left", fontsize=7)
    d = lambda a, b: 2 * (a & b).sum() / (a.sum() + b.sum())
    axs[1].text(2, 130, f"Dice {d(leg.astype(bool), disc(40)):.3f}", color="w", fontsize=6)
    axs[2].text(2, 130, f"init Dice {d(under, disc(40)):.3f} -> {d(new_edge.astype(bool), disc(40)):.3f}", color="w", fontsize=6)
    # (D) sign of the original texture speed C - H inside homogeneous liver, by noise level and intensity window
    from glcm_fast import glcm_maps
    rng = np.random.default_rng(0); sds = [8, 12, 16, 20, 25, 30, 40]; curves = {}
    for lo, hi in ((-1000, 1000), (-100, 400)):
        vals = []
        for sd in sds:
            z = gaussian_filter(rng.normal(0, 1, (160, 160)), 0.6); hu = 60 + z / z.std() * sd
            m = glcm_maps(np.clip((hu - lo) / (hi - lo), 0, 1), window=7, angles=(0,), levels=32)
            vals.append(float((m["contrast"] - m["homogeneity"])[10:-10, 10:-10].mean()))
        curves[(lo, hi)] = vals
    axs[3].axhline(0, color="0.6", lw=0.6)
    axs[3].plot(sds, curves[(-1000, 1000)], "k-o", ms=2.5, lw=1, label="[-1000, 1000] HU")
    axs[3].plot(sds, curves[(-100, 400)], "k--^", ms=2.5, lw=1, mfc="white", label="[-100, 400] HU")
    axs[3].set_xlabel("liver noise SD (HU)"); axs[3].set_ylabel("original speed C - H")
    axs[3].text(9, 3.0, "contour shrinks", fontsize=5.5); axs[3].text(9, -1.9, "contour grows", fontsize=5.5)
    axs[3].set_ylim(-2.4, 7); axs[3].legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    axs[3].set_title("D  Original texture speed", loc="left", fontsize=7)
    fig.tight_layout(); save(fig, "S1_Fig.tif")


QLAB = {"unet": "U-Net", "init_only": "Eroded init.", "texture": "GLCM texture", "edge": "Geodesic edge",
        "region": "Chan-Vese region", "legacy_edge": "Original edge rule"}


def fig_qualitative(unet_dir, results, show=("unet", "init_only", "texture", "edge", "region", "legacy_edge")):
    import cv2
    from level_set import refine, texture_force, dilate, DEFAULTS, legacy_refine
    from data_utils import key2file
    df = pd.read_csv(results); T = pd.read_csv(os.path.join(unet_dir, "test_table.csv")).set_index("slice_key")
    u = df[df.method == "unet"].sort_values("dice").reset_index(drop=True)
    picks = [u.iloc[int(q * (len(u) - 1))].slice_key for q in (0.10, 0.50, 0.90)]  # rule-based selection, not hand-picked
    cols = ["CT and reference", "GLCM contrast", "GLCM homogeneity"] + [QLAB[c] for c in show]
    fig, axs = plt.subplots(3, len(cols), figsize=(7.5, 2.75), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for r, key in enumerate(picks):
        row = T.loc[key]; img = cv2.imread(row.image, 0) / 255.0; gt = cv2.imread(row["mask"], 0) > 0
        un = np.load(os.path.join(unet_dir, "pred", key2file(key))).astype(bool)
        ys, xs = np.nonzero(dilate(un | gt, 30)); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        F, C, H = texture_force(img, dilate(un, DEFAULTS["constraint_radius"]), DEFAULTS)
        crop = lambda a: a[y0:y1, x0:x1]
        for c, name in enumerate(["ct", "con", "hom"] + list(show)):
            ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if name == "con": ax.imshow(crop(C), cmap="gray"); 
            elif name == "hom": ax.imshow(crop(H), cmap="gray")
            else:
                ax.imshow(crop(img), cmap="gray", vmin=0, vmax=1); ax.contour(crop(gt), [0.5], colors="#1baf7a", linewidths=0.7)
                if name != "ct":
                    m = un if name == "unet" else (legacy_refine(img, un, name.replace("legacy_", "")) if name.startswith("legacy") else refine(img, un, name)[0])
                    ax.contour(crop(m.astype(float)), [0.5], colors="#eb6834", linewidths=0.7)
                    dsc = float(df[(df.slice_key == key) & (df.method == name)].dice.iloc[0])
                    ax.text(0.03, 0.04, f"{dsc:.2f}", transform=ax.transAxes, color="white", fontsize=5.5,
                            bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
            if r == 0: ax.set_title(cols[c], fontsize=5.8, pad=2)
        axs[r, 0].set_ylabel(["10th pct", "50th pct", "90th pct"][r], fontsize=5.8)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.93, bottom=0.01)
    save(fig, "Fig_qualitative.tif")


SRC_ORDER = ["LiTS", "CHAOS", "MedSeg"]
SRC_COL = {"LiTS": "#2a78d6", "CHAOS": "#eb6834", "MedSeg": "#1baf7a"}   # validated categorical slots 1-3


def _proto(d):
    return "Held-out patients" if "heldout" in os.path.basename(d) else "Unseen source"


def fig_detector(det_dirs):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for d in det_dirs:
        p = pd.read_csv(os.path.join(d, "test_predictions.csv")); ls = "-" if _proto(d) == "Held-out patients" else "--"
        for s_ in [x for x in SRC_ORDER if x in set(p.source)]:
            g = p[p.source == s_]
            if g.has_liver.nunique() < 2: continue
            f, t, _ = roc_curve(g.has_liver, g.prob)
            ax.plot(f, t, ls, color=SRC_COL.get(s_, "0.3"), lw=1.2, label=f"{_proto(d)}: {s_} (AUC {roc_auc_score(g.has_liver, g.prob):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="0.6", lw=0.6)
    ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=5.2, frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "Fig_detector_roc.tif")
    m = pd.concat([pd.read_csv(os.path.join(d, "detector_metrics.csv")).assign(protocol=_proto(d)) for d in det_dirs])
    m = m[m.rule == "youden_val"].copy(); m["o"] = m.source.map(SRC_ORDER.index)
    m = pd.concat([g.sort_values("o") for _, g in m.groupby("protocol", sort=False)]); n = len(m)
    fig, axs = plt.subplots(1, n, figsize=(7.5, 1.75))
    for ax, (_, r) in zip(np.atleast_1d(axs), m.iterrows()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]]); frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Greys", vmin=0, vmax=1.4)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{int(v):,}\n({frac[i, j]:.1%})", ha="center", va="center", fontsize=5.5, color="white" if frac[i, j] > 0.7 else "black")
        ax.set_xticks([0, 1], ["No liver", "Liver"], fontsize=5.5); ax.set_yticks([0, 1], ["No liver", "Liver"], fontsize=5.5)
        ax.set_xlabel("Predicted", fontsize=6); ax.set_title(f"{r.protocol}:\n{r.source}", fontsize=6)
    np.atleast_1d(axs)[0].set_ylabel("Reference", fontsize=6)
    fig.tight_layout(pad=0.4); save(fig, "Fig_confusion.tif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--unet-dir"); ap.add_argument("--results"); ap.add_argument("--det-dirs", nargs="*")
    a = ap.parse_args()
    if a.what == "static": fig_pipeline(); fig_unet(); fig_s1_verification()
    elif a.what == "qualitative": fig_qualitative(a.unet_dir, a.results)
    elif a.what == "detector": fig_detector(a.det_dirs)
