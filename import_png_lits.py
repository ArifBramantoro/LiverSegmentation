"""Build the LiTS part of the manifest from an existing PNG export with the layout
    <root>/LITS/1_liver/volume_{v}_slice_{s}.png
    <root>/LITS/0_non-liver/volume_{v}_slice_{s}.png
    <root>/LITS_masks/volume_{v}/mask_{v}_slice_{s}.png      (liver slices only)
Use this only if the raw LiTS NIfTI volumes are unavailable. Pixel spacing is unknown for these
PNGs, so surface distances must be reported in pixels (run_refinement.py --pixel-units 1).

--rewindow maps grey levels back to HU assuming the export window [-1000, 1000] HU (as stated in the
thesis) and re-applies the abdominal window [-100, 400] HU. The script first checks this assumption:
the median implied HU inside liver masks must be liver-like (30 to 100 HU), otherwise it stops.
Run: python import_png_lits.py --root "D:/Master/Haziq/Another_attempt" --out data [--rewindow 1]
     then merge with CHAOS/MedSeg: python prepare_data.py --chaos ... --medseg ... --out data --append 1
"""
import argparse, glob, os, re, numpy as np, pandas as pd, cv2

EXPORT_MIN, EXPORT_MAX, WIN_MIN, WIN_MAX, SIZE = -1000.0, 1000.0, -100.0, 400.0, 256
pat = re.compile(r"volume_(\d+)_slice_(\d+)\.png$")


def to_hu(g): return g / 255.0 * (EXPORT_MAX - EXPORT_MIN) + EXPORT_MIN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rewindow", type=int, default=0); a = ap.parse_args()
    files = []
    for lab, sub in [(1, "1_liver"), (0, "0_non-liver")]:
        for f in glob.glob(os.path.join(a.root, "LITS", sub, "*.png")):
            m = pat.search(os.path.basename(f))
            if m: files.append((int(m.group(1)), int(m.group(2)), lab, f))
    print(f"found {len(files)} LiTS PNG slices from {len({v for v, *_ in files})} volumes")
    mask_of = lambda v, s: os.path.join(a.root, "LITS_masks", f"volume_{v}", f"mask_{v}_slice_{s}.png")
    if a.rewindow:   # verify the assumed export window on a sample of liver slices
        rng = np.random.default_rng(0); liv = [x for x in files if x[2] == 1]; vals = []
        for v, s, _, f in [liv[i] for i in rng.choice(len(liv), min(300, len(liv)), replace=False)]:
            g = cv2.imread(f, 0).astype(np.float32); mk = cv2.imread(mask_of(v, s), 0)
            if mk is None: continue
            mk = cv2.resize(mk, g.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            if mk.any(): vals.append(np.median(to_hu(g[mk])))
        med = float(np.median(vals)); print(f"median implied HU inside liver masks: {med:.1f}")
        if not 30 <= med <= 100:
            raise SystemExit("Export window assumption [-1000, 1000] HU not confirmed; run without --rewindow and report the original window.")
    rows = []
    for v, s, lab, f in sorted(files):
        g = cv2.imread(f, 0).astype(np.float32)
        img = np.clip((to_hu(g) - WIN_MIN) / (WIN_MAX - WIN_MIN), 0, 1) if a.rewindow else g / 255.0
        mk = cv2.imread(mask_of(v, s), 0) if lab == 1 else None
        mk = np.zeros(g.shape, np.uint8) if mk is None else (cv2.resize(mk, g.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR); mk = cv2.resize(mk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        d = os.path.join(a.out, "LiTS", f"{v:03d}"); os.makedirs(d + "/img", exist_ok=True); os.makedirs(d + "/mask", exist_ok=True)
        ip, mp = f"{d}/img/{s:04d}.png", f"{d}/mask/{s:04d}.png"
        cv2.imwrite(ip, np.round(img * 255).astype(np.uint8)); cv2.imwrite(mp, mk * 255)
        rows.append(dict(source="LiTS", case_id=f"LiTS:{v:03d}", z=s, image=ip, mask=mp, spacing_y=np.nan, spacing_x=np.nan,
                         has_liver=int(mk.any()), liver_px=int(mk.sum())))
    df = pd.DataFrame(rows); man = os.path.join(a.out, "manifest.csv")
    if os.path.exists(man): df = pd.concat([pd.read_csv(man).query("source != 'LiTS'"), df])
    df.to_csv(man, index=False)
    print(df.groupby("source").agg(cases=("case_id", "nunique"), slices=("z", "size"), liver_slices=("has_liver", "sum")))
    bad = df[(df.source == "LiTS") & (df.has_liver == 0) & df.image.str.contains("1_liver")]
    if len(bad): print(f"warning: {len(bad)} slices in 1_liver have empty masks")


if __name__ == "__main__":
    main()
