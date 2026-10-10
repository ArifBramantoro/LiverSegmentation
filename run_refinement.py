"""Apply every refinement configuration to every liver-containing test slice and write one row per
(slice, method) with Dice, IoU, HD95, ASSD and wall-clock time. Inputs are deterministic (no augmentation).
Run: python run_refinement.py --unet-dir runs/unet --out runs/refine --workers 8 [--legacy 1]"""
import argparse, os, time, json, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from level_set import refine, legacy_refine, texture_force, dilate, DEFAULTS, VARIANTS
from metrics import slice_metrics
from data_utils import key2file

ap = argparse.ArgumentParser()
ap.add_argument("--unet-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--legacy", type=int, default=1, help="also run the original update rules (for explanation)")
ap.add_argument("--params", default="", help="JSON dict overriding level_set.DEFAULTS")
ap.add_argument("--pixel-units", type=int, default=0, help="report HD95/ASSD in pixels (needed if any spacing is unknown)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
P = dict(DEFAULTS); P.update(json.loads(a.params) if a.params else {})
T = pd.read_csv(os.path.join(a.unet_dir, "test_table.csv"))
if a.limit: T = T.iloc[:a.limit]


def one(i):
    r = T.iloc[i]; rows = []
    img = cv2.imread(r.image, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    gt = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 0
    unet = np.load(os.path.join(a.unet_dir, "pred", key2file(r.slice_key))).astype(bool)
    sp = (1.0, 1.0) if a.pixel_units or pd.isna(r.spacing_y) else (r.spacing_y, r.spacing_x)
    base = dict(case_id=r.case_id, source=r.source, z=int(r.z), slice_key=r.slice_key, gt_px=int(gt.sum()))
    rows.append(dict(base, method="unet", seconds=0.0, iterations=0, **slice_metrics(unet, gt, sp)))
    t0 = time.time(); tex = texture_force(img, dilate(unet, P["constraint_radius"]), P) if unet.any() else None
    t_glcm = time.time() - t0
    for v in VARIANTS:
        t0 = time.time(); out, info = refine(img, unet, v, P, texture=tex)
        dt = time.time() - t0 + (t_glcm if "texture" in v else 0.0)
        rows.append(dict(base, method=v, seconds=dt, iterations=info["iterations"], **slice_metrics(out, gt, sp)))
    if a.legacy:
        for v in ["curvature", "edge", "region"]:
            t0 = time.time(); out = legacy_refine(img, unet, v)
            rows.append(dict(base, method="legacy_" + v, seconds=time.time() - t0, iterations=200, **slice_metrics(out, gt, sp)))
    return rows


if __name__ == "__main__":
    with Pool(a.workers) as pool:
        res = []
        for k, rows in enumerate(pool.imap(one, range(len(T)), chunksize=4)):
            res.extend(rows)
            if (k + 1) % 100 == 0: print(f"{k + 1}/{len(T)} slices")
    df = pd.DataFrame(res); df.to_csv(os.path.join(a.out, "per_slice_results.csv"), index=False)
    json.dump(P, open(os.path.join(a.out, "refinement_params.json"), "w"), indent=2, default=list)
    print(df.groupby("method")[["dice", "iou", "hd95_mm", "assd_mm", "seconds"]].median())
