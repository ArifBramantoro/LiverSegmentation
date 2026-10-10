"""Smoke test on synthetic ellipses: checks that the scripts run end to end. Not study data."""
import os, numpy as np, pandas as pd, cv2, subprocess, sys
from scipy.ndimage import gaussian_filter
R = "/tmp/smoke"; os.makedirs(R, exist_ok=True); rng = np.random.default_rng(0); rows = []
yy, xx = np.mgrid[:256, :256]
for c in range(4):
    src = "LiTS" if c < 2 else "CHAOS"; cid = f"{src}:{c:03d}"
    d = os.path.join(R, src, f"{c:03d}"); os.makedirs(d + "/img", exist_ok=True); os.makedirs(d + "/mask", exist_ok=True)
    for z in range(6):
        a, b = 50 + 5 * z, 35 + 3 * c
        gt = (((yy - 120) / a) ** 2 + ((xx - 110) / b) ** 2) <= 1 if z < 5 else np.zeros((256, 256), bool)
        img = np.clip(gaussian_filter(np.where(gt, 0.6, 0.4) + 0.04 * rng.standard_normal((256, 256)), 1), 0, 1)
        ip, mp = f"{d}/img/{z:04d}.png", f"{d}/mask/{z:04d}.png"
        cv2.imwrite(ip, (img * 255).astype(np.uint8)); cv2.imwrite(mp, gt.astype(np.uint8) * 255)
        rows.append(dict(source=src, case_id=cid, z=z, image=ip, mask=mp, spacing_y=1.5, spacing_x=1.5, has_liver=int(gt.any()), liver_px=int(gt.sum())))
m = pd.DataFrame(rows); m.to_csv(f"{R}/manifest.csv", index=False)
pd.DataFrame(dict(case_id=m.case_id.unique(), source=[s.split(":")[0] for s in m.case_id.unique()], split="test")).to_csv(f"{R}/splits.csv", index=False)
# fake U-Net output: GT dilated by 3 px (over-segmentation) to exercise the refinement code
from data_utils import load_table, key2file
te = load_table(f"{R}/manifest.csv", f"{R}/splits.csv", "test", liver_only=True); ta = load_table(f"{R}/manifest.csv", f"{R}/splits.csv", "test")
u = f"{R}/unet"; os.makedirs(u + "/pred", exist_ok=True); te.to_csv(u + "/test_table.csv", index=False); ta.to_csv(u + "/test_table_all_slices.csv", index=False)
from scipy.ndimage import binary_dilation
for _, r in ta.iterrows():
    g = cv2.imread(r["mask"], 0) > 0; np.save(f"{u}/pred/{key2file(r.slice_key)}", binary_dilation(g, iterations=3).astype(np.uint8))
ta[["slice_key"]].assign(prob=ta.has_liver * 0.9 + 0.05).to_csv(f"{R}/det_pred.csv", index=False)
py = sys.executable
subprocess.run([py, "run_refinement.py", "--unet-dir", u, "--out", f"{R}/refine", "--workers", "1"], check=True)
subprocess.run([py, "analyse.py", "--results", f"{R}/refine/per_slice_results.csv", "--out", f"{R}/results", "--unet-dir", u, "--det-pred", f"{R}/det_pred.csv"], check=True)
print(open(f"{R}/results/seg_table.tex").read()[:900]); print(pd.read_csv(f"{R}/results/end_to_end.csv"))
