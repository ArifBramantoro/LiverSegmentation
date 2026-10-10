"""Numerical verification of the corrected solver and diagnosis of the original rule on synthetic discs.
These are code-verification tests on synthetic images only; they are not study results."""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from level_set import refine, legacy_refine, curvature, sdf, DEFAULTS
from scipy.ndimage import gaussian_filter

N = 256
yy, xx = np.mgrid[:N, :N]
def disc(r, cy=128, cx=128): return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
def eq_radius(m): return np.sqrt(m.sum() / np.pi)
def dice(a, b): a = a.astype(bool); b = b.astype(bool); return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-8)

def test_curvature_flow_matches_analytic():
    r0, gamma, dt, iters = 40.0, 1.0, 0.2, 400
    out, info = refine(np.zeros((N, N)), disc(r0), "curvature_only",
                       dict(erosion_radius=0, constraint_radius=12, gamma=gamma, dt=dt, iterations=iters, tol=0))
    r_num = eq_radius(out); r_ana = np.sqrt(r0 ** 2 - 2 * gamma * dt * iters)
    assert abs(r_num - r_ana) < 1.0, (r_num, r_ana)
    return dict(r0=r0, r_numeric=round(float(r_num), 2), r_analytic=round(float(r_ana), 2))

def test_legacy_curvature_is_antidiffusive():
    """Original rule with F_ext = 0: phi += dt * (-gamma * kappa). A disc should shrink; it grows instead."""
    phi = sdf(disc(40)); r_start = eq_radius(phi < 0)
    for _ in range(200):
        phi = phi + 0.1 * (-1.0 * curvature(phi))
    grown = eq_radius(phi < 0)
    assert grown > r_start, (grown, r_start)
    return dict(r_start=round(float(r_start), 2), r_after_200_legacy_steps=round(float(grown), 2))

def synthetic_image(seed=0, r_true=40):
    rng = np.random.default_rng(seed)
    img = np.where(disc(r_true), 0.6, 0.35) + 0.03 * rng.standard_normal((N, N))
    return np.clip(gaussian_filter(img, 1.0), 0, 1), disc(r_true)

def test_legacy_edge_rule_can_only_shrink():
    img, gt = synthetic_image()
    pred = disc(40)                       # perfect "U-Net" mask
    out = legacy_refine(img, pred, "edge")
    d = dice(out, gt)
    assert out.sum() < pred.sum()
    return dict(dice_of_perfect_input=1.0, dice_after_legacy_edge=round(float(d), 4),
                area_ratio=round(float(out.sum() / pred.sum()), 4))

def test_corrected_variants_behave_sensibly():
    img, gt = synthetic_image()
    pred = disc(34)                       # under-segmented "U-Net" mask, true radius 40
    res = {"unet_input_dice": round(float(dice(pred, gt)), 4)}
    for v in ["init_only", "curvature_only", "edge", "region", "texture"]:
        out, info = refine(img, pred, v)
        res[v] = round(float(dice(out, gt)), 4)
    # data-driven variants must recover towards the true boundary; curvature-only must not grow
    assert res["edge"] > res["init_only"] and res["region"] > res["init_only"], res
    assert res["curvature_only"] <= res["init_only"] + 1e-6, res
    return res

if __name__ == "__main__":
    out = {}
    for t in [test_curvature_flow_matches_analytic, test_legacy_curvature_is_antidiffusive,
              test_legacy_edge_rule_can_only_shrink, test_corrected_variants_behave_sensibly]:
        out[t.__name__] = t(); print(t.__name__, "PASSED", out[t.__name__])
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "verification_results.json"), "w"), indent=2)
