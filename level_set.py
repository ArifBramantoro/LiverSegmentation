"""Level set refinement: corrected solver (used for the revised results) and a faithful
reproduction of the original update rule (used only to explain the original results).

Sign convention: phi < 0 inside the liver, phi > 0 outside, |grad phi| = 1 after
re-initialisation. Outward unit normal n = grad phi / |grad phi|; mean curvature
kappa = div(n) (> 0 on a convex boundary). A front moving with outward normal speed V
obeys phi_t = -V |grad phi|.

Corrected evolution (all variants):
    phi_t = |grad phi| * ( gamma * kappa - F_ext ) + A        (Eq. 4 in the revised manuscript)
with F_ext an outward speed normalised to [-1, 1] and A an optional advection term.
    texture (GLCM):   F_tex  = w_h * H - w_c * C                (C, H normalised to [0, 1])
    edge (GAC):       phi_t  = g |grad phi| (gamma kappa - nu - lam_t F_tex) + grad g . grad phi
    region (CV):      F_reg  = -[(I - c_in)^2 - (I - c_out)^2] / max|.|
Curvature-only (gamma kappa |grad phi|) shrinks a disc of radius r at dr/dt = -gamma / r,
which tests/test_level_set.py verifies against the analytic solution.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, binary_erosion, binary_dilation

from glcm_fast import glcm_maps, normalise01


# ----------------------------------------------------------------------------- utilities
def disk(radius):
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def erode(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    return binary_erosion(mask.astype(bool), structure=disk(radius), border_value=0)


def dilate(mask, radius):
    return binary_dilation(mask.astype(bool), structure=disk(radius))


def sdf(mask):
    """Signed distance: negative inside, positive outside, zero level between pixels."""
    m = mask.astype(bool)
    if not m.any():
        return np.full(m.shape, 1e3, dtype=np.float64)
    if m.all():
        return np.full(m.shape, -1e3, dtype=np.float64)
    return distance_transform_edt(~m) - distance_transform_edt(m)


def reinitialise(phi, steps=10, dt=0.3):
    """Sussman-Smereka-Osher re-initialisation, phi_t = S(phi0)(1 - |grad phi|), Godunov upwinding.
    Restores |grad phi| = 1 near the front while keeping the sub-pixel position of the zero level."""
    phi0 = phi.copy()
    S = phi0 / np.sqrt(phi0 ** 2 + 1.0)
    for _ in range(steps):
        p = np.pad(phi, 1, mode="edge")
        a = p[1:-1, 1:-1] - p[1:-1, :-2]; b = p[1:-1, 2:] - p[1:-1, 1:-1]
        c = p[1:-1, 1:-1] - p[:-2, 1:-1]; d = p[2:, 1:-1] - p[1:-1, 1:-1]
        gp = np.sqrt(np.maximum(np.maximum(a, 0) ** 2, np.minimum(b, 0) ** 2) + np.maximum(np.maximum(c, 0) ** 2, np.minimum(d, 0) ** 2))
        gm = np.sqrt(np.maximum(np.minimum(a, 0) ** 2, np.maximum(b, 0) ** 2) + np.maximum(np.minimum(c, 0) ** 2, np.maximum(d, 0) ** 2))
        G = np.where(phi0 > 0, gp, gm)
        phi = phi - dt * S * (G - 1.0)
    return phi


def curvature(phi, eps=1e-8):
    """kappa = div(grad phi / |grad phi|), central differences."""
    py, px = np.gradient(phi)
    norm = np.sqrt(px * px + py * py + eps)
    nyy = np.gradient(py / norm, axis=0)
    nxx = np.gradient(px / norm, axis=1)
    return nxx + nyy


def grad_norm_upwind(phi, speed):
    """Osher-Sethian upwind |grad phi| for phi_t + V|grad phi| = 0 with V = speed."""
    p = np.pad(phi, 1, mode="edge")
    dxm = p[1:-1, 1:-1] - p[1:-1, :-2]; dxp = p[1:-1, 2:] - p[1:-1, 1:-1]
    dym = p[1:-1, 1:-1] - p[:-2, 1:-1]; dyp = p[2:, 1:-1] - p[1:-1, 1:-1]
    gp = np.sqrt(np.maximum(dxm, 0) ** 2 + np.minimum(dxp, 0) ** 2 + np.maximum(dym, 0) ** 2 + np.minimum(dyp, 0) ** 2)
    gm = np.sqrt(np.minimum(dxm, 0) ** 2 + np.maximum(dxp, 0) ** 2 + np.minimum(dym, 0) ** 2 + np.maximum(dyp, 0) ** 2)
    return np.where(speed > 0, gp, gm)


def grad_norm_central(phi, eps=1e-8):
    py, px = np.gradient(phi)
    return np.sqrt(px * px + py * py + eps)


def edge_stop(img01, sigma=1.0, lam=None):
    """g = 1 / (1 + |grad(G_sigma * I)|^2 / lam^2), lam defaults to the 90th percentile of the gradient."""
    sm = gaussian_filter(img01.astype(np.float64), sigma)
    gm = np.hypot(sobel(sm, 1), sobel(sm, 0)) / 8.0
    if lam is None:
        lam = np.percentile(gm, 90) + 1e-8
    return 1.0 / (1.0 + (gm / lam) ** 2)


# ----------------------------------------------------------------------------- corrected solver
DEFAULTS = dict(
    erosion_radius=2,        # px, disc structuring element
    constraint_radius=12,    # px, disc dilation of the U-Net mask
    gamma=0.2,               # curvature weight
    w_c=1.0, w_h=1.0,        # GLCM contrast / homogeneity weights (texture force)
    lam_t=1.0,               # weight of texture force when combined with edge/region terms (0 = texture off)
    nu=0.3,                  # balloon speed of the geodesic (edge) model, outward
    w_r=1.0,                 # region (Chan-Vese) force weight
    dt=0.2,                  # time step (CFL: dt * max|F| <= 0.5, dt * gamma <= 0.25)
    iterations=200,
    reinit_every=10,
    band=3.0,                # narrow-band half-width in px
    tol=1e-3,                # stop when relative area change over 10 iterations < tol
    glcm_window=9, glcm_levels=32, glcm_angles=(0, 45, 90, 135),
)

VARIANTS = {
    # name: which forces are active
    "init_only":        dict(evolve=False),
    "curvature_only":   dict(curv=True),
    "texture":          dict(curv=True, texture=True),
    "edge":             dict(curv=True, edge=True),
    "edge_texture":     dict(curv=True, edge=True, texture=True),
    "region":           dict(curv=True, region=True),
    "region_texture":   dict(curv=True, region=True, texture=True),
}


def texture_force(img01, constraint, p):
    m = glcm_maps(img01, window=p["glcm_window"], levels=p["glcm_levels"], angles=p["glcm_angles"])
    C = normalise01(m["contrast"], constraint)
    Hm = normalise01(m["homogeneity"], constraint)
    F = p["w_h"] * Hm - p["w_c"] * C
    return np.clip(F / (p["w_h"] + p["w_c"]) * 2.0, -1.0, 1.0), C, Hm   # rescaled to [-1, 1]


def refine(img01, unet_mask, variant, params=None, texture=None, return_history=False):
    """Run one refinement variant on one slice. Returns (binary mask, info dict)."""
    p = dict(DEFAULTS); p.update(params or {})
    spec = VARIANTS[variant]
    init = erode(unet_mask, p["erosion_radius"])
    info = {"iterations": 0, "converged": False}
    if not spec.get("evolve", True) or not init.any():
        return init.astype(np.uint8), info
    constraint = dilate(unet_mask, p["constraint_radius"])
    phi = sdf(init)
    I = img01.astype(np.float64)

    F_tex = 0.0
    if spec.get("texture"):
        F_tex = texture[0] if texture is not None else texture_force(I, constraint, p)[0]
    w_tex = p["lam_t"] if (spec.get("edge") or spec.get("region")) else 1.0
    if spec.get("edge"):
        g = edge_stop(I)
        gy, gx = np.gradient(g)

    area_hist = []
    hist = [] if return_history else None
    for it in range(p["iterations"]):
        kappa = curvature(phi)
        gn_c = grad_norm_central(phi)
        if spec.get("edge"):
            F = p["nu"] + (w_tex * F_tex if spec.get("texture") else 0.0)      # outward speed
            curv_term = g * p["gamma"] * kappa * gn_c
            prop = -g * F * grad_norm_upwind(phi, g * F)
            py, px = np.gradient(phi)
            adv_speed_x, adv_speed_y = gx, gy
            # upwind advection grad g . grad phi
            pp = np.pad(phi, 1, mode="edge")
            dxm = pp[1:-1, 1:-1] - pp[1:-1, :-2]; dxp = pp[1:-1, 2:] - pp[1:-1, 1:-1]
            dym = pp[1:-1, 1:-1] - pp[:-2, 1:-1]; dyp = pp[2:, 1:-1] - pp[1:-1, 1:-1]
            adv = np.maximum(gx, 0) * dxp + np.minimum(gx, 0) * dxm + np.maximum(gy, 0) * dyp + np.minimum(gy, 0) * dym
            dphi = curv_term + prop + adv
        else:
            F = np.zeros_like(phi)
            if spec.get("texture"):
                F = F + w_tex * F_tex
            if spec.get("region"):
                inside = phi < 0
                c_in = I[inside & constraint].mean() if np.any(inside & constraint) else I[inside].mean()
                c_out = I[~inside & constraint].mean() if np.any(~inside & constraint) else I[~inside].mean()
                R = (I - c_in) ** 2 - (I - c_out) ** 2
                R = R / (np.abs(R[constraint]).max() + 1e-8)
                F = F - p["w_r"] * R
            dphi = p["gamma"] * kappa * gn_c - F * grad_norm_upwind(phi, F)
        # narrow band and anatomical constraint
        band = np.abs(phi) <= p["band"]
        dphi = np.where(band & constraint, dphi, 0.0)
        phi = phi + p["dt"] * dphi
        phi = np.where(constraint, phi, np.maximum(phi, 1.0))     # contour may not leave the constraint region
        if (it + 1) % p["reinit_every"] == 0:
            phi = reinitialise(phi)
        a = int((phi < 0).sum()); area_hist.append(a)
        if return_history and it % 10 == 0:
            hist.append((it, phi.copy()))
        info["iterations"] = it + 1
        if len(area_hist) > 10 and a > 0 and abs(a - area_hist[-11]) / a < p["tol"]:
            info["converged"] = True
            break
    out = (phi < 0).astype(np.uint8)
    if return_history:
        info["history"] = hist
    return out, info


# ----------------------------------------------------------------------------- original rule (for explanation only)
def legacy_refine(img01, unet_mask, variant, iterations=200, timestep=0.1, gamma=1.0, alpha=1.0, beta=1.0, theta=1.0,
                  glcm_window=7):
    """Faithful reproduction of the evaluation loops in Level_set_test.ipynb (original submission).

    Update: phi += dt * (F_ext * force_mask - gamma * kappa), Gaussian smoothing (sigma = 1) every iteration,
    clip to +/-1000. No |grad phi| factor and the curvature sign is reversed (anti-diffusive).
    Erosion: square 5x5 kernel, one iteration. Force mask: square 25x25 dilation. GLCM: angle 0 only,
    contrast not normalised. 'curvature' and 'gradient' variants are the same rule in the original code.
    """
    import cv2
    pred = unet_mask.astype(np.uint8)
    inner = cv2.erode(pred, np.ones((5, 5), np.uint8), iterations=1)
    force = cv2.dilate(pred, np.ones((25, 25), np.uint8), iterations=1).astype(np.float64)
    phi = distance_transform_edt(inner == 0) - distance_transform_edt(inner)
    I = img01.astype(np.float64)
    if variant in ("curvature", "gradient", "region"):
        m = glcm_maps(I, window=glcm_window, angles=(0,), levels=32)
        con, hom = m["contrast"].astype(np.float64), m["homogeneity"].astype(np.float64)
    if variant == "edge":
        sm = gaussian_filter(I, 1)
        gm = np.hypot(sobel(sm, 1), sobel(sm, 0))
        gm = (gm - gm.min()) / (gm.max() - gm.min() + 1e-8)
        es = 1.0 / (1.0 + gm)

    def kap(ph):
        py, px = np.gradient(ph)
        nrm = np.sqrt(px ** 2 + py ** 2 + 1e-8)
        return np.gradient(px / nrm, axis=1) + np.gradient(py / nrm, axis=0)

    for _ in range(iterations):
        k = kap(phi)
        if variant == "edge":
            F = alpha * es * force
        elif variant == "region":
            ins = phi < 0
            c_in = I[ins].mean() if ins.any() else 0.0
            c_out = I[~ins].mean() if (~ins).any() else 0.0
            F = (alpha * con - beta * hom - theta * ((I - c_in) ** 2 - (I - c_out) ** 2)) * force
        else:
            F = (alpha * con - beta * hom) * force
        phi = phi + timestep * (F - gamma * k)
        phi = gaussian_filter(phi, sigma=1)
        phi = np.clip(phi, -1000, 1000)
    return (phi < 0).astype(np.uint8)
