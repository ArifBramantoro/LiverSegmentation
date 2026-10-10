"""Exact sliding-window GLCM contrast and homogeneity via integral images.

For a symmetric, normalised GLCM built from the pixel pairs (p, p+d) that lie
inside a w x w window, Haralick contrast and homogeneity are expectations over
those pairs:
    contrast    = mean_{pairs} (a - b)^2
    homogeneity = mean_{pairs} 1 / (1 + (a - b)^2)
(symmetrisation does not change either value because both functions are
symmetric in a and b). Each window mean is therefore a rectangle sum of a
per-pair map, which integral images give in O(1) per pixel. Output matches
skimage.feature.graycomatrix/graycoprops exactly (see tests/test_glcm.py),
at a small fraction of the cost of a per-pixel graycomatrix loop.
"""
import numpy as np

ANGLE_OFFSETS = {0: (0, 1), 45: (-1, 1), 90: (-1, 0), 135: (-1, -1)}  # (dy, dx), skimage convention


def quantise(img01, levels=32):
    """Map an image in [0, 1] to integer grey levels 0..levels-1 (same rule as the original code)."""
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * (levels - 1)).astype(np.int64)


def _rect_sum(ii, y0, y1, x0, x1):
    """Inclusive-exclusive rectangle sums from integral image ii (padded with a zero row/col)."""
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def glcm_maps(img01, window=9, distance=1, angles=(0, 45, 90, 135), levels=32):
    """Return dict with 'contrast' and 'homogeneity' maps (angle-averaged), same shape as img01.

    Reflect padding is used at the image border, as in the original implementation.
    """
    q = quantise(img01, levels)
    k = window // 2
    pad = k + distance
    qp = np.pad(q, pad, mode="reflect").astype(np.float64)
    H, W = q.shape
    con = np.zeros((H, W)); hom = np.zeros((H, W))
    cy = np.arange(H)[:, None] + pad
    cx = np.arange(W)[None, :] + pad
    for ang in angles:
        dy, dx = ANGLE_OFFSETS[ang]
        dy *= distance; dx *= distance
        # pair map defined at p for the pair (p, p+d)
        a = qp
        b = np.roll(np.roll(qp, -dy, axis=0), -dx, axis=1)
        diff2 = (a - b) ** 2
        valid = np.ones_like(a)
        # invalidate wrapped rows/cols introduced by roll
        if dy > 0: valid[-dy:, :] = 0
        if dy < 0: valid[:-dy, :] = 0
        if dx > 0: valid[:, -dx:] = 0
        if dx < 0: valid[:, :-dx] = 0
        f_con = diff2 * valid
        f_hom = valid / (1.0 + diff2)
        ii_c = np.pad(np.cumsum(np.cumsum(f_con, 0), 1), ((1, 0), (1, 0)))
        ii_h = np.pad(np.cumsum(np.cumsum(f_hom, 0), 1), ((1, 0), (1, 0)))
        # p must satisfy p in window and p+d in window
        y0 = cy - k + max(0, -dy); y1 = cy + k - max(0, dy) + 1
        x0 = cx - k + max(0, -dx); x1 = cx + k - max(0, dx) + 1
        n = (window - abs(dy)) * (window - abs(dx))
        con += _rect_sum(ii_c, y0, y1, x0, x1) / n
        hom += _rect_sum(ii_h, y0, y1, x0, x1) / n
    con /= len(angles); hom /= len(angles)
    return {"contrast": con.astype(np.float32), "homogeneity": hom.astype(np.float32)}


def normalise01(x, mask=None, eps=1e-8):
    """Min-max normalise to [0, 1], optionally using statistics from a region (e.g. the constraint band)."""
    ref = x[mask > 0] if mask is not None and np.any(mask > 0) else x
    lo, hi = np.percentile(ref, 1), np.percentile(ref, 99)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)
