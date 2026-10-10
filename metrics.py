"""Segmentation metrics. Per-slice Dice, IoU, HD95 and ASSD (in mm), and per-case volumetric Dice.
Surface distances follow the usual definition: boundaries are the mask pixels removed by a
one-pixel erosion; distances are computed with the in-plane pixel spacing of the resized slice."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(pred, gt, empty_value=1.0):
    p = pred.astype(bool); g = gt.astype(bool)
    s = p.sum() + g.sum()
    return empty_value if s == 0 else 2.0 * (p & g).sum() / s


def iou(pred, gt, empty_value=1.0):
    p = pred.astype(bool); g = gt.astype(bool)
    u = (p | g).sum()
    return empty_value if u == 0 else (p & g).sum() / u


def _surface(m):
    return m & ~binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def surface_distances(pred, gt, spacing=(1.0, 1.0)):
    """Return (HD95, ASSD) in the units of spacing. If either mask is empty and the other is not,
    both are set to the image diagonal (worst case), and flag=1 is returned so it can be reported."""
    p = pred.astype(bool); g = gt.astype(bool)
    diag = float(np.hypot(p.shape[0] * spacing[0], p.shape[1] * spacing[1]))
    if not p.any() and not g.any():
        return 0.0, 0.0, 0
    if not p.any() or not g.any():
        return diag, diag, 1
    sp, sg = _surface(p), _surface(g)
    dt_g = distance_transform_edt(~sg, sampling=spacing)
    dt_p = distance_transform_edt(~sp, sampling=spacing)
    d_pg = dt_g[sp]; d_gp = dt_p[sg]
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    assd = (d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp))
    return float(hd95), float(assd), 0


def slice_metrics(pred, gt, spacing):
    hd95, assd, empty = surface_distances(pred, gt, spacing)
    return dict(dice=dice(pred, gt), iou=iou(pred, gt), hd95_mm=hd95, assd_mm=assd, empty_pred=int(empty),
                inter_px=int((pred.astype(bool) & gt.astype(bool)).sum()), pred_px=int(pred.astype(bool).sum()))


def volumetric_dice(pred_slices, gt_slices):
    """3D Dice of a case from its stacked slices (lists or arrays of 2D masks)."""
    P = np.stack(pred_slices).astype(bool); G = np.stack(gt_slices).astype(bool)
    return dice(P, G)


def det_summary(df, t):
    """Slice-level detector metrics at threshold t (df has columns has_liver, prob, case_id)."""
    from sklearn.metrics import roc_auc_score
    y = df.has_liver.values; p = (df.prob.values >= t).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return dict(threshold=t, n=len(y), cases=df.case_id.nunique(), TP=tp, TN=tn, FP=fp, FN=fn, accuracy=(tp + tn) / len(y),
                precision=prec, recall=rec, f1=2 * prec * rec / max(1e-9, prec + rec),
                auc=float(roc_auc_score(y, df.prob)) if len(set(y)) > 1 else float("nan"))
