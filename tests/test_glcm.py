import numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skimage.feature import graycomatrix, graycoprops
from glcm_fast import glcm_maps, quantise, ANGLE_OFFSETS

def reference(img01, window, distance, angles, levels):
    q = quantise(img01, levels).astype(np.uint8)
    k = window // 2
    qp = np.pad(q, k, mode="reflect")
    H, W = q.shape
    rad = [np.deg2rad(a) for a in angles]
    con = np.zeros((H, W)); hom = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            win = qp[i:i + window, j:j + window]
            g = graycomatrix(win, [distance], rad, levels=levels, symmetric=True, normed=True)
            con[i, j] = graycoprops(g, "contrast")[0].mean()
            hom[i, j] = graycoprops(g, "homogeneity")[0].mean()
    return con, hom

def test_matches_skimage():
    rng = np.random.default_rng(0)
    img = rng.random((40, 37))
    img[10:25, 8:30] = 0.6 + 0.02 * rng.random((15, 22))
    for window, angles in [(7, (0,)), (9, (0, 45, 90, 135))]:
        ref_c, ref_h = reference(img, window, 1, angles, 32)
        out = glcm_maps(img, window=window, distance=1, angles=angles, levels=32)
        # interior pixels (window fully inside image) must match exactly; border uses reflect in both
        assert np.allclose(out["contrast"], ref_c, atol=1e-4), np.abs(out["contrast"] - ref_c).max()
        assert np.allclose(out["homogeneity"], ref_h, atol=1e-5), np.abs(out["homogeneity"] - ref_h).max()

if __name__ == "__main__":
    test_matches_skimage(); print("GLCM exactness test passed")
    img = np.random.default_rng(1).random((256, 256))
    t = time.time(); glcm_maps(img); print(f"fast GLCM 256x256, 4 angles, w=9: {time.time()-t:.3f} s")
