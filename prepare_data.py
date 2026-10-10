"""End-to-end data ingestion: raw NIfTI/DICOM volumes -> windowed, resized 2D axial slices with a
manifest that keeps case identity, slice order and pixel spacing (needed for patient-level splits,
surface distances in mm and 3D reconstruction).

Steps per case
  1. load volume and liver label (LiTS/MSD/MedSeg: NIfTI; CHAOS: DICOM series + PNG masks)
  2. reorient to a common radiological axial display (rows anterior->posterior, image left = patient right)
  3. clip to an abdominal window [HU_MIN, HU_MAX] and rescale to [0, 1]
  4. zero-pad to square (aspect ratio preserved) and resize to 256 x 256 (linear for image, nearest for mask)
  5. write image/mask PNGs and one manifest row per slice (source, case_id, z index, spacing after resize, has_liver)
Run:  python prepare_data.py --lits RAW/LiTS --chaos RAW/CHAOS/Train_Sets/CT --medseg RAW/MedSeg --msd RAW/Task03_Liver --out DATA
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, cv2

HU_MIN, HU_MAX, SIZE = -100.0, 400.0, 256


def window01(hu):
    return np.clip((hu - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)


def pad_resize(arr, interp):
    h, w = arr.shape; s = max(h, w)
    out = np.zeros((s, s), arr.dtype); y0 = (s - h) // 2; x0 = (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return cv2.resize(out, (SIZE, SIZE), interpolation=interp), s / SIZE


HEADER_MISMATCH = []   # cases whose label header orientation differs from the image header orientation


def label_header_differs(img_path, lab_path):
    import nibabel as nib
    a = nib.io_orientation(nib.load(img_path).affine); b = nib.io_orientation(nib.load(lab_path).affine)
    return not np.array_equal(a, b)


def nifti_axial_slices(img_path, lab_path, mask_fix="none"):
    """Image reoriented to RAS from its own header; the label ARRAY is voxel-aligned with the image array
    (true for LiTS, MSD and MedSeg), so it receives the image's transform, never its own header's.
    Reorienting each file from its own header mirrored the masks of cases whose label header differs."""
    import nibabel as nib
    im = nib.load(img_path); lb = nib.load(lab_path)
    ornt = nib.io_orientation(im.affine)
    Iraw = np.asanyarray(im.dataobj); Lraw = np.asanyarray(lb.dataobj)
    assert Iraw.shape == Lraw.shape, (img_path, Iraw.shape, Lraw.shape)
    I = nib.orientations.apply_orientation(Iraw, ornt).astype(np.float32)
    L = nib.orientations.apply_orientation(Lraw, ornt) > 0             # label >= 1 is liver (tumour included)
    zooms = [0.0, 0.0, 0.0]
    for ax_in, (ax_out, _) in enumerate(ornt[:3]):
        zooms[int(ax_out)] = float(im.header.get_zooms()[ax_in])
    sx, sy, _ = zooms
    for k in range(I.shape[2]):
        # RAS canonical: axis0 L->R, axis1 P->A. Display: rows A->P, columns R->L
        sl = I[:, :, k].T[::-1, ::-1]; ml = L[:, :, k].T[::-1, ::-1]
        ml = {"none": ml, "flipud": ml[::-1, :], "fliplr": ml[:, ::-1], "rot180": ml[::-1, ::-1]}[mask_fix]
        yield k, sl, ml, (float(sy), float(sx))


def _dicom_index(path):
    m = re.search(r"i(\d+)", os.path.basename(path))       # CHAOS names: i0108,0000b.dcm
    return int(m.group(1)) if m else None


def load_chaos_patient(patient_dir):
    """DICOM volume (HU) and masks, paired by the slice number in the file names
    (i0108,0000b.dcm <-> liver_GT_108.png), as in the original study code."""
    import pydicom
    files = glob.glob(os.path.join(patient_dir, "DICOM_anon", "*.dcm"))
    if all(_dicom_index(f) is not None for f in files):
        files.sort(key=_dicom_index)
    else:
        files.sort(key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber))
    gts = sorted(glob.glob(os.path.join(patient_dir, "Ground", "*.png")), key=num)
    assert len(gts) == len(files), (patient_dir, len(gts), len(files))
    ds = [pydicom.dcmread(f) for f in files]
    vol = np.stack([d.pixel_array * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0)) for d in ds])
    msk = np.stack([cv2.imread(g, cv2.IMREAD_GRAYSCALE) > 0 for g in gts])
    sp = [float(x) for x in ds[0].PixelSpacing]
    return vol, msk, sp


IN_PLANE = {"none": lambda m: m, "flipud": lambda m: m[:, ::-1, :], "fliplr": lambda m: m[:, :, ::-1],
            "rot180": lambda m: m[:, ::-1, ::-1]}


def misalignment(vol, msk):
    """Fraction of mask voxels outside soft-tissue attenuation [-50, 250] HU (air, lung, fat or bone).
    A correctly aligned liver mask has very few such voxels; a flipped one lands on lung, air or spine."""
    v = vol[msk]
    return 1.0 if v.size == 0 else float(np.mean((v < -50) | (v > 250)))


def edge_alignment(vol, msk):
    """Mean image gradient on the mask boundary relative to the mean gradient in the body.
    A correctly aligned liver mask has its boundary on the liver edge, so this ratio is clearly above 1."""
    from scipy.ndimage import binary_erosion, sobel, gaussian_filter
    num_, den_ = 0.0, 0.0
    for k in np.where(msk.reshape(len(msk), -1).any(1))[0][::2]:
        im = gaussian_filter(np.clip(vol[k], -200, 300).astype(np.float32), 1.0)
        g = np.hypot(sobel(im, 0), sobel(im, 1)); m = msk[k]
        b = m & ~binary_erosion(m, iterations=1)
        body = im > -150
        num_ += g[b].mean() if b.any() else 0.0; den_ += g[body].mean() + 1e-6
    return num_ / (den_ + 1e-6)


def chaos_orientation(patient_dirs):
    """Choose ONE in-plane mask orientation for the whole CHAOS dataset (it is a dataset convention)."""
    table = {}
    for pdir in patient_dirs:
        vol, msk, _ = load_chaos_patient(pdir)
        # score: edge alignment of the mask boundary, penalised by mask voxels on air, lung, fat or bone
        table[os.path.basename(pdir)] = {k: edge_alignment(vol, f(msk)) - 5.0 * misalignment(vol, f(msk)) for k, f in IN_PLANE.items()}
    total = {k: np.mean([t[k] for t in table.values()]) for k in IN_PLANE}
    best = max(total, key=total.get)
    print("CHAOS mask orientation score (higher = better aligned): " +
          ", ".join(f"{k} {v:.2f}" for k, v in total.items()) + f"  -> using '{best}'")
    disagree = [p for p, t in table.items() if max(t, key=t.get) != best]
    for p in sorted(disagree, key=int): print(f"  note: patient {p} individually prefers '{max(table[p], key=table[p].get)}'")
    if len(disagree) > 2:
        raise SystemExit("CHAOS orientation is not consistent across patients; send this output for review.")
    return best, table


def chaos_axial_slices(patient_dir, orient):
    vol, msk, sp = load_chaos_patient(patient_dir)
    msk = IN_PLANE[orient](msk)
    print(f"  CHAOS {os.path.basename(patient_dir)}: {len(vol)} slices, edge alignment {edge_alignment(vol, msk):.2f}, "
          f"mask outside soft tissue {misalignment(vol, msk):.3f}")
    for k in range(vol.shape[0]):
        yield k, vol[k], msk[k], (sp[0], sp[1])


def volume_signature(img_path, n=24):
    """Orientation-normalised, resampled, z-scored thumbnail of a CT volume, used to detect the same
    patient scan across datasets even after format conversion or re-export."""
    import nibabel as nib
    from scipy.ndimage import zoom
    im = nib.as_closest_canonical(nib.load(img_path)); a = np.asarray(im.dataobj)
    st = [max(1, d // (4 * n)) for d in a.shape]; a = np.clip(a[::st[0], ::st[1], ::st[2]].astype(np.float32), -200, 400)
    a = zoom(a, [n / d for d in a.shape], order=1)
    return ((a - a.mean()) / (a.std() + 1e-6)).ravel()


_SIG_CACHE = {}


def raw_shape(img_path):
    import nibabel as nib
    return tuple(sorted(nib.load(img_path).shape[:3]))


def _area_pool(a, n, axis):
    """Exact area-weighted mean of a along one axis into n equal-length intervals [i*d/n, (i+1)*d/n],
    with fractional voxel weights at the interval ends. Because the intervals are mirror-symmetric,
    pooling commutes exactly with flipping the axis (unlike interpolation), and it also works for d < n."""
    a = np.moveaxis(a, axis, 0); d = a.shape[0]
    cs = np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), np.cumsum(a, axis=0)], axis=0)
    x = np.arange(n + 1) * d / n; k = np.minimum(np.floor(x).astype(int), d - 1); f = (x - k)[(...,) + (None,) * (a.ndim - 1)]
    S = cs[k] + f * a[k]
    return np.moveaxis((S[1:] - S[:-1]) / (d / n), 0, axis)


def raw_variants(img_path, n=24):
    """Signatures of the stored voxel array (header ignored) under the 16 axis flips/in-plane transposes,
    so the same scan is recognised whatever orientation its header declares. The array is clipped to
    [-200, 400] HU and area-averaged to n^3 voxels; area averaging commutes with flips, so a flipped
    copy of a scan gives the same signature as the original."""
    import nibabel as nib
    dobj = nib.load(img_path).dataobj; parts = []
    for z0 in range(0, dobj.shape[2], 64):           # in chunks of slices to limit memory
        c = np.clip(np.asarray(dobj[:, :, z0:z0 + 64], dtype=np.float64), -200, 400)
        parts.append(_area_pool(_area_pool(c, n, 0), n, 1))
    a = _area_pool(np.concatenate(parts, axis=2), n, 2)
    a = (a - a.mean()) / (a.std() + 1e-6)
    outs = []
    for t in (a, np.transpose(a, (1, 0, 2))):
        for f0 in (False, True):
            for f1 in (False, True):
                for f2 in (False, True):
                    v = t[::-1] if f0 else t; v = v[:, ::-1] if f1 else v; v = v[:, :, ::-1] if f2 else v
                    outs.append(np.ascontiguousarray(v).ravel() / np.sqrt(v.size))
    return np.stack(outs)


def match_to_lits(paths, lits_paths, thr=0.99):
    """Return {path: matching reference case number} for volumes whose stored array matches a reference
    volume (correlation > thr) under any axis flip or in-plane transpose."""
    if not lits_paths or not paths:
        return {}
    key = tuple(sorted(lits_paths))
    if key not in _SIG_CACHE:
        _SIG_CACHE[key] = {num(v): (raw_variants(v)[0], raw_shape(v), v) for v in lits_paths}
    L = _SIG_CACHE[key]
    keys = list(L); M = np.stack([L[k][0] for k in keys])
    out = {}
    for pth in paths:
        shp = raw_shape(pth); V = raw_variants(pth); R_ = V @ M.T              # (16 variants, n refs)
        for j in np.argsort(-R_.max(0)):
            if R_[:, j].max() < thr: break
            if L[keys[j]][1] != shp: continue                                   # a re-saved copy keeps its dimensions
            vi = int(R_[:, j].argmax())
            if float(raw_variants(pth, 48)[vi] @ raw_variants(L[keys[j]][2], 48)[0]) > 0.99:   # fine-resolution confirmation
                out[pth] = int(keys[j]); break
    return out


def _sig(vol01, n=24):
    v = vol01.astype(np.float64)
    for ax in range(3): v = _area_pool(v, n, ax)       # flip-equivariant, like raw_variants
    return ((v - v.mean()) / (v.std() + 1e-6)).ravel() / np.sqrt(n ** 3)


def png_stack_signatures(manifest_rows):
    """Signatures of LiTS cases imported as PNG stacks (axis order z, y, x; intensities as exported)."""
    out = {}
    for cid, g in manifest_rows.groupby("case_id"):
        g = g.sort_values("z"); vol = np.stack([cv2.imread(f, 0) for f in g.image]).astype(np.float32) / 255.0
        out[cid] = _sig(vol)
    return out


def nifti_variants(img_path, n=24):
    """Signatures of a NIfTI volume in the 16 in-plane/axial orientations, windowed like the export (z, y, x order)."""
    import nibabel as nib
    a = np.asarray(nib.as_closest_canonical(nib.load(img_path)).dataobj, dtype=np.float32)
    v = np.transpose(window01(a), (2, 1, 0))
    outs = []
    for zf in (False, True):
        w = v[::-1] if zf else v
        for k in range(4):
            r = np.rot90(w, k, axes=(1, 2)); outs += [_sig(r, n), _sig(r[:, :, ::-1], n)]
    return np.stack(outs)


def match_to_png(paths, sigs, thr=0.95):
    if not sigs or not paths: return {}
    keys = list(sigs); M = np.stack([sigs[k] for k in keys]); out = {}
    for pth in paths:
        r = (nifti_variants(pth) @ M.T).max(0)
        if r.max() > thr: out[pth] = keys[int(r.argmax())]
    return out


def export_case(rows, source, case_id, slices, out):
    d_img = os.path.join(out, source, case_id, "img"); d_msk = os.path.join(out, source, case_id, "mask")
    os.makedirs(d_img, exist_ok=True); os.makedirs(d_msk, exist_ok=True)
    for k, hu, m, (sy, sx) in slices:
        img, f = pad_resize(window01(hu).astype(np.float32), cv2.INTER_LINEAR)
        msk, _ = pad_resize(m.astype(np.uint8), cv2.INTER_NEAREST)
        ip = os.path.join(d_img, f"{k:04d}.png"); mp = os.path.join(d_msk, f"{k:04d}.png")
        cv2.imwrite(ip, np.round(img * 255).astype(np.uint8)); cv2.imwrite(mp, (msk > 0).astype(np.uint8) * 255)
        rows.append(dict(source=source, case_id=f"{source}:{case_id}", z=k, image=ip, mask=mp,
                         spacing_y=sy * f, spacing_x=sx * f, has_liver=int(msk.any()), liver_px=int(msk.sum())))


def reuse_nifti_case(rows, source, case_id, img_path, out, lab_path=None):
    """If a NIfTI case was already exported completely, rebuild its manifest rows from the header and the
    existing PNGs instead of exporting again. Returns True if reused."""
    import nibabel as nib
    if lab_path is not None and label_header_differs(img_path, lab_path):
        HEADER_MISMATCH.append(f"{source}:{case_id}")
        print(f"  {source} {case_id}: label header orientation differs from image; re-exporting with image orientation")
        return False
    im = nib.load(img_path); ornt = nib.io_orientation(im.affine)
    shape = [0, 0, 0]; zoom = [0, 0, 0]
    for ax_in, (ax_out, _) in enumerate(ornt[:3]):
        shape[int(ax_out)] = im.shape[ax_in]; zoom[int(ax_out)] = float(im.header.get_zooms()[ax_in])
    d_img = os.path.join(out, source, case_id, "img"); d_msk = os.path.join(out, source, case_id, "mask")
    n = shape[2]
    if not (os.path.isdir(d_img) and len(glob.glob(d_img + "/*.png")) == n and len(glob.glob(d_msk + "/*.png")) == n):
        return False
    f = max(shape[0], shape[1]) / SIZE
    for k in range(n):
        ip = os.path.join(d_img, f"{k:04d}.png"); mp = os.path.join(d_msk, f"{k:04d}.png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
        rows.append(dict(source=source, case_id=f"{source}:{case_id}", z=k, image=ip, mask=mp,
                         spacing_y=zoom[1] * f, spacing_x=zoom[0] * f, has_liver=int(m.any()), liver_px=int(m.sum())))
    return True


NIFTI_CASES = {}   # manifest case_id -> (source, short id, image path, label path)


def png_alignment(rows):
    """Alignment of one exported case, from its PNGs: score = edge alignment of the mask boundary minus
    5 x fraction of mask pixels outside soft tissue, for the mask as stored and for 3 in-plane flips."""
    from scipy.ndimage import binary_erosion, sobel, gaussian_filter
    liv = [r for r in rows if r["has_liver"]][::2]
    if not liv:
        return None
    imgs = [cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE).astype(np.float32) for r in liv]
    msks = [cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 0 for r in liv]
    grads = [np.hypot(sobel(gaussian_filter(g, 1.0), 0), sobel(gaussian_filter(g, 1.0), 1)) for g in imgs]
    hu = [g / 255.0 * (HU_MAX - HU_MIN) + HU_MIN for g in imgs]
    out = {}
    for name, f in [("none", lambda m: m), ("flipud", lambda m: m[::-1, :]), ("fliplr", lambda m: m[:, ::-1]), ("rot180", lambda m: m[::-1, ::-1])]:
        num_, den_, bad, tot = 0.0, 0.0, 0, 0
        for g, h, m, gi in zip(grads, hu, msks, imgs):
            m = f(m); b = m & ~binary_erosion(m, iterations=1); body = gi > 0
            if b.any(): num_ += g[b].mean()
            den_ += g[body].mean() + 1e-6
            bad += int(((h[m] < -50) | (h[m] > 250)).sum()); tot += int(m.sum())
        out[name] = num_ / (den_ + 1e-6) - 5.0 * bad / max(1, tot)
    return out


def alignment_qc(rows, out_dir):
    """Check every exported case; re-export NIfTI cases whose mask is clearly mirrored; stop if ambiguous."""
    by_case = {}
    for r in rows: by_case.setdefault(r["case_id"], []).append(r)
    log, fixed, unresolved = [], [], []
    for cid, rr in sorted(by_case.items()):
        sc = png_alignment(rr)
        if sc is None: continue
        flips = {k: v for k, v in sc.items() if k != "none"}; best = max(flips, key=flips.get)
        flagged = flips[best] > sc["none"] + 0.5
        entry = dict(case_id=cid, score_as_stored=round(sc["none"], 3), best_flip=best, score_best_flip=round(flips[best], 3),
                     header_mismatch=int(cid in HEADER_MISMATCH), corrected="")
        if flagged:
            if cid in NIFTI_CASES and flips[best] > 1.0 and flips[best] - sc["none"] > 0.8:
                src, short, img, lab = NIFTI_CASES[cid]
                new = []; export_case(new, src, short, nifti_axial_slices(img, lab, best), out_dir)
                sc2 = png_alignment(new)
                if sc2 and sc2["none"] > max(v for k, v in sc2.items() if k != "none"):
                    rr[:] = new; entry["corrected"] = best; entry["score_after"] = round(sc2["none"], 3); fixed.append(cid)
                else:
                    unresolved.append(cid)
            else:
                unresolved.append(cid)
        log.append(entry)
    rows[:] = [r for rr in by_case.values() for r in rr]
    old_f = os.path.join(out_dir, "alignment_check.csv")
    if os.path.exists(old_f):   # keep corrections and header findings made by an earlier, interrupted run
        old = pd.read_csv(old_f, dtype=str).fillna("").set_index("case_id")
        for e in log:
            if e["case_id"] in old.index:
                o = old.loc[e["case_id"]]
                if o.get("corrected", "") and not e["corrected"]:
                    e["corrected"] = o["corrected"]; e["score_after"] = o.get("score_after", "")
                    if e["case_id"] not in fixed: fixed.append(e["case_id"])
                if o.get("header_mismatch", "0") == "1" and e["case_id"] not in HEADER_MISMATCH:
                    HEADER_MISMATCH.append(e["case_id"]); e["header_mismatch"] = 1
    pd.DataFrame(log).to_csv(old_f, index=False)
    msg = (f"Alignment check: {len(log)} cases checked; label header differed from image: {len(HEADER_MISMATCH)} {HEADER_MISMATCH}; "
           f"masks corrected by in-plane flip: {len(fixed)} {fixed}; unresolved: {unresolved}")
    print(msg); open(os.path.join(out_dir, "alignment_check.txt"), "w").write(msg + "\n")
    if unresolved:
        raise SystemExit(f"Mask alignment unresolved for {unresolved}; see {out_dir}/alignment_check.csv and send it for review.")


def nii(pattern, recursive=False):
    """NIfTI files only (.nii or .nii.gz); ignores zipped copies such as volume-0.nii.zip."""
    return [f for f in glob.glob(pattern, recursive=recursive) if f.lower().endswith((".nii", ".nii.gz"))]


def medseg_pairs(root):
    """Find MedSeg image and mask volumes and pair them by case number.
    Image folder: img/imgs/image/images; mask folder: mask/masks/label/labels/seg/segmentation(s).
    Falls back to file names containing 'mask' or 'label' anywhere under root."""
    subs = {os.path.basename(d).lower(): d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)}
    idir = next((subs[k] for k in ("img", "imgs", "image", "images") if k in subs), None)
    mdir = next((subs[k] for k in ("mask", "masks", "label", "labels", "seg", "segmentation", "segmentations") if k in subs), None)
    if idir and mdir:
        imgs = sorted(nii(os.path.join(idir, "**", "*.nii*"), True), key=num)
        masks = {num(f): f for f in nii(os.path.join(mdir, "**", "*.nii*"), True)}
    else:
        allf = nii(os.path.join(root, "**", "*.nii*"), True)
        ism = lambda f: any(k in os.path.relpath(f, root).lower() for k in ("mask", "label", "seg"))
        imgs = sorted([f for f in allf if not ism(f)], key=num); masks = {num(f): f for f in allf if ism(f)}
    paired = [f for f in imgs if num(f) in masks]
    print(f"MedSeg: {len(imgs)} image volumes, {len(masks)} mask volumes, {len(paired)} pairs "
          f"(image folder: {idir or 'auto'}, mask folder: {mdir or 'auto'})")
    if not paired:
        raise SystemExit("No MedSeg image/mask pairs found; check --medseg points to the folder that contains the image and mask folders.")
    return paired, masks


def num(s):
    return int(re.findall(r"\d+", os.path.basename(s))[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lits"); ap.add_argument("--chaos"); ap.add_argument("--medseg"); ap.add_argument("--msd")
    ap.add_argument("--msd-hv", help="optional: MSD Task08_HepaticVessel folder, to record which MedSeg cases come from it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", type=int, default=0, help="keep rows already in OUT/manifest.csv (e.g. LiTS from import_png_lits.py)")
    a = ap.parse_args(); rows = []
    if a.lits:
        vols = sorted(nii(os.path.join(a.lits, "volume-*.nii*")), key=num)
        for v in vols:
            n = num(v); seg = sorted(nii(os.path.join(a.lits, f"segmentation-{n}.nii*")))[0]
            NIFTI_CASES[f"LiTS:{n:03d}"] = ("LiTS", f"{n:03d}", v, seg)
            if reuse_nifti_case(rows, "LiTS", f"{n:03d}", v, a.out, seg): print("LiTS", n, "(already exported, reused)")
            else: print("LiTS", n); export_case(rows, "LiTS", f"{n:03d}", nifti_axial_slices(v, seg), a.out)
    if a.chaos:
        pdirs = [p for p in glob.glob(os.path.join(a.chaos, "*")) if os.path.isdir(p) and os.path.basename(p).isdigit()]
        if not pdirs or not glob.glob(os.path.join(pdirs[0], "DICOM_anon", "*.dcm")):
            raise SystemExit(f"--chaos must point to Train_Sets/CT (numbered patient folders with DICOM_anon and Ground); got {a.chaos}")
        print(f"CHAOS CT patients found: {len(pdirs)}")
        pdirs = sorted(pdirs, key=lambda p: int(os.path.basename(p)))
        orient, _ = chaos_orientation(pdirs)
        for pdir in pdirs:
            export_case(rows, "CHAOS", f"{int(os.path.basename(pdir)):02d}", chaos_axial_slices(pdir, orient), a.out)
    lits_vols = sorted(nii(os.path.join(a.lits, "volume-*.nii*")), key=num) if a.lits else []
    if a.medseg:
        imgs, masks = medseg_pairs(a.medseg)
        # MedSeg images are taken from the Medical Segmentation Decathlon; exclude any case that is a LiTS scan
        if lits_vols:
            dup = match_to_lits(imgs, lits_vols)
        else:   # LiTS imported from PNG (import_png_lits.py): compare against the PNG stacks instead
            man = os.path.join(a.out, "manifest.csv")
            ref = pd.read_csv(man).query("source == 'LiTS'") if os.path.exists(man) else pd.DataFrame()
            if len(ref): dup = match_to_png(imgs, png_stack_signatures(ref))
            else:
                dup = {}; print("  WARNING: no LiTS reference available; MedSeg overlap with LiTS NOT checked")
        msg = f"MedSeg volumes: {len(imgs)}; matching a LiTS volume (excluded): {len(dup)} " + str({num(k): v for k, v in dup.items()})
        if a.msd_hv:   # provenance only: Task08 (MSKCC) cases are independent of LiTS and are kept
            hv = sorted(nii(os.path.join(a.msd_hv, "imagesTr", "hepaticvessel_*.nii*")), key=num)
            if hv:
                m_hv = match_to_lits([v for v in imgs if v not in dup], hv)
                msg += f"\nMedSeg volumes matching MSD Task08 hepatic vessel volumes (kept): {len(m_hv)}"
            else:
                print(f"  --msd-hv given but no hepaticvessel_*.nii* found in {a.msd_hv}/imagesTr; provenance check skipped")
        print(msg); os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, "medseg_lits_overlap.txt"), "w").write(msg + "\n")
        for v in imgs:
            n = num(v)
            if n in masks and v not in dup:
                NIFTI_CASES[f"MedSeg:{n:03d}"] = ("MedSeg", f"{n:03d}", v, masks[n])
                if reuse_nifti_case(rows, "MedSeg", f"{n:03d}", v, a.out, masks[n]): print("MedSeg", n, "(already exported, reused)")
                else: print("MedSeg", n); export_case(rows, "MedSeg", f"{n:03d}", nifti_axial_slices(v, masks[n]), a.out)
    if a.msd and not a.lits and os.path.exists(os.path.join(a.out, "manifest.csv")):
        ref = pd.read_csv(os.path.join(a.out, "manifest.csv")).query("source == 'LiTS'")
        msd = sorted(nii(os.path.join(a.msd, "imagesTr", "liver_*.nii*")), key=num)
        matched = len(match_to_png(msd, png_stack_signatures(ref))) if len(ref) else 0
        msg = f"MSD Task03 training volumes: {len(msd)}; matching a LiTS PNG stack: {matched}"
        print(msg); open(os.path.join(a.out, "msd_lits_overlap.txt"), "w").write(msg + "\n")
    if a.msd and a.lits:
        # quantify overlap between MSD Task03 and LiTS (reviewer 1, point 5)
        msd_all = sorted(nii(os.path.join(a.msd, "imagesTr", "liver_*.nii*")), key=num)
        uniq = {}
        for f in sorted(msd_all, key=lambda f: (num(f), not f.lower().endswith('.nii.gz'))):
            uniq.setdefault(num(f), f)                            # one file per MSD case; the distributed .nii.gz if present
        msd = [uniq[k] for k in sorted(uniq)]
        m = match_to_lits(msd, lits_vols)
        unmatched = sorted(set(uniq) - {num(f) for f in m})
        msg = (f"MSD Task03 training files: {len(msd_all)}; unique cases: {len(msd)}; matching a LiTS training volume "
               f"(any orientation): {len(m)}; unmatched MSD case numbers: {unmatched}")
        print(msg); open(os.path.join(a.out, "msd_lits_overlap.txt"), "w").write(msg + "\n")
    alignment_qc(rows, a.out)
    df = pd.DataFrame(rows); man = os.path.join(a.out, "manifest.csv")
    if a.append and os.path.exists(man):
        old = pd.read_csv(man); df = pd.concat([old[~old.source.isin(df.source.unique())], df])
    df.to_csv(man, index=False)
    print(df.groupby("source").agg(cases=("case_id", "nunique"), slices=("z", "size"), liver_slices=("has_liver", "sum")))


if __name__ == "__main__":
    main()
