# Liver segmentation revision code (PONE-D-26-17147)

Corrected and fully reproducible pipeline for the revised manuscript *Hybrid liver segmentation in computed
tomography using U-Net initialization and GLCM-guided level set refinement*. It replaces the two notebooks in the
original repository and addresses Reviewer 1 points 1-9, 14-17 and Reviewer 2's reproducibility requests.

## What changed relative to the original notebooks
| Issue in original code | Fix here |
|---|---|
| Update `phi += dt*(F - gamma*kappa)`: no `|grad phi|`, curvature sign reversed (anti-diffusive) | `phi_t = |grad phi|(gamma*kappa - F)` with upwinding, CFL-bounded `dt` (`level_set.py`) |
| Gaussian smoothing of `phi` every iteration, clip to +/-1000 | Removed; Sussman PDE re-initialisation every 10 iterations, narrow band 3 px |
| Edge variant: `F = g > 0` everywhere, so the region can only shrink | Geodesic active contour: balloon + curvature weighted by `g`, advection `grad g . grad phi` |
| "Curvature-only" and "gradient-based" loops were the same rule | Seven clearly named configurations incl. eroded-initialisation control and texture-off ablations |
| Erosion 5x5 square x5 (figure) or x1 (evaluation); paper said 3x3 circle | Disc of radius 2 px, configurable |
| GLCM: window 7, angle 0 only, contrast not normalised | Window 9, angles 0/45/90/135, maps normalised to [0, 1]; exact and ~350x faster (`glcm_fast.py`) |
| `random_split` over slices (adjacent slices in train and test) | Patient-level 70/10/20 split per source (`make_splits.py`) |
| Test subset kept `train_transform` (random augmentation at test time) | Deterministic `EVAL_TF` for validation and test |
| U-Net warm-started from checkpoints trained on other splits | Trained from scratch, best epoch by validation Dice, early stopping |
| Sampler weights taken from the first N indices, not the training subset | Weights computed on the training table itself |
| No HD95/ASSD, no statistics | `metrics.py`, `stats_utils.py`, `analyse.py` (patient-level Wilcoxon + Holm, cluster bootstrap CIs) |
| Detector evaluated on full datasets it may have been trained on | Held-out-patient and leave-one-source-out protocols, validation-chosen threshold, ROC curves |

## Running on a Windows laptop (CPU only)
```
pip install -r requirements.txt
python run_pipeline.py --lits-png "C:/.../Another_attempt" --chaos "C:/.../chaos/Train_Sets/CT" --medseg "C:/.../MedSeg"
```
* `--lits-png` imports the existing PNG export (`LITS/1_liver`, `LITS/0_non-liver`, `LITS_masks`) instead of raw NIfTI.
  It re-windows to [-100, 400] HU after checking that liver masks contain liver-like values; pixel spacing is unknown,
  so HD95/ASSD are reported in pixels. The MedSeg and MSD overlap checks compare against the PNG stacks.
* Safe to interrupt: re-running the same command skips finished steps, and training resumes from the last epoch.
* Keep the laptop on mains power with sleep disabled. Expect several days of computation in total.
* On a desktop with an NVIDIA GPU, install the CUDA build of PyTorch first; worker processes are Windows-safe (default 4).

## Data alignment safeguards (added after the first full run)
* NIfTI labels are voxel-aligned with their images, so they now receive the image's orientation transform.
  Reorienting each file from its own header mirrored the masks of LiTS cases whose label header differs.
* `prepare_data.py` checks every exported case (mask boundary on image edges, mask inside soft tissue, compared
  with 3 in-plane flips). A clearly mirrored NIfTI mask is re-exported corrected in voxel space; anything ambiguous
  stops the run. Results: `data/alignment_check.csv` and `data/alignment_check.txt`.
* Scan matching (MSD and MedSeg against LiTS) ignores header orientation, requires identical array dimensions and
  is confirmed at a finer resolution; MSD duplicates (.nii and .nii.gz) are counted once (the .nii.gz copy is used).
  Signatures are area-averaged to 24^3 (and 48^3) voxels, which commutes exactly with axis flips, so a flipped copy
  gives the same signature. `python overlap_check.py --lits ... --msd ... --medseg ...` re-runs only these checks
  and writes the per-volume scores to data/overlap_scores.csv.
* `medseg_duplicates.py` finds repeated scans within MedSeg (nine pairs) and recomputes the detector metrics without
  held-out test cases that have a copy in another partition, counting each scan once when MedSeg is the unseen source.
  The unfiltered files are kept as test_predictions_all.csv and detector_metrics_all.csv.

## Run order
See `run_all.sh`. Expected order of magnitude on the Titan RTX workstation: data export 1-2 h; U-Net 3-6 h;
detector 4 x 1-2 h; refinement of all test slices with 8 CPU workers a few hours (GLCM is now about 25 ms per slice).

## Verification status
* `tests/test_glcm.py`: fast GLCM equals skimage `graycomatrix`/`graycoprops` exactly.
* `tests/test_level_set.py`: corrected curvature flow matches the analytic disc solution (r^2 = r0^2 - 2*gamma*t) within
  0.2 px; the original rule grows the disc; the original edge rule reduces a perfect mask to Dice 0.15 on a synthetic disc.
  Results are written to `tests/verification_results.json` and plotted as S1 Fig.
* `smoke_test.py`: runs refinement, statistics and end-to-end gating on synthetic ellipses.
* `train_unet.py` and `train_detector.py` (both protocols) were run end to end on synthetic data with PyTorch 2.4.1 (CPU),
  followed by refinement, analysis and figure generation. This caught and fixed a bug where the Youden threshold could be
  infinite. A full run still needs the workstation GPU: one 8-slice U-Net step takes about 40 s on a single CPU core.

## Outputs that feed the manuscript
`results/seg_summary.csv` and `seg_table.tex` (Table 6), `runs/det_*/detector_metrics.csv` (Table 5), `results/end_to_end.csv`
(Table 7), `results/runtime.csv`, `figures/*.tif`. Deposit `splits.csv`, `manifest.csv` (without image paths if preferred),
trained weights and per-slice CSVs in a public repository with a DOI (e.g. Zenodo) and cite it in the Data Availability statement.
