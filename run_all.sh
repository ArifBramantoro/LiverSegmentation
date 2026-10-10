#!/usr/bin/env bash
# Full re-run for the PLOS ONE revision (PONE-D-26-17147). Edit the RAW paths first.
set -euo pipefail
RAW=/path/to/raw; DATA=data; RUNS=runs; RES=results
python tests/test_glcm.py && python tests/test_level_set.py          # solver and GLCM verification
python prepare_data.py --lits $RAW/LiTS --chaos $RAW/CHAOS/Train_Sets/CT --medseg $RAW/MedSeg --msd $RAW/Task03_Liver --msd-hv $RAW/Task08_HepaticVessel --out $DATA   # --msd-hv optional
python make_splits.py --manifest $DATA/manifest.csv --out splits.csv --seed 42
python train_unet.py --manifest $DATA/manifest.csv --splits splits.csv --out $RUNS/unet
python train_detector.py --manifest $DATA/manifest.csv --splits splits.csv --protocol heldout --out $RUNS/det_heldout
for H in LiTS CHAOS MedSeg; do
  python train_detector.py --manifest $DATA/manifest.csv --splits splits.csv --protocol loso --holdout $H --out $RUNS/det_loso_$H
done
python run_refinement.py --unet-dir $RUNS/unet --out $RUNS/refine --workers 8 --legacy 1
python analyse.py --results $RUNS/refine/per_slice_results.csv --out $RES --unet-dir $RUNS/unet \
       --det-pred $RUNS/det_heldout/test_predictions.csv --det-threshold $(python -c "import json;print(json.load(open('$RUNS/det_heldout/det_log.json'))['youden_threshold'])")
python make_figures.py static
python make_figures.py qualitative --unet-dir $RUNS/unet --results $RUNS/refine/per_slice_results.csv
python make_figures.py detector --det-dirs $RUNS/det_heldout $RUNS/det_loso_LiTS $RUNS/det_loso_CHAOS $RUNS/det_loso_MedSeg
