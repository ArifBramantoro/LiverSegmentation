"""Cross-platform driver (Windows, macOS, Linux). Re-running it skips steps whose outputs already exist,
and the training scripts resume from their last completed epoch, so it is safe to stop and restart.

Laptop example (LiTS as existing PNGs, CHAOS as DICOM, MedSeg and MSD as NIfTI):
  python run_pipeline.py --lits-png "D:/Master/Haziq/Another_attempt" --chaos "D:/Master/Haziq/chaos/Train_Sets/CT" \
         --medseg "D:/data/MedSeg" --msd "D:/data/Task03_Liver"
With raw LiTS NIfTI instead:  --lits "D:/data/LiTS"  (no --lits-png)
"""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--lits"); ap.add_argument("--lits-png"); ap.add_argument("--rewindow", type=int, default=1)
ap.add_argument("--chaos", required=True); ap.add_argument("--medseg"); ap.add_argument("--msd"); ap.add_argument("--msd-hv")
ap.add_argument("--data", default="data"); ap.add_argument("--runs", default="runs"); ap.add_argument("--results", default="results")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-epochs", type=int, default=0, help="testing only: cap training epochs")
ap.add_argument("--fresh", type=int, default=0, help="1 = delete previous runs/results/figures and redo export checks (keeps data PNGs and splits.csv)")
a = ap.parse_args(); py = sys.executable; D, R = a.data, a.runs
if a.fresh:
    import shutil
    for d in (R, a.results, "figures"):
        if os.path.isdir(d): shutil.rmtree(d); print(f"[fresh] removed {d}")
    if os.path.exists(f"{D}/.export_done"): os.remove(f"{D}/.export_done")
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)


LOG = open("run_log.txt", "a", encoding="utf-8", errors="replace")


def say(msg):
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


say(f"===== run started {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(sys.argv[1:])}")


def step(name, done_file, cmd):
    """Run one step; its full output goes to the screen and to run_log.txt (warnings about package updates omitted)."""
    if done_file and os.path.exists(done_file):
        say(f"[skip] {name}"); return
    say(f"[run ] {name}: {' '.join(cmd)}"); t = time.time()
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1", PYTHONUNBUFFERED="1")
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    for line in pr.stdout:
        if "check_for_updates()" in line or "A new version of Albumentations" in line: continue
        print(line, end="", flush=True); LOG.write(line); LOG.flush()
    if pr.wait() != 0:
        say(f"[FAILED] {name} (exit code {pr.returncode}). Send run_log.txt."); sys.exit(1)
    say(f"[done] {name} in {(time.time() - t) / 3600:.2f} h")


step("verification tests", None, [py, "tests/test_glcm.py"]); step("solver tests", None, [py, "tests/test_level_set.py"])
if a.lits_png:
    step("import LiTS PNGs", f"{D}/manifest.csv", [py, "import_png_lits.py", "--root", a.lits_png, "--out", D, "--rewindow", str(a.rewindow)])
    extra = ["--append", "1"]
else:
    extra = ["--lits", a.lits]
cmd = [py, "prepare_data.py", "--chaos", a.chaos, "--out", D] + extra
for k, v in [("--medseg", a.medseg), ("--msd", a.msd), ("--msd-hv", a.msd_hv)]:
    if v: cmd += [k, v]
step("export CHAOS/MedSeg and overlap checks", f"{D}/.export_done", cmd); open(f"{D}/.export_done", "w").close()
step("patient-level split", "splits.csv", [py, "make_splits.py", "--manifest", f"{D}/manifest.csv", "--out", "splits.csv"])
W = ["--workers", str(a.workers)] + (["--epochs", str(a.max_epochs)] if a.max_epochs else [])
step("U-Net", f"{R}/unet/test_table.csv", [py, "train_unet.py", "--manifest", f"{D}/manifest.csv", "--splits", "splits.csv", "--out", f"{R}/unet"] + W)
srcs = "LiTS,CHAOS,MedSeg" if a.medseg else "LiTS,CHAOS"
step("detector held-out", f"{R}/det_heldout/detector_metrics.csv", [py, "train_detector.py", "--manifest", f"{D}/manifest.csv", "--splits", "splits.csv",
     "--protocol", "heldout", "--sources", srcs, "--out", f"{R}/det_heldout"] + W)
for h in srcs.split(","):
    step(f"detector leave-out {h}", f"{R}/det_loso_{h}/detector_metrics.csv", [py, "train_detector.py", "--manifest", f"{D}/manifest.csv",
         "--splits", "splits.csv", "--protocol", "loso", "--holdout", h, "--sources", srcs, "--out", f"{R}/det_loso_{h}"] + W)
if a.medseg and a.msd and a.lits:   # scan matching with per-volume scores, then repeated MedSeg scans (detector evaluation)
    step("overlap scores", f"{D}/overlap_scores.csv", [py, "overlap_check.py", "--lits", a.lits, "--msd", a.msd, "--medseg", a.medseg, "--out", D])
    step("MedSeg repeated scans", f"{D}/medseg_duplicates.txt", [py, "medseg_duplicates.py", "--overlap", f"{D}/overlap_scores.csv",
         "--manifest", f"{D}/manifest.csv", "--splits", "splits.csv", "--runs", R, "--out", D])
px = "1" if a.lits_png else "0"
step("refinement", f"{R}/refine/per_slice_results.csv", [py, "run_refinement.py", "--unet-dir", f"{R}/unet", "--out", f"{R}/refine",
     "--workers", str(max(1, (os.cpu_count() or 2) - 1)), "--pixel-units", px])
thr = json.load(open(f"{R}/det_heldout/det_log.json"))["youden_threshold"]
step("analysis", None, [py, "analyse.py", "--results", f"{R}/refine/per_slice_results.csv", "--out", a.results, "--unet-dir", f"{R}/unet",
     "--det-pred", f"{R}/det_heldout/test_predictions.csv", "--det-threshold", str(thr)])
step("figures", None, [py, "make_figures.py", "static"])
step("figures qualitative", None, [py, "make_figures.py", "qualitative", "--unet-dir", f"{R}/unet", "--results", f"{R}/refine/per_slice_results.csv"])
step("figures detector", None, [py, "make_figures.py", "detector", "--det-dirs", f"{R}/det_heldout"] + [f"{R}/det_loso_{h}" for h in srcs.split(",")])
say("All steps finished. Upload run_log.txt, data/*.txt, data/alignment_check.csv, data/manifest.csv, splits.csv, and the runs, results and figures folders.")
