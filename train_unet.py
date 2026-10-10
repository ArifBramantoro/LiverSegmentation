"""Train the U-Net from scratch on the training patients of LiTS and CHAOS (no warm start from earlier
checkpoints), select the epoch with the best validation Dice, and write per-slice test predictions.
Run: python train_unet.py --manifest DATA/manifest.csv --splits splits.csv --out runs/unet"""
import argparse, json, os, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from common import *


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); seed_everything(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC = ["LiTS", "CHAOS"]
    tr = load_table(a.manifest, a.splits, "train", SRC, liver_only=True)
    va = load_table(a.manifest, a.splits, "val", SRC, liver_only=True)
    te = load_table(a.manifest, a.splits, "test", SRC, liver_only=True)
    te_all = load_table(a.manifest, a.splits, "test", SRC, liver_only=False)   # all slices, for end-to-end evaluation
    # each source contributes equally in expectation: weight = 1 / (number of training slices of that source)
    w = tr.source.map(1.0 / tr.source.value_counts()).values
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    dl_tr = DataLoader(SliceDataset(tr, SEG_TRAIN_TF), a.batch, sampler=sampler, num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(SliceDataset(va, EVAL_TF), 16, shuffle=False, num_workers=a.workers)
    dl_te = DataLoader(SliceDataset(te, EVAL_TF), 16, shuffle=False, num_workers=a.workers)
    model = UNet().to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr); bce = nn.BCELoss()

    def per_slice_dice(loader):
        model.eval(); out = np.zeros(len(loader.dataset)); probs = {}
        with torch.no_grad():
            for x, y, idx in loader:
                p = model(x.to(dev)).cpu(); b = (p > 0.5).float()
                inter = (b * y).sum((1, 2, 3)); s = b.sum((1, 2, 3)) + y.sum((1, 2, 3))
                out[idx.numpy()] = torch.where(s > 0, 2 * inter / s.clamp(min=1), torch.ones_like(s)).numpy()
        return out

    log = []; best = -1; bad = 0; start = 1
    last = os.path.join(a.out, "unet_last.pth")
    if os.path.exists(last):   # resume after an interruption (laptop sleep, reboot)
        ck = torch.load(last, map_location=dev); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        log, best, bad, start = ck["log"], ck["best"], ck["bad"], ck["epoch"] + 1; print(f"resuming at epoch {start}")
        torch.set_rng_state(ck["rng"])
    for ep in range(start, a.epochs + 1):
        if bad >= a.patience: break
        model.train(); t0 = time.time(); tot = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); p = model(x)
            loss = 0.5 * bce(p, y) + 0.5 * dice_loss(p, y); loss.backward(); opt.step(); tot += loss.item()
        vd = per_slice_dice(dl_va).mean()
        log.append(dict(epoch=ep, train_loss=tot / len(dl_tr), val_dice=float(vd), seconds=time.time() - t0)); print(log[-1])
        if vd > best: best, bad = vd, 0; torch.save(model.state_dict(), os.path.join(a.out, "unet_best.pth"))
        else: bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), log=log, best=best, bad=bad, epoch=ep, rng=torch.get_rng_state()), last)
        if bad >= a.patience: break
    json.dump(dict(args=vars(a), log=log, best_val_dice=float(best), n_train_slices=len(tr), n_val_slices=len(va),
                   n_test_slices=len(te), n_train_cases=tr.case_id.nunique(), n_test_cases=te.case_id.nunique()),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=2)
    # write binary test predictions (used by run_refinement.py)
    model.load_state_dict(torch.load(os.path.join(a.out, "unet_best.pth"), map_location=dev)); model.eval()
    pdir = os.path.join(a.out, "pred"); os.makedirs(pdir, exist_ok=True)
    dl_all = DataLoader(SliceDataset(te_all, EVAL_TF), 16, shuffle=False, num_workers=a.workers)
    with torch.no_grad():
        for x, y, idx in dl_all:
            p = (model(x.to(dev)).cpu().numpy()[:, 0] > 0.5).astype(np.uint8)
            for k, i in enumerate(idx.numpy()):
                np.save(os.path.join(pdir, key2file(te_all.slice_key[i])), p[k])
    te.to_csv(os.path.join(a.out, "test_table.csv"), index=False)
    te_all.to_csv(os.path.join(a.out, "test_table_all_slices.csv"), index=False)
    # GPU inference time per slice (batch 1, model forward only, after warm-up)
    x0 = torch.zeros(1, 1, 256, 256, device=dev)
    with torch.no_grad():
        for _ in range(20): model(x0)
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200): model(x0)
        if dev.type == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) / 200 * 1000
    lg = json.load(open(os.path.join(a.out, "train_log.json"))); lg["inference_ms_per_slice"] = ms; lg["device"] = str(dev)
    lg["torch"] = torch.__version__; lg["cuda"] = torch.version.cuda
    json.dump(lg, open(os.path.join(a.out, "train_log.json"), "w"), indent=2)
    print(f"U-Net inference: {ms:.2f} ms per slice on {dev}")
    print("done; best val Dice", best)


if __name__ == "__main__":   # required on Windows for DataLoader worker processes
    main()
