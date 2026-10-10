"""Liver-slice detector (ResNet50, ImageNet initialisation, gradual unfreezing, weighted BCE).
Two protocols (reviewer 1, point 4):
  --protocol heldout : train on training patients of all sources, test on held-out patients of each source
  --protocol loso    : leave-one-source-out; train on two sources, test on every case of the third
The decision threshold is chosen on the validation set (Youden's J) and is also reported at 0.5.
Run: python train_detector.py --manifest DATA/manifest.csv --splits splits.csv --protocol heldout --out runs/det_heldout"""
import argparse, json, os, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from common import *
from metrics import det_summary


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", choices=["heldout", "loso"], default="heldout"); ap.add_argument("--holdout", default=None)
    ap.add_argument("--sources", default="LiTS,CHAOS,MedSeg"); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="non-liver:liver ratio kept per training case")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--pretrained", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); seed_everything(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = a.sources.split(",")
    if a.protocol == "heldout":
        tr = load_table(a.manifest, a.splits, "train", S); va = load_table(a.manifest, a.splits, "val", S)
        te = load_table(a.manifest, a.splits, "test", S)
    else:
        train_src = [s for s in S if s != a.holdout]
        tr = load_table(a.manifest, a.splits, "train", train_src); va = load_table(a.manifest, a.splits, "val", train_src)
        te = load_table(a.manifest, a.splits, ["train", "val", "test"], [a.holdout])
    # per-case balancing of the training set only (test sets keep their natural class ratio)
    rng = np.random.default_rng(a.seed); keep = []
    for _, g in tr.groupby("case_id"):
        pos = g[g.has_liver == 1]; neg = g[g.has_liver == 0]
        k = min(len(neg), int(round(a.neg_ratio * max(1, len(pos)))))
        keep.append(pd.concat([pos, neg.iloc[rng.choice(len(neg), k, replace=False)] if k else neg.iloc[:0]]))
    tr = pd.concat(keep).reset_index(drop=True)
    pos_weight = torch.tensor([(tr.has_liver == 0).sum() / max(1, (tr.has_liver == 1).sum())], device=dev)
    dl = lambda t, tf, sh: DataLoader(SliceDataset(t, tf, "det"), a.batch, shuffle=sh, num_workers=a.workers)
    dl_tr, dl_va, dl_te = dl(tr, DET_TRAIN_TF, True), dl(va, EVAL_TF, False), dl(te, EVAL_TF, False)
    model = resnet50_detector(bool(a.pretrained)).to(dev); crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def stage(ep):  # gradual unfreezing: head (ep 1-2), + layer4 (3-4), + layer3 (5+)
        groups = {"fc": ep >= 1, "layer4": ep >= 3, "layer3": ep >= 5}
        for n, p in model.named_parameters(): p.requires_grad = any(n.startswith(k) and v for k, v in groups.items())
        head = [p for n, p in model.named_parameters() if n.startswith("fc")]
        body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
        return torch.optim.Adam([{"params": head, "lr": a.lr}] + ([{"params": body, "lr": a.lr / 10}] if body else []))

    def predict(loader):
        model.eval(); pr = np.zeros(len(loader.dataset))
        with torch.no_grad():
            for x, _, idx in loader: pr[idx.numpy()] = torch.sigmoid(model(x.to(dev))).cpu().numpy()[:, 0]
        return pr

    log = []; best = 1e9; bad = 0; start = 1
    last = os.path.join(a.out, "det_last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev); model.load_state_dict(ck["model"])
        log, best, bad, start = ck["log"], ck["best"], ck["bad"], ck["epoch"] + 1; print(f"resuming at epoch {start}")
    for ep in range(start, a.epochs + 1):
        if bad >= a.patience: break
        opt = stage(ep); model.train(); t0 = time.time(); tot = 0
        for x, y, _ in dl_tr:
            opt.zero_grad(); loss = crit(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step(); tot += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for x, y, _ in dl_va: vl += crit(model(x.to(dev)), y.to(dev)).item()
        vl /= max(1, len(dl_va)); log.append(dict(epoch=ep, train_loss=tot / len(dl_tr), val_loss=vl, s=time.time() - t0)); print(log[-1])
        if vl < best: best, bad = vl, 0; torch.save(model.state_dict(), os.path.join(a.out, "det_best.pth"))
        else: bad += 1
        torch.save(dict(model=model.state_dict(), log=log, best=best, bad=bad, epoch=ep), last)
        if bad >= a.patience: break
    model.load_state_dict(torch.load(os.path.join(a.out, "det_best.pth"), map_location=dev))
    pv = predict(dl_va); fpr, tpr, thr = roc_curve(va.has_liver, pv)
    fin = np.isfinite(thr); j = (tpr - fpr)[fin]
    # Youden's J on finite thresholds; fall back to 0.5 if validation data cannot separate the classes
    t_youden = float(thr[fin][np.argmax(j)]) if fin.any() and j.max() > 0 else 0.5
    te["prob"] = predict(dl_te); te.to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)
    x0 = torch.zeros(1, 3, 256, 256, device=dev)
    with torch.no_grad():
        for _ in range(20): model(x0)
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200): model(x0)
        if dev.type == "cuda": torch.cuda.synchronize()
    inf_ms = (time.time() - t0) / 200 * 1000

    res = [dict(source=s, rule=r, **det_summary(g, t)) for s, g in te.groupby("source") for r, t in [("0.5", 0.5), ("youden_val", t_youden)]]
    pd.DataFrame(res).to_csv(os.path.join(a.out, "detector_metrics.csv"), index=False)
    json.dump(dict(args=vars(a), log=log, youden_threshold=t_youden, inference_ms_per_slice=inf_ms, train_sources=sorted(tr.source.unique()),
                   n_train_slices=len(tr), pos_weight=float(pos_weight)), open(os.path.join(a.out, "det_log.json"), "w"), indent=2)
    print(pd.DataFrame(res))


if __name__ == "__main__":   # required on Windows for DataLoader worker processes
    main()
