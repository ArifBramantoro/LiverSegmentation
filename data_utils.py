"""Torch-free helpers shared by all scripts."""
import pandas as pd


def key2file(slice_key):
    return slice_key.replace(":", "_").replace("/", "_") + ".npy"


def load_table(manifest, splits, split, sources=None, liver_only=False):
    m = pd.read_csv(manifest).merge(pd.read_csv(splits)[["case_id", "split"]], on="case_id")
    m = m[m.split.isin([split] if isinstance(split, str) else split)]
    if sources: m = m[m.source.isin(sources)]
    if liver_only: m = m[m.has_liver == 1]
    m["slice_key"] = m.case_id + "/" + m.z.astype(str)
    return m.sort_values(["case_id", "z"]).reset_index(drop=True)
