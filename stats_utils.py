"""Paired statistics for the revised manuscript.

Primary analysis is at patient level (one mean value per test case), because slices from one
case are not independent. Slice-level tests are reported as secondary. Each refinement
configuration is compared with the U-Net baseline by a two-sided Wilcoxon signed-rank test;
p-values are adjusted across configurations with the Holm-Bonferroni procedure. Effect sizes
are median paired differences with 95% cluster (case-level) bootstrap confidence intervals."""
import numpy as np
from scipy.stats import wilcoxon


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj


def wilcoxon_paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return 1.0, 0.0
    res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    n = np.sum(d != 0)
    # matched-pairs rank-biserial correlation as effect size
    ranks = np.argsort(np.argsort(np.abs(d[d != 0]))) + 1
    r = (ranks[d[d != 0] > 0].sum() - ranks[d[d != 0] < 0].sum()) / (n * (n + 1) / 2)
    return float(res.pvalue), float(r)


def cluster_bootstrap_median_diff(df, col, a, b, case_col="case_id", n_boot=2000, seed=0):
    """Median of per-case mean differences (a - b) with a percentile bootstrap CI over cases."""
    rng = np.random.default_rng(seed)
    wide = df.pivot_table(index=[case_col, "slice_key"], columns="method", values=col).dropna(subset=[a, b])
    per_case = (wide[a] - wide[b]).groupby(level=0).mean().values
    est = float(np.median(per_case))
    boots = [np.median(rng.choice(per_case, len(per_case), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def med_iqr(x):
    x = np.asarray(x, float)
    q1, m, q3 = np.nanpercentile(x, [25, 50, 75])
    return m, q1, q3
