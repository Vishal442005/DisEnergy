"""Statistical validation suite: Shapiro-Wilk, Wilcoxon+Holm, Friedman+Nemenyi, Cliff's delta, BCa CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests


def shapiro_levene_audit(groups: dict[str, np.ndarray]) -> dict[str, object]:
    """Audit normality (Shapiro-Wilk per group) and homoscedasticity (Levene)."""
    shapiro_res = {}
    for name, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        sample = arr[:5000] if len(arr) > 5000 else arr
        if len(sample) >= 3 and np.ptp(sample) > 1e-12:
            stat, p = stats.shapiro(sample)
        else:
            stat, p = 1.0, 1.0
        shapiro_res[name] = {"W": float(stat), "p_value": float(p), "normal_at_0.05": bool(p >= 0.05)}

    arrays = [np.asarray(v, dtype=np.float64) for v in groups.values() if len(v) >= 2]
    if len(arrays) >= 2:
        lev_stat, lev_p = stats.levene(*arrays)
    else:
        lev_stat, lev_p = 0.0, 1.0

    return {
        "shapiro": shapiro_res,
        "levene": {"stat": float(lev_stat), "p_value": float(lev_p), "equal_var_0.05": bool(lev_p >= 0.05)},
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute non-parametric Cliff's delta effect size in [-1, 1]."""
    a = np.asarray(x, dtype=np.float64)[:, None]
    b = np.asarray(y, dtype=np.float64)[None, :]
    n_gt = float((a > b).sum())
    n_lt = float((a < b).sum())
    return (n_gt - n_lt) / float(len(x) * len(y))


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bias-corrected bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    point = float(np.mean(arr))
    if len(arr) < 2 or np.ptp(arr) < 1e-12:
        return point, point, point

    res = stats.bootstrap(
        (arr,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    if np.isnan(lo) or np.isnan(hi):
        rng = np.random.default_rng(seed)
        boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
        lo, hi = float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_wilcoxon_holm(
    reference: np.ndarray,
    baselines: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run two-sided Wilcoxon signed-rank tests vs baselines with Holm-Bonferroni correction."""
    ref = np.asarray(reference, dtype=np.float64)
    rows = []
    raw_p = []
    for name, comp in baselines.items():
        arr = np.asarray(comp, dtype=np.float64)
        diff = ref - arr
        if np.all(np.abs(diff) < 1e-12):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(ref, arr, alternative="two-sided", zero_method="wilcox")
        delta = cliffs_delta(ref, arr)
        mean_diff, ci_lo, ci_hi = bootstrap_ci(diff)
        rows.append(
            {
                "baseline": name,
                "wilcoxon_stat": float(stat),
                "p_raw": float(p),
                "cliffs_delta": float(delta),
                "mean_diff": float(mean_diff),
                "ci95_lo": float(ci_lo),
                "ci95_hi": float(ci_hi),
            }
        )
        raw_p.append(float(p))

    if raw_p:
        reject, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method="holm")
        for row, rej, padj in zip(rows, reject, p_adj):
            row["p_holm"] = float(padj)
            row["significant"] = bool(rej)
    return pd.DataFrame(rows)


def friedman_nemenyi(score_matrix: pd.DataFrame) -> dict[str, object]:
    """Omnibus Friedman test across datasets (rows) x methods (cols) + Nemenyi post-hoc."""
    cols = list(score_matrix.columns)
    arrays = [score_matrix[c].to_numpy(dtype=np.float64) for c in cols]
    if len(cols) >= 3 and len(score_matrix) >= 2:
        stat, p_val = stats.friedmanchisquare(*arrays)
        nemenyi_df = sp.posthoc_nemenyi_friedman(score_matrix.to_numpy(dtype=np.float64))
        nemenyi_df.index = cols
        nemenyi_df.columns = cols
    else:
        stat, p_val = 0.0, 1.0
        nemenyi_df = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    # Mean ranks (rank 1 = highest score)
    ranks = score_matrix.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()

    k = len(cols)
    n_ds = len(score_matrix)
    # Approximate Nemenyi critical difference at alpha=0.05 via Studentized range q_alpha / sqrt(2)
    q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0) if k >= 2 else 0.0
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * max(n_ds, 1))))

    return {
        "friedman_stat": float(stat),
        "p_value": float(p_val),
        "mean_ranks": mean_ranks.to_dict(),
        "critical_difference_0.05": cd,
        "nemenyi_p_matrix": nemenyi_df,
    }
