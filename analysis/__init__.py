"""Metrics Reloaded evaluation, statistical significance testing, figures, and LaTeX export."""

from .metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    worst_decile_score,
)
from .stats_tests import (
    bootstrap_ci,
    cliffs_delta,
    friedman_nemenyi,
    paired_wilcoxon_holm,
    shapiro_levene_audit,
)

__all__ = [
    "bootstrap_ci",
    "cliffs_delta",
    "compute_classification_metrics",
    "compute_segmentation_metrics",
    "friedman_nemenyi",
    "paired_wilcoxon_holm",
    "shapiro_levene_audit",
    "worst_decile_score",
]
