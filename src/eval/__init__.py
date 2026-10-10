"""Lightweight M1 evaluation metrics and plots (WP6).

Modules
-------
metrics
    MAE, RMSE, empirical CRPS, and interval coverage on masked cells.
plots
    Forward-noising, mask-pattern, and imputation mean ± band figures.

Classes
-------
EvalTensors
    Packed truth / mean / samples / mask for metric computation.
MetricsTable
    Scalar metric bundle with dict / row export.
ImputationEvaluator
    Compute and format the tiny M1 metrics table.
EvalPlotPaths
    Default figure paths under ``artifacts/m1/``.
ImputationPlotData
    Packed tensors for the imputation band plot.

Functions
---------
masked_mae
    Mean absolute error on missing cells.
masked_rmse
    Root mean squared error on missing cells.
empirical_crps
    Ensemble CRPS averaged over missing cells.
interval_coverage
    Empirical coverage of a central predictive interval.
save_forward_noising_plot
    Save VP-SDE forward-noising panels.
save_mask_patterns_plot
    Save random vs block mask comparison.
save_imputation_bands
    Save ground truth, observations, mean ± quantile bands.
"""

from src.eval.metrics import (
    EvalTensors,
    ImputationEvaluator,
    MetricsTable,
    empirical_crps,
    interval_coverage,
    masked_mae,
    masked_rmse,
)
from src.eval.plots import (
    EvalPlotPaths,
    ImputationPlotData,
    save_forward_noising_plot,
    save_imputation_bands,
    save_mask_patterns_plot,
)

__all__ = [
    "EvalPlotPaths",
    "EvalTensors",
    "ImputationEvaluator",
    "ImputationPlotData",
    "MetricsTable",
    "empirical_crps",
    "interval_coverage",
    "masked_mae",
    "masked_rmse",
    "save_forward_noising_plot",
    "save_imputation_bands",
    "save_mask_patterns_plot",
]
