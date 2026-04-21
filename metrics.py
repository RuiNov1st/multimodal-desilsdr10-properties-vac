"""
Evaluation metrics and visualization utilities.

This module provides:
- Standard regression metrics for model evaluation
- Scatter plot visualization of predictions vs. ground truth

Metrics are designed for continuous physical properties
(e.g., SFR, stellar mass, metallicity) in log space.

The scatter plot includes:
- 1:1 reference line
- 3-sigma outlier boundaries
- Residual distribution

Last updated: 2026-04-20
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------
def compute_metrics(pred_labels, true_labels):
    """
    Compute regression metrics for predicted physical properties.

    Metrics:
    - MSE   : mean squared error
    - RMSE  : root mean squared error
    - NRMSE : normalized RMSE
    - MAE   : mean absolute error
    - STD   : standard deviation of residuals
    - Bias  : mean residual
    - Outlier fraction (|Δ| > 3σ)
    - NMAD  : normalized median absolute deviation (robust scatter)
    - R     : Pearson correlation coefficient

    Parameters
    ----------
    pred_labels : np.ndarray
        Model predictions
    true_labels : np.ndarray
        Ground truth values

    Returns
    -------
    tuple of metrics
    """

    def outlier_compute(deltax, std):
        # Fraction of points outside 3-sigma
        return len(np.where(np.abs(deltax) > 3 * std)[0]) / len(deltax)

    # Residuals
    deltax = pred_labels - true_labels

    # Error metrics
    mse = np.mean(deltax**2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(true_labels) - np.min(true_labels) + 1e-5)
    mae = np.mean(np.abs(deltax))

    # Distribution statistics
    std = np.std(deltax)
    bias = np.mean(deltax)

    # Outlier fraction
    outlier = outlier_compute(deltax, std)

    # Robust scatter estimator
    nmad = 1.4826 * np.nanmedian(np.abs(deltax))

    # Correlation
    r, _ = pearsonr(pred_labels, true_labels)

    return mse, rmse, nrmse, mae, std, bias, outlier, nmad, r


# ------------------------------------------------------------
# Scatter plot visualization
# ------------------------------------------------------------
def make_scatter_plot(preds, labels, std, outlier, config, name):
    """
    Generate prediction vs. truth scatter plot with residuals.

    The figure includes:
    - Top panel: prediction vs. true values
    - Bottom panel: residuals (Δ = pred - true)
    - 1:1 reference line
    - ±3σ outlier boundaries

    Parameters
    ----------
    preds : np.ndarray
        Predicted values
    labels : np.ndarray
        Ground truth values
    std : float
        Standard deviation of residuals
    outlier : float
        Outlier fraction
    config : dict
        Experiment configuration (used for output path)
    name : str
        Property name (e.g., SFR, LGM, OH)
    """

    # Output file path
    file_name = f"./output/{config['Experiment']['Run_name']}/scatter_{name}.png"

    # Create figure (two panels)
    fig = plt.figure(figsize=(3, 3))
    axis = fig.add_axes([0, 0.4, 1, 1])   # main scatter
    axis2 = fig.add_axes([0, 0, 1, 0.3])  # residuals

    # Labels
    axis2.set_xlabel(f'$\log({name})_{{true}}$')
    axis.set_ylabel(f'$\log({name})_{{pred}}$')
    axis2.set_ylabel(f'$\Delta \log({name})$')

    # Axis limits
    min_lim = np.min(labels)
    max_lim = np.max(labels)

    # 1:1 reference line
    axis.plot([min_lim, max_lim], [min_lim, max_lim], 'k-', lw=1)
    axis.set_xlim([min_lim, max_lim])
    axis.set_ylim([min_lim, max_lim])

    # 3-sigma outlier boundaries
    x = np.arange(min_lim, max_lim, 0.1)
    axis.plot(x, x + 3 * std, 'steelblue', linestyle='--', lw=1)
    axis.plot(x, x - 3 * std, 'steelblue', linestyle='--', lw=1)

    # Scatter plot
    axis.scatter(labels, preds, marker='o', color='k', s=0.3, alpha=0.3)

    # Metrics annotation
    axis.text(
        0.1, 0.95,
        f'$\sigma$: {std:.4f}\n $\eta$:{outlier:.4f}',
        transform=axis.transAxes,
        verticalalignment='top',
        horizontalalignment='left'
    )

    # Residual plot
    deltax = preds - labels
    axis2.plot([min_lim, max_lim], [0, 0], 'k-', lw=1)
    axis2.scatter(labels, deltax, color='k', marker='o', s=0.3, alpha=0.3)
    axis2.set_xlim([min_lim, max_lim])
    axis2.set_ylim([np.min(deltax), np.max(deltax)])

    # Save figure
    plt.savefig(file_name, dpi=300, bbox_inches='tight')