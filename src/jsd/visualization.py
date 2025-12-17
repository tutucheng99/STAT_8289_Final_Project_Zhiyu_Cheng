"""
Visualization functions for JSD analysis.

Creates publication-quality plots for:
- JSD distribution
- Sensitivity analysis (raw vs legal)
- JSD curves (Definition A and B)
- Feature importance
- High JSD state analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10


def plot_jsd_distribution(jsd_values: np.ndarray,
                          save_path: Optional[str] = None,
                          title: str = 'Distribution of JSD between pi^H and pi^R') -> plt.Figure:
    """
    Plot JSD distribution histogram with statistics.

    Args:
        jsd_values: (N,) JSD values
        save_path: path to save figure
        title: plot title

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(jsd_values, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Statistics
    median = np.median(jsd_values)
    p95 = np.percentile(jsd_values, 95)
    mean = jsd_values.mean()

    ax.axvline(median, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median:.4f}')
    ax.axvline(p95, color='orange', linestyle='--', linewidth=2,
               label=f'95%: {p95:.4f}')
    ax.axvline(mean, color='green', linestyle=':', linewidth=2,
               label=f'Mean: {mean:.4f}')

    ax.set_xlabel('JSD', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')

    # Add text box with statistics
    stats_text = f'N = {len(jsd_values):,}\nMean = {mean:.4f}\nStd = {jsd_values.std():.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_sensitivity_scatter(jsd_raw: np.ndarray, jsd_legal: np.ndarray,
                             sensitivity_stats: Dict,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot sensitivity analysis: JSD_raw vs JSD_legal scatter plot.

    Args:
        jsd_raw: (N,) JSD with all actions
        jsd_legal: (N,) JSD with legal actions only
        sensitivity_stats: dict from compute_sensitivity_stats
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot (subsample if too many points)
    n = len(jsd_raw)
    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        jsd_raw_plot = jsd_raw[idx]
        jsd_legal_plot = jsd_legal[idx]
    else:
        jsd_raw_plot = jsd_raw
        jsd_legal_plot = jsd_legal

    ax.scatter(jsd_raw_plot, jsd_legal_plot, alpha=0.2, s=1, c='steelblue')

    # y=x reference line
    max_val = max(jsd_raw.max(), jsd_legal.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x')

    ax.set_xlabel('JSD (raw, with smoothed illegal)', fontsize=12)
    ax.set_ylabel('JSD (legal-only, renormalized)', fontsize=12)
    ax.set_title(f'Sensitivity Analysis: corr = {sensitivity_stats["correlation"]:.4f}', fontsize=14)

    # Add statistics text box
    stats_text = (
        f'Mean |diff|: {sensitivity_stats["mean_abs_diff"]:.2e}\n'
        f'Max |diff|: {sensitivity_stats["max_abs_diff"]:.2e}\n'
        f'P95 |diff|: {sensitivity_stats["p95_abs_diff"]:.2e}\n'
        f'Correlation: {sensitivity_stats["correlation"]:.4f}'
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='lower right')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_jsd_curve_A(curve_result: Dict, cov_name: str,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot State-level JSD curve (Definition A) with bootstrap CI.

    Args:
        curve_result: result from fit_jsd_gam_A
        cov_name: covariate name
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    X_grid = curve_result['X_grid']
    point_est = curve_result['point_estimate']
    ci_lower = curve_result['ci_lower']
    ci_upper = curve_result['ci_upper']

    # Confidence band
    ax.fill_between(X_grid, ci_lower, ci_upper, alpha=0.3, color='steelblue',
                    label='95% CI (cluster bootstrap)')

    # Point estimate
    ax.plot(X_grid, point_est, color='steelblue', linewidth=2, label='GAM estimate')

    ax.set_xlabel(cov_name, fontsize=12)
    ax.set_ylabel('E[JSD | X]', fontsize=12)
    ax.set_title(f'JSD Curve A (State-level): {cov_name}', fontsize=14)

    # Add sample info
    info_text = f'n = {curve_result["n_samples"]:,}\neps = {curve_result["n_episodes"]:,}'
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_jsd_curve_B(curve_df: pd.DataFrame, cov_name: str,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Bin-level JSD curve (Definition B).

    Args:
        curve_df: DataFrame from compute_jsd_curve_B
        cov_name: covariate name
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(curve_df['bin_center'], curve_df['jsd_B'],
            'o-', color='darkorange', linewidth=2, markersize=8,
            label='JSD of averaged policies')

    ax.set_xlabel(cov_name, fontsize=12)
    ax.set_ylabel('JSD(E[pi^H|X], E[pi^R|X])', fontsize=12)
    ax.set_title(f'JSD Curve B (Bin-level): {cov_name}', fontsize=14)

    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_jsd_curves_comparison(curve_A: Dict, curve_B: pd.DataFrame,
                               comparison: pd.DataFrame, cov_name: str,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot both JSD curves (A and B) on the same figure for comparison.

    Args:
        curve_A: GAM result for Definition A
        curve_B: DataFrame for Definition B
        comparison: comparison DataFrame
        cov_name: covariate name
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Definition A (GAM curve with CI)
    X_grid = curve_A['X_grid']
    point_est = curve_A['point_estimate']
    ci_lower = curve_A['ci_lower']
    ci_upper = curve_A['ci_upper']

    ax.fill_between(X_grid, ci_lower, ci_upper, alpha=0.2, color='steelblue')
    ax.plot(X_grid, point_est, color='steelblue', linewidth=2,
            label='A: E[JSD(s) | X] (state-level)')

    # Definition B (bin points)
    ax.plot(curve_B['bin_center'], curve_B['jsd_B'],
            'o-', color='darkorange', linewidth=2, markersize=8,
            label='B: JSD(E[pi|X], E[pi|X]) (bin-level)')

    # Annotation
    ax.set_xlabel(cov_name, fontsize=12)
    ax.set_ylabel('JSD', fontsize=12)
    ax.set_title(f'JSD Curves Comparison: {cov_name}\n'
                 f'By Jensen inequality: JSD_B <= JSD_A', fontsize=14)

    # Add text explaining the difference
    mean_diff = comparison['A_minus_B'].mean()
    ax.text(0.95, 0.05, f'Mean(A - B) = {mean_diff:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_jsd_vs_covariates(covariates: np.ndarray, jsd_values: np.ndarray,
                           feature_names: List[str], cov_names: List[str],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot JSD vs multiple covariates in a grid.

    Args:
        covariates: (N, D) covariate matrix
        jsd_values: (N,) JSD values
        feature_names: list of all feature names
        cov_names: list of covariates to plot
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    n_covs = len(cov_names)
    n_cols = min(3, n_covs)
    n_rows = (n_covs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_covs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Subsample for plotting
    n = len(jsd_values)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
    else:
        idx = np.arange(n)

    for i, (ax, cov_name) in enumerate(zip(axes, cov_names)):
        cov_idx = feature_names.index(cov_name)
        x = covariates[idx, cov_idx]
        y = jsd_values[idx]

        ax.scatter(x, y, alpha=0.2, s=1, c='steelblue')
        ax.set_xlabel(cov_name, fontsize=10)
        ax.set_ylabel('JSD', fontsize=10)

        # Add correlation
        corr = np.corrcoef(covariates[:, cov_idx], jsd_values)[0, 1]
        ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide empty axes
    for i in range(len(cov_names), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('JSD vs Covariates', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_feature_importance(importance_df: pd.DataFrame, n_top: int = 15,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance from EBM.

    Args:
        importance_df: DataFrame with 'term' and 'importance' columns
        n_top: number of top features to show
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    top_df = importance_df.head(n_top).iloc[::-1]  # Reverse for horizontal bar

    ax.barh(range(len(top_df)), top_df['importance'], color='steelblue')
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['term'])
    ax.set_xlabel('Importance (mean |score|)', fontsize=12)
    ax.set_title('EBM Feature Importance for JSD Prediction', fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_jsd_heatmap(curve_results: Dict[str, Dict],
                     cov_names: List[str],
                     metric: str = 'point_estimate',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing JSD across covariates.

    Args:
        curve_results: Dict[cov_name, curve_result]
        cov_names: list of covariate names
        metric: 'point_estimate' or 'ci_width'
        save_path: path to save figure

    Returns:
        matplotlib Figure
    """
    set_plot_style()

    # Create matrix: rows = bins, cols = covariates
    n_bins = 10
    data = np.zeros((n_bins, len(cov_names)))

    for j, cov_name in enumerate(cov_names):
        if cov_name in curve_results:
            curve_B = curve_results[cov_name]['curve_B']
            jsd_B = curve_B['jsd_B'].values
            # Pad or truncate to n_bins
            if len(jsd_B) >= n_bins:
                data[:, j] = jsd_B[:n_bins]
            else:
                data[:len(jsd_B), j] = jsd_B
                data[len(jsd_B):, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(cov_names)))
    ax.set_xticklabels(cov_names, rotation=45, ha='right')
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f'Bin {i+1}' for i in range(n_bins)])
    ax.set_ylabel('Covariate Bin (low to high)')

    plt.colorbar(im, ax=ax, label='JSD_B')
    ax.set_title('JSD Heatmap (Bin-level, Definition B)', fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig
