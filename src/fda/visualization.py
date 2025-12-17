"""
Visualization functions for FDA analysis.

Key functions:
- plot_difference_curve: ALR difference curve with CI
- plot_difference_heatmap: Heatmap of mean differences
- plot_legal_rates: QC bar chart of action legal rates
- plot_sample_counts: QC heatmap of sample counts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_difference_curve(result: Dict,
                          cov_name: str,
                          action_name: str,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ALR difference curve Δ_b(x) = ALR^R - ALR^H with bootstrap CI.

    Uses point_estimate_curve (not bootstrap_mean) for the main curve.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    X = result['X_grid']

    # Point estimate curve + Bootstrap Pointwise CI
    ax.plot(X, result['point_estimate_curve'], 'k-', linewidth=2, label='Point Estimate')
    ax.fill_between(X, result['bootstrap_ci_lower'], result['bootstrap_ci_upper'],
                    alpha=0.3, color='blue', label='95% Bootstrap Pointwise CI')

    # Simultaneous band (if available)
    if 'bootstrap_simul_lower' in result:
        ax.fill_between(X, result['bootstrap_simul_lower'], result['bootstrap_simul_upper'],
                        alpha=0.15, color='orange', label='95% Simultaneous Band')

    # Zero line
    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='No difference')

    # Mark significant regions
    sig = result['bootstrap_significant_pointwise']
    y_min, y_max = ax.get_ylim()
    for i in range(len(X) - 1):
        if sig[i]:
            ax.axvspan(X[i], X[i+1], alpha=0.1, color='green')

    ax.set_xlabel(cov_name, fontsize=12)
    ax.set_ylabel(f'Δ = ALR$^R$ - ALR$^H$ ({action_name} / Pass)', fontsize=12)

    pct = result['bootstrap_pct_significant_pointwise']
    n_samples = result.get('n_samples', 'N/A')
    n_eps = result.get('n_episodes', 'N/A')
    ax.set_title(f'{action_name}: {pct*100:.1f}% pointwise significant\n(n={n_samples:,}, episodes={n_eps:,})',
                 fontsize=12)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_difference_heatmap(results_dict: Dict,
                            covariates: List[str],
                            actions: List[str],
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap of mean ALR differences across (covariate, action) pairs.

    Uses point_estimate_curve mean for each cell.
    """
    matrix = np.zeros((len(covariates), len(actions)))

    for i, cov in enumerate(covariates):
        for j, act in enumerate(actions):
            key = (cov, act)
            if key in results_dict:
                # Use point_estimate_curve
                matrix[i, j] = np.mean(results_dict[key]['point_estimate_curve'])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix, xticklabels=actions, yticklabels=covariates,
                cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'Mean ALR Difference'}, ax=ax)

    ax.set_title('Mean ALR Difference Δ = (π$^R$ - π$^H$) [conditioned on legality]', fontsize=14)
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Covariate', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_significance_heatmap(results_dict: Dict,
                              covariates: List[str],
                              actions: List[str],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap of percentage significant (pointwise) for each (covariate, action).
    """
    matrix = np.zeros((len(covariates), len(actions)))

    for i, cov in enumerate(covariates):
        for j, act in enumerate(actions):
            key = (cov, act)
            if key in results_dict:
                matrix[i, j] = results_dict[key]['bootstrap_pct_significant_pointwise'] * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix, xticklabels=actions, yticklabels=covariates,
                cmap='YlOrRd', vmin=0, vmax=100, annot=True, fmt='.1f',
                cbar_kws={'label': '% Pointwise Significant'}, ax=ax)

    ax.set_title('Percentage of X-range with Significant Difference (Pointwise)', fontsize=14)
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Covariate', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_legal_rates(legal_masks: np.ndarray,
                     action_names: List[str],
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    QC plot: Bar chart of action legal rates.
    """
    legal_rates = legal_masks.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ['red' if r < 0.005 else 'steelblue' for r in legal_rates]
    bars = ax.bar(range(len(action_names)), legal_rates, color=colors)

    ax.axhline(0.005, color='red', linestyle='--', alpha=0.7, label='Rare threshold (0.5%)')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='50%')

    ax.set_xticks(range(len(action_names)))
    ax.set_xticklabels(action_names, rotation=45, ha='right')
    ax.set_ylabel('Legal Rate', fontsize=12)
    ax.set_title('Action Legal Rates (QC)', fontsize=14)
    ax.legend(loc='upper right')

    # Annotate rare actions
    for i, rate in enumerate(legal_rates):
        if rate < 0.005:
            ax.annotate(f'{rate:.3f}', (i, rate + 0.01), ha='center', fontsize=8, color='red')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sample_counts(covariates: np.ndarray,
                       legal_masks: np.ndarray,
                       episode_ids: np.ndarray,
                       feature_names: List[str],
                       action_names: List[str],
                       cov_list: List[str],
                       action_list: List[int],
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    QC plot: Heatmap of effective sample/episode counts per (covariate, action).
    """
    sample_matrix = np.zeros((len(cov_list), len(action_list)))
    episode_matrix = np.zeros_like(sample_matrix)

    for i, cov_name in enumerate(cov_list):
        cov_idx = feature_names.index(cov_name)
        X = covariates[:, cov_idx]

        for j, action_idx in enumerate(action_list):
            legal_mask = legal_masks[:, action_idx]
            X_legal = X[legal_mask]
            ep_legal = episode_ids[legal_mask]

            x_lo, x_hi = np.percentile(X_legal, [5, 95])
            in_range = (X_legal >= x_lo) & (X_legal <= x_hi)

            sample_matrix[i, j] = in_range.sum()
            episode_matrix[i, j] = len(np.unique(ep_legal[in_range]))

    action_labels = [action_names[a] for a in action_list]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(sample_matrix, ax=axes[0], annot=True, fmt='.0f',
                xticklabels=action_labels, yticklabels=cov_list,
                cmap='Blues', cbar_kws={'label': 'Samples'})
    axes[0].set_title('Effective Sample Count (5%-95% quantile)', fontsize=12)

    sns.heatmap(episode_matrix, ax=axes[1], annot=True, fmt='.0f',
                xticklabels=action_labels, yticklabels=cov_list,
                cmap='Greens', cbar_kws={'label': 'Episodes'})
    axes[1].set_title('Effective Episode Count', fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_alr_distribution(alr_diff: np.ndarray,
                          legal_masks: np.ndarray,
                          action_names: List[str],
                          action_list: List[int],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Distribution of ALR differences for priority actions.
    """
    n_actions = len(action_list)
    fig, axes = plt.subplots(2, (n_actions + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, action_idx in enumerate(action_list):
        ax = axes[i]
        legal_mask = legal_masks[:, action_idx]
        diff = alr_diff[legal_mask, action_idx]

        ax.hist(diff, bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(diff.mean(), color='orange', linestyle='-', alpha=0.9, label=f'mean={diff.mean():.2f}')

        ax.set_title(f'{action_names[action_idx]} (n={legal_mask.sum():,})', fontsize=11)
        ax.set_xlabel('ALR Difference', fontsize=10)
        ax.legend(fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('ALR Difference Distribution by Action (conditioned on legality)', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
