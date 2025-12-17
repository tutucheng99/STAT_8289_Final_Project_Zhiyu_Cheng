"""
Visualization for Occlusion Analysis.

Creates plots for:
1. Feature group sensitivity ranking (bar chart)
2. Comparison of approximate vs precise methods
3. Policy change distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_group_sensitivity(results: Dict,
                           title: str = "Feature Group Sensitivity",
                           metric: str = 'kl_mean',
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature group sensitivity ranking.

    Args:
        results: Dict with 'group_results' or 'permutation_sensitivity'
        title: Plot title
        metric: Metric to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Extract data
    if 'group_results' in results:
        data = results['group_results']
    elif 'permutation_sensitivity' in results:
        data = results['permutation_sensitivity']
        metric = 'mean_abs_change' if metric == 'kl_mean' else metric
    else:
        data = results

    groups = list(data.keys())
    values = [data[g].get(metric, 0) for g in groups]

    # Sort by value
    sorted_pairs = sorted(zip(groups, values), key=lambda x: x[1], reverse=True)
    groups, values = zip(*sorted_pairs)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(groups)))
    bars = ax.barh(groups, values, color=colors)

    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.invert_yaxis()  # Highest at top

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_approximate_vs_precise(approx_results: Dict,
                                 precise_results: Dict,
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare approximate and precise sensitivity results.

    Args:
        approx_results: Results from approximate analysis
        precise_results: Results from precise analysis
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Extract common groups
    approx_data = approx_results.get('permutation_sensitivity', {})
    precise_data = precise_results.get('group_results', {})

    common_groups = set(approx_data.keys()) & set(precise_data.keys())
    if not common_groups:
        # Try to match observation groups to covariate groups
        group_mapping = {
            'hand': 'hand',
            'bidding': 'bidding',
            'vulnerability': 'context',
            'position': 'context',
        }
        print("Note: Using group mapping for comparison")

    groups = sorted(common_groups)

    approx_vals = [approx_data[g].get('mean_abs_change', 0) for g in groups]
    precise_vals = [precise_data[g].get('kl_mean', 0) for g in groups]

    # Normalize for comparison
    approx_norm = np.array(approx_vals) / (max(approx_vals) + 1e-10)
    precise_norm = np.array(precise_vals) / (max(precise_vals) + 1e-10)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(x - width/2, approx_norm, width, label='Approximate (EBM)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, precise_norm, width, label='Precise (Re-inference)',
                   color='coral', alpha=0.8)

    ax.set_xlabel('Feature Group')
    ax.set_ylabel('Normalized Sensitivity')
    ax.set_title('Approximate vs Precise Sensitivity Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_policy_change_distribution(original_probs: np.ndarray,
                                     perturbed_probs: np.ndarray,
                                     group_name: str,
                                     figsize: Tuple[int, int] = (12, 4),
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of policy changes for a single group.

    Args:
        original_probs: (N, K) original policy
        perturbed_probs: (N, K) perturbed policy
        group_name: Name of occluded group
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    from .metrics import kl_divergence, jsd, tvd

    kl_vals = kl_divergence(original_probs, perturbed_probs)
    jsd_vals = jsd(original_probs, perturbed_probs)
    tvd_vals = tvd(original_probs, perturbed_probs)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # KL histogram
    axes[0].hist(kl_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axvline(np.mean(kl_vals), color='red', linestyle='--',
                    label=f'Mean: {np.mean(kl_vals):.4f}')
    axes[0].set_xlabel('KL Divergence')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'KL Distribution\n(Occluding {group_name})')
    axes[0].legend()

    # JSD histogram
    axes[1].hist(jsd_vals, bins=50, alpha=0.7, color='coral', edgecolor='white')
    axes[1].axvline(np.mean(jsd_vals), color='red', linestyle='--',
                    label=f'Mean: {np.mean(jsd_vals):.4f}')
    axes[1].set_xlabel('JSD')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'JSD Distribution\n(Occluding {group_name})')
    axes[1].legend()

    # TVD histogram
    axes[2].hist(tvd_vals, bins=50, alpha=0.7, color='seagreen', edgecolor='white')
    axes[2].axvline(np.mean(tvd_vals), color='red', linestyle='--',
                    label=f'Mean: {np.mean(tvd_vals):.4f}')
    axes[2].set_xlabel('Total Variation Distance')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'TVD Distribution\n(Occluding {group_name})')
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_combined_summary(results: Dict,
                          title: str = "Occlusion Analysis Summary",
                          figsize: Tuple[int, int] = (14, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create combined summary plot with multiple panels.

    Args:
        results: Full analysis results
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Group sensitivity ranking
    ax1 = fig.add_subplot(gs[0, 0])
    if 'group_results' in results:
        data = results['group_results']
        metric = 'kl_mean'
    else:
        data = results.get('permutation_sensitivity', {})
        metric = 'mean_abs_change'

    groups = list(data.keys())
    values = [data[g].get(metric, 0) for g in groups]
    sorted_pairs = sorted(zip(groups, values), key=lambda x: x[1], reverse=True)

    if sorted_pairs:
        groups, values = zip(*sorted_pairs)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(groups)))
        bars = ax1.barh(groups, values, color=colors)
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_title('Feature Group Sensitivity')
        ax1.invert_yaxis()

    # Panel 2: Method comparison (if both available)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'ranking' in results:
        ranking = results['ranking']
        names, scores = zip(*ranking[:5])  # Top 5
        ax2.bar(names, scores, color='coral', alpha=0.8)
        ax2.set_xlabel('Feature Group')
        ax2.set_ylabel('Sensitivity Score')
        ax2.set_title('Top 5 Most Sensitive Groups')
        ax2.tick_params(axis='x', rotation=45)

    # Panel 3: Summary statistics
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    summary_text = "Analysis Summary\n" + "="*40 + "\n\n"
    if 'n_samples' in results:
        summary_text += f"Samples analyzed: {results['n_samples']}\n"
    if 'method' in results:
        summary_text += f"Method: {results['method']}\n"
    if 'cv_r2' in results and results['cv_r2'] is not None:
        summary_text += f"EBM CV R²: {results['cv_r2']:.4f}\n"

    summary_text += "\nTop Sensitive Groups:\n"
    for name, score in results.get('ranking', [])[:5]:
        summary_text += f"  • {name}: {score:.4f}\n"

    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 4: Feature count by group
    ax4 = fig.add_subplot(gs[1, 1])
    from .feature_groups import OBSERVATION_GROUPS
    group_sizes = {name: end - start for name, (start, end) in OBSERVATION_GROUPS.items()}
    ax4.pie(group_sizes.values(), labels=group_sizes.keys(),
            autopct='%1.0f%%', colors=plt.cm.Pastel1(np.linspace(0, 1, len(group_sizes))))
    ax4.set_title('Observation Dimensions by Group')

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig
