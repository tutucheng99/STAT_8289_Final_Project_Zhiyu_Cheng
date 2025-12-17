"""
Visualization for Policy Distillation.

Creates plots for:
1. Decision tree visualization
2. GAM shape functions
3. Fidelity vs complexity trade-off
4. Confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_fidelity_vs_complexity(results: Dict,
                                 metric: str = 'val_accuracy',
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot fidelity vs tree complexity.

    Args:
        results: Results from train_trees_multiple_depths
        metric: Metric to plot ('val_accuracy', 'top3_agreement')
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    depths = []
    values = []
    n_leaves = []

    for depth_name, data in results.items():
        if metric in data and data[metric] is not None:
            depths.append(depth_name)
            values.append(data[metric])
            n_leaves.append(data['n_leaves'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Fidelity vs depth
    ax1.plot(range(len(depths)), values, 'o-', markersize=8, linewidth=2)
    ax1.set_xticks(range(len(depths)))
    ax1.set_xticklabels(depths)
    ax1.set_xlabel('Max Depth')
    ax1.set_ylabel(metric.replace('_', ' ').title())
    ax1.set_title('Fidelity vs. Tree Depth')
    ax1.grid(True, alpha=0.3)

    # Fidelity vs leaves
    ax2.plot(n_leaves, values, 'o-', markersize=8, linewidth=2, color='coral')
    ax2.set_xlabel('Number of Leaves')
    ax2.set_ylabel(metric.replace('_', ' ').title())
    ax2.set_title('Fidelity vs. Model Complexity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_tree_feature_importance(importance_df,
                                  top_k: int = 15,
                                  figsize: Tuple[int, int] = (10, 8),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance from decision tree.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_k: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    df = importance_df.head(top_k)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(df['feature'], df['importance'], color=colors)

    ax.set_xlabel('Importance')
    ax.set_title('Decision Tree Feature Importance')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, df['importance']):
        ax.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_gam_shape_functions(gam_results: Dict,
                              feature_names: List[str],
                              X: np.ndarray,
                              action: int,
                              top_features: int = 9,
                              figsize: Tuple[int, int] = (15, 12),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot GAM shape functions for a single action.

    Args:
        gam_results: Results from train_gam_per_action
        feature_names: List of feature names
        X: Sample data for range estimation
        action: Action index to plot
        top_features: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    if action not in gam_results or 'gam' not in gam_results[action]:
        print(f"No GAM available for action {action}")
        return None

    gam = gam_results[action]['gam']

    # Get feature importances (approximated by range of partial dependence)
    importances = []
    for i in range(len(feature_names)):
        try:
            XX = gam.generate_X_grid(term=i, n=50)
            pdep = gam.partial_dependence(term=i, X=XX)
            importances.append((i, np.ptp(pdep)))  # peak-to-peak range
        except:
            importances.append((i, 0))

    # Sort by importance
    importances.sort(key=lambda x: x[1], reverse=True)
    top_indices = [imp[0] for imp in importances[:top_features]]

    # Create subplots
    n_cols = 3
    n_rows = (top_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, feat_idx in enumerate(top_indices):
        ax = axes[idx]
        try:
            XX = gam.generate_X_grid(term=feat_idx, n=100)
            pdep = gam.partial_dependence(term=feat_idx, X=XX)

            ax.plot(XX[:, feat_idx], pdep, 'b-', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(feature_names[feat_idx])
            ax.set_ylabel('Partial Dependence')
            ax.set_title(feature_names[feat_idx])
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes,
                    ha='center', va='center')

    # Hide empty subplots
    for idx in range(top_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'GAM Shape Functions for Action {action}', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_action_confusion_matrix(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  action_names: List[str] = None,
                                  top_k: int = 15,
                                  figsize: Tuple[int, int] = (12, 10),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix for top-k actions.

    Args:
        y_true: True actions
        y_pred: Predicted actions
        action_names: Optional action names
        top_k: Number of top actions to show
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import confusion_matrix

    # Get top-k actions by frequency
    unique, counts = np.unique(y_true, return_counts=True)
    top_actions = unique[np.argsort(counts)[::-1][:top_k]]

    # Filter to top actions
    mask = np.isin(y_true, top_actions)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Compute confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_actions)

    # Normalize
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

    # Labels
    if action_names:
        labels = [action_names[a] for a in top_actions]
    else:
        labels = [str(a) for a in top_actions]

    ax.set_xticks(range(len(top_actions)))
    ax.set_yticks(range(len(top_actions)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('True Action')
    ax.set_title('Action Confusion Matrix (Normalized)')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(top_actions)):
        for j in range(len(top_actions)):
            val = cm_normalized[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig


def plot_distillation_summary(tree_results: Dict,
                               gam_results: Dict,
                               figsize: Tuple[int, int] = (14, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create summary plot comparing distillation methods.

    Args:
        tree_results: Results from decision tree distillation
        gam_results: Results from GAM distillation
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Tree accuracy vs depth
    ax1 = fig.add_subplot(gs[0, 0])
    depths = list(tree_results.keys())
    accs = [tree_results[d]['val_accuracy'] for d in depths]
    ax1.bar(range(len(depths)), accs, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(depths)))
    ax1.set_xticklabels(depths)
    ax1.set_xlabel('Max Depth')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Decision Tree: Accuracy vs Depth')

    # Panel 2: GAM per-action accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    gam_actions = [a for a in gam_results.keys() if 'gam' in gam_results[a]]
    gam_accs = [gam_results[a]['val_accuracy'] for a in gam_actions]
    ax2.bar(range(len(gam_actions)), gam_accs, color='coral', alpha=0.8)
    ax2.set_xticks(range(len(gam_actions)))
    ax2.set_xticklabels([str(a) for a in gam_actions], rotation=45)
    ax2.set_xlabel('Action')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('GAM: Per-Action Accuracy')

    # Panel 3: Summary text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    # Find best tree
    best_depth = max(tree_results.keys(),
                     key=lambda d: tree_results[d]['val_accuracy'])
    best_tree_acc = tree_results[best_depth]['val_accuracy']
    best_tree_leaves = tree_results[best_depth]['n_leaves']

    summary_text = "DISTILLATION SUMMARY\n" + "="*50 + "\n\n"
    summary_text += "Decision Tree:\n"
    summary_text += f"  Best depth: {best_depth}\n"
    summary_text += f"  Validation accuracy: {best_tree_acc:.4f}\n"
    summary_text += f"  Number of leaves: {best_tree_leaves}\n\n"

    summary_text += "GAM (per-action):\n"
    summary_text += f"  Actions modeled: {len(gam_actions)}\n"
    if gam_accs:
        summary_text += f"  Mean accuracy: {np.mean(gam_accs):.4f}\n"
        summary_text += f"  Best action accuracy: {max(gam_accs):.4f}\n"

    ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Policy Distillation Results', fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {save_path}")

    return fig
