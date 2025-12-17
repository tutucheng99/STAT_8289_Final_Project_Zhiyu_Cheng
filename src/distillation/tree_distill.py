"""
Decision Tree Distillation for Policy Interpretability.

Trains decision trees to approximate the black-box policy π^R,
extracting interpretable rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path


def train_decision_tree(X_train: np.ndarray,
                        y_train: np.ndarray,
                        max_depth: Optional[int] = None,
                        min_samples_leaf: int = 50,
                        class_weight: str = 'balanced',
                        seed: int = 42) -> DecisionTreeClassifier:
    """
    Train a decision tree to distill π^R.

    Args:
        X_train: Training features, shape (N, D)
        y_train: Training labels (argmax of π^R), shape (N,)
        max_depth: Maximum tree depth (None for unlimited)
        min_samples_leaf: Minimum samples per leaf
        class_weight: Class weighting strategy
        seed: Random seed

    Returns:
        Trained DecisionTreeClassifier
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
    )
    tree.fit(X_train, y_train)
    return tree


def evaluate_tree(tree: DecisionTreeClassifier,
                  X: np.ndarray,
                  y: np.ndarray,
                  pi_R: np.ndarray = None) -> Dict:
    """
    Evaluate decision tree performance.

    Args:
        tree: Trained decision tree
        X: Features
        y: True labels
        pi_R: Original policy probabilities (optional)

    Returns:
        Dict with evaluation metrics
    """
    y_pred = tree.predict(X)
    accuracy = (y_pred == y).mean()

    # Tree complexity metrics
    n_leaves = tree.get_n_leaves()
    depth = tree.get_depth()

    metrics = {
        'accuracy': float(accuracy),
        'n_leaves': int(n_leaves),
        'depth': int(depth),
        'n_features_used': int(len(set(tree.tree_.feature[tree.tree_.feature >= 0]))),
    }

    # Top-k agreement if pi_R provided
    if pi_R is not None:
        top3 = np.argsort(pi_R, axis=1)[:, -3:]
        top3_agreement = np.mean([y_pred[i] in top3[i] for i in range(len(y_pred))])
        metrics['top3_agreement'] = float(top3_agreement)

        top5 = np.argsort(pi_R, axis=1)[:, -5:]
        top5_agreement = np.mean([y_pred[i] in top5[i] for i in range(len(y_pred))])
        metrics['top5_agreement'] = float(top5_agreement)

    return metrics


def extract_rules(tree: DecisionTreeClassifier,
                  feature_names: List[str],
                  action_names: List[str] = None,
                  max_depth: int = None) -> str:
    """
    Extract human-readable rules from decision tree.

    Args:
        tree: Trained decision tree
        feature_names: List of feature names
        action_names: List of action names (optional)
        max_depth: Maximum depth to show

    Returns:
        String representation of rules
    """
    return export_text(tree, feature_names=feature_names, max_depth=max_depth)


def get_feature_importance(tree: DecisionTreeClassifier,
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importances from decision tree.

    Args:
        tree: Trained decision tree
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances
    """
    importances = tree.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    return df


def train_trees_multiple_depths(X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 pi_R_val: np.ndarray = None,
                                 depths: List[int] = None,
                                 seed: int = 42,
                                 verbose: bool = True) -> Dict:
    """
    Train decision trees with multiple depths for complexity analysis.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        pi_R_val: Original policy on validation set
        depths: List of max_depth values to try
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with trees and metrics for each depth
    """
    if depths is None:
        depths = [3, 5, 7, 10, 15, None]

    results = {}

    for depth in depths:
        depth_name = str(depth) if depth is not None else 'unlimited'
        if verbose:
            print(f"  Training tree with max_depth={depth_name}...")

        tree = train_decision_tree(X_train, y_train, max_depth=depth, seed=seed)

        train_metrics = evaluate_tree(tree, X_train, y_train)
        val_metrics = evaluate_tree(tree, X_val, y_val, pi_R_val)

        results[depth_name] = {
            'tree': tree,
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'top3_agreement': val_metrics.get('top3_agreement'),
            'n_leaves': val_metrics['n_leaves'],
            'depth': val_metrics['depth'],
            'n_features_used': val_metrics['n_features_used'],
        }

        if verbose:
            print(f"    Train acc: {train_metrics['accuracy']:.4f}, "
                  f"Val acc: {val_metrics['accuracy']:.4f}, "
                  f"Leaves: {val_metrics['n_leaves']}")

    return results


def save_tree(tree: DecisionTreeClassifier, path: str):
    """Save decision tree to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"[OK] Saved tree: {path}")


def load_tree(path: str) -> DecisionTreeClassifier:
    """Load decision tree from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
