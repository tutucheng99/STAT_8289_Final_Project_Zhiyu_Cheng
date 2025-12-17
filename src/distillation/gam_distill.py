"""
GAM (Generalized Additive Model) Distillation for Policy Interpretability.

Trains GAM models to approximate policy decisions for individual actions,
providing interpretable shape functions showing feature effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pygam import LogisticGAM, s, l
import pickle
from pathlib import Path


def get_top_k_actions(y: np.ndarray, k: int = 10) -> List[int]:
    """
    Get the k most frequent actions.

    Args:
        y: Action labels
        k: Number of top actions

    Returns:
        List of action indices sorted by frequency
    """
    unique, counts = np.unique(y, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    return unique[sorted_idx[:k]].tolist()


def train_gam_for_action(X: np.ndarray,
                         y: np.ndarray,
                         action: int,
                         n_splines: int = 10,
                         lam: float = 0.6,
                         feature_indices: List[int] = None) -> LogisticGAM:
    """
    Train a GAM for binary classification of a single action.

    Args:
        X: Features, shape (N, D)
        y: Action labels, shape (N,)
        action: Target action index
        n_splines: Number of splines per feature
        lam: Regularization parameter (default 0.6 to skip gridsearch)
        feature_indices: Optional subset of feature indices to use

    Returns:
        Trained LogisticGAM
    """
    # Binary labels
    y_binary = (y == action).astype(int)

    # Select features
    if feature_indices is not None:
        X_subset = X[:, feature_indices]
        n_features = len(feature_indices)
    else:
        X_subset = X
        n_features = X.shape[1]

    # Build GAM terms
    terms = s(0, n_splines=n_splines)
    for i in range(1, n_features):
        terms += s(i, n_splines=n_splines)

    # Use fixed lambda to avoid slow gridsearch
    gam = LogisticGAM(terms, lam=lam).fit(X_subset, y_binary)

    return gam


def train_gam_per_action(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray,
                         top_k: int = 10,
                         n_splines: int = 10,
                         verbose: bool = True) -> Dict:
    """
    Train GAM models for top-k actions.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        top_k: Number of top actions to model
        n_splines: Number of splines per feature
        verbose: Print progress

    Returns:
        Dict with GAM models and metrics
    """
    top_actions = get_top_k_actions(y_train, k=top_k)

    if verbose:
        print(f"Training GAM for top {top_k} actions: {top_actions}")

    results = {}

    for action in top_actions:
        if verbose:
            print(f"  Training GAM for action {action}...")

        try:
            gam = train_gam_for_action(X_train, y_train, action, n_splines=n_splines)

            # Evaluate
            y_train_binary = (y_train == action).astype(int)
            y_val_binary = (y_val == action).astype(int)

            train_acc = gam.accuracy(X_train, y_train_binary)
            val_acc = gam.accuracy(X_val, y_val_binary)

            # AUC if possible
            try:
                from sklearn.metrics import roc_auc_score
                train_proba = gam.predict_proba(X_train)
                val_proba = gam.predict_proba(X_val)
                train_auc = roc_auc_score(y_train_binary, train_proba)
                val_auc = roc_auc_score(y_val_binary, val_proba)
            except:
                train_auc = val_auc = None

            results[action] = {
                'gam': gam,
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'train_auc': float(train_auc) if train_auc else None,
                'val_auc': float(val_auc) if val_auc else None,
                'n_positive_train': int(y_train_binary.sum()),
                'n_positive_val': int(y_val_binary.sum()),
            }

            if verbose:
                print(f"    Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")
            results[action] = {'error': str(e)}

    return results


def get_shape_functions(gam: LogisticGAM,
                        feature_idx: int,
                        X: np.ndarray,
                        n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract shape function for a single feature.

    Args:
        gam: Trained GAM model
        feature_idx: Index of feature
        X: Sample data (for range estimation)
        n_points: Number of points to evaluate

    Returns:
        (x_values, y_values, confidence_intervals)
    """
    # Generate grid for this feature
    XX = gam.generate_X_grid(term=feature_idx, n=n_points)
    x_values = XX[:, feature_idx]

    # Get partial dependence
    pdep = gam.partial_dependence(term=feature_idx, X=XX)

    # Get confidence intervals
    pdep_ci = gam.partial_dependence(term=feature_idx, X=XX, width=0.95)

    return x_values, pdep, pdep_ci


def extract_all_shape_functions(gam: LogisticGAM,
                                 feature_names: List[str],
                                 X: np.ndarray) -> Dict:
    """
    Extract shape functions for all features.

    Args:
        gam: Trained GAM model
        feature_names: List of feature names
        X: Sample data

    Returns:
        Dict[feature_name, {x, y, ci}]
    """
    results = {}
    for i, name in enumerate(feature_names):
        try:
            x, y, ci = get_shape_functions(gam, i, X)
            results[name] = {
                'x': x,
                'y': y,
                'ci': ci,
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    return results


def predict_with_gams(gam_results: Dict,
                      X: np.ndarray,
                      n_actions: int = 38) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict actions using ensemble of per-action GAMs.

    Args:
        gam_results: Dict from train_gam_per_action
        X: Features to predict
        n_actions: Total number of actions

    Returns:
        (predicted_actions, probability_matrix)
    """
    N = len(X)
    proba_matrix = np.zeros((N, n_actions))

    # Fill in probabilities for modeled actions
    for action, result in gam_results.items():
        if 'gam' in result:
            proba_matrix[:, action] = result['gam'].predict_proba(X)

    # Predict action with highest probability (among modeled actions)
    modeled_actions = [a for a, r in gam_results.items() if 'gam' in r]
    proba_subset = proba_matrix[:, modeled_actions]
    pred_idx = np.argmax(proba_subset, axis=1)
    predictions = np.array([modeled_actions[i] for i in pred_idx])

    return predictions, proba_matrix


def save_gam_results(gam_results: Dict, path: str):
    """Save GAM results to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(gam_results, f)
    print(f"[OK] Saved GAM results: {path}")


def load_gam_results(path: str) -> Dict:
    """Load GAM results from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
