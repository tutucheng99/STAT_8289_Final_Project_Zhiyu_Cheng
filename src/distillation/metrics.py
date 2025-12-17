"""
Fidelity Metrics for Policy Distillation.

Measures how well the distilled model π^D approximates the original π^R.
"""

import numpy as np
from typing import Dict, Optional
from scipy.special import xlogy


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute KL divergence: KL(p || q)

    Args:
        p: Reference distribution (π^R), shape (..., K)
        q: Approximation (π^D), shape (..., K)
        eps: Small constant for numerical stability

    Returns:
        KL divergence values
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Renormalize
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)

    return np.sum(xlogy(p, p) - xlogy(p, q), axis=-1)


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute Jensen-Shannon Divergence."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (p.sum(axis=-1, keepdims=True) + eps)
    q = q / (q.sum(axis=-1, keepdims=True) + eps)

    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def action_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute action agreement rate (accuracy).

    Args:
        y_true: True actions (from π^R)
        y_pred: Predicted actions (from π^D)

    Returns:
        Agreement rate [0, 1]
    """
    return np.mean(y_true == y_pred)


def top_k_agreement(pi_R: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """
    Check if predicted action is in top-k of π^R.

    Args:
        pi_R: Original policy probabilities, shape (N, K)
        y_pred: Predicted actions, shape (N,)
        k: Number of top actions to consider

    Returns:
        Agreement rate [0, 1]
    """
    top_k_actions = np.argsort(pi_R, axis=1)[:, -k:]
    in_top_k = np.array([y_pred[i] in top_k_actions[i] for i in range(len(y_pred))])
    return np.mean(in_top_k)


def compute_fidelity_metrics(pi_R: np.ndarray,
                              pi_D: np.ndarray,
                              y_pred: np.ndarray,
                              legal_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive fidelity metrics.

    Args:
        pi_R: Original policy probabilities, shape (N, K)
        pi_D: Distilled policy probabilities, shape (N, K) or None
        y_pred: Predicted actions from distilled model, shape (N,)
        legal_mask: Optional legal action mask, shape (N, K)

    Returns:
        Dict with fidelity metrics
    """
    y_true = np.argmax(pi_R, axis=1)

    metrics = {
        'top1_agreement': action_agreement(y_true, y_pred),
        'top3_agreement': top_k_agreement(pi_R, y_pred, k=3),
        'top5_agreement': top_k_agreement(pi_R, y_pred, k=5),
    }

    # KL/JSD if we have probability outputs
    if pi_D is not None:
        # Apply legal mask if provided
        if legal_mask is not None:
            pi_R_masked = np.where(legal_mask, pi_R, 0)
            pi_D_masked = np.where(legal_mask, pi_D, 0)
            pi_R_masked = pi_R_masked / (pi_R_masked.sum(axis=-1, keepdims=True) + 1e-10)
            pi_D_masked = pi_D_masked / (pi_D_masked.sum(axis=-1, keepdims=True) + 1e-10)
        else:
            pi_R_masked = pi_R
            pi_D_masked = pi_D

        kl_vals = kl_divergence(pi_R_masked, pi_D_masked)
        jsd_vals = jsd(pi_R_masked, pi_D_masked)

        metrics['kl_mean'] = float(np.mean(kl_vals))
        metrics['kl_std'] = float(np.std(kl_vals))
        metrics['kl_median'] = float(np.median(kl_vals))
        metrics['jsd_mean'] = float(np.mean(jsd_vals))
        metrics['jsd_std'] = float(np.std(jsd_vals))

    return metrics


def per_action_accuracy(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        n_actions: int = 38) -> Dict[int, Dict]:
    """
    Compute per-action accuracy metrics.

    Args:
        y_true: True actions
        y_pred: Predicted actions
        n_actions: Number of actions

    Returns:
        Dict[action_id, {count, correct, accuracy}]
    """
    results = {}
    for action in range(n_actions):
        mask = y_true == action
        count = mask.sum()
        if count > 0:
            correct = (y_pred[mask] == action).sum()
            results[action] = {
                'count': int(count),
                'correct': int(correct),
                'accuracy': float(correct / count),
            }
    return results
