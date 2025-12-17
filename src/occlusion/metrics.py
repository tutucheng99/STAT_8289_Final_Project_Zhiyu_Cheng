"""
Metrics for Occlusion Analysis.

Provides various metrics to quantify policy changes under occlusion:
- KL divergence (asymmetric)
- Jensen-Shannon divergence (symmetric)
- Total variation distance
- Action change rate
"""

import numpy as np
from typing import Dict, Tuple
from scipy.special import xlogy


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute KL divergence: KL(p || q) = sum_i p_i * log(p_i / q_i)

    Args:
        p: Original policy, shape (..., K)
        q: Perturbed policy, shape (..., K)
        eps: Small constant for numerical stability

    Returns:
        KL divergence, shape (...)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Clip to avoid log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Renormalize
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)

    # KL = sum_i p_i * log(p_i / q_i)
    return np.sum(xlogy(p, p) - xlogy(p, q), axis=-1)


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute Jensen-Shannon divergence.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5 * (p + q)

    Args:
        p: First distribution, shape (..., K)
        q: Second distribution, shape (..., K)
        eps: Small constant for numerical stability

    Returns:
        JSD value, shape (...)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Renormalize
    p = p / (p.sum(axis=-1, keepdims=True) + eps)
    q = q / (q.sum(axis=-1, keepdims=True) + eps)

    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute Total Variation Distance.

    TVD(p, q) = 0.5 * sum_i |p_i - q_i|

    Args:
        p: First distribution, shape (..., K)
        q: Second distribution, shape (..., K)

    Returns:
        TVD value, shape (...)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Renormalize
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-10)
    q = q / (q.sum(axis=-1, keepdims=True) + 1e-10)

    return 0.5 * np.sum(np.abs(p - q), axis=-1)


def action_change_rate(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute rate of argmax action changes.

    Args:
        p: Original policy, shape (N, K)
        q: Perturbed policy, shape (N, K)

    Returns:
        Fraction of samples where argmax changes
    """
    p = np.asarray(p)
    q = np.asarray(q)

    action_p = np.argmax(p, axis=-1)
    action_q = np.argmax(q, axis=-1)

    return np.mean(action_p != action_q)


def policy_change_metrics(p_original: np.ndarray,
                          p_perturbed: np.ndarray,
                          legal_mask: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive policy change metrics.

    Args:
        p_original: Original policy, shape (N, K)
        p_perturbed: Perturbed policy, shape (N, K)
        legal_mask: Optional legal action mask, shape (N, K)

    Returns:
        Dict with:
        - kl_mean, kl_std, kl_median, kl_p95: KL divergence statistics
        - jsd_mean, jsd_std: JSD statistics
        - tvd_mean, tvd_std: TVD statistics
        - action_change_rate: Fraction of argmax changes
    """
    p_original = np.asarray(p_original, dtype=np.float64)
    p_perturbed = np.asarray(p_perturbed, dtype=np.float64)

    # Apply legal mask if provided
    if legal_mask is not None:
        legal_mask = np.asarray(legal_mask, dtype=bool)
        p_original = np.where(legal_mask, p_original, 0.0)
        p_perturbed = np.where(legal_mask, p_perturbed, 0.0)

        # Renormalize
        p_original = p_original / (p_original.sum(axis=-1, keepdims=True) + 1e-10)
        p_perturbed = p_perturbed / (p_perturbed.sum(axis=-1, keepdims=True) + 1e-10)

    # Compute metrics
    kl_vals = kl_divergence(p_original, p_perturbed)
    jsd_vals = jsd(p_original, p_perturbed)
    tvd_vals = tvd(p_original, p_perturbed)
    acr = action_change_rate(p_original, p_perturbed)

    return {
        'kl_mean': float(np.mean(kl_vals)),
        'kl_std': float(np.std(kl_vals)),
        'kl_median': float(np.median(kl_vals)),
        'kl_p95': float(np.percentile(kl_vals, 95)),
        'jsd_mean': float(np.mean(jsd_vals)),
        'jsd_std': float(np.std(jsd_vals)),
        'tvd_mean': float(np.mean(tvd_vals)),
        'tvd_std': float(np.std(tvd_vals)),
        'action_change_rate': float(acr),
    }


def sensitivity_ranking(results: Dict[str, Dict[str, float]],
                        metric: str = 'kl_mean') -> list:
    """
    Rank feature groups by sensitivity.

    Args:
        results: Dict[group_name, metrics_dict]
        metric: Metric to rank by (default: 'kl_mean')

    Returns:
        List of (group_name, value) sorted descending by sensitivity
    """
    ranked = [(name, metrics[metric]) for name, metrics in results.items()
              if metric in metrics]
    return sorted(ranked, key=lambda x: x[1], reverse=True)
