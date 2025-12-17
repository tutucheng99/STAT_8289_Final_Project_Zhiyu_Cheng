"""
JSD (Jensen-Shannon Divergence) computation with numerical stability.

Uses scipy.special.xlogy to handle 0*log(0) = 0 correctly without eps bias.

Key functions:
- jsd: Compute JSD between two distributions
- jsd_legal_only: Compute JSD only on legal actions (renormalized)
- compute_sensitivity_stats: Compare JSD_raw vs JSD_legal
"""

import numpy as np
from scipy.special import xlogy
from typing import Dict, Tuple
import pandas as pd


def jsd(p: np.ndarray, q: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Compute Jensen-Shannon Divergence (numerically stable version).

    Uses xlogy(x, y) = x * log(y), which returns 0 when x=0,
    avoiding artificial eps bias.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p: (N, K) or (K,) probability distribution
        q: (N, K) or (K,) probability distribution
        base: logarithm base (2 -> [0, 1], e -> [0, ln2])

    Returns:
        JSD values: (N,) or scalar

    Properties:
        - JSD in [0, 1] when base=2
        - JSD = 0 iff P = Q
        - Symmetric: JSD(P || Q) = JSD(Q || P)
    """
    # Ensure arrays
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Mixture distribution
    m = 0.5 * (p + q)

    # KL(P || M) = sum(p * log(p/m)) = sum(p * log(p) - p * log(m))
    # Using xlogy: xlogy(p, p) = p * log(p), returns 0 when p=0
    if base == 2:
        log_scale = np.log(2)
        kl_pm = np.sum(xlogy(p, p) - xlogy(p, m), axis=-1) / log_scale
        kl_qm = np.sum(xlogy(q, q) - xlogy(q, m), axis=-1) / log_scale
    else:
        kl_pm = np.sum(xlogy(p, p) - xlogy(p, m), axis=-1)
        kl_qm = np.sum(xlogy(q, q) - xlogy(q, m), axis=-1)

    jsd_val = 0.5 * (kl_pm + kl_qm)

    # Clip small numerical errors
    return np.clip(jsd_val, 0.0, 1.0 if base == 2 else np.log(2))


def jsd_legal_only(pi_H: np.ndarray, pi_R: np.ndarray,
                   legal_masks: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Compute JSD only on legal actions (renormalized).

    For each sample, extracts legal action probabilities, renormalizes to sum=1,
    then computes JSD.

    Args:
        pi_H: (N, K) policy probabilities for human proxy
        pi_R: (N, K) policy probabilities for RL policy
        legal_masks: (N, K) bool array of legal actions
        base: logarithm base

    Returns:
        JSD values: (N,)
    """
    N = len(pi_H)
    jsd_values = np.zeros(N, dtype=np.float64)

    for i in range(N):
        mask = legal_masks[i]
        n_legal = mask.sum()

        if n_legal <= 1:
            # Only one legal action -> JSD = 0 (both policies must put all mass there)
            jsd_values[i] = 0.0
            continue

        p = pi_H[i, mask].astype(np.float64)
        q = pi_R[i, mask].astype(np.float64)

        # Renormalize
        p_sum = p.sum()
        q_sum = q.sum()

        if p_sum > 0:
            p = p / p_sum
        if q_sum > 0:
            q = q / q_sum

        jsd_values[i] = jsd(p, q, base=base)

    return jsd_values


def jsd_batch_legal(pi_H: np.ndarray, pi_R: np.ndarray,
                    legal_masks: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Vectorized JSD on legal actions (faster than loop for large N).

    Uses masked operations for efficiency.

    Args:
        pi_H: (N, K) policy probabilities
        pi_R: (N, K) policy probabilities
        legal_masks: (N, K) bool array
        base: logarithm base

    Returns:
        JSD values: (N,)
    """
    # Apply legal mask
    pi_H_legal = np.where(legal_masks, pi_H, 0.0).astype(np.float64)
    pi_R_legal = np.where(legal_masks, pi_R, 0.0).astype(np.float64)

    # Renormalize
    H_sum = pi_H_legal.sum(axis=1, keepdims=True)
    R_sum = pi_R_legal.sum(axis=1, keepdims=True)

    # Avoid division by zero
    H_sum = np.where(H_sum > 0, H_sum, 1.0)
    R_sum = np.where(R_sum > 0, R_sum, 1.0)

    pi_H_norm = pi_H_legal / H_sum
    pi_R_norm = pi_R_legal / R_sum

    return jsd(pi_H_norm, pi_R_norm, base=base)


def compute_sensitivity_stats(jsd_raw: np.ndarray, jsd_legal: np.ndarray) -> Dict:
    """
    Compute sensitivity statistics comparing JSD_raw vs JSD_legal.

    This is a required analysis to quantify the impact of using raw vs
    legal-only JSD computation.

    Args:
        jsd_raw: (N,) JSD values with all actions (including smoothed illegal)
        jsd_legal: (N,) JSD values with only legal actions (renormalized)

    Returns:
        Dict with sensitivity statistics:
        - mean_abs_diff: Mean absolute difference
        - max_abs_diff: Maximum absolute difference
        - p50/p90/p95/p99_abs_diff: Percentile of absolute differences
        - correlation: Pearson correlation
        - mean_relative_diff: Mean |raw - legal| / (raw + 1e-10)
    """
    diff = np.abs(jsd_raw - jsd_legal)

    stats = {
        'mean_abs_diff': float(diff.mean()),
        'std_abs_diff': float(diff.std()),
        'max_abs_diff': float(diff.max()),
        'p50_abs_diff': float(np.percentile(diff, 50)),
        'p90_abs_diff': float(np.percentile(diff, 90)),
        'p95_abs_diff': float(np.percentile(diff, 95)),
        'p99_abs_diff': float(np.percentile(diff, 99)),
        'correlation': float(np.corrcoef(jsd_raw, jsd_legal)[0, 1]),
        'mean_relative_diff': float(np.mean(diff / (jsd_raw + 1e-10))),
    }

    return stats


def jsd_decomposition(p: np.ndarray, q: np.ndarray, base: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose JSD into its two KL divergence components.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    Useful for understanding which policy contributes more to divergence.

    Args:
        p: (N, K) or (K,) probability distribution
        q: (N, K) or (K,) probability distribution
        base: logarithm base

    Returns:
        (kl_pm, kl_qm): Each is (N,) or scalar
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    m = 0.5 * (p + q)

    if base == 2:
        log_scale = np.log(2)
        kl_pm = np.sum(xlogy(p, p) - xlogy(p, m), axis=-1) / log_scale
        kl_qm = np.sum(xlogy(q, q) - xlogy(q, m), axis=-1) / log_scale
    else:
        kl_pm = np.sum(xlogy(p, p) - xlogy(p, m), axis=-1)
        kl_qm = np.sum(xlogy(q, q) - xlogy(q, m), axis=-1)

    return np.clip(kl_pm, 0, None), np.clip(kl_qm, 0, None)
