"""
GAM fitting for ALR difference curves.

Key function:
- fit_difference_curve: Fit GAM to ALR difference (Δ_b = ALR^R - ALR^H)
"""

import numpy as np
from pygam import LinearGAM, s
from typing import Dict, Optional


def fit_difference_curve(X: np.ndarray,
                         alr_diff: np.ndarray,
                         legal_mask: np.ndarray,
                         episode_ids: np.ndarray,
                         n_splines: int = 10,
                         lam: Optional[float] = None) -> Dict:
    """
    Fit GAM to ALR difference curve Δ_b(x) = ALR^R - ALR^H.

    This directly fits the difference (more stable than fitting two curves separately).

    Args:
        X: (N,) single covariate
        alr_diff: (N,) ALR difference for single action
        legal_mask: (N,) bool - only use legal samples
        episode_ids: (N,) int - for tracking episodes
        n_splines: number of splines for GAM
        lam: smoothing parameter (None = gridsearch)

    Returns:
        Dict with:
        - X_grid: (100,) prediction grid
        - pred_diff: (100,) predicted difference curve
        - gam: fitted GAM model
        - lam: smoothing parameter used
        - n_samples: number of samples used
        - n_episodes: number of unique episodes
        - x_range: (x_lo, x_hi) analysis range
    """
    # Filter to legal samples
    X_legal = X[legal_mask]
    diff_legal = alr_diff[legal_mask]
    ep_legal = episode_ids[legal_mask]

    # Use 5%-95% quantile range (avoid extrapolation risk)
    x_lo, x_hi = np.percentile(X_legal, [5, 95])
    in_range = (X_legal >= x_lo) & (X_legal <= x_hi)
    X_fit = X_legal[in_range]
    diff_fit = diff_legal[in_range]
    ep_fit = ep_legal[in_range]

    # Fit GAM
    if lam is not None:
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(X_fit.reshape(-1, 1), diff_fit)
    else:
        gam = LinearGAM(s(0, n_splines=n_splines)).gridsearch(X_fit.reshape(-1, 1), diff_fit)
        lam = gam.lam[0] if isinstance(gam.lam, (list, np.ndarray)) else gam.lam

    # Predict on grid
    X_grid = np.linspace(x_lo, x_hi, 100)
    pred_diff = gam.predict(X_grid.reshape(-1, 1))

    # Ensure lam is a scalar
    if isinstance(lam, (list, np.ndarray)):
        lam = float(lam[0])
    else:
        lam = float(lam)

    return {
        'X_grid': X_grid,
        'pred_diff': pred_diff,
        'gam': gam,
        'lam': lam,
        'n_samples': len(X_fit),
        'n_episodes': len(np.unique(ep_fit)),
        'x_range': (x_lo, x_hi),
    }


def batch_fit_difference_curves(covariates: np.ndarray,
                                alr_diff: np.ndarray,
                                legal_masks: np.ndarray,
                                episode_ids: np.ndarray,
                                feature_names: list,
                                action_names: list,
                                cov_list: list,
                                action_list: list,
                                n_splines: int = 10) -> Dict:
    """
    Batch fit GAM curves for multiple (covariate, action) combinations.

    Args:
        covariates: (N, F) all covariates
        alr_diff: (N, K) ALR differences for all actions
        legal_masks: (N, K) legal masks
        episode_ids: (N,) episode IDs
        feature_names: list of feature names
        action_names: list of action names
        cov_list: list of covariate names to analyze
        action_list: list of action indices to analyze
        n_splines: number of splines

    Returns:
        Dict[(cov_name, action_name)] -> fit result
    """
    results = {}

    for cov_name in cov_list:
        cov_idx = feature_names.index(cov_name)

        for action_idx in action_list:
            action_name = action_names[action_idx]
            key = (cov_name, action_name)

            result = fit_difference_curve(
                X=covariates[:, cov_idx],
                alr_diff=alr_diff[:, action_idx],
                legal_mask=legal_masks[:, action_idx],
                episode_ids=episode_ids,
                n_splines=n_splines,
            )

            results[key] = result
            print(f"  {cov_name} x {action_name}: n={result['n_samples']:,}, eps={result['n_episodes']:,}, lam={result['lam']:.2e}")

    return results
