"""
GAM fitting for JSD curves.

Two definitions of JSD curves:
- Definition A (State-level): E[JSD(pi_H(s), pi_R(s)) | X = t]
  - First compute JSD for each state, then average by covariate bin
  - Fitted using GAM with cluster bootstrap CI

- Definition B (Bin-level): JSD(E[pi_H | X=t], E[pi_R | X=t])
  - First average policies by covariate bin, then compute JSD
  - Computed directly via binning (no GAM needed)

By Jensen's inequality: JSD_B(t) <= JSD_A(t) in general.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pygam import LinearGAM, s

from .metrics import jsd


def fit_jsd_gam_A(X: np.ndarray, jsd_values: np.ndarray,
                  episode_ids: np.ndarray,
                  n_splines: int = 10,
                  n_bootstrap: int = 100,
                  seed: int = 42,
                  percentile_range: Tuple[float, float] = (5, 95),
                  verbose: bool = False) -> Dict:
    """
    Fit State-level JSD curve (Definition A) using GAM with cluster bootstrap CI.

    JSD_A(t) = E[JSD(pi_H(s), pi_R(s)) | X = t]

    Args:
        X: (N,) covariate values
        jsd_values: (N,) JSD values for each state
        episode_ids: (N,) episode indices for cluster bootstrap
        n_splines: number of splines for GAM
        n_bootstrap: number of bootstrap iterations
        seed: random seed
        percentile_range: (lo, hi) percentile range to fit
        verbose: print progress

    Returns:
        Dict with:
        - X_grid: (100,) evaluation grid
        - point_estimate: (100,) GAM prediction
        - ci_lower, ci_upper: (100,) 95% bootstrap CI
        - n_samples, n_episodes: sample counts
        - gam: fitted GAM object
        - lam: regularization parameter
    """
    rng = np.random.default_rng(seed)

    # Filter to percentile range
    x_lo, x_hi = np.percentile(X, percentile_range)
    in_range = (X >= x_lo) & (X <= x_hi)
    X_fit = X[in_range]
    jsd_fit = jsd_values[in_range]
    ep_fit = episode_ids[in_range]

    n_samples = len(X_fit)
    unique_eps = np.unique(ep_fit)
    n_episodes = len(unique_eps)

    if verbose:
        print(f"  Fitting GAM: {n_samples} samples, {n_episodes} episodes")

    # Point estimate with gridsearch for lambda
    gam = LinearGAM(s(0, n_splines=n_splines)).gridsearch(
        X_fit.reshape(-1, 1), jsd_fit, progress=False
    )
    lam = gam.lam[0] if hasattr(gam.lam, '__len__') else gam.lam

    # Prediction grid
    X_grid = np.linspace(x_lo, x_hi, 100)
    point_estimate = gam.predict(X_grid.reshape(-1, 1))

    # Cluster bootstrap
    ep_to_idx = {ep: np.where(ep_fit == ep)[0] for ep in unique_eps}

    boot_curves = []
    for b in range(n_bootstrap):
        sampled_eps = rng.choice(unique_eps, size=n_episodes, replace=True)
        boot_idx = np.concatenate([ep_to_idx[ep] for ep in sampled_eps])

        try:
            gam_b = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(
                X_fit[boot_idx].reshape(-1, 1), jsd_fit[boot_idx]
            )
            boot_curves.append(gam_b.predict(X_grid.reshape(-1, 1)))
        except Exception:
            continue

    boot_curves = np.array(boot_curves)
    if len(boot_curves) > 0:
        ci_lower = np.percentile(boot_curves, 2.5, axis=0)
        ci_upper = np.percentile(boot_curves, 97.5, axis=0)
    else:
        ci_lower = ci_upper = point_estimate

    return {
        'X_grid': X_grid,
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_samples': n_samples,
        'n_episodes': n_episodes,
        'gam': gam,
        'lam': lam,
        'x_range': (x_lo, x_hi),
    }


def compute_jsd_curve_B(covariates: np.ndarray, pi_H: np.ndarray,
                        pi_R: np.ndarray, cov_idx: int,
                        n_bins: int = 10) -> pd.DataFrame:
    """
    Compute Bin-level JSD curve (Definition B).

    JSD_B(t) = JSD(E[pi_H | X in bin], E[pi_R | X in bin])

    First averages policies within each covariate bin, then computes JSD
    between the averaged policies.

    Args:
        covariates: (N, D) covariate matrix
        pi_H: (N, K) human proxy policy
        pi_R: (N, K) RL policy
        cov_idx: index of covariate to bin on
        n_bins: number of quantile bins

    Returns:
        DataFrame with columns:
        - bin_center: center of each bin
        - bin_lo, bin_hi: bin boundaries
        - jsd_B: JSD of averaged policies
        - n_samples: samples in bin
        - n_episodes: unique episodes in bin (if episode_ids provided)
    """
    X = covariates[:, cov_idx]

    # Create quantile bins - convert to Series first
    X_series = pd.Series(X)
    try:
        bins = pd.qcut(X_series, q=n_bins, duplicates='drop')
    except ValueError:
        # Fall back to equal-width bins if quantile fails
        bins = pd.cut(X_series, bins=n_bins, duplicates='drop')

    results = []
    for bin_label in bins.cat.categories:
        mask = (bins == bin_label).values
        n_samples = mask.sum()

        if n_samples < 10:
            continue

        # Average policies in bin
        avg_pi_H = pi_H[mask].mean(axis=0)
        avg_pi_R = pi_R[mask].mean(axis=0)

        # Renormalize (should be close to 1 already)
        avg_pi_H = avg_pi_H / avg_pi_H.sum()
        avg_pi_R = avg_pi_R / avg_pi_R.sum()

        # JSD of averaged policies
        jsd_B = jsd(avg_pi_H, avg_pi_R)

        results.append({
            'bin_center': X[mask].mean(),
            'bin_lo': bin_label.left,
            'bin_hi': bin_label.right,
            'jsd_B': float(jsd_B),
            'n_samples': n_samples,
        })

    return pd.DataFrame(results)


def compute_jsd_curves_both(covariates: np.ndarray, pi_H: np.ndarray,
                            pi_R: np.ndarray, jsd_values: np.ndarray,
                            cov_idx: int, episode_ids: np.ndarray,
                            n_bins: int = 10,
                            n_splines: int = 10,
                            n_bootstrap: int = 100,
                            seed: int = 42) -> Dict:
    """
    Compute both JSD curve definitions for a given covariate.

    Definition A: State-level average (GAM fitted)
    Definition B: Bin-level system (direct computation)

    Args:
        covariates: (N, D) covariate matrix
        pi_H: (N, K) human proxy policy
        pi_R: (N, K) RL policy
        jsd_values: (N,) pre-computed JSD values
        cov_idx: covariate index
        episode_ids: (N,) episode indices
        n_bins: number of bins for Definition B
        n_splines: splines for GAM
        n_bootstrap: bootstrap iterations
        seed: random seed

    Returns:
        Dict with:
        - curve_A: GAM results for Definition A
        - curve_B: DataFrame for Definition B
        - comparison: DataFrame comparing A and B at bin centers
    """
    X = covariates[:, cov_idx]

    # Definition A: State-level GAM
    curve_A = fit_jsd_gam_A(
        X, jsd_values, episode_ids,
        n_splines=n_splines, n_bootstrap=n_bootstrap, seed=seed
    )

    # Definition B: Bin-level
    curve_B = compute_jsd_curve_B(covariates, pi_H, pi_R, cov_idx, n_bins=n_bins)

    # Also compute Definition A at bin centers for comparison
    # (average JSD within each bin, not GAM prediction)
    X_series = pd.Series(X)
    try:
        bins = pd.qcut(X_series, q=n_bins, duplicates='drop')
    except ValueError:
        bins = pd.cut(X_series, bins=n_bins, duplicates='drop')

    comparison = []
    for bin_label in bins.cat.categories:
        mask = (bins == bin_label).values
        if mask.sum() < 10:
            continue

        jsd_A_mean = jsd_values[mask].mean()
        jsd_A_std = jsd_values[mask].std()

        # Find corresponding B value
        bin_center = X[mask].mean()
        b_row = curve_B[curve_B['bin_center'].apply(
            lambda x: abs(x - bin_center) < (X.max() - X.min()) / n_bins
        )]
        jsd_B_val = b_row['jsd_B'].values[0] if len(b_row) > 0 else np.nan

        comparison.append({
            'bin_center': bin_center,
            'jsd_A_mean': jsd_A_mean,
            'jsd_A_std': jsd_A_std,
            'jsd_B': jsd_B_val,
            'A_minus_B': jsd_A_mean - jsd_B_val if not np.isnan(jsd_B_val) else np.nan,
            'n_samples': mask.sum(),
        })

    comparison_df = pd.DataFrame(comparison)

    return {
        'curve_A': curve_A,
        'curve_B': curve_B,
        'comparison': comparison_df,
    }


def batch_fit_jsd_curves(covariates: np.ndarray, pi_H: np.ndarray,
                         pi_R: np.ndarray, jsd_values: np.ndarray,
                         episode_ids: np.ndarray,
                         feature_names: List[str],
                         cov_names: List[str],
                         n_bins: int = 10,
                         n_splines: int = 10,
                         n_bootstrap: int = 100,
                         seed: int = 42,
                         verbose: bool = True) -> Dict[str, Dict]:
    """
    Fit JSD curves for multiple covariates.

    Args:
        covariates: (N, D) covariate matrix
        pi_H, pi_R: (N, K) policies
        jsd_values: (N,) JSD values
        episode_ids: (N,) episode indices
        feature_names: list of all feature names
        cov_names: list of covariate names to analyze
        n_bins, n_splines, n_bootstrap, seed: fitting parameters
        verbose: print progress

    Returns:
        Dict[covariate_name, curve_results]
    """
    results = {}

    for cov_name in cov_names:
        if verbose:
            print(f"  Fitting curves for {cov_name}...")

        cov_idx = feature_names.index(cov_name)
        results[cov_name] = compute_jsd_curves_both(
            covariates, pi_H, pi_R, jsd_values, cov_idx, episode_ids,
            n_bins=n_bins, n_splines=n_splines,
            n_bootstrap=n_bootstrap, seed=seed
        )

    return results
