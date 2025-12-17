"""
Cluster bootstrap and permutation tests for FDA statistical inference.

Key functions:
- bootstrap_difference_curve: Cluster bootstrap for confidence intervals
- perm_test_curve_gam: Curve-based sign-flip permutation test for p-values
"""

import numpy as np
from pygam import LinearGAM, s
from typing import Dict, Tuple, Optional
from tqdm.auto import tqdm


def bootstrap_difference_curve(X: np.ndarray,
                               alr_diff: np.ndarray,
                               legal_mask: np.ndarray,
                               episode_ids: np.ndarray,
                               n_bootstrap: int = 500,
                               n_splines: int = 10,
                               lam: Optional[float] = None,
                               seed: int = 42,
                               verbose: bool = False) -> Dict:
    """
    Cluster bootstrap for ALR difference curve confidence intervals.

    CRITICAL: Preserves episode multiplicity (not just np.isin)!

    Args:
        X: (N,) single covariate
        alr_diff: (N,) ALR difference for single action
        legal_mask: (N,) bool
        episode_ids: (N,) int
        n_bootstrap: number of bootstrap iterations
        n_splines: GAM splines
        lam: fixed smoothing parameter (None = gridsearch on original data)
        seed: random seed
        verbose: show progress bar

    Returns:
        Dict with point_estimate_curve, bootstrap CI, simultaneous band, etc.
    """
    rng = np.random.default_rng(seed)

    # Filter to legal samples + 5%-95% range
    X_legal = X[legal_mask]
    diff_legal = alr_diff[legal_mask]
    ep_legal = episode_ids[legal_mask]

    x_lo, x_hi = np.percentile(X_legal, [5, 95])
    in_range = (X_legal >= x_lo) & (X_legal <= x_hi)
    X_fit = X_legal[in_range]
    diff_fit = diff_legal[in_range]
    ep_fit = ep_legal[in_range]

    # Prediction grid
    X_grid = np.linspace(x_lo, x_hi, 100)

    # 0. Point estimate (GAM on full data)
    if lam is not None:
        gam_point = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(X_fit.reshape(-1, 1), diff_fit)
    else:
        gam_point = LinearGAM(s(0, n_splines=n_splines)).gridsearch(X_fit.reshape(-1, 1), diff_fit)
        lam = gam_point.lam[0] if isinstance(gam_point.lam, (list, np.ndarray)) else gam_point.lam

    point_estimate_curve = gam_point.predict(X_grid.reshape(-1, 1))

    # 1. Cluster Bootstrap
    unique_eps = np.unique(ep_fit)
    ep_to_idx = {ep: np.where(ep_fit == ep)[0] for ep in unique_eps}

    diff_curves = []
    iterator = tqdm(range(n_bootstrap), desc='Bootstrap', disable=not verbose)

    for _ in iterator:
        # Sample episodes with replacement
        sampled_eps = rng.choice(unique_eps, size=len(unique_eps), replace=True)

        # Concat indices (preserves multiplicity!)
        boot_idx = np.concatenate([ep_to_idx[ep] for ep in sampled_eps], axis=0)

        X_boot = X_fit[boot_idx]
        diff_boot = diff_fit[boot_idx]

        try:
            gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(X_boot.reshape(-1, 1), diff_boot)
            pred = gam.predict(X_grid.reshape(-1, 1))
            diff_curves.append(pred)
        except Exception:
            continue

    diff_curves = np.array(diff_curves)

    # 2. Pointwise CI
    ci_lower = np.percentile(diff_curves, 2.5, axis=0)
    ci_upper = np.percentile(diff_curves, 97.5, axis=0)
    bootstrap_mean = np.mean(diff_curves, axis=0)

    significant_pointwise = (ci_lower > 0) | (ci_upper < 0)

    # 3. Simultaneous band (centered on point estimate)
    deviations = np.abs(diff_curves - point_estimate_curve)
    max_deviations = deviations.max(axis=1)
    q95 = np.percentile(max_deviations, 95)

    simul_lower = point_estimate_curve - q95
    simul_upper = point_estimate_curve + q95
    significant_simultaneous = (simul_lower > 0) | (simul_upper < 0)

    return {
        'X_grid': X_grid,
        'point_estimate_curve': point_estimate_curve,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_ci_lower': ci_lower,
        'bootstrap_ci_upper': ci_upper,
        'bootstrap_significant_pointwise': significant_pointwise,
        'bootstrap_pct_significant_pointwise': significant_pointwise.mean(),
        'bootstrap_simul_lower': simul_lower,
        'bootstrap_simul_upper': simul_upper,
        'bootstrap_significant_simultaneous': significant_simultaneous,
        'bootstrap_pct_significant_simultaneous': significant_simultaneous.mean(),
        'n_bootstrap_success': len(diff_curves),
        'n_samples': len(X_fit),
        'n_episodes': len(unique_eps),
        'lam': lam,
        'x_range': (x_lo, x_hi),
    }


def perm_test_curve_gam(X_fit: np.ndarray,
                        diff_fit: np.ndarray,
                        ep_fit: np.ndarray,
                        x_lo: float,
                        x_hi: float,
                        n_perm: int = 500,
                        n_splines: int = 10,
                        lam: Optional[float] = None,
                        seed: int = 42,
                        verbose: bool = False) -> Tuple[float, float, np.ndarray]:
    """
    Curve-based Sign-Flip Permutation Test.

    H0: f(X) ≡ 0 (difference curve is zero everywhere)
    H1: f(X) ≠ 0 (difference curve is non-zero somewhere)

    CRITICAL: Refits GAM for each permutation! This ensures p-value depends on X.

    Args:
        X_fit: (n,) covariate (already filtered for legal + range)
        diff_fit: (n,) ALR difference
        ep_fit: (n,) episode IDs
        x_lo, x_hi: covariate range for prediction grid
        n_perm: number of permutations
        n_splines: GAM splines
        lam: fixed smoothing parameter
        seed: random seed
        verbose: show progress bar

    Returns:
        p_value: one-sided p-value
        T_obs: observed test statistic
        T_null: null distribution of test statistic
    """
    rng = np.random.default_rng(seed)
    X_grid = np.linspace(x_lo, x_hi, 100).reshape(-1, 1)

    # 1. Observed curve statistic
    if lam is not None:
        gam_obs = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(X_fit.reshape(-1, 1), diff_fit)
    else:
        gam_obs = LinearGAM(s(0, n_splines=n_splines)).gridsearch(X_fit.reshape(-1, 1), diff_fit)
        lam = gam_obs.lam[0] if isinstance(gam_obs.lam, (list, np.ndarray)) else gam_obs.lam

    f_obs = gam_obs.predict(X_grid)
    T_obs = np.mean(np.abs(f_obs))  # Curve statistic: mean|f(X)|

    # 2. Episode-level sign flip
    unique_eps = np.unique(ep_fit)
    ep_to_idx = {ep: np.where(ep_fit == ep)[0] for ep in unique_eps}

    T_null = []
    iterator = tqdm(range(n_perm), desc='Permutation', disable=not verbose)

    for _ in iterator:
        # Random signs for each episode
        signs = rng.choice([-1, 1], size=len(unique_eps))

        # Flip signs per episode
        diff_perm = diff_fit.copy()
        for ep, sgn in zip(unique_eps, signs):
            diff_perm[ep_to_idx[ep]] *= sgn

        # Refit GAM (CRITICAL!)
        try:
            gam_p = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(X_fit.reshape(-1, 1), diff_perm)
            f_p = gam_p.predict(X_grid)
            T_null.append(np.mean(np.abs(f_p)))
        except Exception:
            continue

    T_null = np.asarray(T_null)

    # 3. p-value (conservative: +1 to avoid p=0)
    p_value = (1 + np.sum(T_null >= T_obs)) / (len(T_null) + 1)

    return p_value, T_obs, T_null


def run_permutation_tests(covariates: np.ndarray,
                          alr_diff: np.ndarray,
                          legal_masks: np.ndarray,
                          episode_ids: np.ndarray,
                          feature_names: list,
                          action_names: list,
                          cov_list: list,
                          action_list: list,
                          n_perm: int = 500,
                          n_splines: int = 10,
                          seed: int = 42) -> list:
    """
    Run permutation tests for all (covariate, action) combinations.

    Returns:
        List of dicts with covariate, action, T_obs, p_raw
    """
    results = []

    for cov_name in cov_list:
        cov_idx = feature_names.index(cov_name)

        for action_idx in action_list:
            action_name = action_names[action_idx]

            # Filter to legal + range
            legal_mask = legal_masks[:, action_idx]
            X_legal = covariates[legal_mask, cov_idx]
            diff_legal = alr_diff[legal_mask, action_idx]
            ep_legal = episode_ids[legal_mask]

            x_lo, x_hi = np.percentile(X_legal, [5, 95])
            in_range = (X_legal >= x_lo) & (X_legal <= x_hi)
            X_fit = X_legal[in_range]
            diff_fit = diff_legal[in_range]
            ep_fit = ep_legal[in_range]

            # Run permutation test
            p, T_obs, T_null = perm_test_curve_gam(
                X_fit, diff_fit, ep_fit, x_lo, x_hi,
                n_perm=n_perm, n_splines=n_splines, seed=seed
            )

            results.append({
                'covariate': cov_name,
                'action': action_name,
                'action_idx': action_idx,
                'T_obs': T_obs,
                'p_raw': p,
                'n_samples': len(X_fit),
                'n_episodes': len(np.unique(ep_fit)),
            })

            print(f"  {cov_name} x {action_name}: T={T_obs:.4f}, p={p:.4f}")

    return results
