"""
EBM (Explainable Boosting Machine) analysis for JSD prediction.

Uses InterpretML's EBM to:
1. Predict JSD from covariates
2. Extract feature importances
3. Visualize shape functions

EBM is chosen for interpretability - it provides:
- Additive structure: f(x) = sum_i f_i(x_i) + sum_{i,j} f_{i,j}(x_i, x_j)
- Feature importance: mean absolute score contribution
- Shape functions: partial dependence plots
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path


def train_ebm_jsd(covariates: np.ndarray,
                  jsd_values: np.ndarray,
                  feature_names: List[str],
                  n_interactions: int = 10,
                  max_bins: int = 256,
                  learning_rate: float = 0.01,
                  seed: int = 42,
                  verbose: bool = True) -> Dict:
    """
    Train EBM to predict JSD from covariates.

    Args:
        covariates: (N, D) covariate matrix
        jsd_values: (N,) JSD values to predict
        feature_names: list of feature names
        n_interactions: number of pairwise interactions to include
        max_bins: maximum bins per feature
        learning_rate: boosting learning rate
        seed: random seed
        verbose: print progress

    Returns:
        Dict with:
        - ebm: trained EBM model
        - feature_importances: DataFrame of importances
        - cv_r2: cross-validation R^2 score
        - cv_rmse: cross-validation RMSE
    """
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.model_selection import cross_val_score

    if verbose:
        print("Training EBM model...")

    # Train EBM
    ebm = ExplainableBoostingRegressor(
        feature_names=feature_names,
        interactions=n_interactions,
        max_bins=max_bins,
        learning_rate=learning_rate,
        n_jobs=-1,
        random_state=seed,
    )
    ebm.fit(covariates, jsd_values)

    # Cross-validation scores
    if verbose:
        print("Computing cross-validation scores...")
    cv_r2 = cross_val_score(ebm, covariates, jsd_values, cv=5, scoring='r2')
    cv_rmse = -cross_val_score(ebm, covariates, jsd_values, cv=5,
                                scoring='neg_root_mean_squared_error')

    if verbose:
        print(f"  CV R^2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
        print(f"  CV RMSE: {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")

    # Feature importances (API changed: use term_importances instead of feature_importances_)
    importances = ebm.term_importances()

    # Map term names to importances
    term_names = ebm.term_names_
    importance_df = pd.DataFrame({
        'term': term_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    return {
        'ebm': ebm,
        'feature_importances': importance_df,
        'cv_r2_mean': float(cv_r2.mean()),
        'cv_r2_std': float(cv_r2.std()),
        'cv_rmse_mean': float(cv_rmse.mean()),
        'cv_rmse_std': float(cv_rmse.std()),
    }


def get_top_features(ebm_result: Dict, n_top: int = 10) -> pd.DataFrame:
    """
    Get top N most important features from EBM.

    Args:
        ebm_result: result from train_ebm_jsd
        n_top: number of top features

    Returns:
        DataFrame with top features
    """
    return ebm_result['feature_importances'].head(n_top)


def extract_shape_function(ebm, feature_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract shape function for a single feature.

    Args:
        ebm: trained EBM model
        feature_idx: index of feature

    Returns:
        (x_values, y_values): shape function data
    """
    # Get term data
    term_data = ebm.explain_global().data(feature_idx)

    # For main effects
    if 'names' in term_data:
        x_values = np.array(term_data['names'])
        y_values = np.array(term_data['scores'])
        return x_values, y_values

    return None, None


def save_ebm_model(ebm_result: Dict, save_path: str):
    """Save EBM model and results to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(ebm_result, f)

    print(f"[OK] Saved EBM model: {path}")


def load_ebm_model(load_path: str) -> Dict:
    """Load EBM model and results from disk."""
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def analyze_high_jsd_features(covariates: np.ndarray,
                              jsd_values: np.ndarray,
                              feature_names: List[str],
                              percentile_threshold: float = 95) -> pd.DataFrame:
    """
    Compare feature distributions between high-JSD and all samples.

    Args:
        covariates: (N, D) covariate matrix
        jsd_values: (N,) JSD values
        feature_names: list of feature names
        percentile_threshold: percentile for "high JSD" cutoff

    Returns:
        DataFrame comparing feature means with effect sizes
    """
    threshold = np.percentile(jsd_values, percentile_threshold)
    high_jsd_mask = jsd_values >= threshold

    high_jsd_covs = covariates[high_jsd_mask]
    all_covs = covariates

    comparison = []
    for i, name in enumerate(feature_names):
        high_mean = high_jsd_covs[:, i].mean()
        all_mean = all_covs[:, i].mean()
        all_std = all_covs[:, i].std()

        effect_size = (high_mean - all_mean) / (all_std + 1e-8)

        comparison.append({
            'feature': name,
            'high_jsd_mean': high_mean,
            'all_mean': all_mean,
            'all_std': all_std,
            'effect_size': effect_size,
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('effect_size', key=abs, ascending=False)

    return df
