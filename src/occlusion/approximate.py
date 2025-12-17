"""
Approximate Sensitivity Analysis using EBM.

Uses the EBM model trained in JSD analysis to approximate feature sensitivity
without requiring model re-inference. This is faster but less precise than
direct occlusion analysis.

Method:
1. Load trained EBM model from JSD analysis
2. Extract feature importances and shape functions
3. Aggregate by feature groups
4. Simulate occlusion effects using EBM predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from .feature_groups import COVARIATE_GROUPS, get_group_indices, get_all_feature_names


def load_ebm_results(ebm_path: str) -> Dict:
    """Load EBM model and results from disk."""
    with open(ebm_path, 'rb') as f:
        return pickle.load(f)


def extract_ebm_importances(ebm_result: Dict) -> pd.DataFrame:
    """
    Extract feature importances from EBM.

    Returns DataFrame with columns: term, importance
    """
    return ebm_result['feature_importances']


def aggregate_importance_by_group(importance_df: pd.DataFrame,
                                   feature_names: List[str]) -> Dict[str, float]:
    """
    Aggregate feature importances by covariate group.

    Args:
        importance_df: DataFrame with 'term' and 'importance' columns
        feature_names: List of feature names in order

    Returns:
        Dict[group_name, total_importance]
    """
    group_importance = {}

    for group_name, group_features in COVARIATE_GROUPS.items():
        total = 0.0
        for feat in group_features:
            # Match exact feature name or interaction containing it
            matches = importance_df[importance_df['term'] == feat]
            if len(matches) > 0:
                total += matches['importance'].values[0]

        group_importance[group_name] = total

    return group_importance


def permutation_sensitivity_ebm(ebm_result: Dict,
                                 covariates: np.ndarray,
                                 feature_names: List[str],
                                 n_permutations: int = 10,
                                 seed: int = 42) -> Dict[str, Dict]:
    """
    Estimate feature group sensitivity via EBM permutation importance.

    For each feature group, shuffle the features and measure the change
    in EBM predictions.

    Args:
        ebm_result: Trained EBM result dict
        covariates: (N, D) covariate matrix
        feature_names: List of feature names
        n_permutations: Number of permutation iterations
        seed: Random seed

    Returns:
        Dict[group_name, sensitivity_metrics]
    """
    rng = np.random.default_rng(seed)
    ebm = ebm_result['ebm']

    # Baseline predictions
    baseline_pred = ebm.predict(covariates)

    results = {}

    for group_name, group_features in COVARIATE_GROUPS.items():
        indices = get_group_indices(feature_names, group_features)
        if len(indices) == 0:
            continue

        pred_diffs = []
        for _ in range(n_permutations):
            # Shuffle group features
            covariates_perm = covariates.copy()
            for idx in indices:
                covariates_perm[:, idx] = rng.permutation(covariates[:, idx])

            # Get new predictions
            perm_pred = ebm.predict(covariates_perm)
            pred_diffs.append(np.abs(perm_pred - baseline_pred))

        pred_diffs = np.array(pred_diffs)

        results[group_name] = {
            'mean_abs_change': float(np.mean(pred_diffs)),
            'std_abs_change': float(np.std(pred_diffs)),
            'max_abs_change': float(np.max(pred_diffs)),
            'median_abs_change': float(np.median(pred_diffs)),
            'n_features': len(indices),
            'feature_names': list(group_features),
        }

    return results


def approximate_sensitivity_analysis(ebm_path: str,
                                      covariates: np.ndarray,
                                      feature_names: List[str],
                                      n_permutations: int = 10,
                                      seed: int = 42,
                                      verbose: bool = True) -> Dict:
    """
    Run full approximate sensitivity analysis.

    Args:
        ebm_path: Path to saved EBM model
        covariates: (N, D) covariate matrix
        feature_names: List of feature names
        n_permutations: Number of permutation iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with:
        - ebm_importances: Feature importance DataFrame
        - group_importances: Dict[group_name, total_importance]
        - permutation_sensitivity: Dict[group_name, sensitivity_metrics]
        - ranking: List of (group_name, importance) sorted descending
    """
    if verbose:
        print("Loading EBM model...")
    ebm_result = load_ebm_results(ebm_path)

    if verbose:
        print("Extracting feature importances...")
    importance_df = extract_ebm_importances(ebm_result)

    if verbose:
        print("Aggregating by feature group...")
    group_importance = aggregate_importance_by_group(importance_df, feature_names)

    if verbose:
        print(f"Running permutation sensitivity ({n_permutations} iterations)...")
    perm_sensitivity = permutation_sensitivity_ebm(
        ebm_result, covariates, feature_names, n_permutations, seed
    )

    # Rank groups by permutation sensitivity
    ranking = sorted(
        [(name, metrics['mean_abs_change']) for name, metrics in perm_sensitivity.items()],
        key=lambda x: x[1],
        reverse=True
    )

    if verbose:
        print("\nFeature Group Ranking (by permutation sensitivity):")
        for name, score in ranking:
            print(f"  {name}: {score:.6f}")

    return {
        'ebm_importances': importance_df,
        'group_importances': group_importance,
        'permutation_sensitivity': perm_sensitivity,
        'ranking': ranking,
        'cv_r2': ebm_result.get('cv_r2_mean'),
        'cv_rmse': ebm_result.get('cv_rmse_mean'),
    }


def compare_approximate_vs_ebm(ebm_result: Dict,
                                feature_names: List[str]) -> pd.DataFrame:
    """
    Compare individual feature vs group-level importances.

    Returns DataFrame with both levels of analysis.
    """
    importance_df = extract_ebm_importances(ebm_result)

    # Add group labels
    def get_group(term):
        for group_name, features in COVARIATE_GROUPS.items():
            if term in features:
                return group_name
        # Check if it's an interaction
        if ' & ' in term:
            parts = term.split(' & ')
            groups = [get_group(p.strip()) for p in parts]
            return ' × '.join(sorted(set(g for g in groups if g)))
        return 'interaction'

    importance_df = importance_df.copy()
    importance_df['group'] = importance_df['term'].apply(get_group)

    return importance_df
