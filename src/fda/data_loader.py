"""
Data loading and ALR transform for FDA analysis.

Key functions:
- load_fda_data: Load policy samples and metadata
- compute_alr_transform: Compute ALR transform with Pass as reference
- get_analyzable_actions: Filter rare/illegal actions
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_fda_data(data_dir: str = 'data/processed/policy_samples',
                  data_file: str = '100K_pi_H_v2_policy_samples.npz',
                  meta_file: str = '100K_pi_H_v2_metadata.json') -> Dict:
    """
    Load policy samples and metadata for FDA analysis.

    Returns:
        Dict with keys:
        - covariates: (N, 48) float32
        - pi_H: (N, 38) float32
        - pi_R: (N, 38) float32
        - legal_masks: (N, 38) bool
        - episode_ids: (N,) int32
        - feature_names: List[str]
        - action_names: List[str]
        - ref_action_idx: int (0 for Pass)
        - meta: full metadata dict
    """
    data_path = Path(data_dir)

    # Load data
    data = np.load(data_path / data_file)
    with open(data_path / meta_file) as f:
        meta = json.load(f)

    # Extract arrays
    covariates = data['covariates']
    pi_H = data['pi_H']
    pi_R = data['pi_R']
    legal_masks = data['legal_masks']
    episode_ids = data['episode_ids']

    # Get names from metadata
    feature_names = meta['feature_names']
    action_names = meta['action_names']
    ref_action_idx = meta['ref_action_idx']

    # Assertions for consistency
    assert ref_action_idx == 0, f"Expected ref_action_idx=0, got {ref_action_idx}"
    assert action_names[0] == "Pass", f"Expected action_names[0]='Pass', got {action_names[0]}"
    assert covariates.shape[1] == len(feature_names), "Feature count mismatch!"
    assert pi_H.shape[1] == len(action_names), "Action count mismatch!"

    print(f"[OK] Data loaded: {len(covariates):,} samples, {len(feature_names)} features, {len(action_names)} actions")
    print(f"  Episodes: {len(np.unique(episode_ids)):,}")

    return {
        'covariates': covariates,
        'pi_H': pi_H,
        'pi_R': pi_R,
        'legal_masks': legal_masks,
        'episode_ids': episode_ids,
        'feature_names': feature_names,
        'action_names': action_names,
        'ref_action_idx': ref_action_idx,
        'meta': meta,
    }


def compute_alr_transform(pi_H: np.ndarray, pi_R: np.ndarray,
                          ref_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Additive Log-Ratio (ALR) transform.

    ALR_b(s) = log(π(b|s) / π(ref|s))

    Args:
        pi_H: (N, K) policy probabilities for human proxy
        pi_R: (N, K) policy probabilities for RL policy
        ref_idx: reference action index (default 0 = Pass)

    Returns:
        alr_H: (N, K) ALR transform of pi_H
        alr_R: (N, K) ALR transform of pi_R
        alr_diff: (N, K) = alr_R - alr_H (the main analysis target)
    """
    # ALR transform
    alr_H = np.log(pi_H / pi_H[:, ref_idx:ref_idx+1])
    alr_R = np.log(pi_R / pi_R[:, ref_idx:ref_idx+1])

    # Difference (main analysis target)
    alr_diff = alr_R - alr_H

    return alr_H, alr_R, alr_diff


def get_analyzable_actions(legal_masks: np.ndarray,
                           episode_ids: np.ndarray,
                           action_names: List[str],
                           ref_idx: int = 0,
                           min_legal_rate: float = 0.005,
                           min_unique_eps: int = 200,
                           rare_actions: Optional[set] = None) -> List[int]:
    """
    Get list of analyzable action indices (excluding Pass and rare actions).

    Args:
        legal_masks: (N, K) bool
        episode_ids: (N,) int
        action_names: List[str]
        ref_idx: reference action index to exclude
        min_legal_rate: minimum legal rate threshold (default 0.5%)
        min_unique_eps: minimum unique episodes required
        rare_actions: set of action indices to exclude

    Returns:
        List of analyzable action indices
    """
    if rare_actions is None:
        rare_actions = set()

    analyzable = []
    skipped = []

    for action_idx in range(len(action_names)):
        if action_idx == ref_idx:
            continue  # Skip reference action (Pass)
        if action_idx in rare_actions:
            skipped.append((action_names[action_idx], 'rare'))
            continue

        legal_mask = legal_masks[:, action_idx]
        legal_rate = legal_mask.mean()

        if legal_rate < min_legal_rate:
            skipped.append((action_names[action_idx], f'legal_rate={legal_rate:.3f}'))
            continue

        unique_eps = len(np.unique(episode_ids[legal_mask]))
        if unique_eps < min_unique_eps:
            skipped.append((action_names[action_idx], f'unique_eps={unique_eps}'))
            continue

        analyzable.append(action_idx)

    print(f"[OK] Analyzable actions: {len(analyzable)} / {len(action_names)-1}")
    if skipped:
        print(f"  Skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    return analyzable


def get_feature_index(feature_names: List[str], name: str) -> int:
    """Get feature index by name."""
    return feature_names.index(name)


# Continuous covariates suitable for GAM spline fitting
CONTINUOUS_COVARIATES = [
    'hcp_total',
    'controls_total',
    'ltc',
    'quick_tricks',
    'n_contracts_bid',
]

# Priority actions for analysis (excluding Pass)
PRIORITY_ACTIONS = ['Dbl', '1C', '1D', '1H', '1S', '1NT', '2C']
