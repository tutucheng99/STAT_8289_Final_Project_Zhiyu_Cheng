"""
Feature Group Definitions for Occlusion Analysis.

Defines two levels of feature grouping:
1. OBSERVATION_GROUPS: Groups for 480-dim raw observations
2. COVARIATE_GROUPS: Groups for 48-dim extracted covariates

The 480-dim observation structure:
- [0:4]     Vulnerability encoding
- [4:8]     Position (passes before opening)
- [8:428]   Bidding history (35 contracts × 3 states × 4 players)
- [428:480] Hand encoding (52 cards)
"""

from typing import Dict, List, Tuple
import numpy as np

# =============================================================================
# Observation-level groups (480-dim)
# =============================================================================

OBSERVATION_GROUPS: Dict[str, Tuple[int, int]] = {
    'vulnerability': (0, 4),      # 4 bits: [NOT_we, we, NOT_they, they]
    'position': (4, 8),           # 4 bits: passes before opening per player
    'bidding': (8, 428),          # 420 bits: 35 contracts × 3 states × 4 players
    'hand': (428, 480),           # 52 bits: card encoding
}

# Subgroups within bidding (for finer analysis)
BIDDING_SUBGROUPS: Dict[str, List[int]] = {
    'level_1': list(range(8, 8 + 5*3*4)),        # 1C-1NT bids
    'level_2': list(range(8 + 5*3*4, 8 + 10*3*4)),  # 2C-2NT bids
    'level_3': list(range(8 + 10*3*4, 8 + 15*3*4)), # 3C-3NT bids
    'level_4_plus': list(range(8 + 15*3*4, 428)),   # 4+ level bids
}

# =============================================================================
# Covariate-level groups (48-dim)
# =============================================================================

# Hand features (28 total)
HAND_FEATURES = [
    # HCP (5)
    'hcp_total', 'hcp_spade', 'hcp_heart', 'hcp_diamond', 'hcp_club',
    # Distribution (9)
    'len_spade', 'len_heart', 'len_diamond', 'len_club',
    'longest_suit', 'shortest_suit', 'is_balanced', 'n_singletons', 'n_voids',
    # Controls & Tricks (7)
    'controls_total', 'controls_spade', 'controls_heart', 'controls_diamond', 'controls_club',
    'ltc', 'quick_tricks',
    # Honors (7)
    'n_aces', 'n_kings', 'n_queens', 'n_jacks',
    'n_honors_in_long_suits', 'has_ak_in_any_suit', 'n_suits_with_honors',
]

# Bidding features (13 total)
BIDDING_FEATURES = [
    'n_contracts_bid', 'auction_level', 'contract_strain',
    'is_doubled', 'is_redoubled', 'is_competitive', 'is_passout', 'has_contract',
    'self_opened', 'partner_opened', 'rho_opened', 'lho_opened',
    'double_status',
]

# Context features (7 total)
CONTEXT_FEATURES = [
    'we_vulnerable', 'they_vulnerable', 'both_vulnerable', 'none_vulnerable',
    'favorable_vul', 'unfavorable_vul', 'n_passed_before_opening',
]

COVARIATE_GROUPS: Dict[str, List[str]] = {
    'hand': HAND_FEATURES,
    'bidding': BIDDING_FEATURES,
    'context': CONTEXT_FEATURES,
}

# Feature subgroups for finer analysis
COVARIATE_SUBGROUPS: Dict[str, List[str]] = {
    'hcp': ['hcp_total', 'hcp_spade', 'hcp_heart', 'hcp_diamond', 'hcp_club'],
    'distribution': ['len_spade', 'len_heart', 'len_diamond', 'len_club',
                     'longest_suit', 'shortest_suit', 'is_balanced', 'n_singletons', 'n_voids'],
    'controls': ['controls_total', 'controls_spade', 'controls_heart', 'controls_diamond', 'controls_club',
                 'ltc', 'quick_tricks'],
    'honors': ['n_aces', 'n_kings', 'n_queens', 'n_jacks',
               'n_honors_in_long_suits', 'has_ak_in_any_suit', 'n_suits_with_honors'],
    'auction_state': ['n_contracts_bid', 'auction_level', 'contract_strain',
                      'is_doubled', 'is_redoubled', 'is_competitive', 'is_passout', 'has_contract'],
    'opener_info': ['self_opened', 'partner_opened', 'rho_opened', 'lho_opened'],
    'vulnerability': ['we_vulnerable', 'they_vulnerable', 'both_vulnerable', 'none_vulnerable',
                      'favorable_vul', 'unfavorable_vul'],
}


def get_group_indices(feature_names: List[str], group_features: List[str]) -> np.ndarray:
    """
    Get indices of group features in feature_names array.

    Args:
        feature_names: List of all feature names
        group_features: List of features in the group

    Returns:
        Array of indices
    """
    indices = []
    for feat in group_features:
        if feat in feature_names:
            indices.append(feature_names.index(feat))
    return np.array(indices, dtype=np.int32)


def get_observation_mask(group_name: str) -> np.ndarray:
    """
    Get boolean mask for observation group.

    Args:
        group_name: Name of the observation group

    Returns:
        Boolean mask of shape (480,)
    """
    mask = np.zeros(480, dtype=bool)
    if group_name in OBSERVATION_GROUPS:
        start, end = OBSERVATION_GROUPS[group_name]
        mask[start:end] = True
    return mask


def get_all_feature_names() -> List[str]:
    """Get all 48 feature names in canonical order."""
    return HAND_FEATURES + BIDDING_FEATURES + CONTEXT_FEATURES
