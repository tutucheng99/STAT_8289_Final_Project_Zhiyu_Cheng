"""
Occlusion Analysis Module for Bridge Bidding Policy Interpretability.

This module provides tools for understanding feature importance through
occlusion (permutation) analysis at both approximate and precise levels.
"""

from .feature_groups import OBSERVATION_GROUPS, COVARIATE_GROUPS, get_group_indices
from .metrics import kl_divergence, policy_change_metrics
from .approximate import approximate_sensitivity_analysis
from .precise import precise_occlusion_analysis

__all__ = [
    'OBSERVATION_GROUPS',
    'COVARIATE_GROUPS',
    'get_group_indices',
    'kl_divergence',
    'policy_change_metrics',
    'approximate_sensitivity_analysis',
    'precise_occlusion_analysis',
]
