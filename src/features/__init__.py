"""
Bridge Bidding Feature Extraction Module

This module extracts interpretable statistical covariates from
PGX 480-dim observations for FDA analysis and interpretable distillation.
"""
from .constants import SUIT_ORDER, RANK_ORDER, HCP_WEIGHTS, CONTROL_WEIGHTS
from .hand_features import extract_hand_features
from .bidding_features import extract_bidding_features
from .context_features import extract_context_features
from .feature_extractor import BridgeFeatureExtractor

__all__ = [
    'SUIT_ORDER',
    'RANK_ORDER',
    'HCP_WEIGHTS',
    'CONTROL_WEIGHTS',
    'extract_hand_features',
    'extract_bidding_features',
    'extract_context_features',
    'BridgeFeatureExtractor',
]
