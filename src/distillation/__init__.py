"""
Distillation Module for Bridge Bidding Policy Interpretability.

This module provides tools for distilling black-box neural network policies
into interpretable models (Decision Trees, GAMs).
"""

from .tree_distill import train_decision_tree, extract_rules, evaluate_tree
from .gam_distill import train_gam_per_action, get_shape_functions
from .metrics import compute_fidelity_metrics, action_agreement

__all__ = [
    'train_decision_tree',
    'extract_rules',
    'evaluate_tree',
    'train_gam_per_action',
    'get_shape_functions',
    'compute_fidelity_metrics',
    'action_agreement',
]
