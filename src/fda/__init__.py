# FDA (Functional Data Analysis) module for bridge bidding policy analysis
"""
Compositional FDA analysis for comparing bridge bidding policies.

Key components:
- data_loader: Load and preprocess policy samples, compute ALR transforms
- gam_fitting: Fit GAM models for ALR difference curves
- bootstrap: Cluster bootstrap for confidence intervals, permutation tests for p-values
- visualization: Publication-quality plots
"""

from .data_loader import load_fda_data, compute_alr_transform
from .gam_fitting import fit_difference_curve
from .bootstrap import bootstrap_difference_curve, perm_test_curve_gam

__all__ = [
    'load_fda_data',
    'compute_alr_transform',
    'fit_difference_curve',
    'bootstrap_difference_curve',
    'perm_test_curve_gam',
]
