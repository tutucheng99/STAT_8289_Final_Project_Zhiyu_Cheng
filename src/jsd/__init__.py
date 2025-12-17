# JSD (Jensen-Shannon Divergence) module for bridge bidding policy analysis
"""
JSD analysis for comparing bridge bidding policies.

Key components:
- metrics: JSD computation using scipy.special.xlogy for numerical stability
- gam_fitting: Fit GAM models for JSD curves (Definition A: state-level, Definition B: bin-level)
- ebm_analysis: EBM interpretable model for JSD prediction
- visualization: Publication-quality plots
"""

from .metrics import jsd, jsd_legal_only, jsd_batch_legal, compute_sensitivity_stats
from .gam_fitting import (
    fit_jsd_gam_A, compute_jsd_curve_B, compute_jsd_curves_both, batch_fit_jsd_curves
)
from .ebm_analysis import train_ebm_jsd, get_top_features, analyze_high_jsd_features
from .visualization import (
    plot_jsd_distribution, plot_sensitivity_scatter, plot_jsd_curve_A,
    plot_jsd_curve_B, plot_jsd_curves_comparison, plot_jsd_vs_covariates,
    plot_feature_importance, plot_jsd_heatmap
)

__all__ = [
    # Metrics
    'jsd',
    'jsd_legal_only',
    'jsd_batch_legal',
    'compute_sensitivity_stats',
    # GAM fitting
    'fit_jsd_gam_A',
    'compute_jsd_curve_B',
    'compute_jsd_curves_both',
    'batch_fit_jsd_curves',
    # EBM analysis
    'train_ebm_jsd',
    'get_top_features',
    'analyze_high_jsd_features',
    # Visualization
    'plot_jsd_distribution',
    'plot_sensitivity_scatter',
    'plot_jsd_curve_A',
    'plot_jsd_curve_B',
    'plot_jsd_curves_comparison',
    'plot_jsd_vs_covariates',
    'plot_feature_importance',
    'plot_jsd_heatmap',
]
