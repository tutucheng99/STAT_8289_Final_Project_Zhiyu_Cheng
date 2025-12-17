#!/usr/bin/env python3
"""
JSD Analysis for Bridge Bidding Policy Comparison

This script runs the complete JSD analysis pipeline:
1. Load data and compute JSD values
2. Sensitivity analysis (JSD_raw vs JSD_legal)
3. EDA (distribution, high JSD states)
4. JSD curve fitting (Definition A: state-level, Definition B: bin-level)
5. EBM interpretable model
6. Report generation

Usage:
    python scripts/run_jsd_analysis.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fda.data_loader import load_fda_data, CONTINUOUS_COVARIATES
from jsd.metrics import jsd, jsd_legal_only, jsd_batch_legal, compute_sensitivity_stats
from jsd.gam_fitting import batch_fit_jsd_curves
from jsd.ebm_analysis import train_ebm_jsd, analyze_high_jsd_features, save_ebm_model
from jsd.visualization import (
    plot_jsd_distribution, plot_sensitivity_scatter, plot_jsd_curve_A,
    plot_jsd_curve_B, plot_jsd_curves_comparison, plot_jsd_vs_covariates,
    plot_feature_importance, plot_jsd_heatmap
)


# Configuration
CONFIG = {
    'data_dir': 'data/processed/policy_samples',
    'data_file': '100K_pi_H_v2_policy_samples.npz',
    'meta_file': '100K_pi_H_v2_metadata.json',
    'results_dir': 'results/jsd',
    'n_bootstrap': 100,
    'n_bins': 10,
    'n_splines': 10,
    'seed': 42,
    'ebm_interactions': 10,
}

# Additional covariates for JSD analysis (beyond CONTINUOUS_COVARIATES)
ADDITIONAL_COVARIATES = ['auction_level']


def main():
    print("=" * 60)
    print("JSD Analysis for Bridge Bidding Policy Comparison")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results_dir = Path(CONFIG['results_dir'])

    # ========== Step 5.0: Load Data ==========
    print("\n" + "=" * 40)
    print("Step 5.0: Load Data")
    print("=" * 40)

    data = load_fda_data(
        data_dir=CONFIG['data_dir'],
        data_file=CONFIG['data_file'],
        meta_file=CONFIG['meta_file']
    )

    covariates = data['covariates']
    pi_H = data['pi_H']
    pi_R = data['pi_R']
    legal_masks = data['legal_masks']
    episode_ids = data['episode_ids']
    feature_names = data['feature_names']
    action_names = data['action_names']

    print(f"\nData shape: {covariates.shape}")

    # ========== Step 5.1: Compute JSD & Sensitivity Analysis ==========
    print("\n" + "=" * 40)
    print("Step 5.1: JSD Computation & Sensitivity Analysis")
    print("=" * 40)

    eda_dir = results_dir / 'eda'
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Compute raw JSD (all actions, including smoothed illegal)
    print("\nComputing JSD_raw (all actions)...")
    jsd_raw = jsd(pi_H, pi_R)
    print(f"[OK] JSD_raw: mean={jsd_raw.mean():.4f}, median={np.median(jsd_raw):.4f}")

    # Compute legal-only JSD (renormalized)
    print("\nComputing JSD_legal (legal actions only)...")
    jsd_legal = jsd_batch_legal(pi_H, pi_R, legal_masks)
    print(f"[OK] JSD_legal: mean={jsd_legal.mean():.4f}, median={np.median(jsd_legal):.4f}")

    # Sensitivity analysis
    print("\n--- Sensitivity Analysis: JSD_raw vs JSD_legal ---")
    sensitivity_stats = compute_sensitivity_stats(jsd_raw, jsd_legal)

    print(f"  Mean |diff|:  {sensitivity_stats['mean_abs_diff']:.2e}")
    print(f"  Max |diff|:   {sensitivity_stats['max_abs_diff']:.2e}")
    print(f"  P90 |diff|:   {sensitivity_stats['p90_abs_diff']:.2e}")
    print(f"  P95 |diff|:   {sensitivity_stats['p95_abs_diff']:.2e}")
    print(f"  P99 |diff|:   {sensitivity_stats['p99_abs_diff']:.2e}")
    print(f"  Correlation:  {sensitivity_stats['correlation']:.4f}")

    # Save sensitivity stats
    sens_df = pd.DataFrame([sensitivity_stats])
    sens_df.to_csv(eda_dir / 'sensitivity_stats.csv', index=False)
    print(f"\n[OK] Saved: {eda_dir / 'sensitivity_stats.csv'}")

    # Sensitivity scatter plot
    print("\nGenerating sensitivity scatter plot...")
    fig = plot_sensitivity_scatter(jsd_raw, jsd_legal, sensitivity_stats,
                                   save_path=str(eda_dir / 'sensitivity_legal_vs_raw.png'))
    plt.close(fig)

    # Decide which JSD to use for downstream analysis
    # If correlation > 0.99 and mean_abs_diff < 0.01, use jsd_raw for simplicity
    if sensitivity_stats['correlation'] > 0.99 and sensitivity_stats['mean_abs_diff'] < 0.01:
        print("\n[INFO] JSD_raw and JSD_legal are highly correlated. Using JSD_raw for analysis.")
        jsd_values = jsd_raw
        jsd_type = 'raw'
    else:
        print("\n[INFO] Using JSD_legal for analysis (significant difference from JSD_raw).")
        jsd_values = jsd_legal
        jsd_type = 'legal'

    # ========== Step 5.1 (cont): EDA ==========
    print("\n" + "=" * 40)
    print("Step 5.1 (cont): Exploratory Data Analysis")
    print("=" * 40)

    # JSD distribution
    print("\nGenerating JSD distribution plot...")
    fig = plot_jsd_distribution(jsd_values,
                                save_path=str(eda_dir / 'jsd_distribution.png'))
    plt.close(fig)

    # JSD statistics
    jsd_stats = {
        'jsd_type': jsd_type,
        'mean': float(jsd_values.mean()),
        'std': float(jsd_values.std()),
        'median': float(np.median(jsd_values)),
        'min': float(jsd_values.min()),
        'max': float(jsd_values.max()),
        'p90': float(np.percentile(jsd_values, 90)),
        'p95': float(np.percentile(jsd_values, 95)),
        'p99': float(np.percentile(jsd_values, 99)),
    }
    print("\nJSD Statistics:")
    for k, v in jsd_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # JSD vs covariates scatter
    print("\nGenerating JSD vs covariates plot...")
    all_covariates = CONTINUOUS_COVARIATES + ADDITIONAL_COVARIATES
    # Filter to covariates that exist
    plot_covs = [c for c in all_covariates if c in feature_names]
    fig = plot_jsd_vs_covariates(covariates, jsd_values, feature_names, plot_covs,
                                 save_path=str(eda_dir / 'jsd_vs_covariates.png'))
    plt.close(fig)

    # High JSD state analysis
    print("\nAnalyzing high JSD states...")
    high_jsd_analysis = analyze_high_jsd_features(covariates, jsd_values, feature_names,
                                                  percentile_threshold=95)
    high_jsd_analysis.to_csv(eda_dir / 'high_jsd_features.csv', index=False)
    print(f"[OK] Saved: {eda_dir / 'high_jsd_features.csv'}")

    print("\nTop features distinguishing high-JSD states (by effect size):")
    print(high_jsd_analysis[['feature', 'effect_size', 'high_jsd_mean', 'all_mean']].head(10).to_string(index=False))

    # ========== Step 5.2: JSD Curves (A & B) ==========
    print("\n" + "=" * 40)
    print("Step 5.2: JSD Curve Fitting (Definitions A & B)")
    print("=" * 40)

    curves_dir = results_dir / 'curves'
    curves_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = results_dir / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Save JSD quantiles (moved here after stats_dir creation)
    quantiles_df = pd.DataFrame([jsd_stats])
    quantiles_df.to_csv(stats_dir / 'jsd_quantiles.csv', index=False)
    print(f"[OK] Saved: {stats_dir / 'jsd_quantiles.csv'}")

    print(f"\nFitting JSD curves for {len(CONTINUOUS_COVARIATES)} covariates...")
    curve_results = batch_fit_jsd_curves(
        covariates, pi_H, pi_R, jsd_values, episode_ids,
        feature_names, CONTINUOUS_COVARIATES,
        n_bins=CONFIG['n_bins'],
        n_splines=CONFIG['n_splines'],
        n_bootstrap=CONFIG['n_bootstrap'],
        seed=CONFIG['seed'],
        verbose=True
    )

    # Generate individual curve plots
    print("\nGenerating curve plots...")
    for cov_name, result in curve_results.items():
        # Definition A
        fig = plot_jsd_curve_A(result['curve_A'], cov_name,
                               save_path=str(curves_dir / f'jsd_A_vs_{cov_name}.png'))
        plt.close(fig)

        # Definition B
        fig = plot_jsd_curve_B(result['curve_B'], cov_name,
                               save_path=str(curves_dir / f'jsd_B_vs_{cov_name}.png'))
        plt.close(fig)

        # Comparison (A vs B)
        fig = plot_jsd_curves_comparison(result['curve_A'], result['curve_B'],
                                          result['comparison'], cov_name,
                                          save_path=str(curves_dir / f'jsd_AB_comparison_{cov_name}.png'))
        plt.close(fig)

    print(f"[OK] Saved {len(curve_results) * 3} curve plots to {curves_dir}")

    # Heatmap
    fig = plot_jsd_heatmap(curve_results, CONTINUOUS_COVARIATES,
                           save_path=str(curves_dir / 'jsd_heatmap.png'))
    plt.close(fig)

    # Summary statistics table
    summary_rows = []
    for cov_name, result in curve_results.items():
        curve_A = result['curve_A']
        curve_B = result['curve_B']
        comparison = result['comparison']

        summary_rows.append({
            'covariate': cov_name,
            'jsd_A_mean': curve_A['point_estimate'].mean(),
            'jsd_A_min': curve_A['point_estimate'].min(),
            'jsd_A_max': curve_A['point_estimate'].max(),
            'jsd_B_mean': curve_B['jsd_B'].mean(),
            'jsd_B_min': curve_B['jsd_B'].min(),
            'jsd_B_max': curve_B['jsd_B'].max(),
            'mean_A_minus_B': comparison['A_minus_B'].mean(),
            'n_samples': curve_A['n_samples'],
            'n_episodes': curve_A['n_episodes'],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(stats_dir / 'jsd_by_covariate.csv', index=False)
    print(f"\n[OK] Saved: {stats_dir / 'jsd_by_covariate.csv'}")

    # Save curve data
    models_dir = results_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save without GAM objects
    curve_results_save = {}
    for cov_name, result in curve_results.items():
        curve_results_save[cov_name] = {
            'curve_A': {k: v for k, v in result['curve_A'].items() if k != 'gam'},
            'curve_B': result['curve_B'].to_dict(),
            'comparison': result['comparison'].to_dict(),
        }

    with open(models_dir / 'jsd_curves.pkl', 'wb') as f:
        pickle.dump(curve_results_save, f)
    print(f"[OK] Saved: {models_dir / 'jsd_curves.pkl'}")

    # ========== Step 5.3: EBM Interpretable Model ==========
    print("\n" + "=" * 40)
    print("Step 5.3: EBM Interpretable Model")
    print("=" * 40)

    print("\nTraining EBM model...")
    ebm_result = train_ebm_jsd(
        covariates, jsd_values, feature_names,
        n_interactions=CONFIG['ebm_interactions'],
        seed=CONFIG['seed'],
        verbose=True
    )

    # Save EBM model
    save_ebm_model(ebm_result, str(models_dir / 'ebm_jsd_model.pkl'))

    # Feature importance
    importance_df = ebm_result['feature_importances']
    importance_df.to_csv(models_dir / 'feature_importance.csv', index=False)
    print(f"[OK] Saved: {models_dir / 'feature_importance.csv'}")

    print("\nTop 10 features by importance:")
    print(importance_df.head(10).to_string(index=False))

    # Feature importance plot
    fig = plot_feature_importance(importance_df, n_top=15,
                                  save_path=str(curves_dir / 'feature_importance.png'))
    plt.close(fig)

    # ========== Step 5.4: Report Generation ==========
    print("\n" + "=" * 40)
    print("Step 5.4: Report Generation")
    print("=" * 40)

    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'data_summary': {
            'n_samples': len(covariates),
            'n_episodes': len(np.unique(episode_ids)),
            'n_features': len(feature_names),
            'n_actions': len(action_names),
        },
        'sensitivity_analysis': sensitivity_stats,
        'jsd_used': jsd_type,
        'jsd_statistics': jsd_stats,
        'curve_summary': {
            'covariates_analyzed': CONTINUOUS_COVARIATES,
            'n_bins': CONFIG['n_bins'],
            'n_bootstrap': CONFIG['n_bootstrap'],
        },
        'ebm_summary': {
            'cv_r2_mean': ebm_result['cv_r2_mean'],
            'cv_r2_std': ebm_result['cv_r2_std'],
            'cv_rmse_mean': ebm_result['cv_rmse_mean'],
            'cv_rmse_std': ebm_result['cv_rmse_std'],
            'top_features': importance_df.head(10).to_dict('records'),
        },
        'high_jsd_features': high_jsd_analysis.head(10).to_dict('records'),
    }

    with open(results_dir / 'jsd_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] Saved: {results_dir / 'jsd_report.json'}")

    # Final summary
    print("\n" + "=" * 60)
    print("JSD Analysis Complete!")
    print("=" * 60)
    print(f"\nResults directory: {results_dir}")
    print(f"  - EDA: {eda_dir}")
    print(f"  - Curves: {curves_dir}")
    print(f"  - Statistics: {stats_dir}")
    print(f"  - Models: {models_dir}")
    print(f"\nKey findings:")
    print(f"  - JSD type used: {jsd_type}")
    print(f"  - Mean JSD: {jsd_stats['mean']:.4f}")
    print(f"  - Median JSD: {jsd_stats['median']:.4f}")
    print(f"  - 95th percentile: {jsd_stats['p95']:.4f}")
    print(f"  - EBM CV R^2: {ebm_result['cv_r2_mean']:.4f} +/- {ebm_result['cv_r2_std']:.4f}")
    print(f"  - Top feature: {importance_df.iloc[0]['term']}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    main()
