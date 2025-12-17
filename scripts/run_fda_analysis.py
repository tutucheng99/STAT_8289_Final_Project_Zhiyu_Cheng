#!/usr/bin/env python3
"""
FDA Analysis for Bridge Bidding Policy Comparison

This script runs the complete FDA analysis pipeline:
1. Load data and compute ALR transforms
2. Exploratory Data Analysis (EDA)
3. GAM fitting for difference curves
4. Statistical inference (Bootstrap CI + Permutation p-values)
5. Visualization
6. Report generation

Usage:
    python scripts/run_fda_analysis.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from statsmodels.stats.multitest import multipletests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fda.data_loader import (
    load_fda_data, compute_alr_transform, get_analyzable_actions,
    CONTINUOUS_COVARIATES, PRIORITY_ACTIONS
)
from fda.gam_fitting import fit_difference_curve, batch_fit_difference_curves
from fda.bootstrap import bootstrap_difference_curve, run_permutation_tests
from fda.visualization import (
    plot_difference_curve, plot_difference_heatmap, plot_significance_heatmap,
    plot_legal_rates, plot_sample_counts, plot_alr_distribution
)


# Configuration
# Note: For production, use n_bootstrap=500, n_permutations=500
# For quick testing, use smaller values (50-100)
CONFIG = {
    'data_dir': 'data/processed/policy_samples',
    'data_file': '100K_pi_H_v2_policy_samples.npz',
    'meta_file': '100K_pi_H_v2_metadata.json',
    'results_dir': 'results/fda',
    'n_bootstrap': 100,       # Production run
    'n_permutations': 100,    # Production run
    'n_splines': 10,
    'seed': 42,
    'fdr_alpha': 0.05,
}


def main():
    print("=" * 60)
    print("FDA Analysis for Bridge Bidding Policy Comparison")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results_dir = Path(CONFIG['results_dir'])

    # ========== Step 4.0: Load Data ==========
    print("\n" + "=" * 40)
    print("Step 4.0: Load Data & Preprocessing")
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
    ref_idx = data['ref_action_idx']

    # Compute ALR transform
    print("\nComputing ALR transform...")
    alr_H, alr_R, alr_diff = compute_alr_transform(pi_H, pi_R, ref_idx)
    print(f"[OK] ALR transform computed: shape={alr_diff.shape}")

    # Get analyzable actions
    print("\nFiltering analyzable actions...")
    analyzable_actions = get_analyzable_actions(
        legal_masks, episode_ids, action_names, ref_idx
    )

    # Map priority action names to indices
    priority_action_indices = []
    for name in PRIORITY_ACTIONS:
        if name in action_names:
            idx = action_names.index(name)
            if idx in analyzable_actions:
                priority_action_indices.append(idx)

    print(f"Priority actions for analysis: {[action_names[i] for i in priority_action_indices]}")

    # ========== Step 4.1: EDA ==========
    print("\n" + "=" * 40)
    print("Step 4.1: Exploratory Data Analysis")
    print("=" * 40)

    eda_dir = results_dir / 'eda'
    eda_dir.mkdir(parents=True, exist_ok=True)

    # 1. Legal rates plot
    print("\nGenerating legal rates plot...")
    fig = plot_legal_rates(legal_masks, action_names,
                           save_path=str(eda_dir / 'qc_legal_rates.png'))
    plt.close(fig)
    print(f"[OK] Saved: {eda_dir / 'qc_legal_rates.png'}")

    # 2. Sample counts heatmap
    print("\nGenerating sample counts heatmap...")
    fig = plot_sample_counts(covariates, legal_masks, episode_ids,
                             feature_names, action_names,
                             CONTINUOUS_COVARIATES, priority_action_indices,
                             save_path=str(eda_dir / 'qc_sample_counts.png'))
    plt.close(fig)
    print(f"[OK] Saved: {eda_dir / 'qc_sample_counts.png'}")

    # 3. ALR distribution
    print("\nGenerating ALR distribution plot...")
    fig = plot_alr_distribution(alr_diff, legal_masks, action_names,
                                priority_action_indices,
                                save_path=str(eda_dir / 'alr_diff_distribution.png'))
    plt.close(fig)
    print(f"[OK] Saved: {eda_dir / 'alr_diff_distribution.png'}")

    # 4. EDA statistics table
    print("\nComputing EDA statistics...")
    eda_stats = []
    for action_idx in priority_action_indices:
        legal_mask = legal_masks[:, action_idx]
        diff = alr_diff[legal_mask, action_idx]
        eda_stats.append({
            'action': action_names[action_idx],
            'action_idx': action_idx,
            'mean_diff': diff.mean(),
            'std_diff': diff.std(),
            'median_diff': np.median(diff),
            'n_legal': legal_mask.sum(),
            'legal_rate': legal_mask.mean(),
            'n_episodes': len(np.unique(episode_ids[legal_mask])),
        })

    eda_df = pd.DataFrame(eda_stats)
    eda_df = eda_df.sort_values('mean_diff', key=abs, ascending=False)
    eda_df.to_csv(eda_dir / 'eda_statistics.csv', index=False)
    print(f"[OK] Saved: {eda_dir / 'eda_statistics.csv'}")
    print("\nTop actions by |mean_diff|:")
    print(eda_df[['action', 'mean_diff', 'std_diff', 'n_legal']].head(7).to_string(index=False))

    # ========== Step 4.2: GAM Fitting ==========
    print("\n" + "=" * 40)
    print("Step 4.2: GAM Fitting (Point Estimates)")
    print("=" * 40)

    print(f"\nFitting {len(CONTINUOUS_COVARIATES)} covariates x {len(priority_action_indices)} actions...")
    gam_results = batch_fit_difference_curves(
        covariates, alr_diff, legal_masks, episode_ids,
        feature_names, action_names,
        CONTINUOUS_COVARIATES, priority_action_indices,
        n_splines=CONFIG['n_splines']
    )

    # Save GAM results
    models_dir = results_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save without the GAM objects (they're not picklable easily)
    gam_results_save = {}
    for key, val in gam_results.items():
        gam_results_save[key] = {k: v for k, v in val.items() if k != 'gam'}
    with open(models_dir / 'gam_point_estimates.pkl', 'wb') as f:
        pickle.dump(gam_results_save, f)
    print(f"[OK] Saved: {models_dir / 'gam_point_estimates.pkl'}")

    # ========== Step 4.3: Bootstrap CI + Permutation Tests ==========
    print("\n" + "=" * 40)
    print("Step 4.3: Statistical Inference")
    print("=" * 40)

    bootstrap_results = {}
    stats_dir = results_dir / 'statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning cluster bootstrap (n={CONFIG['n_bootstrap']})...")
    for cov_name in CONTINUOUS_COVARIATES:
        cov_idx = feature_names.index(cov_name)
        print(f"\n  Covariate: {cov_name}")

        for action_idx in priority_action_indices:
            action_name = action_names[action_idx]
            key = (cov_name, action_name)

            # Get lam from GAM fit
            lam = gam_results[key]['lam']

            result = bootstrap_difference_curve(
                X=covariates[:, cov_idx],
                alr_diff=alr_diff[:, action_idx],
                legal_mask=legal_masks[:, action_idx],
                episode_ids=episode_ids,
                n_bootstrap=CONFIG['n_bootstrap'],
                n_splines=CONFIG['n_splines'],
                lam=lam,
                seed=CONFIG['seed'],
                verbose=False
            )
            bootstrap_results[key] = result
            print(f"    {action_name}: {result['bootstrap_pct_significant_pointwise']*100:.1f}% significant")

    # Save bootstrap results
    with open(stats_dir / 'bootstrap_results.pkl', 'wb') as f:
        pickle.dump(bootstrap_results, f)
    print(f"\n[OK] Saved: {stats_dir / 'bootstrap_results.pkl'}")

    # Permutation tests
    print(f"\nRunning permutation tests (n={CONFIG['n_permutations']})...")
    perm_results = run_permutation_tests(
        covariates, alr_diff, legal_masks, episode_ids,
        feature_names, action_names,
        CONTINUOUS_COVARIATES, priority_action_indices,
        n_perm=CONFIG['n_permutations'],
        n_splines=CONFIG['n_splines'],
        seed=CONFIG['seed']
    )

    # BH-FDR correction
    p_values = [r['p_raw'] for r in perm_results]
    rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh', alpha=CONFIG['fdr_alpha'])

    for i, r in enumerate(perm_results):
        r['p_adjusted'] = p_adjusted[i]
        r['significant_fdr'] = rejected[i]

    perm_df = pd.DataFrame(perm_results)
    perm_df = perm_df.sort_values('p_adjusted')
    perm_df.to_csv(stats_dir / 'permutation_pvalues.csv', index=False)
    print(f"[OK] Saved: {stats_dir / 'permutation_pvalues.csv'}")

    n_sig = sum(rejected)
    print(f"\nSignificant after BH-FDR correction (alpha={CONFIG['fdr_alpha']}): {n_sig}/{len(p_values)}")

    # ========== Step 4.4: Visualization ==========
    print("\n" + "=" * 40)
    print("Step 4.4: Visualization")
    print("=" * 40)

    curves_dir = results_dir / 'alr_curves'
    curves_dir.mkdir(parents=True, exist_ok=True)

    # Individual curves
    print("\nGenerating individual curve plots...")
    for key, result in bootstrap_results.items():
        cov_name, action_name = key
        fig = plot_difference_curve(result, cov_name, action_name,
                                    save_path=str(curves_dir / f'{cov_name}_{action_name}.png'))
        plt.close(fig)
    print(f"[OK] Saved {len(bootstrap_results)} curve plots to {curves_dir}")

    # Heatmaps
    print("\nGenerating heatmaps...")
    action_labels = [action_names[i] for i in priority_action_indices]

    fig = plot_difference_heatmap(bootstrap_results, CONTINUOUS_COVARIATES, action_labels,
                                  save_path=str(curves_dir / 'summary_heatmap_diff.png'))
    plt.close(fig)
    print(f"[OK] Saved: {curves_dir / 'summary_heatmap_diff.png'}")

    fig = plot_significance_heatmap(bootstrap_results, CONTINUOUS_COVARIATES, action_labels,
                                    save_path=str(curves_dir / 'summary_heatmap_sig.png'))
    plt.close(fig)
    print(f"[OK] Saved: {curves_dir / 'summary_heatmap_sig.png'}")

    # ========== Step 4.5: Report Generation ==========
    print("\n" + "=" * 40)
    print("Step 4.5: Report Generation")
    print("=" * 40)

    # Summary statistics table
    summary_stats = []
    for key, result in bootstrap_results.items():
        cov_name, action_name = key
        perm_row = perm_df[(perm_df['covariate'] == cov_name) & (perm_df['action'] == action_name)]

        summary_stats.append({
            'covariate': cov_name,
            'action': action_name,
            'mean_diff': np.mean(result['point_estimate_curve']),
            'pct_significant_pointwise': result['bootstrap_pct_significant_pointwise'],
            'pct_significant_simultaneous': result['bootstrap_pct_significant_simultaneous'],
            'p_raw': perm_row['p_raw'].values[0] if len(perm_row) > 0 else np.nan,
            'p_adjusted': perm_row['p_adjusted'].values[0] if len(perm_row) > 0 else np.nan,
            'significant_fdr': perm_row['significant_fdr'].values[0] if len(perm_row) > 0 else False,
            'n_samples': result['n_samples'],
            'n_episodes': result['n_episodes'],
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('p_adjusted')
    summary_df.to_csv(stats_dir / 'curve_differences_alr.csv', index=False)
    print(f"[OK] Saved: {stats_dir / 'curve_differences_alr.csv'}")

    # JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'data_summary': {
            'n_samples': len(covariates),
            'n_episodes': len(np.unique(episode_ids)),
            'n_features': len(feature_names),
            'n_actions': len(action_names),
        },
        'analysis_summary': {
            'covariates_analyzed': CONTINUOUS_COVARIATES,
            'actions_analyzed': [action_names[i] for i in priority_action_indices],
            'n_combinations': len(bootstrap_results),
            'n_significant_fdr': int(sum(rejected)),
            'fdr_alpha': CONFIG['fdr_alpha'],
        },
        'top_effects': summary_df[summary_df['significant_fdr']].head(10).to_dict('records'),
    }

    with open(results_dir / 'fda_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] Saved: {results_dir / 'fda_report.json'}")

    # Final summary
    print("\n" + "=" * 60)
    print("FDA Analysis Complete!")
    print("=" * 60)
    print(f"\nResults directory: {results_dir}")
    print(f"  - EDA plots: {eda_dir}")
    print(f"  - ALR curves: {curves_dir}")
    print(f"  - Statistics: {stats_dir}")
    print(f"  - Models: {models_dir}")
    print(f"\nSignificant effects (FDR < {CONFIG['fdr_alpha']}):")
    sig_df = summary_df[summary_df['significant_fdr']]
    if len(sig_df) > 0:
        print(sig_df[['covariate', 'action', 'mean_diff', 'p_adjusted']].to_string(index=False))
    else:
        print("  No significant effects after FDR correction.")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    main()
