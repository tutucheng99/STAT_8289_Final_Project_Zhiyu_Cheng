#!/usr/bin/env python3
"""
Step 6: Occlusion Sensitivity Analysis

This script performs both approximate and precise occlusion analysis:

1. Approximate Analysis (EBM-based):
   - Uses the EBM model from JSD analysis
   - Permutation importance on 48-dim covariates
   - Fast but less precise

2. Precise Analysis (Model re-inference):
   - Occludes 480-dim observations
   - Re-runs neural network inference
   - Measures KL divergence of policy changes

Usage:
    python scripts/run_occlusion.py --mode approximate
    python scripts/run_occlusion.py --mode precise
    python scripts/run_occlusion.py --mode both

Output:
    results/occlusion/
    ├── approximate_sensitivity.json
    ├── precise_sensitivity.json
    ├── group_sensitivity_approximate.png
    ├── group_sensitivity_precise.png
    └── occlusion_report.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.occlusion.feature_groups import (
    OBSERVATION_GROUPS, COVARIATE_GROUPS, get_all_feature_names
)
from src.occlusion.approximate import approximate_sensitivity_analysis
from src.occlusion.visualization import (
    plot_group_sensitivity, plot_combined_summary
)


def run_approximate_analysis(data_path: str,
                              ebm_path: str,
                              results_dir: Path,
                              n_permutations: int = 10,
                              seed: int = 42) -> dict:
    """Run approximate sensitivity analysis using EBM."""
    print("\n" + "="*60)
    print("APPROXIMATE SENSITIVITY ANALYSIS (EBM-based)")
    print("="*60)

    # Load data
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    covariates = data['covariates']
    print(f"  Covariates shape: {covariates.shape}")

    # Load metadata for feature names
    meta_path = Path(data_path).parent / Path(data_path).name.replace('_policy_samples.npz', '_metadata.json')
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        feature_names = metadata.get('feature_names', get_all_feature_names())
    else:
        feature_names = get_all_feature_names()
    print(f"  Features: {len(feature_names)}")

    # Run analysis
    results = approximate_sensitivity_analysis(
        ebm_path=ebm_path,
        covariates=covariates,
        feature_names=feature_names,
        n_permutations=n_permutations,
        seed=seed,
        verbose=True
    )

    # Save results
    results_path = results_dir / 'approximate_sensitivity.json'

    # Convert to JSON-serializable format
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'method': 'approximate_ebm',
        'n_samples': len(covariates),
        'n_permutations': n_permutations,
        'cv_r2': results['cv_r2'],
        'cv_rmse': results['cv_rmse'],
        'group_importances': results['group_importances'],
        'permutation_sensitivity': results['permutation_sensitivity'],
        'ranking': results['ranking'],
    }

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[OK] Saved: {results_path}")

    # Plot
    plot_path = results_dir / 'group_sensitivity_approximate.png'
    plot_group_sensitivity(
        results,
        title="Feature Group Sensitivity (Approximate - EBM)",
        metric='mean_abs_change',
        save_path=str(plot_path)
    )
    plt.close()

    return results_json


def run_precise_analysis(data_path: str,
                          pi_H_path: str,
                          results_dir: Path,
                          method: str = 'permutation',
                          n_permutations: int = 5,
                          n_samples: int = 10000,
                          batch_size: int = 1024,
                          seed: int = 42) -> dict:
    """Run precise occlusion analysis with model re-inference."""
    print("\n" + "="*60)
    print("PRECISE SENSITIVITY ANALYSIS (Model Re-inference)")
    print("="*60)

    # Import JAX-dependent modules here
    import jax
    from src.policy_loader import PolicyWrapper
    from src.occlusion.precise import precise_occlusion_analysis
    from src.occlusion.metrics import policy_change_metrics

    # Load data
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    observations = data['observations'].astype(np.float32)
    legal_masks = data['legal_masks']
    pi_H = data['pi_H']
    print(f"  Observations shape: {observations.shape}")
    print(f"  Legal masks shape: {legal_masks.shape}")

    # Subsample for efficiency
    if n_samples is not None and n_samples < len(observations):
        print(f"\nSubsampling to {n_samples} samples...")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(observations), n_samples, replace=False)
        observations = observations[indices]
        legal_masks = legal_masks[indices]
        pi_H = pi_H[indices]

    # Load model
    print(f"\nLoading pi_H model from {pi_H_path}...")
    policy = PolicyWrapper(pi_H_path)

    # Run analysis
    print(f"\nRunning precise occlusion analysis ({method} method)...")
    results = precise_occlusion_analysis(
        policy.forward,
        policy.params,
        observations,
        legal_masks,
        pi_H,
        groups=list(OBSERVATION_GROUPS.keys()),
        method=method,
        n_permutations=n_permutations,
        batch_size=batch_size,
        seed=seed,
        verbose=True
    )

    # Save results
    results_path = results_dir / 'precise_sensitivity.json'

    results_json = {
        'timestamp': datetime.now().isoformat(),
        'method': f'precise_{method}',
        'n_samples': len(observations),
        'n_permutations': n_permutations if method == 'permutation' else 1,
        'group_results': results['group_results'],
        'ranking': results['ranking'],
    }

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[OK] Saved: {results_path}")

    # Plot
    plot_path = results_dir / 'group_sensitivity_precise.png'
    plot_group_sensitivity(
        results,
        title="Feature Group Sensitivity (Precise - Re-inference)",
        metric='kl_mean',
        save_path=str(plot_path)
    )
    plt.close()

    return results_json


def generate_report(approx_results: dict,
                     precise_results: dict,
                     results_dir: Path) -> dict:
    """Generate combined analysis report."""
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'observation_groups': {
                name: {'start': start, 'end': end, 'size': end - start}
                for name, (start, end) in OBSERVATION_GROUPS.items()
            },
            'covariate_groups': {
                name: {'features': features, 'size': len(features)}
                for name, features in COVARIATE_GROUPS.items()
            },
        },
    }

    if approx_results:
        report['approximate_analysis'] = {
            'method': approx_results['method'],
            'n_samples': approx_results['n_samples'],
            'ebm_cv_r2': approx_results.get('cv_r2'),
            'ranking': approx_results['ranking'],
        }

    if precise_results:
        report['precise_analysis'] = {
            'method': precise_results['method'],
            'n_samples': precise_results['n_samples'],
            'ranking': precise_results['ranking'],
        }

    # Key findings
    findings = []

    if approx_results and approx_results['ranking']:
        top_approx = approx_results['ranking'][0]
        findings.append(f"Approximate: Most sensitive group is '{top_approx[0]}' "
                       f"(importance={top_approx[1]:.4f})")

    if precise_results and precise_results['ranking']:
        top_precise = precise_results['ranking'][0]
        findings.append(f"Precise: Most sensitive group is '{top_precise[0]}' "
                       f"(KL={top_precise[1]:.4f})")

    report['key_findings'] = findings

    # Save report
    report_path = results_dir / 'occlusion_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved: {report_path}")

    # Print summary
    print("\n" + "-"*40)
    print("KEY FINDINGS:")
    for finding in findings:
        print(f"  • {finding}")
    print("-"*40)

    return report


def main():
    parser = argparse.ArgumentParser(description='Step 6: Occlusion Analysis')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['approximate', 'precise', 'both'],
                        help='Analysis mode')
    parser.add_argument('--data-dir', type=str, default='data/processed/policy_samples',
                        help='Directory containing policy samples')
    parser.add_argument('--data-file', type=str, default='100K_pi_H_v2_policy_samples.npz',
                        help='Policy samples file')
    parser.add_argument('--ebm-path', type=str, default='results/jsd/ebm_model.pkl',
                        help='Path to trained EBM model')
    parser.add_argument('--pi-H-path', type=str, default='checkpoints/pi_H',
                        help='Path to pi_H checkpoint')
    parser.add_argument('--results-dir', type=str, default='results/occlusion',
                        help='Output directory')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples for precise analysis')
    parser.add_argument('--n-permutations', type=int, default=10,
                        help='Number of permutation iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup paths
    data_path = PROJECT_ROOT / args.data_dir / args.data_file
    ebm_path = PROJECT_ROOT / args.ebm_path
    pi_H_path = PROJECT_ROOT / args.pi_H_path
    results_dir = PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STEP 6: OCCLUSION SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"\nMode: {args.mode}")
    print(f"Data: {data_path}")
    print(f"EBM model: {ebm_path}")
    print(f"pi_H checkpoint: {pi_H_path}")
    print(f"Results dir: {results_dir}")

    approx_results = None
    precise_results = None

    # Run approximate analysis
    if args.mode in ['approximate', 'both']:
        if ebm_path.exists():
            approx_results = run_approximate_analysis(
                str(data_path),
                str(ebm_path),
                results_dir,
                n_permutations=args.n_permutations,
                seed=args.seed
            )
        else:
            print(f"\n[WARN] EBM model not found at {ebm_path}")
            print("       Run JSD analysis first or skip approximate mode")

    # Run precise analysis
    if args.mode in ['precise', 'both']:
        if pi_H_path.exists():
            precise_results = run_precise_analysis(
                str(data_path),
                str(pi_H_path),
                results_dir,
                method='permutation',
                n_permutations=args.n_permutations,
                n_samples=args.n_samples,
                seed=args.seed
            )
        else:
            print(f"\n[WARN] pi_H checkpoint not found at {pi_H_path}")
            print("       Skip precise mode or provide correct path")

    # Generate report
    if approx_results or precise_results:
        generate_report(approx_results, precise_results, results_dir)

    print("\n" + "="*60)
    print("STEP 6 COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
