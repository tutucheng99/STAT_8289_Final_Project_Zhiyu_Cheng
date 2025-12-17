#!/usr/bin/env python3
"""
Step 7: Rule Distillation

Distills the black-box π^R policy into interpretable models:
1. Decision Tree - extracts discrete rules
2. GAM - shows continuous feature effects via shape functions

Usage:
    python scripts/run_distillation.py --mode all
    python scripts/run_distillation.py --mode tree
    python scripts/run_distillation.py --mode gam

Output:
    results/distillation/
    ├── models/
    │   ├── decision_trees.pkl
    │   └── gam_models.pkl
    ├── analysis/
    │   ├── tree_comparison.csv
    │   ├── feature_importance.csv
    │   └── fidelity_metrics.csv
    ├── plots/
    │   ├── fidelity_vs_complexity.png
    │   ├── tree_feature_importance.png
    │   ├── gam_shape_functions_*.png
    │   └── confusion_matrix.png
    └── distillation_report.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation.tree_distill import (
    train_trees_multiple_depths, extract_rules, get_feature_importance, save_tree
)
from src.distillation.gam_distill import (
    train_gam_per_action, get_top_k_actions, save_gam_results
)
from src.distillation.metrics import compute_fidelity_metrics, per_action_accuracy
from src.distillation.visualization import (
    plot_fidelity_vs_complexity, plot_tree_feature_importance,
    plot_gam_shape_functions, plot_action_confusion_matrix,
    plot_distillation_summary
)

# Action names for bridge bidding
ACTION_NAMES = ['Pass', 'Dbl', 'RDbl'] + [
    f'{level}{strain}' for level in range(1, 8)
    for strain in ['C', 'D', 'H', 'S', 'NT']
]


def load_data(data_path: str, meta_path: str = None) -> dict:
    """Load policy samples data."""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    # Load metadata for feature names
    if meta_path and Path(meta_path).exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        feature_names = metadata.get('feature_names', None)
    else:
        feature_names = None

    return {
        'covariates': data['covariates'],
        'pi_R': data['pi_R'],
        'pi_H': data['pi_H'],
        'legal_masks': data['legal_masks'],
        'feature_names': feature_names,
    }


def run_tree_distillation(X_train, y_train, X_val, y_val,
                           pi_R_val, feature_names, results_dir, verbose=True):
    """Run decision tree distillation."""
    print("\n" + "="*60)
    print("DECISION TREE DISTILLATION")
    print("="*60)

    # Train trees with multiple depths
    depths = [3, 5, 7, 10, 15, None]
    if verbose:
        print(f"\nTraining decision trees with depths: {depths}")

    tree_results = train_trees_multiple_depths(
        X_train, y_train, X_val, y_val, pi_R_val,
        depths=depths, verbose=verbose
    )

    # Save trees
    models_dir = results_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    for depth_name, data in tree_results.items():
        save_tree(data['tree'], models_dir / f'tree_depth_{depth_name}.pkl')

    # Feature importance from best tree (depth=10)
    best_depth = '10'
    if best_depth in tree_results:
        best_tree = tree_results[best_depth]['tree']
        importance_df = get_feature_importance(best_tree, feature_names)
        importance_df.to_csv(results_dir / 'analysis' / 'tree_feature_importance.csv', index=False)

        # Extract rules
        rules_text = extract_rules(best_tree, feature_names, max_depth=5)
        with open(results_dir / 'analysis' / 'tree_rules.txt', 'w') as f:
            f.write(rules_text)
        print(f"\n[OK] Saved tree rules (depth 10, showing max_depth=5)")

    # Comparison table
    comparison = []
    for depth_name, data in tree_results.items():
        comparison.append({
            'depth': depth_name,
            'train_accuracy': data['train_accuracy'],
            'val_accuracy': data['val_accuracy'],
            'top3_agreement': data.get('top3_agreement'),
            'n_leaves': data['n_leaves'],
            'actual_depth': data['depth'],
            'n_features_used': data['n_features_used'],
        })
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(results_dir / 'analysis' / 'tree_comparison.csv', index=False)
    print(f"\n[OK] Saved: tree_comparison.csv")

    # Plots
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_fidelity_vs_complexity(
        tree_results, metric='val_accuracy',
        save_path=str(plots_dir / 'fidelity_vs_complexity.png')
    )
    plt.close()

    if best_depth in tree_results:
        plot_tree_feature_importance(
            importance_df, top_k=15,
            save_path=str(plots_dir / 'tree_feature_importance.png')
        )
        plt.close()

    return tree_results


def run_gam_distillation(X_train, y_train, X_val, y_val,
                          feature_names, results_dir, top_k=10, verbose=True):
    """Run GAM distillation."""
    print("\n" + "="*60)
    print("GAM DISTILLATION")
    print("="*60)

    # Train GAM per action
    gam_results = train_gam_per_action(
        X_train, y_train, X_val, y_val,
        top_k=top_k, n_splines=10, verbose=verbose
    )

    # Save GAM models
    models_dir = results_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    save_gam_results(gam_results, models_dir / 'gam_models.pkl')

    # Summary table
    summary = []
    for action, data in gam_results.items():
        if 'gam' in data:
            summary.append({
                'action': action,
                'action_name': ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
                'train_accuracy': data['train_accuracy'],
                'val_accuracy': data['val_accuracy'],
                'train_auc': data.get('train_auc'),
                'val_auc': data.get('val_auc'),
                'n_positive_train': data['n_positive_train'],
                'n_positive_val': data['n_positive_val'],
            })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / 'analysis' / 'gam_summary.csv', index=False)
    print(f"\n[OK] Saved: gam_summary.csv")

    # Plot shape functions for top action
    plots_dir = results_dir / 'plots'
    if summary:
        top_action = summary[0]['action']
        plot_gam_shape_functions(
            gam_results, feature_names, X_train,
            action=top_action, top_features=9,
            save_path=str(plots_dir / f'gam_shape_functions_action_{top_action}.png')
        )
        plt.close()

    return gam_results


def run_evaluation(tree_results, gam_results, X_val, y_val, pi_R_val,
                   results_dir, verbose=True):
    """Evaluate distillation quality."""
    print("\n" + "="*60)
    print("FIDELITY EVALUATION")
    print("="*60)

    metrics_list = []

    # Evaluate best tree (depth=10)
    if '10' in tree_results:
        tree = tree_results['10']['tree']
        y_pred_tree = tree.predict(X_val)

        tree_metrics = compute_fidelity_metrics(pi_R_val, None, y_pred_tree)
        tree_metrics['model'] = 'Decision Tree (depth=10)'
        metrics_list.append(tree_metrics)

        if verbose:
            print(f"\nDecision Tree (depth=10):")
            print(f"  Top-1 Agreement: {tree_metrics['top1_agreement']:.4f}")
            print(f"  Top-3 Agreement: {tree_metrics['top3_agreement']:.4f}")

        # Confusion matrix
        plots_dir = results_dir / 'plots'
        plot_action_confusion_matrix(
            y_val, y_pred_tree, ACTION_NAMES, top_k=15,
            save_path=str(plots_dir / 'confusion_matrix_tree.png')
        )
        plt.close()

    # Evaluate unlimited tree
    if 'unlimited' in tree_results:
        tree = tree_results['unlimited']['tree']
        y_pred_tree = tree.predict(X_val)

        tree_metrics = compute_fidelity_metrics(pi_R_val, None, y_pred_tree)
        tree_metrics['model'] = 'Decision Tree (unlimited)'
        metrics_list.append(tree_metrics)

        if verbose:
            print(f"\nDecision Tree (unlimited):")
            print(f"  Top-1 Agreement: {tree_metrics['top1_agreement']:.4f}")
            print(f"  Top-3 Agreement: {tree_metrics['top3_agreement']:.4f}")

    # Save metrics
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(results_dir / 'analysis' / 'fidelity_metrics.csv', index=False)
        print(f"\n[OK] Saved: fidelity_metrics.csv")

    return metrics_list


def generate_report(tree_results, gam_results, metrics_list, results_dir):
    """Generate JSON report."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'decision_tree': {},
        'gam': {},
        'fidelity_summary': {},
    }

    # Tree summary
    for depth_name, data in tree_results.items():
        report['decision_tree'][depth_name] = {
            'train_accuracy': data['train_accuracy'],
            'val_accuracy': data['val_accuracy'],
            'top3_agreement': data.get('top3_agreement'),
            'n_leaves': data['n_leaves'],
            'depth': data['depth'],
        }

    # GAM summary
    for action, data in gam_results.items():
        if 'gam' in data:
            report['gam'][str(action)] = {
                'action_name': ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
                'train_accuracy': data['train_accuracy'],
                'val_accuracy': data['val_accuracy'],
            }

    # Best models
    if tree_results:
        best_tree_depth = max(tree_results.keys(),
                              key=lambda d: tree_results[d]['val_accuracy'])
        report['best_tree'] = {
            'depth': best_tree_depth,
            'val_accuracy': tree_results[best_tree_depth]['val_accuracy'],
        }

    # Fidelity
    if metrics_list:
        report['fidelity_summary'] = metrics_list

    # Save report
    report_path = results_dir / 'distillation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Saved: {report_path}")

    # Print key findings
    print("\n" + "-"*40)
    print("KEY FINDINGS:")
    if 'best_tree' in report:
        print(f"  Best Tree: depth={report['best_tree']['depth']}, "
              f"accuracy={report['best_tree']['val_accuracy']:.4f}")
    if metrics_list:
        best_fidelity = max(metrics_list, key=lambda x: x['top1_agreement'])
        print(f"  Best Fidelity: {best_fidelity['model']}, "
              f"top1={best_fidelity['top1_agreement']:.4f}")
    print("-"*40)

    return report


def main():
    parser = argparse.ArgumentParser(description='Step 7: Rule Distillation')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'tree', 'gam'],
                        help='Distillation mode')
    parser.add_argument('--data-dir', type=str, default='data/processed/policy_samples',
                        help='Directory containing policy samples')
    parser.add_argument('--data-file', type=str, default='100K_pi_H_v2_policy_samples.npz',
                        help='Policy samples file')
    parser.add_argument('--results-dir', type=str, default='results/distillation',
                        help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Validation set size')
    parser.add_argument('--top-k-gam', type=int, default=10,
                        help='Number of top actions for GAM')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup paths
    data_path = PROJECT_ROOT / args.data_dir / args.data_file
    meta_path = data_path.parent / data_path.name.replace('_policy_samples.npz', '_metadata.json')
    results_dir = PROJECT_ROOT / args.results_dir
    (results_dir / 'models').mkdir(parents=True, exist_ok=True)
    (results_dir / 'analysis').mkdir(parents=True, exist_ok=True)
    (results_dir / 'plots').mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STEP 7: RULE DISTILLATION")
    print("="*60)
    print(f"\nMode: {args.mode}")
    print(f"Data: {data_path}")
    print(f"Results dir: {results_dir}")

    # Load data
    data = load_data(str(data_path), str(meta_path))
    X = data['covariates']
    pi_R = data['pi_R']
    y = np.argmax(pi_R, axis=1)  # Hard labels

    # Get feature names
    if data['feature_names']:
        feature_names = data['feature_names']
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Unique actions: {len(np.unique(y))}")

    # Train/val split
    X_train, X_val, y_train, y_val, pi_R_train, pi_R_val = train_test_split(
        X, y, pi_R, test_size=args.test_size, random_state=args.seed
    )
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    tree_results = {}
    gam_results = {}

    # Run distillation
    if args.mode in ['all', 'tree']:
        tree_results = run_tree_distillation(
            X_train, y_train, X_val, y_val, pi_R_val,
            feature_names, results_dir, verbose=True
        )

    if args.mode in ['all', 'gam']:
        gam_results = run_gam_distillation(
            X_train, y_train, X_val, y_val,
            feature_names, results_dir, top_k=args.top_k_gam, verbose=True
        )

    # Evaluation
    metrics_list = []
    if tree_results:
        metrics_list = run_evaluation(
            tree_results, gam_results, X_val, y_val, pi_R_val,
            results_dir, verbose=True
        )

    # Summary plot
    if tree_results and gam_results:
        plot_distillation_summary(
            tree_results, gam_results,
            save_path=str(results_dir / 'plots' / 'distillation_summary.png')
        )
        plt.close()

    # Report
    generate_report(tree_results, gam_results, metrics_list, results_dir)

    print("\n" + "="*60)
    print("STEP 7 COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
