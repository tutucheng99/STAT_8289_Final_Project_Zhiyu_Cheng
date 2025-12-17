"""
Model Validation Script

This script validates that the loaded models work correctly by:
1. Loading pi_H and pi_R models
2. Collecting legal states from environment rollout (NOT random generation)
3. Validating policy outputs (shape, normalization, illegal action masking)
4. Comparing policy behaviors (top-1 agreement, KL divergence)

Usage:
    python scripts/validate_models.py
    python scripts/validate_models.py --n-states 500 --seed 42
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def collect_legal_states(env, n_states: int = 1000, seed: int = 42) -> list:
    """
    Collect legal states from environment rollout.

    IMPORTANT: Bridge observations are NOT arbitrary 480-bit vectors.
    Random generation produces invalid states. We must sample from
    actual environment trajectories.

    Args:
        env: PGX BridgeBidding environment
        n_states: Number of states to collect
        seed: Random seed for reproducibility

    Returns:
        List of dicts with 'obs' and 'legal_mask' arrays
    """
    key = jax.random.PRNGKey(seed)
    states_data = []
    n_episodes = 0

    print(f"Collecting {n_states} legal states from environment rollout...")

    while len(states_data) < n_states:
        key, subkey = jax.random.split(key)
        state = env.init(subkey)
        n_episodes += 1

        # Rollout until terminal
        while not state.terminated:
            obs = state.observation
            legal_mask = state.legal_action_mask

            # Save state
            states_data.append({
                "obs": np.array(obs),
                "legal_mask": np.array(legal_mask),
            })

            if len(states_data) >= n_states:
                break

            # Random action to continue
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(legal_mask)[0]

            if len(legal_actions) == 0:
                break

            action_idx = jax.random.choice(subkey, len(legal_actions))
            action = legal_actions[action_idx]

            key, subkey = jax.random.split(key)
            state = env.step(state, action, subkey)

    print(f"  Collected {len(states_data)} states from {n_episodes} episodes")
    return states_data[:n_states]


def validate_policy(policy, states_data: list, name: str = "policy") -> dict:
    """
    Validate a single policy on collected states.

    Checks:
    - Output shape is (38,)
    - Probabilities sum to 1
    - Illegal actions have near-zero probability

    Args:
        policy: PolicyWrapper instance
        states_data: List of state dicts
        name: Policy name for logging

    Returns:
        Dictionary of validation results
    """
    print(f"\nValidating {name}...")

    results = {
        "name": name,
        "n_states": len(states_data),
        "shape_ok": True,
        "sum_to_one": True,
        "illegal_prob_mass": [],
        "entropy": [],
        "n_legal_actions": [],
    }

    for i, data in enumerate(states_data):
        obs = jnp.array(data["obs"])
        legal_mask = jnp.array(data["legal_mask"])

        probs, value = policy.get_probs(obs, legal_mask)

        # Check shape
        if probs.shape != (38,):
            results["shape_ok"] = False
            print(f"  WARNING: State {i} has wrong shape: {probs.shape}")

        # Check normalization
        prob_sum = float(probs.sum())
        if abs(prob_sum - 1.0) > 1e-5:
            results["sum_to_one"] = False
            print(f"  WARNING: State {i} probs sum to {prob_sum}")

        # Check illegal action probability
        illegal_prob = float(probs[~legal_mask].sum())
        results["illegal_prob_mass"].append(illegal_prob)

        # Compute entropy
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
        results["entropy"].append(entropy)

        # Count legal actions
        n_legal = int(legal_mask.sum())
        results["n_legal_actions"].append(n_legal)

    # Compute summary statistics
    results["illegal_prob_mean"] = np.mean(results["illegal_prob_mass"])
    results["illegal_prob_max"] = np.max(results["illegal_prob_mass"])
    results["entropy_mean"] = np.mean(results["entropy"])
    results["entropy_std"] = np.std(results["entropy"])
    results["n_legal_mean"] = np.mean(results["n_legal_actions"])

    # Print summary
    print(f"  Shape OK: {results['shape_ok']}")
    print(f"  Sum to 1: {results['sum_to_one']}")
    print(f"  Illegal prob mass (mean): {results['illegal_prob_mean']:.2e}")
    print(f"  Illegal prob mass (max):  {results['illegal_prob_max']:.2e}")
    print(f"  Entropy (mean +/- std): {results['entropy_mean']:.3f} +/- {results['entropy_std']:.3f}")
    print(f"  Legal actions (mean): {results['n_legal_mean']:.1f}")

    return results


def compare_policies(
    pi_H,
    pi_R,
    states_data: list,
    epsilon: float = 1e-5,
) -> dict:
    """
    Compare two policies on the same states.

    Computes:
    - Top-1 agreement rate
    - KL divergence (pi_R || pi_H) with full statistics
    - Jensen-Shannon divergence with full statistics
    - High-divergence examples for analysis

    Args:
        pi_H: Human proxy policy
        pi_R: RL policy
        states_data: List of state dicts
        epsilon: Smoothing epsilon for KL computation (consistent with CoDA FDA)

    Returns:
        Dictionary of comparison metrics
    """
    print("\nComparing pi_H vs pi_R...")
    print(f"  (Using epsilon={epsilon} for additive smoothing, consistent with CoDA)")

    top1_agreements = []
    kl_divergences = []
    jsd_values = []
    state_divergence_info = []  # For high-KL examples

    for i, data in enumerate(states_data):
        obs = jnp.array(data["obs"])
        legal_mask = jnp.array(data["legal_mask"])

        probs_H, _ = pi_H.get_probs(obs, legal_mask)
        probs_R, _ = pi_R.get_probs(obs, legal_mask)

        # Additive smoothing (consistent with FDA analysis)
        n_actions = 38
        probs_H_smooth = (probs_H + epsilon) / (1 + n_actions * epsilon)
        probs_R_smooth = (probs_R + epsilon) / (1 + n_actions * epsilon)

        # Top-1 agreement (only among legal actions)
        top1_H = jnp.argmax(jnp.where(legal_mask, probs_H, -jnp.inf))
        top1_R = jnp.argmax(jnp.where(legal_mask, probs_R, -jnp.inf))
        top1_agreements.append(int(top1_H == top1_R))

        # KL divergence (only on legal actions)
        legal_probs_H = probs_H_smooth[legal_mask]
        legal_probs_R = probs_R_smooth[legal_mask]

        # Re-normalize for KL
        legal_probs_H = legal_probs_H / legal_probs_H.sum()
        legal_probs_R = legal_probs_R / legal_probs_R.sum()

        kl = float(jnp.sum(legal_probs_R * jnp.log(legal_probs_R / legal_probs_H)))
        kl_divergences.append(kl)

        # JSD
        m = (legal_probs_H + legal_probs_R) / 2
        jsd = 0.5 * float(jnp.sum(legal_probs_H * jnp.log(legal_probs_H / m)))
        jsd += 0.5 * float(jnp.sum(legal_probs_R * jnp.log(legal_probs_R / m)))
        jsd_values.append(jsd)

        # Store info for high-KL analysis
        legal_actions = jnp.where(legal_mask)[0]
        state_divergence_info.append({
            "index": i,
            "kl": kl,
            "jsd": jsd,
            "top1_H": int(top1_H),
            "top1_R": int(top1_R),
            "n_legal": int(legal_mask.sum()),
            "top5_H": [(int(a), float(probs_H[a])) for a in np.argsort(probs_H)[-5:][::-1]],
            "top5_R": [(int(a), float(probs_R[a])) for a in np.argsort(probs_R)[-5:][::-1]],
        })

    # Compute full statistics
    kl_arr = np.array(kl_divergences)
    jsd_arr = np.array(jsd_values)

    results = {
        "top1_agreement": np.mean(top1_agreements),
        "epsilon": epsilon,
        # KL statistics
        "kl_mean": np.mean(kl_arr),
        "kl_std": np.std(kl_arr),
        "kl_median": np.median(kl_arr),
        "kl_q90": np.percentile(kl_arr, 90),
        "kl_q95": np.percentile(kl_arr, 95),
        "kl_max": np.max(kl_arr),
        # JSD statistics
        "jsd_mean": np.mean(jsd_arr),
        "jsd_std": np.std(jsd_arr),
        "jsd_median": np.median(jsd_arr),
        "jsd_q90": np.percentile(jsd_arr, 90),
        "jsd_q95": np.percentile(jsd_arr, 95),
        "jsd_max": np.max(jsd_arr),
    }

    # Print summary
    print(f"  Top-1 Agreement: {results['top1_agreement']:.1%}")
    print(f"  KL(pi_R || pi_H):")
    print(f"    mean: {results['kl_mean']:.4f} +/- {results['kl_std']:.4f}")
    print(f"    median: {results['kl_median']:.4f}")
    print(f"    90%/95%/max: {results['kl_q90']:.4f} / {results['kl_q95']:.4f} / {results['kl_max']:.4f}")
    print(f"  JSD:")
    print(f"    mean: {results['jsd_mean']:.4f} +/- {results['jsd_std']:.4f}")
    print(f"    median: {results['jsd_median']:.4f}")
    print(f"    90%/95%/max: {results['jsd_q90']:.4f} / {results['jsd_q95']:.4f} / {results['jsd_max']:.4f}")

    # Show high-KL examples
    sorted_by_kl = sorted(state_divergence_info, key=lambda x: x["kl"], reverse=True)
    print(f"\n  Top 3 High-KL States:")
    for j, info in enumerate(sorted_by_kl[:3]):
        print(f"    [{j+1}] State {info['index']}: KL={info['kl']:.4f}, JSD={info['jsd']:.4f}")
        print(f"        Top-1: pi_H={info['top1_H']}, pi_R={info['top1_R']}")
        print(f"        pi_H top-5: {info['top5_H']}")
        print(f"        pi_R top-5: {info['top5_R']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate bridge bidding models")
    parser.add_argument(
        "--n-states",
        type=int,
        default=1000,
        help="Number of states to collect (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory",
    )
    args = parser.parse_args()

    # Auto-detect project root
    if args.project_root is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
    else:
        project_root = Path(args.project_root)

    # Add project root to path
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("Bridge Bidding Model Validation")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"N states: {args.n_states}")
    print(f"Seed: {args.seed}")

    # Import after path setup
    try:
        from pgx.bridge_bidding import BridgeBidding
    except ImportError as e:
        print(f"\nERROR: Failed to import PGX BridgeBidding: {e}")
        print("Make sure PGX is installed: pip install pgx")
        return 1

    try:
        from src.policy_loader import PolicyWrapper
    except ImportError as e:
        print(f"\nERROR: Failed to import PolicyWrapper: {e}")
        print("Make sure the project is installed: pip install -e .")
        return 1

    # Load environment with DDS data
    dds_path = project_root / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    if not dds_path.exists():
        print(f"\nERROR: DDS file not found: {dds_path}")
        print("Run 'python scripts/prepare_data.py' first")
        return 1

    print(f"\nLoading BridgeBidding environment...")
    print(f"  DDS path: {dds_path}")
    env = BridgeBidding(dds_results_table_path=str(dds_path))

    # Collect legal states
    states_data = collect_legal_states(env, n_states=args.n_states, seed=args.seed)

    # Load policies
    print("\nLoading policies...")
    pi_H_dir = project_root / "checkpoints" / "pi_H"
    pi_R_dir = project_root / "checkpoints" / "pi_R"

    try:
        pi_H = PolicyWrapper(pi_H_dir)
        print(f"  pi_H loaded from {pi_H_dir}")
    except Exception as e:
        print(f"  ERROR loading pi_H: {e}")
        return 1

    try:
        pi_R = PolicyWrapper(pi_R_dir)
        print(f"  pi_R loaded from {pi_R_dir}")
    except Exception as e:
        print(f"  ERROR loading pi_R: {e}")
        return 1

    # Validate each policy
    results_H = validate_policy(pi_H, states_data, name="pi_H (Human Proxy)")
    results_R = validate_policy(pi_R, states_data, name="pi_R (RL)")

    # Compare policies
    comparison = compare_policies(pi_H, pi_R, states_data)

    # Final summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    all_ok = True

    # Check pi_H
    if results_H["shape_ok"] and results_H["sum_to_one"] and results_H["illegal_prob_max"] < 1e-6:
        print("  pi_H: PASS")
    else:
        print("  pi_H: FAIL")
        all_ok = False

    # Check pi_R
    if results_R["shape_ok"] and results_R["sum_to_one"] and results_R["illegal_prob_max"] < 1e-6:
        print("  pi_R: PASS")
    else:
        print("  pi_R: FAIL")
        all_ok = False

    # Policy divergence sanity check
    if comparison["top1_agreement"] < 0.1:
        print("  WARNING: Very low top-1 agreement (<10%), policies may be mismatched")
    elif comparison["top1_agreement"] > 0.99:
        print("  WARNING: Very high top-1 agreement (>99%), policies may be identical")
    else:
        print(f"  Policy divergence: {1 - comparison['top1_agreement']:.1%} (reasonable)")

    print("=" * 60)

    # Save results to logs/validation/
    import json
    from datetime import datetime

    logs_dir = project_root / "logs" / "validation"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = logs_dir / f"validation_{timestamp}.json"

    # Helper to convert numpy types to Python native types for JSON serialization
    def to_native(val):
        if hasattr(val, 'item'):  # numpy scalar
            return val.item()
        return val

    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "n_states": args.n_states,
        "seed": args.seed,
        "pi_H": {
            "pass": bool(results_H["shape_ok"] and results_H["sum_to_one"] and results_H["illegal_prob_max"] < 1e-6),
            "shape_ok": bool(results_H["shape_ok"]),
            "sum_to_one": bool(results_H["sum_to_one"]),
            "illegal_prob_max": to_native(results_H["illegal_prob_max"]),
            "entropy_mean": to_native(results_H["entropy_mean"]),
            "entropy_std": to_native(results_H["entropy_std"]),
        },
        "pi_R": {
            "pass": bool(results_R["shape_ok"] and results_R["sum_to_one"] and results_R["illegal_prob_max"] < 1e-6),
            "shape_ok": bool(results_R["shape_ok"]),
            "sum_to_one": bool(results_R["sum_to_one"]),
            "illegal_prob_max": to_native(results_R["illegal_prob_max"]),
            "entropy_mean": to_native(results_R["entropy_mean"]),
            "entropy_std": to_native(results_R["entropy_std"]),
        },
        "comparison": {
            "top1_agreement": to_native(comparison["top1_agreement"]),
            "kl_mean": to_native(comparison["kl_mean"]),
            "kl_std": to_native(comparison["kl_std"]),
            "kl_median": to_native(comparison["kl_median"]),
            "kl_q90": to_native(comparison["kl_q90"]),
            "kl_q95": to_native(comparison["kl_q95"]),
            "kl_max": to_native(comparison["kl_max"]),
            "jsd_mean": to_native(comparison["jsd_mean"]),
            "jsd_std": to_native(comparison["jsd_std"]),
            "jsd_median": to_native(comparison["jsd_median"]),
            "jsd_q90": to_native(comparison["jsd_q90"]),
            "jsd_q95": to_native(comparison["jsd_q95"]),
            "jsd_max": to_native(comparison["jsd_max"]),
            "epsilon": to_native(comparison["epsilon"]),
        },
        "all_passed": bool(all_ok),
    }

    with open(results_file, "w") as f:
        json.dump(validation_report, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    if all_ok:
        print("All validations PASSED")
        return 0
    else:
        print("Some validations FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
