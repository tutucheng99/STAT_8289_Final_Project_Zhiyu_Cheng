"""
Policy Sampling Script

Collects policy behavior samples from π^H and π^R for FDA analysis.

Usage:
    python scripts/sample_policies.py --n_samples 100000 --seed 42

For full 1M samples on Colab A100, use the notebook instead.

Output:
    data/processed/policy_samples/<run_id>_policy_samples.npz
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

# Check device
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

from pgx.bridge_bidding import BridgeBidding

from src.policy_loader import PolicyWrapper
from src.features.feature_extractor import BridgeFeatureExtractor
from src.sampling.sampler import PolicySampler, SamplingConfig


def main():
    parser = argparse.ArgumentParser(description="Sample policy behavior for FDA analysis")

    # Sampling parameters
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of samples to collect (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--behavior_policy", type=str, default="random",
                        choices=["random", "mixed", "pi_H", "pi_R"],
                        help="Behavior policy for action selection (default: random)")

    # Smoothing
    parser.add_argument("--smoothing_epsilon", type=float, default=1e-5,
                        help="Additive smoothing epsilon (default: 1e-5)")

    # Batch sizes
    parser.add_argument("--collection_batch_size", type=int, default=64,
                        help="Episodes per collection batch (default: 64)")

    # Output
    parser.add_argument("--output_dir", type=str, default="data/processed/policy_samples",
                        help="Output directory (default: data/processed/policy_samples)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run ID for output files (default: timestamp)")

    # Paths
    parser.add_argument("--dds_path", type=str,
                        default="data/raw/dds_results/dds_results_100K_eval.npy",
                        help="Path to DDS results file")
    parser.add_argument("--pi_H_checkpoint", type=str, default="checkpoints/pi_H",
                        help="Path to π^H checkpoint directory")
    parser.add_argument("--pi_R_checkpoint", type=str, default="checkpoints/pi_R",
                        help="Path to π^R checkpoint directory")

    args = parser.parse_args()

    print("=" * 60)
    print("Policy Sampling for FDA Analysis")
    print("=" * 60)

    # Resolve paths
    dds_path = PROJECT_ROOT / args.dds_path
    pi_H_path = PROJECT_ROOT / args.pi_H_checkpoint
    pi_R_path = PROJECT_ROOT / args.pi_R_checkpoint

    # Load environment
    print(f"\nLoading environment...")
    print(f"  DDS path: {args.dds_path}")  # Use relative path to avoid Unicode
    env = BridgeBidding(dds_results_table_path=str(dds_path))

    # Load policies
    print(f"\nLoading policies...")
    print(f"  pi_H: {args.pi_H_checkpoint}")  # Use relative path
    pi_H = PolicyWrapper(
        pi_H_path,
        model_type="DeepMind",
        activation="relu",
        model_file="model-sl.pkl",
    )

    print(f"  pi_R: {args.pi_R_checkpoint}")  # Use relative path
    pi_R = PolicyWrapper(
        pi_R_path,
        model_type="DeepMind",
        activation="relu",
        model_file="model-pretrained-rl-with-fsp.pkl",
    )

    # Create extractor
    extractor = BridgeFeatureExtractor(normalize=False)
    print(f"\nFeature extractor: {len(extractor.get_feature_names())} features")

    # Create config
    config = SamplingConfig(
        n_samples=args.n_samples,
        smoothing_epsilon=args.smoothing_epsilon,
        collection_batch_size=args.collection_batch_size,
        seed=args.seed,
        output_dir=str(PROJECT_ROOT / args.output_dir),
        run_id=args.run_id,
    )

    print(f"\nSampling configuration:")
    print(f"  Target samples: {config.n_samples:,}")
    print(f"  Behavior policy: {args.behavior_policy}")
    print(f"  Smoothing epsilon: {config.smoothing_epsilon}")
    print(f"  Seed: {config.seed}")
    print(f"  Output dir: {args.output_dir}")  # Use relative path
    print(f"  Run ID: {config.run_id}")

    # Create sampler
    sampler = PolicySampler(
        pi_H=pi_H,
        pi_R=pi_R,
        env=env,
        config=config,
        extractor=extractor,
    )

    # Run sampling
    print("\n" + "=" * 60)
    samples = sampler.run_sampling(behavior_policy=args.behavior_policy)

    # Verify samples
    print("\n" + "=" * 60)
    print("Verifying samples...")
    report = sampler.verify_samples(samples)

    print(f"\nVerification report:")
    for check in report['checks']:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        for key, val in check.items():
            if key not in ['name', 'passed']:
                print(f"         {key}: {val}")

    if report['all_passed']:
        print(f"\nAll verification checks passed!")
    else:
        print(f"\nWARNING: Some verification checks failed!")

    # Save samples
    print("\n" + "=" * 60)
    output_path = sampler.save_samples(samples)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    # Print relative path to avoid Unicode issues
    try:
        rel_path = output_path.relative_to(PROJECT_ROOT)
        print(f"\nOutput: {rel_path}")
    except ValueError:
        print(f"\nOutput saved successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
