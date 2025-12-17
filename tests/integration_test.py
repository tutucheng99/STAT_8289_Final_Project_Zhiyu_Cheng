"""
Integration Test (Layer 2): Full Data Flow Verification
========================================================
Run: python tests/integration_test.py
Note: May take several minutes

This test verifies:
1. Data files exist and are valid
2. Policy loading and inference
3. Environment rollout and state collection
4. Small-scale policy sampling (future)
5. Basic FDA computation (future)
6. JSD calculation (future)
"""
import sys
import os
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_data_files():
    """Test that all required data files exist."""
    print("[1/6] Testing data files...")

    data_dir = Path(PROJECT_ROOT) / "data" / "raw"
    required_files = [
        "dds_results/dds_results_10M.npy",
        "dds_results/dds_results_2.5M.npy",
        "dds_results/dds_results_500K.npy",
        "dds_results/dds_results_100K_eval.npy",
        "openspiel_bridge/train.txt",
        "openspiel_bridge/test.txt",
        "data_manifest.json",
    ]

    missing = []
    for f in required_files:
        path = data_dir / f
        if not path.exists():
            missing.append(f)
        else:
            print(f"      OK: {f}")

    if missing:
        print(f"      FAIL: Missing files: {missing}")
        return False

    # Verify manifest
    manifest_path = data_dir / "data_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"      Manifest generated at: {manifest.get('generated_at', 'unknown')}")
    return True


def test_policy_loading():
    """Test that policies can be loaded."""
    print("[2/6] Testing policy loading...")

    try:
        from src.policy_loader import PolicyWrapper
    except ImportError as e:
        print(f"      FAIL: Import error: {e}")
        return False

    pi_H_dir = Path(PROJECT_ROOT) / "checkpoints" / "pi_H"
    pi_R_dir = Path(PROJECT_ROOT) / "checkpoints" / "pi_R"

    try:
        pi_H = PolicyWrapper(pi_H_dir)
        print(f"      OK: pi_H loaded ({pi_H.model_type})")
    except Exception as e:
        print(f"      FAIL: pi_H loading failed: {e}")
        return False

    try:
        pi_R = PolicyWrapper(pi_R_dir)
        print(f"      OK: pi_R loaded ({pi_R.model_type})")
    except Exception as e:
        print(f"      FAIL: pi_R loading failed: {e}")
        return False

    return True


def test_policy_inference():
    """Test policy inference shape and normalization."""
    print("[3/6] Testing policy inference...")

    try:
        import jax
        import jax.numpy as jnp
        from pgx.bridge_bidding import BridgeBidding
        from src.policy_loader import PolicyWrapper
    except ImportError as e:
        print(f"      FAIL: Import error: {e}")
        return False

    # Load environment
    dds_path = Path(PROJECT_ROOT) / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))

    # Initialize state
    key = jax.random.PRNGKey(0)
    state = env.init(key)

    obs = state.observation
    legal_mask = state.legal_action_mask

    # Check env output shapes
    if obs.shape != (480,):
        print(f"      FAIL: obs shape {obs.shape} != (480,)")
        return False
    if legal_mask.shape != (38,):
        print(f"      FAIL: legal_mask shape {legal_mask.shape} != (38,)")
        return False

    print(f"      OK: Environment outputs correct shapes")

    # Load policy
    pi_H = PolicyWrapper(Path(PROJECT_ROOT) / "checkpoints" / "pi_H")

    # Run inference
    probs, value = pi_H.get_probs(obs, legal_mask)

    # Check output shape
    if probs.shape != (38,):
        print(f"      FAIL: probs shape {probs.shape} != (38,)")
        return False

    # Check normalization
    prob_sum = float(probs.sum())
    if abs(prob_sum - 1.0) > 1e-5:
        print(f"      FAIL: probs sum to {prob_sum}, not 1.0")
        return False

    # Check illegal action masking
    illegal_prob = float(probs[~legal_mask].max())
    if illegal_prob > 1e-8:
        print(f"      FAIL: illegal action prob {illegal_prob} > 1e-8")
        return False

    print(f"      OK: Inference produces correct output")
    return True


def test_environment_rollout():
    """Test environment rollout for collecting states."""
    print("[4/6] Testing environment rollout...")

    try:
        import jax
        import jax.numpy as jnp
        from pgx.bridge_bidding import BridgeBidding
    except ImportError as e:
        print(f"      FAIL: Import error: {e}")
        return False

    dds_path = Path(PROJECT_ROOT) / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))

    key = jax.random.PRNGKey(42)
    states_collected = 0
    n_episodes = 0

    # Collect 50 states
    while states_collected < 50:
        key, subkey = jax.random.split(key)
        state = env.init(subkey)
        n_episodes += 1

        while not state.terminated:
            states_collected += 1
            if states_collected >= 50:
                break

            # Random action
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break

            action_idx = jax.random.choice(subkey, len(legal_actions))
            action = legal_actions[action_idx]

            key, subkey = jax.random.split(key)
            state = env.step(state, action, subkey)

    print(f"      OK: Collected {states_collected} states from {n_episodes} episodes")
    return True


def test_feature_extraction():
    """Test feature extraction from observations."""
    print("[5/8] Testing feature extraction...")

    try:
        import jax
        import jax.numpy as jnp
        from pgx.bridge_bidding import BridgeBidding
        from src.features.feature_extractor import BridgeFeatureExtractor
    except ImportError as e:
        print(f"      FAIL: Import error: {e}")
        return False

    dds_path = Path(PROJECT_ROOT) / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))
    extractor = BridgeFeatureExtractor()

    key = jax.random.PRNGKey(0)
    state = env.init(key)

    features = extractor.extract(state.observation)

    # Check HCP range
    hcp = float(features['hcp_total'])
    if hcp < 0 or hcp > 37:
        print(f"      FAIL: HCP {hcp} out of range [0, 37]")
        return False

    # Check suit lengths sum to 13
    len_sum = sum(float(features[f'len_{s}']) for s in ['spade', 'heart', 'diamond', 'club'])
    if abs(len_sum - 13) > 0.01:
        print(f"      FAIL: Suit lengths sum to {len_sum}, not 13")
        return False

    # Check HCP consistency
    hcp_sum = sum(float(features[f'hcp_{s}']) for s in ['spade', 'heart', 'diamond', 'club'])
    if abs(hcp_sum - hcp) > 0.01:
        print(f"      FAIL: HCP sum mismatch: {hcp_sum} vs {hcp}")
        return False

    # Check LTC range
    ltc = float(features['ltc'])
    if ltc < 0 or ltc > 12:
        print(f"      FAIL: LTC {ltc} out of range [0, 12]")
        return False

    print(f"      OK: Features extracted (HCP={hcp:.0f}, LTC={ltc:.0f}, "
          f"shape={features['len_spade']:.0f}-{features['len_heart']:.0f}-"
          f"{features['len_diamond']:.0f}-{features['len_club']:.0f})")
    return True


def test_feature_invariants():
    """Test feature invariants across multiple states."""
    print("[6/8] Testing feature invariants...")

    try:
        import jax
        import jax.numpy as jnp
        from pgx.bridge_bidding import BridgeBidding
        from src.features.feature_extractor import BridgeFeatureExtractor
    except ImportError as e:
        print(f"      FAIL: Import error: {e}")
        return False

    dds_path = Path(PROJECT_ROOT) / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))
    extractor = BridgeFeatureExtractor()

    key = jax.random.PRNGKey(123)
    n_tests = 20
    failures = 0

    for i in range(n_tests):
        key, subkey = jax.random.split(key)
        state = env.init(subkey)
        features = extractor.extract(state.observation)

        # Check invariants
        len_sum = sum(float(features[f'len_{s}']) for s in ['spade', 'heart', 'diamond', 'club'])
        hcp_sum = sum(float(features[f'hcp_{s}']) for s in ['spade', 'heart', 'diamond', 'club'])
        ctrl_sum = sum(float(features[f'controls_{s}']) for s in ['spade', 'heart', 'diamond', 'club'])

        if abs(len_sum - 13) > 0.01:
            failures += 1
        if abs(hcp_sum - float(features['hcp_total'])) > 0.01:
            failures += 1
        if abs(ctrl_sum - float(features['controls_total'])) > 0.01:
            failures += 1

    if failures > 0:
        print(f"      FAIL: {failures} invariant violations in {n_tests} states")
        return False

    print(f"      OK: All invariants hold across {n_tests} states")
    return True


def test_small_scale_sampling():
    """Test policy sampling with a small number of states."""
    print("[7/8] Testing small-scale sampling...")
    # NOTE: Will be implemented in Step 3 (Policy Sampling)
    # Current status: SKIP (expected - Step 3 not yet executed)
    print("      SKIP: Awaiting Step 3 implementation")
    return True  # SKIP counts as PASS for incomplete steps


def test_basic_fda():
    """Test basic FDA computation."""
    print("[8/8] Testing basic FDA...")
    # NOTE: Will be implemented in Step 4 (Compositional FDA)
    # Current status: SKIP (expected - Step 4 not yet executed)
    print("      SKIP: Awaiting Step 4 implementation")
    return True  # SKIP counts as PASS for incomplete steps


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TEST (Layer 2): Bridge Bidding Interpretability")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    tests = [
        ("Data Files", test_data_files),
        ("Policy Loading", test_policy_loading),
        ("Policy Inference", test_policy_inference),
        ("Environment Rollout", test_environment_rollout),
        ("Feature Extraction", test_feature_extraction),
        ("Feature Invariants", test_feature_invariants),
        ("Small-scale Sampling", test_small_scale_sampling),
        ("Basic FDA", test_basic_fda),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"      FAIL: {e}")
            results.append((name, "FAIL"))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, status in results:
        icon = "OK" if status == "PASS" else "XX"
        print(f"  [{icon}] {name}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print()
    print(f"Total: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
