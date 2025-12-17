"""
Feature Validation & Statistics Script

This script:
1. Tests structural invariants (catches reshape/indexing errors)
2. Tests feature value invariants (catches computation errors)
3. Computes feature statistics (fills metadata)
4. Identifies edge cases (pass-out, competitive, doubled)

Run: python scripts/validate_features.py
"""
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import json
from pgx.bridge_bidding import BridgeBidding

from src.features.feature_extractor import BridgeFeatureExtractor
from src.features.hand_features import parse_hand_from_obs
from src.features.bidding_features import parse_bidding_history


def test_hand_invariants(obs: jnp.ndarray) -> dict:
    """
    Test hand structural invariants.

    These catch reshape/indexing errors.
    """
    results = {'passed': True, 'errors': []}

    hand = parse_hand_from_obs(obs)
    hand_bits = hand['raw']
    cards = hand['cards']

    # Invariant 1: Hand has exactly 13 cards
    hand_sum = int(jnp.sum(hand_bits))
    if hand_sum != 13:
        results['passed'] = False
        results['errors'].append(f"hand_sum={hand_sum}, expected 13")

    # Invariant 2: Suit lengths sum to 13
    suit_lens = jnp.sum(cards.astype(jnp.int32), axis=-1)
    suit_sum = int(jnp.sum(suit_lens))
    if suit_sum != 13:
        results['passed'] = False
        results['errors'].append(f"suit_sum={suit_sum}, expected 13")

    # Invariant 3: Each suit length in [0, 13]
    for i, length in enumerate(suit_lens):
        length_val = int(length)
        if length_val < 0 or length_val > 13:
            results['passed'] = False
            results['errors'].append(f"suit {i} length={length_val}, expected [0,13]")

    return results


def test_bidding_invariants(obs: jnp.ndarray) -> dict:
    """
    Test bidding structural invariants.

    These catch reshape axis order errors.

    NOTE: bidding shape is (35, 3, 4) = [contract, state, player]
    where state: 0=bid, 1=double, 2=redouble
    """
    results = {'passed': True, 'errors': []}

    parsed = parse_bidding_history(obs)
    bidding = parsed['bidding']  # (35, 3, 4) = [contract, state, player]

    # Invariant 1: Each contract bid by at most one player
    # bidding[:, 0, :] = bid bits for all contracts and players
    bids_made = bidding[:, 0, :].astype(jnp.int32)  # (35, 4)
    bids_per_contract = jnp.sum(bids_made, axis=1)  # (35,)

    for i in range(35):
        count = int(bids_per_contract[i])
        if count > 1:
            results['passed'] = False
            results['errors'].append(f"contract {i}: {count} players bid (max 1)")

    # Invariant 2: Each contract doubled by at most one player
    doubles = bidding[:, 1, :].astype(jnp.int32)  # (35, 4)
    doubles_per_contract = jnp.sum(doubles, axis=1)  # (35,)

    for i in range(35):
        count = int(doubles_per_contract[i])
        if count > 1:
            results['passed'] = False
            results['errors'].append(f"contract {i}: {count} players doubled (max 1)")

    # Invariant 3: Redouble requires prior double
    redoubles = bidding[:, 2, :].astype(jnp.int32)  # (35, 4)
    for i in range(35):
        has_redouble = bool(jnp.any(redoubles[i]))
        has_double = bool(jnp.any(doubles[i]))
        if has_redouble and not has_double:
            results['passed'] = False
            results['errors'].append(f"contract {i}: redouble without double")

    # Invariant 4: Double/Redouble requires prior bid
    # (dbl_any | rdbl_any) => bid_any
    bid_any = jnp.any(bids_made, axis=-1)  # (35,)
    dbl_any = jnp.any(doubles, axis=-1)    # (35,)
    rdbl_any = jnp.any(redoubles, axis=-1) # (35,)

    for i in range(35):
        if (bool(dbl_any[i]) or bool(rdbl_any[i])) and not bool(bid_any[i]):
            results['passed'] = False
            results['errors'].append(f"contract {i}: double/redouble without bid")

    return results


def test_vulnerability_invariants(obs: jnp.ndarray) -> dict:
    """
    Test vulnerability encoding invariants.

    PGX encoding: [NOT_we_vul, we_vul, NOT_they_vul, they_vul]
    So obs[0]+obs[1]=1 and obs[2]+obs[3]=1
    """
    results = {'passed': True, 'errors': []}

    vuln = obs[0:4].astype(jnp.int32)

    # Invariant: we_vul bits are complementary
    if int(vuln[0]) + int(vuln[1]) != 1:
        results['passed'] = False
        results['errors'].append(f"we_vul not complementary: {vuln[0:2].tolist()}")

    # Invariant: they_vul bits are complementary
    if int(vuln[2]) + int(vuln[3]) != 1:
        results['passed'] = False
        results['errors'].append(f"they_vul not complementary: {vuln[2:4].tolist()}")

    return results


def test_opener_invariants(features: dict) -> dict:
    """
    Test opener one-hot invariants.

    If has_contract=1: exactly one opener flag should be 1
    If is_passout=1: all opener flags should be 0
    """
    results = {'passed': True, 'errors': []}

    opener_sum = (float(features.get('self_opened', 0)) +
                  float(features.get('partner_opened', 0)) +
                  float(features.get('rho_opened', 0)) +
                  float(features.get('lho_opened', 0)))

    has_contract = float(features.get('has_contract', 0))
    is_passout = float(features.get('is_passout', 0))

    if has_contract == 1.0 and abs(opener_sum - 1.0) > 0.01:
        results['passed'] = False
        results['errors'].append(f"has_contract=1 but opener_sum={opener_sum}")

    if is_passout == 1.0 and abs(opener_sum) > 0.01:
        results['passed'] = False
        results['errors'].append(f"is_passout=1 but opener_sum={opener_sum}")

    return results


def test_feature_invariants(features: dict) -> dict:
    """
    Test feature value invariants.

    These catch computation errors.
    """
    results = {'passed': True, 'errors': []}

    # HCP total = sum of per-suit HCP
    hcp_sum = sum(float(features.get(f'hcp_{s}', 0)) for s in ['spade', 'heart', 'diamond', 'club'])
    hcp_total = float(features.get('hcp_total', 0))
    if abs(hcp_sum - hcp_total) > 0.01:
        results['passed'] = False
        results['errors'].append(f"hcp_sum={hcp_sum:.2f} != hcp_total={hcp_total:.2f}")

    # Suit lengths sum to 13
    len_sum = sum(float(features.get(f'len_{s}', 0)) for s in ['spade', 'heart', 'diamond', 'club'])
    if abs(len_sum - 13) > 0.01:
        results['passed'] = False
        results['errors'].append(f"len_sum={len_sum:.2f} != 13")

    # Controls total = sum of per-suit controls
    ctrl_sum = sum(float(features.get(f'controls_{s}', 0)) for s in ['spade', 'heart', 'diamond', 'club'])
    ctrl_total = float(features.get('controls_total', 0))
    if abs(ctrl_sum - ctrl_total) > 0.01:
        results['passed'] = False
        results['errors'].append(f"controls_sum={ctrl_sum:.2f} != controls_total={ctrl_total:.2f}")

    # HCP in valid range
    if hcp_total < 0 or hcp_total > 37:
        results['passed'] = False
        results['errors'].append(f"hcp_total={hcp_total:.2f} out of [0,37]")

    # LTC in valid range
    ltc = float(features.get('ltc', 0))
    if ltc < 0 or ltc > 12:
        results['passed'] = False
        results['errors'].append(f"ltc={ltc:.2f} out of [0,12]")

    return results


def compute_feature_statistics(n_states: int = 5000, seed: int = 42):
    """Compute feature statistics and save to JSON."""
    print(f"\nComputing feature statistics on {n_states} states...")
    print("=" * 60)

    dds_path = PROJECT_ROOT / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))
    extractor = BridgeFeatureExtractor(normalize=False)

    key = jax.random.PRNGKey(seed)
    all_features = defaultdict(list)

    # Edge case counters
    edge_cases = {
        'passout': 0,
        'competitive': 0,
        'doubled': 0,
        'redoubled': 0,
    }

    states_collected = 0
    invariant_failures = 0
    episodes = 0

    while states_collected < n_states:
        key, subkey = jax.random.split(key)
        state = env.init(subkey)
        episodes += 1

        while not state.terminated and states_collected < n_states:
            obs = state.observation

            # Test invariants
            hand_inv = test_hand_invariants(obs)
            bid_inv = test_bidding_invariants(obs)
            vuln_inv = test_vulnerability_invariants(obs)

            # Extract features first for opener invariant
            features = extractor.extract(obs)
            opener_inv = test_opener_invariants(features)

            all_inv_passed = (hand_inv['passed'] and bid_inv['passed'] and
                              vuln_inv['passed'] and opener_inv['passed'])

            if not all_inv_passed:
                invariant_failures += 1
                if invariant_failures <= 3:
                    print(f"  Invariant failure #{invariant_failures}:")
                    if hand_inv['errors']:
                        print(f"    Hand: {hand_inv['errors']}")
                    if bid_inv['errors']:
                        print(f"    Bidding: {bid_inv['errors']}")
                    if vuln_inv['errors']:
                        print(f"    Vulnerability: {vuln_inv['errors']}")
                    if opener_inv['errors']:
                        print(f"    Opener: {opener_inv['errors']}")

            # Test feature invariants
            feat_inv = test_feature_invariants(features)
            if not feat_inv['passed'] and invariant_failures <= 3:
                print(f"  Feature invariant failure: {feat_inv['errors']}")

            # Collect statistics
            for name, value in features.items():
                all_features[name].append(float(value))

            # Count edge cases
            if features.get('is_passout', 0) > 0.5:
                edge_cases['passout'] += 1
            if features.get('is_competitive', 0) > 0.5:
                edge_cases['competitive'] += 1
            if features.get('is_doubled', 0) > 0.5:
                edge_cases['doubled'] += 1
            if features.get('is_redoubled', 0) > 0.5:
                edge_cases['redoubled'] += 1

            states_collected += 1

            # Random action to continue
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break
            action = legal_actions[jax.random.choice(subkey, len(legal_actions))]
            key, subkey = jax.random.split(key)
            state = env.step(state, action, subkey)

    # Report
    print(f"\nCollected {states_collected} states from {episodes} episodes")
    print(f"Invariant failures: {invariant_failures}/{states_collected}")

    print(f"\nEdge cases:")
    for case, count in edge_cases.items():
        pct = 100 * count / states_collected if states_collected > 0 else 0
        print(f"  {case}: {count} ({pct:.1f}%)")

    # Compute statistics
    print(f"\nFeature Statistics:")
    print("-" * 70)

    statistics = {}
    for name in sorted(all_features.keys()):
        values = np.array(all_features[name])
        stats = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'p5': float(np.percentile(values, 5)),
            'p95': float(np.percentile(values, 95)),
        }
        statistics[name] = stats

        print(f"{name:30s}: mean={stats['mean']:7.2f}, std={stats['std']:6.2f}, "
              f"[{stats['min']:5.1f}, {stats['max']:5.1f}]")

    # Save statistics
    output_dir = PROJECT_ROOT / "logs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "feature_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2)
    # Use relative path to avoid Unicode issues on Windows
    print(f"\nStatistics saved to: logs/features/feature_statistics.json")

    return statistics, invariant_failures


def main():
    print("=" * 60)
    print("Feature Validation & Statistics")
    print("=" * 60)

    stats, failures = compute_feature_statistics(n_states=1000)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if failures == 0:
        print("  All invariant checks PASSED")
    else:
        print(f"  WARNING: {failures} invariant failures detected")

    print(f"  Features extracted: {len(stats)}")
    print(f"  Statistics saved to: logs/features/feature_statistics.json")


if __name__ == "__main__":
    main()
