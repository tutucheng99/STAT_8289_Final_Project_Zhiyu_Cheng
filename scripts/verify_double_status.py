"""
Double/Redouble Status Verification Script

Verifies that our computed double_status matches PGX state._call_x and state._call_xx.

PGX state fields:
- _call_x: Was the last bid doubled? (bool)
- _call_xx: Was the last bid redoubled? (bool)

Our feature:
- double_status: 0=undoubled, 1=doubled, 2=redoubled

Run: python scripts/verify_double_status.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding

from src.features.feature_extractor import BridgeFeatureExtractor


def verify_double_status(n_states=500, seed=42):
    """Verify double_status feature against PGX state fields."""
    print("=" * 60)
    print(f"VERIFYING DOUBLE STATUS WITH {n_states} PGX STATES")
    print("=" * 60)

    dds_path = PROJECT_ROOT / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))
    extractor = BridgeFeatureExtractor(normalize=False)

    key = jax.random.PRNGKey(seed)

    matches = 0
    mismatches = 0
    mismatch_details = []

    # Track statistics
    state_doubled = 0
    state_redoubled = 0
    feature_doubled = 0
    feature_redoubled = 0
    passout_count = 0

    states_checked = 0
    episodes = 0

    while states_checked < n_states:
        key, subkey = jax.random.split(key)
        state = env.init(subkey)
        episodes += 1

        while not state.terminated and states_checked < n_states:
            obs = state.observation
            features = extractor.extract(obs)

            # Get ground truth from state
            is_doubled_state = bool(state._call_x)
            is_redoubled_state = bool(state._call_xx)

            if is_redoubled_state:
                expected_status = 2
                state_redoubled += 1
            elif is_doubled_state:
                expected_status = 1
                state_doubled += 1
            else:
                expected_status = 0

            # Get our computed value
            computed_status = int(features.get('double_status', 0))
            computed_doubled = int(features.get('is_doubled', 0))
            computed_redoubled = int(features.get('is_redoubled', 0))

            if computed_doubled > 0.5:
                feature_doubled += 1
            if computed_redoubled > 0.5:
                feature_redoubled += 1

            is_passout = int(features.get('is_passout', 0)) > 0.5
            if is_passout:
                passout_count += 1

            if computed_status == expected_status:
                matches += 1
            else:
                mismatches += 1
                if len(mismatch_details) < 5:
                    mismatch_details.append({
                        'state_idx': states_checked,
                        'state_call_x': is_doubled_state,
                        'state_call_xx': is_redoubled_state,
                        'expected_status': expected_status,
                        'computed_status': computed_status,
                        'is_passout': is_passout,
                        'has_contract': int(features.get('has_contract', 0)),
                    })

            states_checked += 1

            # Take random action
            key, subkey = jax.random.split(key)
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) == 0:
                break
            action = legal_actions[jax.random.choice(subkey, len(legal_actions))]
            key, subkey = jax.random.split(key)
            state = env.step(state, action, subkey)

    print(f"\nResults:")
    print(f"  States checked: {states_checked}")
    print(f"  Matches: {matches} ({100*matches/states_checked:.1f}%)")
    print(f"  Mismatches: {mismatches} ({100*mismatches/states_checked:.1f}%)")

    print(f"\nState field statistics (ground truth):")
    print(f"  Doubled (state._call_x): {state_doubled} ({100*state_doubled/states_checked:.1f}%)")
    print(f"  Redoubled (state._call_xx): {state_redoubled} ({100*state_redoubled/states_checked:.1f}%)")

    print(f"\nFeature statistics:")
    print(f"  is_doubled=1: {feature_doubled} ({100*feature_doubled/states_checked:.1f}%)")
    print(f"  is_redoubled=1: {feature_redoubled} ({100*feature_redoubled/states_checked:.1f}%)")
    print(f"  is_passout=1: {passout_count} ({100*passout_count/states_checked:.1f}%)")

    if mismatch_details:
        print(f"\nFirst {len(mismatch_details)} mismatch details:")
        for d in mismatch_details:
            print(f"  State {d['state_idx']}:")
            print(f"    state._call_x={d['state_call_x']}, state._call_xx={d['state_call_xx']}")
            print(f"    expected_status={d['expected_status']}, computed_status={d['computed_status']}")
            print(f"    is_passout={d['is_passout']}, has_contract={d['has_contract']}")

    return matches, mismatches


def main():
    matches, mismatches = verify_double_status(n_states=1000)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if mismatches == 0:
        print("double_status feature matches PGX state perfectly!")
    else:
        print(f"WARNING: {mismatches} mismatches found!")
        print("The double_status computation needs to be investigated.")


if __name__ == "__main__":
    main()
