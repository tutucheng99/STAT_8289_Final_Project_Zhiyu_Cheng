"""
Verify PGX State Structure and Action Encoding

This script checks:
1. What fields are available in PGX BridgeBidding state
2. Whether board_id/hand_id is exposed
3. Action encoding order (Pass=0, Dbl=1, Rdbl=2, ...)

Run: python scripts/verify_pgx_state.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from pgx.bridge_bidding import BridgeBidding

# PGX expected action encoding (from documentation)
EXPECTED_ACTION_NAMES = [
    "Pass",      # 0
    "Dbl",       # 1
    "Rdbl",      # 2
    "1C",        # 3
    "1D",        # 4
    "1H",        # 5
    "1S",        # 6
    "1NT",       # 7
    "2C",        # 8
    "2D",        # 9
    "2H",        # 10
    "2S",        # 11
    "2NT",       # 12
    "3C",        # 13
    "3D",        # 14
    "3H",        # 15
    "3S",        # 16
    "3NT",       # 17
    "4C",        # 18
    "4D",        # 19
    "4H",        # 20
    "4S",        # 21
    "4NT",       # 22
    "5C",        # 23
    "5D",        # 24
    "5H",        # 25
    "5S",        # 26
    "5NT",       # 27
    "6C",        # 28
    "6D",        # 29
    "6H",        # 30
    "6S",        # 31
    "6NT",       # 32
    "7C",        # 33
    "7D",        # 34
    "7H",        # 35
    "7S",        # 36
    "7NT",       # 37
]


def main():
    print("=" * 60)
    print("PGX State Structure Verification")
    print("=" * 60)

    # Load environment
    dds_path = PROJECT_ROOT / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    print(f"\nLoading environment...")
    print(f"  DDS path exists: {dds_path.exists()}")

    env = BridgeBidding(dds_results_table_path=str(dds_path))
    print(f"  Environment loaded: {type(env)}")

    # Initialize a state
    key = jax.random.PRNGKey(42)
    state = env.init(key)

    print("\n" + "=" * 60)
    print("1. State Fields (dir(state))")
    print("=" * 60)

    # Get all attributes
    state_attrs = [attr for attr in dir(state) if not attr.startswith('_')]
    print(f"\nPublic attributes ({len(state_attrs)}):")
    for attr in sorted(state_attrs):
        try:
            val = getattr(state, attr)
            if hasattr(val, 'shape'):
                print(f"  {attr}: shape={val.shape}, dtype={val.dtype}")
            elif callable(val):
                print(f"  {attr}: <method>")
            else:
                print(f"  {attr}: {type(val).__name__} = {val}")
        except Exception as e:
            print(f"  {attr}: <error: {e}>")

    # Check for board_id related fields
    print("\n" + "=" * 60)
    print("2. Board ID / Hand ID Fields")
    print("=" * 60)

    board_id_candidates = ['board_id', 'hand_id', 'deal_id', 'deal_idx',
                           '_deal_id', '_board_id', '_hand_id', '_idx',
                           'table_idx', '_table_idx', 'dds_idx', '_dds_idx']

    found_any = False
    for candidate in board_id_candidates:
        if hasattr(state, candidate):
            val = getattr(state, candidate)
            print(f"  FOUND: {candidate} = {val}")
            found_any = True

    if not found_any:
        print("  No board_id/hand_id field found in state")
        print("\n  Checking internal attributes (starting with _):")
        internal_attrs = [attr for attr in dir(state) if attr.startswith('_') and not attr.startswith('__')]
        for attr in sorted(internal_attrs)[:20]:  # Limit to first 20
            try:
                val = getattr(state, attr)
                if hasattr(val, 'shape'):
                    print(f"    {attr}: shape={val.shape}")
                elif isinstance(val, (int, float, bool, type(None))):
                    print(f"    {attr}: {val}")
            except:
                pass

    # Check for _hand field (deal representation)
    print("\n" + "=" * 60)
    print("3. Hand/Deal Representation")
    print("=" * 60)

    if hasattr(state, '_hand'):
        hand = state._hand
        print(f"  _hand: shape={hand.shape}, dtype={hand.dtype}")
        print(f"  _hand values: {hand}")
    else:
        print("  No _hand field found")

    # Verify action encoding
    print("\n" + "=" * 60)
    print("4. Action Encoding Verification")
    print("=" * 60)

    print(f"\nTotal actions: {env.num_actions}")
    print(f"Expected: 38 (Pass + Dbl + Rdbl + 35 bids)")

    # Check legal_action_mask at initial state
    mask = state.legal_action_mask
    print(f"\nInitial state legal_action_mask: shape={mask.shape}")
    legal_actions = jnp.where(mask)[0]
    print(f"Legal actions at initial state: {legal_actions.tolist()}")

    # At initial state, should be able to: Pass (0), and all bids (3-37)
    # Should NOT be able to: Dbl (1), Rdbl (2)
    print("\nExpected at initial state:")
    print("  - Pass (0): legal")
    print("  - Dbl (1): illegal (no contract to double)")
    print("  - Rdbl (2): illegal (no double to redouble)")
    print("  - All bids (3-37): legal")

    print("\nActual:")
    print(f"  - Pass (0): {'legal' if mask[0] else 'ILLEGAL'}")
    print(f"  - Dbl (1): {'legal' if mask[1] else 'illegal'}")
    print(f"  - Rdbl (2): {'legal' if mask[2] else 'illegal'}")
    print(f"  - 1C (3): {'legal' if mask[3] else 'ILLEGAL'}")
    print(f"  - 7NT (37): {'legal' if mask[37] else 'ILLEGAL'}")

    # Verify by making a bid and checking double legality
    print("\n" + "=" * 60)
    print("5. Action Sequence Verification")
    print("=" * 60)

    # Make a 1C bid (action 3)
    state1 = env.step(state, jnp.array(3))  # 1C
    mask1 = state1.legal_action_mask
    print(f"\nAfter 1C bid by dealer:")
    print(f"  - Pass (0): {'legal' if mask1[0] else 'illegal'}")
    print(f"  - Dbl (1): {'legal' if mask1[1] else 'illegal'} (opponent can double)")
    print(f"  - Rdbl (2): {'legal' if mask1[2] else 'illegal'}")

    # Make opponent double
    state2 = env.step(state1, jnp.array(1))  # Double
    mask2 = state2.legal_action_mask
    print(f"\nAfter opponent doubles:")
    print(f"  - Pass (0): {'legal' if mask2[0] else 'illegal'}")
    print(f"  - Dbl (1): {'legal' if mask2[1] else 'illegal'}")
    print(f"  - Rdbl (2): {'legal' if mask2[2] else 'illegal'} (partner can redouble)")

    # Summary
    print("\n" + "=" * 60)
    print("6. Summary")
    print("=" * 60)

    # Check if action encoding matches expectation
    encoding_ok = (
        mask[0] == True and      # Pass legal at start
        mask[1] == False and     # Dbl illegal at start
        mask[2] == False and     # Rdbl illegal at start
        mask[3] == True and      # 1C legal at start
        mask1[1] == True and     # Dbl legal after opponent bid
        mask2[2] == True         # Rdbl legal after double
    )

    print(f"\nAction encoding matches PGX docs: {'YES' if encoding_ok else 'NO'}")
    print(f"  Pass = action 0: {'CONFIRMED' if encoding_ok else 'NEEDS CHECK'}")
    print(f"  ref_action_idx should be: 0")

    # Board ID availability
    print(f"\nBoard ID available in state: {'YES' if found_any else 'NO'}")
    if not found_any:
        print("  -> No direct board_id field")
        print("  -> Can use _hand (52,) as unique board identifier for rejection sampling")
        print("  -> Or use episode_ids only (simpler fallback)")

    print("\n" + "=" * 60)
    print("Action Names (for metadata)")
    print("=" * 60)
    print(f"\naction_names = {EXPECTED_ACTION_NAMES}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
