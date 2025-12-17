"""
Hand Encoding Verification Script V2

This script performs definitive verification of the 52-bit hand encoding
by comparing observation hand bits against state._hand using PGX's own
card conversion function.

The key insight:
- PGX internal: card = suit*13 + rank (suit: 0=S,1=H,2=D,3=C; rank: 0=A,1=K,...,12=2)
- OpenSpiel (obs): index = suit + rank*4 (suit: 0=C,1=D,2=H,3=S; rank: 0=2,...,12=A)
- So obs is RANK-MAJOR, not suit-major!

Run: python scripts/verify_hand_encoding_v2.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding


def _convert_card_pgx_to_openspiel(card):
    """
    Replicate PGX's card conversion exactly.

    PGX internal card encoding:
      card = suit * 13 + rank
      suit: 0=S, 1=H, 2=D, 3=C
      rank: 0=A, 1=K, 2=Q, 3=J, 4=10, 5=9, ..., 12=2

    OpenSpiel card encoding:
      index = suit + rank * 4
      suit: 0=C, 1=D, 2=H, 3=S
      rank: 0=2, 1=3, ..., 11=K, 12=A
    """
    OPEN_SPIEL_SUIT_NUM = jnp.array([3, 2, 1, 0], dtype=jnp.int32)
    OPEN_SPIEL_RANK_NUM = jnp.array([12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=jnp.int32)

    pgx_suit = card // 13
    pgx_rank = card % 13

    openspiel_suit = OPEN_SPIEL_SUIT_NUM[pgx_suit]
    openspiel_rank = OPEN_SPIEL_RANK_NUM[pgx_rank]

    return openspiel_suit + openspiel_rank * 4


def verify_hand_bits_match_state(state, obs, player):
    """
    Verify that obs[428:480] hand bits exactly match state._hand for current player.

    Returns: (match, expected_indices, obs_indices)
    """
    # Get player's position using PGX's method
    # position = argmax(state._shuffled_players == player)
    position = int(jnp.argmax(state._shuffled_players == player))

    # Get this player's cards from state._hand
    # state._hand is organized by position (0-3), 13 cards each
    player_cards = state._hand[position * 13 : (position + 1) * 13]

    # Convert each card to OpenSpiel index
    expected_indices = set()
    for card in player_cards:
        card_int = int(card)
        openspiel_idx = int(_convert_card_pgx_to_openspiel(jnp.int32(card_int)))
        expected_indices.add(openspiel_idx)

    # Get indices where obs hand bits are set
    hand_bits = obs[428:480]
    obs_indices = set(int(i) for i in jnp.where(hand_bits)[0])

    match = expected_indices == obs_indices
    return match, sorted(expected_indices), sorted(obs_indices)


def openspiel_idx_to_card_name(idx):
    """Convert OpenSpiel index to human-readable card name."""
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    suit = idx % 4
    rank = idx // 4

    return f"{ranks[rank]}{suits[suit]}"


def analyze_encoding_structure():
    """Analyze what reshape(4, 13) vs reshape(13, 4) produces."""
    print("\n" + "=" * 60)
    print("ENCODING STRUCTURE ANALYSIS")
    print("=" * 60)

    # OpenSpiel encoding: index = suit + rank * 4
    # suit: 0=C, 1=D, 2=H, 3=S
    # rank: 0=2, 1=3, ..., 12=A

    print("\nOpenSpiel 52-bit hand encoding:")
    print("index = suit + rank * 4")
    print("suit: 0=C, 1=D, 2=H, 3=S")
    print("rank: 0=2, 1=3, ..., 11=K, 12=A")

    print("\nBit layout:")
    print("bits 0-3:   2C(0), 2D(1), 2H(2), 2S(3)")
    print("bits 4-7:   3C(4), 3D(5), 3H(6), 3S(7)")
    print("...")
    print("bits 48-51: AC(48), AD(49), AH(50), AS(51)")

    print("\nIf we reshape(4, 13) (WRONG - current code):")
    print("  Row 0 = bits 0-12  = 2C,2D,2H,2S,3C,3D,3H,3S,4C,4D,4H,4S,5C")
    print("  Row 1 = bits 13-25 = 5D,5H,5S,6C,6D,6H,6S,7C,7D,7H,7S,8C,8D")
    print("  ... (mixes suits and ranks!)")

    print("\nIf we reshape(13, 4) (CORRECT):")
    print("  Row 0 = bits 0-3   = 2C, 2D, 2H, 2S  (all 2s)")
    print("  Row 1 = bits 4-7   = 3C, 3D, 3H, 3S  (all 3s)")
    print("  ...")
    print("  Row 12 = bits 48-51 = AC, AD, AH, AS (all As)")
    print("  -> Shape: (13 ranks, 4 suits) where suits are C-D-H-S")

    print("\nTo get (4 suits, 13 ranks) in S-H-D-C order with A-K-...-2:")
    print("  1. reshape(13, 4)             -> (ranks 2..A, suits C-D-H-S)")
    print("  2. transpose to (4, 13)       -> (suits C-D-H-S, ranks 2..A)")
    print("  3. flip axis 0 [::-1]         -> (suits S-H-D-C, ranks 2..A)")
    print("  4. flip axis 1 [::-1]         -> (suits S-H-D-C, ranks A..2)")


def verify_with_pgx_states(n_states=100, seed=42):
    """Verify hand encoding against actual PGX states."""
    print("\n" + "=" * 60)
    print(f"VERIFYING HAND ENCODING WITH {n_states} PGX STATES")
    print("=" * 60)

    dds_path = PROJECT_ROOT / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))

    key = jax.random.PRNGKey(seed)

    matches = 0
    mismatches = 0
    mismatch_details = []

    states_checked = 0
    episodes = 0

    while states_checked < n_states:
        key, subkey = jax.random.split(key)
        state = env.init(subkey)
        episodes += 1

        while not state.terminated and states_checked < n_states:
            obs = state.observation
            player = int(state.current_player)

            match, expected, actual = verify_hand_bits_match_state(state, obs, player)

            if match:
                matches += 1
            else:
                mismatches += 1
                if len(mismatch_details) < 3:
                    mismatch_details.append({
                        'state_idx': states_checked,
                        'player': player,
                        'expected': expected,
                        'actual': actual,
                        'expected_cards': [openspiel_idx_to_card_name(i) for i in expected],
                        'actual_cards': [openspiel_idx_to_card_name(i) for i in actual],
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

    if mismatch_details:
        print(f"\nFirst {len(mismatch_details)} mismatch details:")
        for d in mismatch_details:
            print(f"  State {d['state_idx']}, player {d['player']}:")
            print(f"    Expected: {d['expected_cards']}")
            print(f"    Actual:   {d['actual_cards']}")

    return matches, mismatches


def test_current_vs_corrected_parsing():
    """Compare current parsing vs corrected parsing."""
    print("\n" + "=" * 60)
    print("COMPARING CURRENT VS CORRECTED PARSING")
    print("=" * 60)

    # Create a test hand with known cards
    # Say: AS, KH, QD, JC (one card per suit, different ranks)
    # In OpenSpiel encoding:
    # AS: suit=3, rank=12 -> index = 3 + 12*4 = 51
    # KH: suit=2, rank=11 -> index = 2 + 11*4 = 46
    # QD: suit=1, rank=10 -> index = 1 + 10*4 = 41
    # JC: suit=0, rank=9  -> index = 0 + 9*4  = 36

    test_bits = jnp.zeros(52, dtype=jnp.bool_)
    test_bits = test_bits.at[51].set(True)  # AS
    test_bits = test_bits.at[46].set(True)  # KH
    test_bits = test_bits.at[41].set(True)  # QD
    test_bits = test_bits.at[36].set(True)  # JC

    print("\nTest hand: AS, KH, QD, JC")
    print(f"OpenSpiel indices: 51 (AS), 46 (KH), 41 (QD), 36 (JC)")

    # Current (wrong) parsing: reshape(4, 13)
    current_cards = test_bits.reshape(4, 13)
    print(f"\nCurrent parsing reshape(4, 13):")
    print(f"  Shape: {current_cards.shape}")
    for i in range(4):
        set_bits = jnp.where(current_cards[i])[0].tolist()
        print(f"  Row {i} (intended as suit {['S','H','D','C'][i]}): bits {set_bits}")

    # Corrected parsing: reshape(13, 4) then transform
    rank_major = test_bits.reshape(13, 4)  # (ranks 2..A, suits C-D-H-S)
    suit_major = rank_major.T              # (suits C-D-H-S, ranks 2..A)
    suit_major = suit_major[::-1]          # (suits S-H-D-C, ranks 2..A)
    suit_major = suit_major[:, ::-1]       # (suits S-H-D-C, ranks A..2)

    print(f"\nCorrected parsing reshape(13,4).T[::-1,::-1]:")
    print(f"  Shape: {suit_major.shape}")
    for i in range(4):
        set_bits = jnp.where(suit_major[i])[0].tolist()
        ranks = ['A','K','Q','J','10','9','8','7','6','5','4','3','2']
        cards = [ranks[b] for b in set_bits]
        print(f"  Row {i} (suit {['S','H','D','C'][i]}): positions {set_bits} = {cards}")

    # Verify corrected parsing is correct
    print("\nExpected result with corrected parsing:")
    print("  Row 0 (S): position 0 = A  -> AS")
    print("  Row 1 (H): position 1 = K  -> KH")
    print("  Row 2 (D): position 2 = Q  -> QD")
    print("  Row 3 (C): position 3 = J  -> JC")


def test_parse_hand_from_obs():
    """Test that parse_hand_from_obs produces correct results after the fix."""
    print("\n" + "=" * 60)
    print("TESTING parse_hand_from_obs (FIXED VERSION)")
    print("=" * 60)

    from src.features.hand_features import parse_hand_from_obs

    # Create test observation with AS, KH, QD, JC
    # OpenSpiel indices: AS=51, KH=46, QD=41, JC=36
    test_obs = jnp.zeros(480, dtype=jnp.float32)
    test_obs = test_obs.at[428 + 51].set(1.0)  # AS
    test_obs = test_obs.at[428 + 46].set(1.0)  # KH
    test_obs = test_obs.at[428 + 41].set(1.0)  # QD
    test_obs = test_obs.at[428 + 36].set(1.0)  # JC

    result = parse_hand_from_obs(test_obs)
    cards = result['cards']

    print(f"\nTest hand: AS, KH, QD, JC")
    print(f"Cards shape: {cards.shape}")

    # Expected: cards[0, 0] = AS, cards[1, 1] = KH, cards[2, 2] = QD, cards[3, 3] = JC
    # Suit order: 0=S, 1=H, 2=D, 3=C
    # Rank order: 0=A, 1=K, 2=Q, 3=J, ...

    suit_names = ['S', 'H', 'D', 'C']
    rank_names = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

    print("\nParsed cards:")
    for suit_idx in range(4):
        set_ranks = jnp.where(cards[suit_idx])[0].tolist()
        rank_strs = [rank_names[r] for r in set_ranks]
        print(f"  {suit_names[suit_idx]}: {rank_strs}")

    # Verify expected cards
    expected_set = {
        (0, 0),  # AS: suit=S(0), rank=A(0)
        (1, 1),  # KH: suit=H(1), rank=K(1)
        (2, 2),  # QD: suit=D(2), rank=Q(2)
        (3, 3),  # JC: suit=C(3), rank=J(3)
    }

    actual_set = set()
    for suit_idx in range(4):
        for rank_idx in range(13):
            if cards[suit_idx, rank_idx]:
                actual_set.add((suit_idx, rank_idx))

    if expected_set == actual_set:
        print("\nPASS: parse_hand_from_obs produces correct output!")
        return True
    else:
        print(f"\nFAIL: Expected {expected_set}, got {actual_set}")
        return False


def test_hcp_computation():
    """Test that HCP is computed correctly with the fixed parsing."""
    print("\n" + "=" * 60)
    print("TESTING HCP COMPUTATION")
    print("=" * 60)

    from src.features.hand_features import parse_hand_from_obs, compute_hcp

    # Create a hand: AK of spades, QJ of hearts (4+3+2+1 = 10 HCP)
    # AS=51, KS=47, QH=42, JH=38
    test_obs = jnp.zeros(480, dtype=jnp.float32)
    test_obs = test_obs.at[428 + 51].set(1.0)  # AS
    test_obs = test_obs.at[428 + 47].set(1.0)  # KS
    test_obs = test_obs.at[428 + 42].set(1.0)  # QH
    test_obs = test_obs.at[428 + 38].set(1.0)  # JH

    result = parse_hand_from_obs(test_obs)
    hcp = compute_hcp(result['cards'])

    print(f"\nTest hand: AS, KS, QH, JH")
    print(f"Expected HCP: A(4) + K(3) + Q(2) + J(1) = 10")
    print(f"Computed total HCP: {int(hcp['hcp_total'])}")
    print(f"Computed per-suit HCP:")
    print(f"  Spade: {int(hcp['hcp_spade'])} (expected: 7 = A+K)")
    print(f"  Heart: {int(hcp['hcp_heart'])} (expected: 3 = Q+J)")
    print(f"  Diamond: {int(hcp['hcp_diamond'])} (expected: 0)")
    print(f"  Club: {int(hcp['hcp_club'])} (expected: 0)")

    if (int(hcp['hcp_total']) == 10 and
        int(hcp['hcp_spade']) == 7 and
        int(hcp['hcp_heart']) == 3 and
        int(hcp['hcp_diamond']) == 0 and
        int(hcp['hcp_club']) == 0):
        print("\nPASS: HCP computation is correct!")
        return True
    else:
        print("\nFAIL: HCP computation is incorrect!")
        return False


def main():
    analyze_encoding_structure()
    test_current_vs_corrected_parsing()
    matches, mismatches = verify_with_pgx_states(n_states=200)

    # Test the actual implementation
    parse_ok = test_parse_hand_from_obs()
    hcp_ok = test_hcp_computation()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if mismatches == 0:
        print("Hand bits match state._hand perfectly.")
        print("The encoding verification is CONFIRMED.")
    else:
        print(f"WARNING: {mismatches} mismatches found!")
        print("The encoding needs to be re-investigated.")

    if parse_ok and hcp_ok:
        print("\nAll implementation tests PASSED!")
    else:
        print("\nSome implementation tests FAILED!")


if __name__ == "__main__":
    main()
