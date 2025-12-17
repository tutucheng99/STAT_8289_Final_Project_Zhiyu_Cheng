"""
Hand Feature Extractor

Extracts bridge statistics from 52-bit hand encoding in PGX observation.

Key features:
- HCP (High Card Points): A=4, K=3, Q=2, J=1
- Distribution: suit lengths, shape classification
- Controls: A=2, K=1
- LTC (Losing Trick Count): with corrected short-suit handling
- Quick Tricks: fast winners
- Honor structure: AKQJ counts, combinations

Implementation notes:
- All bitwise operations use bool dtype
- All numeric computations use int32/float32
- Output is uniformly float32 for downstream compatibility

CRITICAL: PGX hand encoding (2025-12-15 fix)
- PGX uses OpenSpiel card index: index = suit + rank * 4
- Suit: 0=C, 1=D, 2=H, 3=S (Club, Diamond, Heart, Spade)
- Rank: 0=2, 1=3, ..., 11=K, 12=A (2 to Ace)
- This is RANK-MAJOR order: bits 0-3 = 2C,2D,2H,2S; bits 48-51 = AC,AD,AH,AS
- To get (4 suits, 13 ranks) in S-H-D-C order with A-K-...-2:
  reshape(13, 4).T[::-1, ::-1]
"""
import jax.numpy as jnp
from typing import Dict

from .constants import (
    SUIT_ORDER,
    HCP_WEIGHTS,
    CONTROL_WEIGHTS,
    OBS_HAND_START,
    OBS_HAND_END,
)


def parse_hand_from_obs(obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Parse hand cards from 480-dim observation.

    Args:
        obs: shape (480,) or (batch, 480), dtype may be bool or float32

    Returns:
        Dict with:
        - 'cards': shape (4, 13) or (batch, 4, 13), dtype bool
          cards[suit, rank] where suit: 0=S,1=H,2=D,3=C; rank: 0=A,1=K,...,12=2
        - 'raw': shape (52,) or (batch, 52), dtype bool

    CRITICAL: PGX uses OpenSpiel RANK-MAJOR encoding:
        index = openspiel_suit + openspiel_rank * 4
        openspiel_suit: 0=C, 1=D, 2=H, 3=S
        openspiel_rank: 0=2, 1=3, ..., 12=A

    To convert to our (4 suits S-H-D-C, 13 ranks A-K-...-2) format:
        1. reshape(13, 4)     -> (ranks 2..A, suits C-D-H-S)
        2. transpose          -> (suits C-D-H-S, ranks 2..A)
        3. flip axis 0 [::-1] -> (suits S-H-D-C, ranks 2..A)
        4. flip axis 1 [::-1] -> (suits S-H-D-C, ranks A..2)
    """
    hand_bits = obs[..., OBS_HAND_START:OBS_HAND_END]

    # Convert to bool for consistent bitwise operations
    hand_bits = hand_bits.astype(jnp.bool_)

    # CORRECTED: Handle rank-major OpenSpiel encoding
    # Step 1: Reshape to (13 ranks, 4 suits) in OpenSpiel order
    rank_major = hand_bits.reshape(*hand_bits.shape[:-1], 13, 4)

    # Step 2-4: Transform to (4 suits S-H-D-C, 13 ranks A-K-...-2)
    # For batch case, we need to handle the axes carefully
    if hand_bits.ndim == 1:  # Single observation
        # (13, 4) -> transpose -> (4, 13) -> flip both axes
        cards = rank_major.T[::-1, ::-1]
    else:  # Batch case: (batch, 13, 4)
        # Transpose last two dims: (batch, 13, 4) -> (batch, 4, 13)
        cards = jnp.swapaxes(rank_major, -2, -1)
        # Flip suits: S-H-D-C order (reverse axis -2)
        cards = jnp.flip(cards, axis=-2)
        # Flip ranks: A-K-...-2 order (reverse axis -1)
        cards = jnp.flip(cards, axis=-1)

    return {'cards': cards, 'raw': hand_bits}


def compute_hcp(cards: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute High Card Points.

    HCP: A=4, K=3, Q=2, J=1

    Args:
        cards: shape (4, 13) or (batch, 4, 13), dtype bool

    Returns:
        Dict with hcp_total and per-suit HCP
    """
    cards_int = cards.astype(jnp.int32)
    hcp_weights = jnp.array(HCP_WEIGHTS, dtype=jnp.int32)

    # Per-suit HCP: sum over ranks
    hcp_per_suit = jnp.sum(cards_int * hcp_weights, axis=-1)  # (4,) or (batch, 4)
    hcp_total = jnp.sum(hcp_per_suit, axis=-1)  # scalar or (batch,)

    result = {'hcp_total': hcp_total}
    for i, suit in enumerate(SUIT_ORDER):
        result[f'hcp_{suit}'] = hcp_per_suit[..., i]

    return result


def compute_distribution(cards: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute distribution features.

    Features:
    - len_<suit>: length of each suit (0-13)
    - longest_suit: length of longest suit
    - shortest_suit: length of shortest suit
    - is_balanced: 1 if 4333/4432/5332
    - n_singletons: count of singleton suits (0-3)
    - n_voids: count of void suits (0-3)
    """
    cards_int = cards.astype(jnp.int32)
    suit_lengths = jnp.sum(cards_int, axis=-1)  # (4,) or (batch, 4)

    longest = jnp.max(suit_lengths, axis=-1)
    shortest = jnp.min(suit_lengths, axis=-1)

    # Balanced hand: 4333, 4432, 5332
    sorted_lens = jnp.sort(suit_lengths, axis=-1)
    is_4333 = jnp.all(sorted_lens == jnp.array([3, 3, 3, 4]), axis=-1)
    is_4432 = jnp.all(sorted_lens == jnp.array([2, 3, 4, 4]), axis=-1)
    is_5332 = jnp.all(sorted_lens == jnp.array([2, 3, 3, 5]), axis=-1)
    is_balanced = (is_4333 | is_4432 | is_5332).astype(jnp.int32)

    # Singletons and voids
    n_singletons = jnp.sum(suit_lengths == 1, axis=-1)
    n_voids = jnp.sum(suit_lengths == 0, axis=-1)

    result = {
        'longest_suit': longest,
        'shortest_suit': shortest,
        'is_balanced': is_balanced,
        'n_singletons': n_singletons,
        'n_voids': n_voids,
    }
    for i, suit in enumerate(SUIT_ORDER):
        result[f'len_{suit}'] = suit_lengths[..., i]

    return result


def compute_controls(cards: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute controls.

    Controls: A=2, K=1
    """
    cards_int = cards.astype(jnp.int32)
    control_weights = jnp.array(CONTROL_WEIGHTS, dtype=jnp.int32)

    controls_per_suit = jnp.sum(cards_int * control_weights, axis=-1)
    controls_total = jnp.sum(controls_per_suit, axis=-1)

    result = {'controls_total': controls_total}
    for i, suit in enumerate(SUIT_ORDER):
        result[f'controls_{suit}'] = controls_per_suit[..., i]

    return result


def compute_ltc(cards: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Losing Trick Count (corrected version).

    IMPORTANT: Correct handling of short suits:
    - Void (L=0): 0 losers
    - Singleton A: 0 losers
    - Singleton non-A: 1 loser
    - Doubleton AK: 0 losers
    - Doubleton Ax/Kx: 1 loser
    - Doubleton xx: 2 losers
    - 3+ cards: 3 - (A + K + Q count) losers

    Args:
        cards: shape (4, 13) or (batch, 4, 13), dtype bool

    Returns:
        Total LTC across all suits
    """
    cards_int = cards.astype(jnp.int32)
    L = jnp.sum(cards_int, axis=-1)  # suit length, (4,) or (batch, 4)

    has_A = cards_int[..., 0]  # (4,) or (batch, 4)
    has_K = cards_int[..., 1]
    has_Q = cards_int[..., 2]

    # L=0: 0 losers
    losers_0 = jnp.zeros_like(L)

    # L=1: 1 - A (singleton A = 0, singleton non-A = 1)
    losers_1 = jnp.maximum(1 - has_A, 0)

    # L=2: 2 - A - K (AK = 0, Ax/Kx = 1, xx = 2)
    losers_2 = jnp.maximum(2 - has_A - has_K, 0)

    # L>=3: 3 - A - K - Q
    losers_3 = jnp.maximum(3 - has_A - has_K - has_Q, 0)

    suit_ltc = jnp.where(L == 0, losers_0,
                jnp.where(L == 1, losers_1,
                  jnp.where(L == 2, losers_2, losers_3)))

    return jnp.sum(suit_ltc, axis=-1)


def compute_quick_tricks(cards: jnp.ndarray) -> jnp.ndarray:
    """
    Compute quick tricks.

    Rules:
    - AK = 2.0
    - AQ (no K) = 1.5
    - A (no K, no Q) = 1.0
    - KQ (no A) = 1.0
    - Kx (no A, no Q, len >= 2) = 0.5

    Theoretical max: 4 suits × 2 = 8
    """
    cards_bool = cards.astype(jnp.bool_)
    cards_int = cards.astype(jnp.int32)

    has_A = cards_bool[..., 0]
    has_K = cards_bool[..., 1]
    has_Q = cards_bool[..., 2]
    suit_len = jnp.sum(cards_int, axis=-1)

    # AK = 2
    qt_ak = (has_A & has_K).astype(jnp.float32) * 2.0

    # AQ without K = 1.5
    qt_aq = (has_A & (~has_K) & has_Q).astype(jnp.float32) * 1.5

    # A without K, without Q = 1.0
    qt_a = (has_A & (~has_K) & (~has_Q)).astype(jnp.float32) * 1.0

    # KQ without A = 1.0
    qt_kq = ((~has_A) & has_K & has_Q).astype(jnp.float32) * 1.0

    # Kx (K without A, without Q, suit >= 2) = 0.5
    qt_kx = ((~has_A) & has_K & (~has_Q) & (suit_len >= 2)).astype(jnp.float32) * 0.5

    suit_qt = qt_ak + qt_aq + qt_a + qt_kq + qt_kx
    return jnp.sum(suit_qt, axis=-1)


def compute_honor_structure(cards: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute honor structure features.

    Features:
    - n_aces, n_kings, n_queens, n_jacks: count of each honor
    - n_honors_in_long_suits: honors (AKQJ) in suits with 4+ cards
    - has_ak_in_any_suit: 1 if any suit has both A and K
    - n_suits_with_honors: count of suits with at least one AKQJ
    """
    cards_int = cards.astype(jnp.int32)
    cards_bool = cards.astype(jnp.bool_)

    # Count honors
    n_aces = jnp.sum(cards_int[..., 0], axis=-1)
    n_kings = jnp.sum(cards_int[..., 1], axis=-1)
    n_queens = jnp.sum(cards_int[..., 2], axis=-1)
    n_jacks = jnp.sum(cards_int[..., 3], axis=-1)

    # Honors in long suits (length >= 4)
    suit_lengths = jnp.sum(cards_int, axis=-1)
    is_long = (suit_lengths >= 4).astype(jnp.int32)
    honors_per_suit = jnp.sum(cards_int[..., :4], axis=-1)  # AKQJ count per suit
    honors_in_long = jnp.sum(honors_per_suit * is_long, axis=-1)

    # AK combination in any suit
    has_ak = cards_bool[..., 0] & cards_bool[..., 1]
    has_ak_any = jnp.any(has_ak, axis=-1).astype(jnp.int32)

    # Suits with honors
    has_honors = honors_per_suit > 0
    n_suits_with_honors = jnp.sum(has_honors.astype(jnp.int32), axis=-1)

    return {
        'n_aces': n_aces,
        'n_kings': n_kings,
        'n_queens': n_queens,
        'n_jacks': n_jacks,
        'n_honors_in_long_suits': honors_in_long,
        'has_ak_in_any_suit': has_ak_any,
        'n_suits_with_honors': n_suits_with_honors,
    }


def extract_hand_features(obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Extract all hand features from observation.

    Args:
        obs: shape (480,) or (batch, 480)

    Returns:
        Dict with all hand features, uniformly float32
    """
    hand = parse_hand_from_obs(obs)
    cards = hand['cards']

    features = {}
    features.update(compute_hcp(cards))
    features.update(compute_distribution(cards))
    features.update(compute_controls(cards))
    features['ltc'] = compute_ltc(cards)
    features['quick_tricks'] = compute_quick_tricks(cards)
    features.update(compute_honor_structure(cards))

    # Convert all to float32
    features = {k: v.astype(jnp.float32) for k, v in features.items()}

    return features
