"""
Bidding History Feature Extractor

Extracts auction state features from 420-bit bidding history in PGX observation.

Key corrections (2025-12-15 review):
1. reshape order: (35, 3, 4) = [contract, state, player], NOT (35, 4, 3)
2. 12-bit block structure: [bid_p0-p3, dbl_p0-p3, rdbl_p0-p3]
3. is_doubled/is_redoubled: only checks the HIGHEST contract
4. opener detection: correctly handles players who never bid
5. batch support: properly implemented via vmap-compatible indexing

PGX bidding encoding:
- obs[8:428] = 35 contracts × 12 bits = 420 bits
- Each 12-bit block: [bid×4, double×4, redouble×4] for players 0-3
- Contract index: 0=1C, 1=1D, 2=1H, 3=1S, 4=1NT, ..., 34=7NT
"""
import jax
import jax.numpy as jnp
from typing import Dict

from .constants import (
    OBS_BIDDING_START,
    OBS_BIDDING_END,
    N_CONTRACTS,
    N_PLAYERS,
    N_BID_STATES,
)


def parse_bidding_history(obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Parse bidding history from 480-dim observation.

    Args:
        obs: shape (480,) or (batch, 480)

    Returns:
        Dict with:
        - 'bidding': shape (35, 3, 4) or (batch, 35, 3, 4), dtype bool
          bidding[contract_idx, state, player] where state: 0=bid, 1=double, 2=redouble
        - 'raw': shape (420,) or (batch, 420), dtype bool

    NOTE: The 12-bit block per contract is organized as:
        [bid_p0, bid_p1, bid_p2, bid_p3,   <- state 0
         dbl_p0, dbl_p1, dbl_p2, dbl_p3,   <- state 1
         rdbl_p0, rdbl_p1, rdbl_p2, rdbl_p3] <- state 2
    So reshape is (35, 3, 4) = [contract, state, player]
    """
    bidding_bits = obs[..., OBS_BIDDING_START:OBS_BIDDING_END]
    bidding_bits = bidding_bits.astype(jnp.bool_)

    # CORRECTED: Reshape to (35 contracts, 3 states, 4 players)
    # This matches the actual PGX encoding: [bid×4, dbl×4, rdbl×4]
    bidding = bidding_bits.reshape(
        *bidding_bits.shape[:-1], N_CONTRACTS, N_BID_STATES, N_PLAYERS
    )

    return {'bidding': bidding, 'raw': bidding_bits}


def compute_auction_features(bidding: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute auction progress features.

    Args:
        bidding: shape (35, 3, 4) or (batch, 35, 3, 4), dtype bool
                 [contract, state, player] where state: 0=bid, 1=double, 2=redouble

    Returns:
        Dict with auction features
    """
    # Bid indicators: which contracts were bid by any player
    # bidding[..., 0, :] = bid bits for all players
    bids_made = bidding[..., :, 0, :]  # (35, 4) or (batch, 35, 4)
    any_bid_per_contract = jnp.any(bids_made, axis=-1)  # (35,) or (batch, 35)

    n_contracts_bid = jnp.sum(any_bid_per_contract.astype(jnp.int32), axis=-1)

    # Find highest bid contract index
    bid_indices = jnp.arange(N_CONTRACTS)
    highest_bid_idx = jnp.where(
        any_bid_per_contract,
        bid_indices,
        jnp.full_like(bid_indices, -1)
    ).max(axis=-1)

    has_any_bid = highest_bid_idx >= 0

    # Contract level and strain
    # Level = (idx // 5) + 1 (1-7), Strain = idx % 5 (0=C, 1=D, 2=H, 3=S, 4=NT)
    auction_level = jnp.where(has_any_bid, (highest_bid_idx // 5) + 1, 0)
    contract_strain = jnp.where(has_any_bid, highest_bid_idx % 5, -1)

    # CORRECTED: is_doubled/is_redoubled only for the HIGHEST contract
    # Get doubled/redoubled state for the highest contract only
    safe_idx = jnp.maximum(highest_bid_idx, 0)

    # Handle both single and batch cases uniformly
    if bidding.ndim == 3:  # (35, 3, 4) - single observation
        # bidding[safe_idx, 1, :] = double bits for highest contract
        highest_contract_doubles = bidding[safe_idx, 1, :]  # (4,)
        highest_contract_redoubles = bidding[safe_idx, 2, :]  # (4,)
        is_doubled = jnp.any(highest_contract_doubles) & has_any_bid
        is_redoubled = jnp.any(highest_contract_redoubles) & has_any_bid
    else:  # (batch, 35, 3, 4) - batch of observations
        # Use advanced indexing for batch
        batch_size = bidding.shape[0]
        batch_indices = jnp.arange(batch_size)
        # bidding[batch_idx, contract_idx, state, :]
        highest_contract_doubles = bidding[batch_indices, safe_idx, 1, :]  # (batch, 4)
        highest_contract_redoubles = bidding[batch_indices, safe_idx, 2, :]  # (batch, 4)
        is_doubled = jnp.any(highest_contract_doubles, axis=-1) & has_any_bid
        is_redoubled = jnp.any(highest_contract_redoubles, axis=-1) & has_any_bid

    # Competitive auction: both partnerships have bid
    # Players: 0=self, 1=LHO, 2=partner, 3=RHO (clockwise from self)
    # Our side = players 0, 2; Their side = players 1, 3
    our_side_bid = bids_made[..., 0] | bids_made[..., 2]  # (35,) or (batch, 35)
    their_side_bid = bids_made[..., 1] | bids_made[..., 3]  # (35,) or (batch, 35)
    our_any = jnp.any(our_side_bid, axis=-1)
    their_any = jnp.any(their_side_bid, axis=-1)
    is_competitive = (our_any & their_any).astype(jnp.int32)

    # Pass-out: no bids at all
    is_passout = (~has_any_bid).astype(jnp.int32)

    # Double status: 0=undoubled, 1=doubled, 2=redoubled
    # This is cleaner for downstream analysis than separate flags
    double_status = jnp.where(
        is_redoubled, 2,
        jnp.where(is_doubled, 1, 0)
    )

    return {
        'n_contracts_bid': n_contracts_bid.astype(jnp.float32),
        'auction_level': auction_level.astype(jnp.float32),
        'contract_strain': contract_strain.astype(jnp.float32),
        'is_doubled': is_doubled.astype(jnp.float32),
        'is_redoubled': is_redoubled.astype(jnp.float32),
        'double_status': double_status.astype(jnp.float32),  # 0=undoubled, 1=doubled, 2=redoubled
        'is_competitive': is_competitive.astype(jnp.float32),
        'is_passout': is_passout.astype(jnp.float32),
        'has_contract': has_any_bid.astype(jnp.float32),
    }


def compute_opener_features(bidding: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute opener-related features.

    CORRECTED: Properly handles players who never bid.

    Args:
        bidding: shape (35, 3, 4) or (batch, 35, 3, 4), dtype bool
                 [contract, state, player]

    Returns:
        Dict with opener features
    """
    # bids_made: bidding[..., 0, :] = bid bits (state 0)
    bids_made = bidding[..., :, 0, :]  # (35, 4) or (batch, 35, 4)
    any_bid_per_contract = jnp.any(bids_made, axis=-1)  # (35,) or (batch, 35)

    # Find first bid contract index (min index where any player bid)
    bid_indices = jnp.arange(N_CONTRACTS)
    first_bid_idx = jnp.where(
        any_bid_per_contract,
        bid_indices,
        jnp.full_like(bid_indices, N_CONTRACTS)  # 35 means no bid
    ).min(axis=-1)

    has_opening = first_bid_idx < N_CONTRACTS

    # Find opener player (who made the first bid)
    if bidding.ndim == 3:  # Single observation (35, 3, 4)
        # Safe index for accessing bids_made
        safe_first_idx = jnp.minimum(first_bid_idx, N_CONTRACTS - 1)
        first_bidders = bids_made[safe_first_idx, :]  # (4,) bool

        # Find opener: argmax returns 0 for all-False, so we check any() first
        opener_player = jnp.where(
            has_opening & jnp.any(first_bidders),
            jnp.argmax(first_bidders.astype(jnp.int32)),
            -1
        )
    else:  # Batch case (batch, 35, 3, 4)
        batch_size = bidding.shape[0]
        batch_indices = jnp.arange(batch_size)
        safe_first_idx = jnp.minimum(first_bid_idx, N_CONTRACTS - 1)
        first_bidders = bids_made[batch_indices, safe_first_idx, :]  # (batch, 4)

        # Find opener for each batch element
        opener_player = jnp.where(
            has_opening & jnp.any(first_bidders, axis=-1),
            jnp.argmax(first_bidders.astype(jnp.int32), axis=-1),
            -1
        )

    # Determine who opened relative to current player (assumed to be player 0)
    # PGX relative_bidder: 0=self, 1=LHO, 2=partner, 3=RHO (clockwise bidding order)
    self_opened = (opener_player == 0).astype(jnp.int32)
    lho_opened = (opener_player == 1).astype(jnp.int32)
    partner_opened = (opener_player == 2).astype(jnp.int32)
    rho_opened = (opener_player == 3).astype(jnp.int32)

    return {
        'self_opened': self_opened.astype(jnp.float32),
        'partner_opened': partner_opened.astype(jnp.float32),
        'rho_opened': rho_opened.astype(jnp.float32),
        'lho_opened': lho_opened.astype(jnp.float32),
    }


def extract_bidding_features(obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Extract all bidding features from observation.

    Args:
        obs: shape (480,) or (batch, 480)

    Returns:
        Dict with all bidding features, uniformly float32
    """
    parsed = parse_bidding_history(obs)
    bidding = parsed['bidding']

    features = {}
    features.update(compute_auction_features(bidding))
    features.update(compute_opener_features(bidding))

    return features
