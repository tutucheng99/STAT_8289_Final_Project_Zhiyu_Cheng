"""
Feature Extraction Constants

CRITICAL UPDATE (2025-12-15): Hand encoding corrected via verify_hand_encoding_v2.py

PGX Bridge Bidding observation encoding (480 bits):
- obs[0:4]     = Vulnerability [NOT_we, we, NOT_they, they]
- obs[4:8]     = Passed before opening per relative player
- obs[8:428]   = Bidding history (35 contracts × 3 states × 4 players)
- obs[428:480] = Hand cards (52 bits in OpenSpiel RANK-MAJOR format)

Hand card encoding (OpenSpiel format):
- index = suit + rank * 4
- suit: 0=Club, 1=Diamond, 2=Heart, 3=Spade
- rank: 0=2, 1=3, ..., 11=King, 12=Ace
- Bits 0-3 = 2C,2D,2H,2S; Bits 48-51 = AC,AD,AH,AS

After parsing in hand_features.py, cards array is:
- Shape: (4 suits, 13 ranks)
- Suit order: Spade, Heart, Diamond, Club (index 0-3)
- Rank order: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2 (index 0-12)
"""

# Suit order (verified: SHDC)
SUIT_ORDER = ['spade', 'heart', 'diamond', 'club']
SUIT_INDICES = {s: i for i, s in enumerate(SUIT_ORDER)}
SUIT_SYMBOLS = {'spade': '♠', 'heart': '♥', 'diamond': '♦', 'club': '♣'}

# Rank order (verified: A-K-Q-J-10-...-2)
RANK_ORDER = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
RANK_INDICES = {r: i for i, r in enumerate(RANK_ORDER)}

# HCP weights: A=4, K=3, Q=2, J=1, others=0
HCP_WEIGHTS = [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Control weights: A=2, K=1, others=0
CONTROL_WEIGHTS = [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Observation indices
OBS_VULNERABILITY_START = 0
OBS_VULNERABILITY_END = 4
OBS_PASSED_BEFORE_START = 4
OBS_PASSED_BEFORE_END = 8
OBS_BIDDING_START = 8
OBS_BIDDING_END = 428
OBS_HAND_START = 428
OBS_HAND_END = 480

# Bidding constants
N_CONTRACTS = 35  # 1C, 1D, 1H, 1S, 1NT, ..., 7NT
N_PLAYERS = 4
N_BID_STATES = 3  # bid_made, doubled, redoubled

# Contract strain mapping (0=C, 1=D, 2=H, 3=S, 4=NT)
# NOTE: This is DIFFERENT from SUIT_ORDER (S-H-D-C)!
STRAIN_ORDER = ['club', 'diamond', 'heart', 'spade', 'notrump']
STRAIN_SYMBOLS = {'club': '♣', 'diamond': '♦', 'heart': '♥', 'spade': '♠', 'notrump': 'NT'}

# Mapping from contract strain index to hand suit index
# Use this when relating contract_strain to len_{suit} or hcp_{suit}
# strain_idx -> suit_idx: C(0)->3, D(1)->2, H(2)->1, S(3)->0, NT(4)->None
STRAIN_TO_SUIT_IDX = {0: 3, 1: 2, 2: 1, 3: 0, 4: None}  # NT has no suit
SUIT_TO_STRAIN_IDX = {3: 0, 2: 1, 1: 2, 0: 3}  # club->C, diamond->D, heart->H, spade->S


def contract_idx_to_level_strain(idx: int):
    """Convert contract index (0-34) to (level, strain)."""
    if idx < 0 or idx >= 35:
        return None, None
    level = (idx // 5) + 1  # 1-7
    strain = idx % 5  # 0-4 (C, D, H, S, NT)
    return level, strain


def level_strain_to_contract_idx(level: int, strain: int) -> int:
    """Convert (level, strain) to contract index (0-34)."""
    return (level - 1) * 5 + strain
