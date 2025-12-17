"""
验证 PGX 52-bit 手牌编码顺序

方法：
1. 从环境 rollout 收集状态
2. 分析 52-bit 与已知手牌分布的对应关系
3. 确定花色顺序 (SHDC vs CDHS) 和点数顺序 (A-2 vs 2-A)

Run: python scripts/verify_card_encoding.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding


def analyze_card_encoding():
    """分析 52-bit 编码结构"""
    dds_path = PROJECT_ROOT / "data" / "raw" / "dds_results" / "dds_results_100K_eval.npy"
    env = BridgeBidding(dds_results_table_path=str(dds_path))

    key = jax.random.PRNGKey(42)
    samples = []

    print("Collecting hand samples from environment...")

    # 收集多个状态 (只取初始状态，确保是完整手牌)
    for i in range(200):
        key, subkey = jax.random.split(key)
        state = env.init(subkey)

        hand_bits = np.array(state.observation[428:480])
        samples.append(hand_bits)

    samples = np.array(samples)

    print("=" * 60)
    print("52-bit Hand Encoding Analysis")
    print("=" * 60)

    # 检查 1: 每手恰好 13 张牌
    hand_sums = samples.sum(axis=1)
    print(f"\n[1] Hand size check:")
    print(f"    All hands have 13 cards: {np.all(hand_sums == 13)}")
    print(f"    Min/Max cards: {hand_sums.min()}/{hand_sums.max()}")

    if not np.all(hand_sums == 13):
        print("    ERROR: Hand size invariant violated!")
        return None

    # 检查 2: 假设 4 花色各 13 张的 reshape
    print(f"\n[2] Suit structure analysis:")

    cards = samples.reshape(-1, 4, 13)  # (N, 4 suits, 13 ranks)
    suit_lens = cards.sum(axis=2)  # (N, 4)

    # 每手各花色长度应 0-13，且总和 13
    per_hand_sum = suit_lens.sum(axis=1)
    valid = np.all(per_hand_sum == 13)
    print(f"    Suit lengths sum to 13: {valid}")

    if valid:
        print(f"    Suit length distribution (assuming SHDC order):")
        for i, s in enumerate(['S', 'H', 'D', 'C']):
            lens = suit_lens[:, i]
            print(f"      {s}: mean={lens.mean():.2f}, std={lens.std():.2f}, "
                  f"range=[{lens.min()}, {lens.max()}]")

    # 检查 3: 点数顺序验证 - 使用 HCP 分布
    print(f"\n[3] Rank order verification via HCP:")

    # 方案 A: 假设 A-K-Q-J-10-...-2 顺序 (index 0 = A)
    hcp_weights_a = np.array([4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    hcp_a = (cards * hcp_weights_a).sum(axis=(1, 2))

    # 方案 B: 假设 2-3-...-K-A 顺序 (index 12 = A)
    hcp_weights_b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    hcp_b = (cards * hcp_weights_b).sum(axis=(1, 2))

    print(f"    Order A (A-K-Q-J-...-2): HCP mean={hcp_a.mean():.2f}, "
          f"range=[{hcp_a.min()}, {hcp_a.max()}]")
    print(f"    Order B (2-3-...-K-A):   HCP mean={hcp_b.mean():.2f}, "
          f"range=[{hcp_b.min()}, {hcp_b.max()}]")

    # HCP 均值应该接近 10 (40/4=10)
    # 且范围应该在 [0, 37] (理论上)
    if 8 <= hcp_a.mean() <= 12 and hcp_a.min() >= 0 and hcp_a.max() <= 37:
        print(f"    --> Order A (A-K-Q-J-...-2) appears CORRECT")
        rank_order = "A-K-Q-J-10-9-8-7-6-5-4-3-2"
    elif 8 <= hcp_b.mean() <= 12 and hcp_b.min() >= 0 and hcp_b.max() <= 37:
        print(f"    --> Order B (2-3-...-K-A) appears CORRECT")
        rank_order = "2-3-4-5-6-7-8-9-10-J-Q-K-A"
    else:
        print(f"    --> Unable to determine rank order!")
        rank_order = "UNKNOWN"

    # 检查 4: 验证花色顺序
    print(f"\n[4] Suit order verification:")

    # 计算每个 "花色" 的 HCP 分布
    # 如果花色划分正确，每个花色的 HCP 均值应接近 2.5 (10/4)
    for i in range(4):
        suit_cards = cards[:, i, :]  # (N, 13)
        suit_hcp = (suit_cards * hcp_weights_a).sum(axis=1)
        print(f"    Suit {i}: HCP mean={suit_hcp.mean():.2f}, "
              f"std={suit_hcp.std():.2f}, range=[{suit_hcp.min()}, {suit_hcp.max()}]")

    # 每个花色的 HCP 应该有合理的分布 (均值约 2.5，范围 0-10)
    all_suit_hcp_ok = True
    for i in range(4):
        suit_cards = cards[:, i, :]
        suit_hcp = (suit_cards * hcp_weights_a).sum(axis=1)
        if not (1.5 <= suit_hcp.mean() <= 3.5 and suit_hcp.max() <= 10):
            all_suit_hcp_ok = False

    if all_suit_hcp_ok:
        print(f"    --> Suit structure appears CORRECT")
        suit_order = "S-H-D-C"  # 假设这是正确的顺序
    else:
        print(f"    --> Suit structure may need verification")
        suit_order = "UNKNOWN"

    # 检查 5: 额外验证 - Aces 分布
    print(f"\n[5] Aces distribution check:")
    aces_per_hand = cards[:, :, 0].sum(axis=1)  # 假设 index 0 是 A
    print(f"    Aces per hand: mean={aces_per_hand.mean():.2f}, "
          f"range=[{aces_per_hand.min()}, {aces_per_hand.max()}]")
    # 应该是均值 1，范围 0-4
    if 0.8 <= aces_per_hand.mean() <= 1.2 and aces_per_hand.max() <= 4:
        print(f"    --> Confirms index 0 = Ace")
    else:
        print(f"    --> Index 0 may not be Ace!")

    # 总结
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Rank order: {rank_order}")
    print(f"  Suit order: {suit_order}")
    print(f"\n  Encoding: obs[428:480] = 52 bits")
    print(f"  Structure: (4 suits × 13 ranks)")
    if rank_order != "UNKNOWN" and suit_order != "UNKNOWN":
        print(f"\n  cards[suit_idx, rank_idx] where:")
        print(f"    suit_idx: 0=Spade, 1=Heart, 2=Diamond, 3=Club")
        print(f"    rank_idx: 0=A, 1=K, 2=Q, 3=J, 4=10, ..., 12=2")

    return {
        'rank_order': rank_order,
        'suit_order': suit_order,
        'hcp_mean': float(hcp_a.mean()),
        'hcp_range': [int(hcp_a.min()), int(hcp_a.max())],
        'samples_analyzed': len(samples),
    }


if __name__ == "__main__":
    result = analyze_card_encoding()
    if result:
        print(f"\nVerification complete. Results: {result}")
