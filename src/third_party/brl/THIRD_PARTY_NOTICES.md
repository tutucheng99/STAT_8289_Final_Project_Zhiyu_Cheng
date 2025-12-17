# Third Party Notices

This directory contains vendored code from external repositories for reproducibility.

## BRL (Bridge Reinforcement Learning)

**Source Repository**: https://github.com/harukaki/brl

**Paper**: Kita, H., Koyamada, S., Yamaguchi, Y., & Ishii, S. (2024). "A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI". IEEE Conference on Games.

**License**: Apache License 2.0 (see LICENSE file)

### Vendored Files

| File | Description | Original Location |
|------|-------------|-------------------|
| `models.py` | ActorCritic neural network (DeepMind/FAIR architectures) | `brl/src/models.py` |

### Pre-trained Weights

The following pre-trained model weights are used in this project:

| Model File | Source | Description |
|------------|--------|-------------|
| `model-sl.pkl` | `brl/bridge_models/model-sl.pkl` | Supervised learning baseline (-0.56 IMPs/b vs WBridge5) |
| `model-pretrained-rl-with-fsp.pkl` | `brl/bridge_models/model-pretrained-rl-with-fsp.pkl` | SL + PPO + FSP (+1.24 IMPs/b vs WBridge5) |

### Modifications

**None** - The vendored code is used as-is without modification to ensure compatibility with pre-trained weights.

### Version Information

- **Vendored Date**: 2025-12-15
- **brl Recommended PGX Version**: 1.4.0
- **brl Recommended JAX Version**: 0.4.23

### Important Notes

1. The model architecture in `models.py` must remain unchanged to load the pre-trained `.pkl` files correctly.

2. The pre-trained weights were created using `pgx==1.4.0`. Using a different PGX version may affect environment behavior consistency.

3. See the original repository for training details and additional model variants.
