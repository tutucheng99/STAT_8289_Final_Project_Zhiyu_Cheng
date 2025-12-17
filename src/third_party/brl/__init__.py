"""
BRL (Bridge Reinforcement Learning) - Vendored Code

Source: https://github.com/harukaki/brl
Paper: Kita et al. (2024) "A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI"

This module contains vendored code from the brl repository for reproducibility.
The original LICENSE is preserved in this directory.
"""

from src.third_party.brl.models import ActorCritic, make_forward_pass

__all__ = ["ActorCritic", "make_forward_pass"]
