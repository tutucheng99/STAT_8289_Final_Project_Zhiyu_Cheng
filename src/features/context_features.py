"""
Context Feature Extractor

Extracts vulnerability and position features from PGX observation.

Features:
- Vulnerability: we_vulnerable, they_vulnerable, both, none, favorable, unfavorable
- Position: n_passed_before_opening (from obs[4:8])

Note: Full pass sequence within auction cannot be reconstructed from
the observation. Only "passed before opening" (obs[4:8]) is available.

PGX vulnerability encoding (from _observe in bridge_bidding.py):
  obs[0] = NOT we_vulnerable
  obs[1] = we_vulnerable
  obs[2] = NOT they_vulnerable
  obs[3] = they_vulnerable
"""
import jax.numpy as jnp
from typing import Dict

from .constants import (
    OBS_VULNERABILITY_START,
    OBS_VULNERABILITY_END,
    OBS_PASSED_BEFORE_START,
    OBS_PASSED_BEFORE_END,
)


def extract_context_features(obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Extract vulnerability and position features.

    PGX vulnerability encoding:
      obs[0] = NOT we_vulnerable
      obs[1] = we_vulnerable
      obs[2] = NOT they_vulnerable
      obs[3] = they_vulnerable

    obs[4:8] = passed before opening per relative player (bool)

    Args:
        obs: shape (480,) or (batch, 480)

    Returns:
        Dict with context features, uniformly float32
    """
    # Convert to bool for consistent bitwise operations
    vulnerability = obs[..., OBS_VULNERABILITY_START:OBS_VULNERABILITY_END].astype(jnp.bool_)
    passed_before = obs[..., OBS_PASSED_BEFORE_START:OBS_PASSED_BEFORE_END].astype(jnp.bool_)

    # CORRECTED: Use correct indices based on PGX encoding
    # obs[1] = we_vulnerable, obs[3] = they_vulnerable
    we_vul = vulnerability[..., 1]
    they_vul = vulnerability[..., 3]

    # Vulnerability combinations
    both_vul = we_vul & they_vul
    none_vul = (~we_vul) & (~they_vul)
    favorable_vul = (~we_vul) & they_vul  # We NV, they V
    unfavorable_vul = we_vul & (~they_vul)  # We V, they NV

    # Count of players who passed before opening (0-4)
    # This is the only pass count available from the observation
    n_passed_before = jnp.sum(passed_before.astype(jnp.int32), axis=-1)

    return {
        'we_vulnerable': we_vul.astype(jnp.float32),
        'they_vulnerable': they_vul.astype(jnp.float32),
        'both_vulnerable': both_vul.astype(jnp.float32),
        'none_vulnerable': none_vul.astype(jnp.float32),
        'favorable_vul': favorable_vul.astype(jnp.float32),
        'unfavorable_vul': unfavorable_vul.astype(jnp.float32),
        'n_passed_before_opening': n_passed_before.astype(jnp.float32),
    }
