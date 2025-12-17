"""
Model Architecture Wrapper

This module provides a unified interface for loading and using bridge bidding models.
It wraps the vendored brl code to ensure compatibility with pre-trained weights.

Architecture Options:
- DeepMind: 4×1024 MLP (default, used by pre-trained models)
- FAIR: Residual network with 200 units

Usage:
    from src.models import load_brl_model, get_policy_probs

    # Load model
    forward, params = load_brl_model("checkpoints/pi_H/active.pkl")

    # Get policy probabilities
    probs, value = get_policy_probs(forward, params, obs, legal_mask)
"""

import pickle
from pathlib import Path
from typing import Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp

from src.third_party.brl.models import ActorCritic, make_forward_pass


def load_brl_model(
    path: Union[str, Path],
    model_type: str = "DeepMind",
    activation: str = "relu",
) -> Tuple[hk.Transformed, dict]:
    """
    Load a pre-trained brl model.

    Args:
        path: Path to the pickle file containing model parameters
        model_type: "DeepMind" (4×1024 MLP) or "FAIR" (residual network)
        activation: "relu" or "tanh"

    Returns:
        forward: Haiku transformed forward function
        params: Model parameters dictionary

    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the model type is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if model_type not in ["DeepMind", "FAIR"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'DeepMind' or 'FAIR'")

    # Create forward function with matching architecture
    forward = make_forward_pass(activation, model_type)

    # Load parameters
    with open(path, "rb") as f:
        params = pickle.load(f)

    return forward, params


def get_policy_probs(
    forward: hk.Transformed,
    params: dict,
    obs: jnp.ndarray,
    legal_mask: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get policy probability distribution over actions.

    This function applies the model to get logits, masks illegal actions,
    and converts to probabilities via softmax.

    Args:
        forward: Haiku transformed forward function
        params: Model parameters
        obs: Observation array, shape (480,) or (batch, 480)
        legal_mask: Boolean mask of legal actions, shape (38,) or (batch, 38)
        temperature: Softmax temperature (default 1.0)

    Returns:
        probs: Probability distribution, shape (38,) or (batch, 38)
        value: Value estimate, shape () or (batch,)
    """
    # Handle single observation (add batch dimension)
    single_obs = obs.ndim == 1
    if single_obs:
        obs = obs[None, :]
        legal_mask = legal_mask[None, :]

    # Forward pass
    logits, value = forward.apply(params, obs)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Mask illegal actions with large negative value
    masked_logits = jnp.where(legal_mask, logits, -1e9)

    # Softmax to probabilities
    probs = jax.nn.softmax(masked_logits, axis=-1)

    # Remove batch dimension if single observation
    if single_obs:
        probs = probs[0]
        value = value[0]

    return probs, value


def get_action_logits(
    forward: hk.Transformed,
    params: dict,
    obs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get raw action logits without masking.

    Args:
        forward: Haiku transformed forward function
        params: Model parameters
        obs: Observation array, shape (480,) or (batch, 480)

    Returns:
        logits: Raw logits, shape (38,) or (batch, 38)
        value: Value estimate, shape () or (batch,)
    """
    single_obs = obs.ndim == 1
    if single_obs:
        obs = obs[None, :]

    logits, value = forward.apply(params, obs)

    if single_obs:
        logits = logits[0]
        value = value[0]

    return logits, value


def sample_action(
    forward: hk.Transformed,
    params: dict,
    obs: jnp.ndarray,
    legal_mask: jnp.ndarray,
    key: jax.random.PRNGKey,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample an action from the policy.

    Args:
        forward: Haiku transformed forward function
        params: Model parameters
        obs: Observation array, shape (480,) or (batch, 480)
        legal_mask: Boolean mask of legal actions, shape (38,) or (batch, 38)
        key: JAX random key
        temperature: Softmax temperature

    Returns:
        action: Sampled action index
        probs: Full probability distribution
        value: Value estimate
    """
    probs, value = get_policy_probs(forward, params, obs, legal_mask, temperature)

    # Sample action
    action = jax.random.categorical(key, jnp.log(probs + 1e-10))

    return action, probs, value


def greedy_action(
    forward: hk.Transformed,
    params: dict,
    obs: jnp.ndarray,
    legal_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get the greedy (argmax) action from the policy.

    Args:
        forward: Haiku transformed forward function
        params: Model parameters
        obs: Observation array, shape (480,) or (batch, 480)
        legal_mask: Boolean mask of legal actions, shape (38,) or (batch, 38)

    Returns:
        action: Greedy action index
        probs: Full probability distribution
        value: Value estimate
    """
    probs, value = get_policy_probs(forward, params, obs, legal_mask)

    # Greedy action (among legal actions only)
    masked_probs = jnp.where(legal_mask, probs, -jnp.inf)
    action = jnp.argmax(masked_probs, axis=-1)

    return action, probs, value


# Re-export for convenience
__all__ = [
    "ActorCritic",
    "make_forward_pass",
    "load_brl_model",
    "get_policy_probs",
    "get_action_logits",
    "sample_action",
    "greedy_action",
]
