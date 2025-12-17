"""
Common Utilities
================
Shared utility functions for the project.
"""
import os
import random
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    random.seed(seed)
    np.random.seed(seed)

    if JAX_AVAILABLE:
        # JAX uses explicit PRNG keys, but we can set numpy seed
        # which affects some operations
        pass

    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path


def load_config(config_path: str, overrides: list = None):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file
        overrides: List of CLI overrides (e.g., ["repro.seed=123"])

    Returns:
        OmegaConf DictConfig object
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


# Bridge-specific constants
BRIDGE_ACTIONS = {
    0: "Pass",
    1: "Dbl",
    2: "RDbl",
    # Bids: 3-37 (1C, 1D, 1H, 1S, 1NT, 2C, ..., 7NT)
}

NUM_ACTIONS = 38

# Suit indices
CLUBS = 0
DIAMONDS = 1
HEARTS = 2
SPADES = 3

# Card values for HCP
HCP_VALUES = {
    'A': 4,
    'K': 3,
    'Q': 2,
    'J': 1,
}

# Control values
CONTROL_VALUES = {
    'A': 2,
    'K': 1,
}
