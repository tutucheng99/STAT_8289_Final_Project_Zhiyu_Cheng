"""
Policy Loader and Manager

This module provides a high-level interface for loading and managing bridge bidding policies.
It handles model loading, metadata tracking, and provides a unified API for policy inference.

Usage:
    from src.policy_loader import PolicyWrapper

    # Load human proxy policy
    pi_H = PolicyWrapper("checkpoints/pi_H")

    # Get action probabilities
    probs, value = pi_H.get_probs(obs, legal_mask)

    # Get metadata
    print(pi_H.metadata)
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp

from src.models import get_policy_probs, load_brl_model


class PolicyWrapper:
    """
    High-level wrapper for bridge bidding policies.

    This class manages model loading, metadata, and provides a clean interface
    for policy inference.

    Attributes:
        checkpoint_dir: Path to the checkpoint directory
        forward: Haiku transformed forward function
        params: Model parameters
        metadata: Dictionary of model metadata
        model_type: Architecture type ("DeepMind" or "FAIR")
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_type: str = "DeepMind",
        activation: str = "relu",
        model_file: str = "active.pkl",
    ):
        """
        Initialize a policy wrapper.

        Args:
            checkpoint_dir: Directory containing the model checkpoint
            model_type: "DeepMind" (4×1024 MLP) or "FAIR" (residual)
            activation: "relu" or "tanh"
            model_file: Name of the model file (default: "active.pkl")
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.activation = activation

        # Load model
        model_path = self.checkpoint_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Available files: {list(self.checkpoint_dir.glob('*.pkl'))}"
            )

        self.forward, self.params = load_brl_model(
            model_path,
            model_type=model_type,
            activation=activation,
        )

        # Load or create metadata
        meta_path = self.checkpoint_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = self._generate_metadata(model_path)

    def _generate_metadata(self, model_path: Path) -> Dict:
        """Generate metadata for the model."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_file": model_path.name,
            "model_type": self.model_type,
            "activation": self.activation,
            "file_hash_sha256": self.compute_file_hash(model_path),
            "loaded_at": datetime.now().isoformat(),
        }

    @staticmethod
    def compute_file_hash(path: Union[str, Path], algorithm: str = "sha256") -> str:
        """Compute hash of a file."""
        hash_obj = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def get_probs(
        self,
        obs: jnp.ndarray,
        legal_mask: jnp.ndarray,
        temperature: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get policy probability distribution.

        Args:
            obs: Observation array, shape (480,) or (batch, 480)
            legal_mask: Boolean mask of legal actions, shape (38,) or (batch, 38)
            temperature: Softmax temperature (default 1.0)

        Returns:
            probs: Probability distribution, shape (38,) or (batch, 38)
            value: Value estimate, shape () or (batch,)
        """
        return get_policy_probs(
            self.forward, self.params, obs, legal_mask, temperature
        )

    def get_batch_probs(
        self,
        obs_batch: jnp.ndarray,
        legal_mask_batch: jnp.ndarray,
        temperature: float = 1.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get policy probabilities for a batch of observations.

        Args:
            obs_batch: Batch of observations, shape (batch, 480)
            legal_mask_batch: Batch of legal masks, shape (batch, 38)
            temperature: Softmax temperature

        Returns:
            probs_batch: Batch of probability distributions, shape (batch, 38)
            value_batch: Batch of value estimates, shape (batch,)
        """
        return get_policy_probs(
            self.forward, self.params, obs_batch, legal_mask_batch, temperature
        )

    def save_metadata(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save metadata to JSON file.

        Args:
            path: Output path (default: checkpoint_dir/metadata.json)
        """
        if path is None:
            path = self.checkpoint_dir / "metadata.json"
        else:
            path = Path(path)

        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def __repr__(self) -> str:
        return (
            f"PolicyWrapper(\n"
            f"  checkpoint_dir={self.checkpoint_dir},\n"
            f"  model_type={self.model_type},\n"
            f"  activation={self.activation}\n"
            f")"
        )


def create_metadata_file(
    checkpoint_dir: Union[str, Path],
    model_file: str,
    source: str,
    description: str,
    model_type: str = "DeepMind",
    activation: str = "relu",
    extra_info: Optional[Dict] = None,
) -> Dict:
    """
    Create a metadata JSON file for a model checkpoint.

    Args:
        checkpoint_dir: Directory containing the model
        model_file: Name of the model pickle file
        source: Source of the model (e.g., "brl/bridge_models")
        description: Human-readable description
        model_type: Architecture type
        activation: Activation function
        extra_info: Additional metadata to include

    Returns:
        metadata: The generated metadata dictionary
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / model_file

    metadata = {
        "model_file": model_file,
        "source": source,
        "description": description,
        "model_type": model_type,
        "activation": activation,
        "file_hash_sha256": PolicyWrapper.compute_file_hash(model_path),
        "created_at": datetime.now().isoformat(),
    }

    if extra_info:
        metadata.update(extra_info)

    meta_path = checkpoint_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


__all__ = ["PolicyWrapper", "create_metadata_file"]
