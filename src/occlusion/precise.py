"""
Precise Occlusion Analysis with Model Re-inference.

Performs occlusion analysis by:
1. Zeroing/permuting observation features at the raw 480-dim level
2. Re-running neural network inference
3. Measuring policy changes via KL divergence

This is more computationally expensive but gives precise sensitivity estimates.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm

from .feature_groups import OBSERVATION_GROUPS, get_observation_mask
from .metrics import policy_change_metrics, kl_divergence, jsd


def zero_occlusion(observations: np.ndarray,
                   group_name: str) -> np.ndarray:
    """
    Occlude features by zeroing them.

    Args:
        observations: (N, 480) observation array
        group_name: Name of observation group to occlude

    Returns:
        (N, 480) occluded observations
    """
    mask = get_observation_mask(group_name)
    occluded = observations.copy()
    occluded[:, mask] = 0
    return occluded


def permutation_occlusion(observations: np.ndarray,
                          group_name: str,
                          rng: np.random.Generator = None) -> np.ndarray:
    """
    Occlude features by permuting them across samples.

    Args:
        observations: (N, 480) observation array
        group_name: Name of observation group to occlude
        rng: Random generator (default: new generator)

    Returns:
        (N, 480) occluded observations
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = get_observation_mask(group_name)
    occluded = observations.copy()

    # Permute the group features
    indices = np.where(mask)[0]
    perm_idx = rng.permutation(len(observations))
    occluded[:, mask] = observations[perm_idx][:, mask]

    return occluded


def batch_inference(forward_fn,
                    params: Dict,
                    observations: np.ndarray,
                    legal_masks: np.ndarray,
                    batch_size: int = 1024) -> np.ndarray:
    """
    Run batched neural network inference.

    Args:
        forward_fn: Haiku Transformed forward function
        params: Model parameters
        observations: (N, 480) observation array
        legal_masks: (N, 38) legal action masks
        batch_size: Batch size for inference

    Returns:
        (N, 38) policy probabilities
    """
    n_samples = len(observations)
    all_probs = []

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        obs_batch = jnp.array(observations[i:end_idx], dtype=jnp.float32)
        mask_batch = jnp.array(legal_masks[i:end_idx], dtype=jnp.bool_)

        # Forward pass - use .apply() for Haiku Transformed
        logits, _ = forward_fn.apply(params, obs_batch)

        # Apply legal mask and softmax
        masked_logits = jnp.where(mask_batch, logits, -1e9)
        probs = jax.nn.softmax(masked_logits, axis=-1)
        all_probs.append(np.array(probs))

    return np.concatenate(all_probs, axis=0)


def precise_occlusion_single_group(forward_fn: Callable,
                                    params: Dict,
                                    observations: np.ndarray,
                                    legal_masks: np.ndarray,
                                    original_probs: np.ndarray,
                                    group_name: str,
                                    method: str = 'permutation',
                                    n_permutations: int = 5,
                                    batch_size: int = 1024,
                                    seed: int = 42,
                                    verbose: bool = False) -> Dict:
    """
    Compute occlusion sensitivity for a single feature group.

    Args:
        forward_fn: JAX forward function
        params: Model parameters
        observations: (N, 480) observation array
        legal_masks: (N, 38) legal action masks
        original_probs: (N, 38) original policy probabilities
        group_name: Name of observation group
        method: 'zero' or 'permutation'
        n_permutations: Number of permutation iterations (for permutation method)
        batch_size: Batch size for inference
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with sensitivity metrics
    """
    rng = np.random.default_rng(seed)

    if method == 'zero':
        # Single occlusion
        occluded_obs = zero_occlusion(observations, group_name)
        occluded_probs = batch_inference(forward_fn, params, occluded_obs, legal_masks, batch_size)
        metrics = policy_change_metrics(original_probs, occluded_probs, legal_masks)
        metrics['method'] = 'zero'
        return metrics

    elif method == 'permutation':
        # Multiple permutations
        all_metrics = []
        for i in range(n_permutations):
            occluded_obs = permutation_occlusion(observations, group_name, rng)
            occluded_probs = batch_inference(forward_fn, params, occluded_obs, legal_masks, batch_size)
            metrics = policy_change_metrics(original_probs, occluded_probs, legal_masks)
            all_metrics.append(metrics)

        # Aggregate
        aggregated = {
            'kl_mean': np.mean([m['kl_mean'] for m in all_metrics]),
            'kl_std': np.std([m['kl_mean'] for m in all_metrics]),
            'jsd_mean': np.mean([m['jsd_mean'] for m in all_metrics]),
            'jsd_std': np.std([m['jsd_mean'] for m in all_metrics]),
            'tvd_mean': np.mean([m['tvd_mean'] for m in all_metrics]),
            'action_change_rate': np.mean([m['action_change_rate'] for m in all_metrics]),
            'method': 'permutation',
            'n_permutations': n_permutations,
        }
        return aggregated

    else:
        raise ValueError(f"Unknown method: {method}")


def precise_occlusion_analysis(forward_fn: Callable,
                               params: Dict,
                               observations: np.ndarray,
                               legal_masks: np.ndarray,
                               original_probs: np.ndarray = None,
                               groups: List[str] = None,
                               method: str = 'permutation',
                               n_permutations: int = 5,
                               batch_size: int = 1024,
                               seed: int = 42,
                               verbose: bool = True) -> Dict:
    """
    Run precise occlusion analysis for all feature groups.

    Args:
        forward_fn: JAX forward function
        params: Model parameters
        observations: (N, 480) observation array
        legal_masks: (N, 38) legal action masks
        original_probs: (N, 38) original policy probabilities (optional, will compute if None)
        groups: List of group names to analyze (default: all groups)
        method: 'zero' or 'permutation'
        n_permutations: Number of permutation iterations
        batch_size: Batch size for inference
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with:
        - group_results: Dict[group_name, sensitivity_metrics]
        - ranking: List of (group_name, kl_mean) sorted descending
    """
    if groups is None:
        groups = list(OBSERVATION_GROUPS.keys())

    # Compute original probs if not provided
    if original_probs is None:
        if verbose:
            print("Computing original policy probabilities...")
        original_probs = batch_inference(forward_fn, params, observations, legal_masks, batch_size)

    group_results = {}

    iterator = tqdm(groups, desc="Occlusion analysis") if verbose else groups
    for group_name in iterator:
        if verbose:
            tqdm.write(f"  Analyzing {group_name}...")

        result = precise_occlusion_single_group(
            forward_fn, params, observations, legal_masks, original_probs,
            group_name, method, n_permutations, batch_size, seed, verbose
        )
        group_results[group_name] = result

    # Rank by KL divergence
    ranking = sorted(
        [(name, result['kl_mean']) for name, result in group_results.items()],
        key=lambda x: x[1],
        reverse=True
    )

    if verbose:
        print("\nFeature Group Ranking (by KL divergence):")
        for name, score in ranking:
            print(f"  {name}: {score:.6f}")

    return {
        'group_results': group_results,
        'ranking': ranking,
        'method': method,
        'n_samples': len(observations),
    }


def run_precise_analysis_from_data(data_path: str,
                                    pi_H_path: str,
                                    pi_R_path: str = None,
                                    policy: str = 'pi_H',
                                    method: str = 'permutation',
                                    n_permutations: int = 5,
                                    n_samples: int = None,
                                    seed: int = 42,
                                    verbose: bool = True) -> Dict:
    """
    Convenience function to run precise occlusion analysis from data files.

    Args:
        data_path: Path to policy samples NPZ file
        pi_H_path: Path to pi_H checkpoint directory
        pi_R_path: Path to pi_R checkpoint directory (optional)
        policy: Which policy to analyze ('pi_H' or 'pi_R')
        method: 'zero' or 'permutation'
        n_permutations: Number of permutation iterations
        n_samples: Number of samples to use (None = all)
        seed: Random seed
        verbose: Print progress

    Returns:
        Analysis results dict
    """
    from src.policy_loader import PolicyWrapper

    if verbose:
        print(f"Loading data from {data_path}...")

    # Load data
    data = np.load(data_path)
    observations = data['observations']
    legal_masks = data['legal_masks']

    if policy == 'pi_H':
        original_probs = data['pi_H']
    else:
        original_probs = data['pi_R']

    # Subsample if requested
    if n_samples is not None and n_samples < len(observations):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(observations), n_samples, replace=False)
        observations = observations[indices]
        legal_masks = legal_masks[indices]
        original_probs = original_probs[indices]

    if verbose:
        print(f"Using {len(observations)} samples")

    # Load policy model
    if verbose:
        print(f"Loading {policy} model from {pi_H_path if policy == 'pi_H' else pi_R_path}...")

    if policy == 'pi_H':
        policy_wrapper = PolicyWrapper(pi_H_path)
    else:
        policy_wrapper = PolicyWrapper(pi_R_path)

    # Run analysis
    results = precise_occlusion_analysis(
        policy_wrapper.forward,
        policy_wrapper.params,
        observations.astype(np.float32),
        legal_masks,
        original_probs,
        method=method,
        n_permutations=n_permutations,
        seed=seed,
        verbose=verbose
    )

    results['policy'] = policy
    results['data_path'] = data_path

    return results
