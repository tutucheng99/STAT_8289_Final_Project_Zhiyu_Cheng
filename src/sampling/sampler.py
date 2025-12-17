"""
Policy Sampler for Bridge Bidding

Collects policy behavior samples from π^H and π^R for FDA analysis.

Key features:
- JIT-compiled batch inference for GPU acceleration
- Additive smoothing for numerical stability (ALR transform)
- Without-replacement board sampling using _hand fingerprint
- Episode/board tracking for cluster bootstrap

Output format (NPZ):
- observations: (N, 480) bool - raw PGX observations
- covariates: (N, 48) float32 - extracted features
- pi_H: (N, 38) float32 - smoothed π^H probabilities
- pi_R: (N, 38) float32 - smoothed π^R probabilities
- legal_masks: (N, 38) bool - legal action masks
- episode_ids: (N,) int32 - episode index for cluster bootstrap
- board_ids: (N,) int32 - unique board index
- timestep_in_episode: (N,) int16 - bidding round within episode

Metadata includes:
- action_names: 38 action names in PGX order (Pass, Dbl, Rdbl, 1C, ...)
- ref_action: "Pass" (ALR reference action)
- ref_action_idx: 0 (PGX encoding)
- action_legal_rates: per-action legality rates
- rare_actions: actions with legal_rate < 0.5%
- states_per_episode: {min, median, max, mean}
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.policy_loader import PolicyWrapper
from src.features.feature_extractor import BridgeFeatureExtractor


# PGX action encoding (verified via scripts/verify_pgx_state.py)
# action 0 = Pass, 1 = Dbl, 2 = Rdbl, 3-37 = 1C..7NT
PGX_ACTION_NAMES = [
    "Pass",   # 0
    "Dbl",    # 1
    "Rdbl",   # 2
    "1C", "1D", "1H", "1S", "1NT",      # 3-7
    "2C", "2D", "2H", "2S", "2NT",      # 8-12
    "3C", "3D", "3H", "3S", "3NT",      # 13-17
    "4C", "4D", "4H", "4S", "4NT",      # 18-22
    "5C", "5D", "5H", "5S", "5NT",      # 23-27
    "6C", "6D", "6H", "6S", "6NT",      # 28-32
    "7C", "7D", "7H", "7S", "7NT",      # 33-37
]
REF_ACTION = "Pass"
REF_ACTION_IDX = 0  # Pass is always action 0 in PGX


@dataclass
class SamplingConfig:
    """Configuration for policy sampling."""

    # Sampling targets
    n_samples: int = 1_000_000

    # Smoothing for numerical stability
    smoothing_epsilon: float = 1e-5

    # Batch sizes
    inference_batch_size: int = 2048
    collection_batch_size: int = 64  # Episodes per batch

    # Without-replacement sampling
    without_replacement: bool = True  # Use _hand fingerprint to avoid duplicate boards

    # Random seed
    seed: int = 42

    # Output
    output_dir: str = "data/processed/policy_samples"
    run_id: Optional[str] = None

    # Rare action threshold
    rare_action_threshold: float = 0.005  # 0.5%

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


class PolicySampler:
    """
    Samples policy behavior from π^H and π^R.

    Collects observations, features, and policy probabilities
    for downstream FDA analysis.
    """

    def __init__(
        self,
        pi_H: PolicyWrapper,
        pi_R: PolicyWrapper,
        env,
        config: SamplingConfig,
        extractor: Optional[BridgeFeatureExtractor] = None,
    ):
        """
        Initialize the sampler.

        Args:
            pi_H: Human proxy policy wrapper
            pi_R: RL policy wrapper
            env: PGX BridgeBidding environment
            config: Sampling configuration
            extractor: Feature extractor (optional, created if not provided)
        """
        self.pi_H = pi_H
        self.pi_R = pi_R
        self.env = env
        self.config = config
        self.extractor = extractor or BridgeFeatureExtractor(normalize=False)

        # Build JIT-compiled inference functions
        self._build_jit_functions()

    def _build_jit_functions(self):
        """Build JIT-compiled inference functions for GPU acceleration."""

        # Cache forward functions and params
        forward_H = self.pi_H.forward
        forward_R = self.pi_R.forward
        params_H = self.pi_H.params
        params_R = self.pi_R.params

        @jax.jit
        def batch_inference(obs_batch, mask_batch):
            """JIT-compiled batch inference for both policies."""
            # π^H inference
            logits_H, _ = forward_H.apply(params_H, obs_batch)
            masked_logits_H = jnp.where(mask_batch, logits_H, -1e9)
            probs_H = jax.nn.softmax(masked_logits_H, axis=-1)

            # π^R inference
            logits_R, _ = forward_R.apply(params_R, obs_batch)
            masked_logits_R = jnp.where(mask_batch, logits_R, -1e9)
            probs_R = jax.nn.softmax(masked_logits_R, axis=-1)

            return probs_H, probs_R

        self._batch_inference = batch_inference

    def smooth_probs(self, probs: jnp.ndarray) -> jnp.ndarray:
        """
        Apply additive smoothing to probabilities.

        Ensures numerical stability for ALR transform in FDA.

        Args:
            probs: (N, 38) probability matrix

        Returns:
            Smoothed probabilities that still sum to 1
        """
        eps = self.config.smoothing_epsilon
        K = probs.shape[-1]  # 38 actions
        smoothed = (probs + eps) / (1 + K * eps)
        return smoothed

    def extract_covariates(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Extract covariates from observation.

        Args:
            obs: (480,) observation

        Returns:
            (48,) covariate vector
        """
        features = self.extractor.extract(obs)
        # Convert to ordered array
        arr, names = self.extractor.to_array(features)
        return arr

    def extract_covariates_batch(self, obs_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Extract covariates from batch of observations.

        Args:
            obs_batch: (N, 480) observations

        Returns:
            (N, 48) covariate matrix
        """
        # For now, extract one at a time (feature extraction is CPU-bound anyway)
        # Could be optimized with vmap if needed
        covariates = []
        for i in range(len(obs_batch)):
            cov = self.extract_covariates(obs_batch[i])
            covariates.append(cov)
        return jnp.stack(covariates, axis=0)

    def sample_episodes(
        self,
        n_episodes: int,
        key: jax.random.PRNGKey,
        behavior_policy: str = "mixed",
        seen_boards: Optional[set] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample states from multiple episodes.

        Args:
            n_episodes: Number of episodes to sample
            key: JAX random key
            behavior_policy: How to select actions:
                - "mixed": 50/50 π^H and π^R
                - "random": Random legal action
                - "pi_H": Always use π^H
                - "pi_R": Always use π^R
            seen_boards: Set of board fingerprints already sampled (for without-replacement)

        Returns:
            Dictionary with observations, legal_masks, episode_ids, board_ids, timestep_in_episode
        """
        if seen_boards is None:
            seen_boards = set()

        all_obs = []
        all_masks = []
        all_episode_ids = []
        all_board_ids = []
        all_timesteps = []

        ep_idx = 0
        board_idx = len(seen_boards)  # Start from next available board index
        max_retries = n_episodes * 10  # Avoid infinite loop
        retries = 0

        while ep_idx < n_episodes and retries < max_retries:
            key, init_key = jax.random.split(key)
            state = self.env.init(init_key)

            # Get board fingerprint from _hand
            board_fingerprint = tuple(np.array(state._hand).tolist())

            # Without-replacement: skip if board already seen
            if self.config.without_replacement and board_fingerprint in seen_boards:
                retries += 1
                continue

            # Mark this board as seen
            seen_boards.add(board_fingerprint)
            current_board_id = board_idx
            board_idx += 1

            timestep = 0
            while not state.terminated:
                # Collect state data
                obs = state.observation
                mask = state.legal_action_mask

                all_obs.append(obs)
                all_masks.append(mask)
                all_episode_ids.append(ep_idx)
                all_board_ids.append(current_board_id)
                all_timesteps.append(timestep)

                # Select action based on behavior policy
                key, action_key = jax.random.split(key)
                action = self._select_action(state, action_key, behavior_policy)

                # Step environment
                key, step_key = jax.random.split(key)
                state = self.env.step(state, action, step_key)

                timestep += 1

            ep_idx += 1

        if retries >= max_retries:
            print(f"Warning: Hit max retries ({max_retries}) for without-replacement sampling")

        if len(all_obs) == 0:
            return {
                'observations': jnp.zeros((0, 480), dtype=bool),
                'legal_masks': jnp.zeros((0, 38), dtype=bool),
                'episode_ids': jnp.array([], dtype=jnp.int32),
                'board_ids': jnp.array([], dtype=jnp.int32),
                'timestep_in_episode': jnp.array([], dtype=jnp.int16),
            }

        return {
            'observations': jnp.stack(all_obs, axis=0),
            'legal_masks': jnp.stack(all_masks, axis=0),
            'episode_ids': jnp.array(all_episode_ids, dtype=jnp.int32),
            'board_ids': jnp.array(all_board_ids, dtype=jnp.int32),
            'timestep_in_episode': jnp.array(all_timesteps, dtype=jnp.int16),
        }

    def _select_action(
        self,
        state,
        key: jax.random.PRNGKey,
        behavior_policy: str,
    ) -> int:
        """Select action based on behavior policy."""
        obs = state.observation
        mask = state.legal_action_mask
        legal_actions = jnp.where(mask)[0]

        if len(legal_actions) == 0:
            return 0  # Shouldn't happen

        if behavior_policy == "random":
            idx = jax.random.choice(key, len(legal_actions))
            return int(legal_actions[idx])

        elif behavior_policy == "mixed":
            # 50/50 between π^H and π^R
            key1, key2 = jax.random.split(key)
            use_H = jax.random.uniform(key1) < 0.5
            if use_H:
                probs, _ = self.pi_H.get_probs(obs, mask)
            else:
                probs, _ = self.pi_R.get_probs(obs, mask)
            return int(jax.random.categorical(key2, jnp.log(probs + 1e-10)))

        elif behavior_policy == "pi_H":
            probs, _ = self.pi_H.get_probs(obs, mask)
            return int(jax.random.categorical(key, jnp.log(probs + 1e-10)))

        elif behavior_policy == "pi_R":
            probs, _ = self.pi_R.get_probs(obs, mask)
            return int(jax.random.categorical(key, jnp.log(probs + 1e-10)))

        else:
            raise ValueError(f"Unknown behavior policy: {behavior_policy}")

    def run_sampling(
        self,
        behavior_policy: str = "random",
        progress_callback=None,
    ) -> Dict[str, np.ndarray]:
        """
        Run the full sampling pipeline.

        Args:
            behavior_policy: How to select actions during rollout
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dictionary with all sample data (numpy arrays)
        """
        print(f"Starting policy sampling...")
        print(f"  Target: {self.config.n_samples:,} samples")
        print(f"  Behavior policy: {behavior_policy}")
        print(f"  Smoothing epsilon: {self.config.smoothing_epsilon}")
        print(f"  Without-replacement: {self.config.without_replacement}")

        key = jax.random.PRNGKey(self.config.seed)

        # Estimate episodes needed (~12 states per episode on average)
        states_per_episode = 12
        n_episodes_needed = int(self.config.n_samples / states_per_episode * 1.1)  # 10% buffer

        print(f"  Estimated episodes needed: {n_episodes_needed:,}")

        # Collect states in batches
        all_obs = []
        all_masks = []
        all_covariates = []
        all_pi_H = []
        all_pi_R = []
        all_episode_ids = []
        all_board_ids = []
        all_timesteps = []

        # Track seen boards for without-replacement sampling
        seen_boards = set() if self.config.without_replacement else None

        total_states = 0
        total_episodes = 0
        batch_idx = 0
        start_time = time.time()

        while total_states < self.config.n_samples:
            batch_idx += 1
            key, batch_key = jax.random.split(key)

            # Sample a batch of episodes
            batch_data = self.sample_episodes(
                n_episodes=self.config.collection_batch_size,
                key=batch_key,
                behavior_policy=behavior_policy,
                seen_boards=seen_boards,
            )

            n_batch_states = len(batch_data['observations'])

            if n_batch_states == 0:
                continue

            # Run inference on batch
            obs_batch = batch_data['observations']
            mask_batch = batch_data['legal_masks']

            # Cast observations to float32 for model inference
            obs_f32 = obs_batch.astype(jnp.float32)
            probs_H, probs_R = self._batch_inference(obs_f32, mask_batch)

            # Apply smoothing
            probs_H = self.smooth_probs(probs_H)
            probs_R = self.smooth_probs(probs_R)

            # Extract covariates (this is slower, CPU-bound)
            covariates = self.extract_covariates_batch(obs_f32)

            # Adjust episode_ids to be globally unique (not per-batch)
            batch_episode_ids = np.array(batch_data['episode_ids']) + total_episodes

            # Count unique episodes in this batch
            n_batch_episodes = len(np.unique(batch_data['episode_ids']))
            total_episodes += n_batch_episodes

            # Collect all data
            all_obs.append(np.array(obs_batch))  # Keep as bool
            all_masks.append(np.array(mask_batch))
            all_pi_H.append(np.array(probs_H))
            all_pi_R.append(np.array(probs_R))
            all_covariates.append(np.array(covariates))
            all_episode_ids.append(batch_episode_ids)
            all_board_ids.append(np.array(batch_data['board_ids']))
            all_timesteps.append(np.array(batch_data['timestep_in_episode']))

            total_states += n_batch_states

            # Progress update
            if batch_idx % 10 == 0 or total_states >= self.config.n_samples:
                elapsed = time.time() - start_time
                rate = total_states / elapsed if elapsed > 0 else 0
                remaining = (self.config.n_samples - total_states) / rate if rate > 0 else 0
                n_boards = len(seen_boards) if seen_boards else "N/A"
                print(f"  Batch {batch_idx}: {total_states:,}/{self.config.n_samples:,} "
                      f"({100*total_states/self.config.n_samples:.1f}%) "
                      f"- {rate:.1f} states/s - ETA: {remaining/60:.1f} min "
                      f"- Boards: {n_boards}")

            if progress_callback:
                progress_callback(total_states, self.config.n_samples)

        # Concatenate all batches
        print(f"\nConcatenating {len(all_obs)} batches...")

        result = {
            'observations': np.concatenate(all_obs, axis=0),
            'legal_masks': np.concatenate(all_masks, axis=0),
            'covariates': np.concatenate(all_covariates, axis=0),
            'pi_H': np.concatenate(all_pi_H, axis=0),
            'pi_R': np.concatenate(all_pi_R, axis=0),
            'episode_ids': np.concatenate(all_episode_ids, axis=0),
            'board_ids': np.concatenate(all_board_ids, axis=0),
            'timestep_in_episode': np.concatenate(all_timesteps, axis=0),
        }

        # Trim to exact target
        if len(result['observations']) > self.config.n_samples:
            for k in result:
                result[k] = result[k][:self.config.n_samples]

        total_time = time.time() - start_time
        n_unique_boards = len(set(result['board_ids'].tolist()))
        n_unique_episodes = len(set(result['episode_ids'].tolist()))

        print(f"\nSampling complete!")
        print(f"  Total samples: {len(result['observations']):,}")
        print(f"  Unique episodes: {n_unique_episodes:,}")
        print(f"  Unique boards: {n_unique_boards:,}")
        print(f"  Total time: {total_time/60:.1f} min")
        print(f"  Rate: {len(result['observations'])/total_time:.1f} states/s")

        return result

    def verify_samples(self, samples: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Verify sample quality and invariants.

        Includes E. Hard-core checks (Critical):
        - Pass always legal
        - States per episode distribution
        - isfinite checks
        - Board no duplicates

        Args:
            samples: Sample dictionary from run_sampling()

        Returns:
            Verification report dictionary
        """
        report = {
            'n_samples': len(samples['observations']),
            'checks': [],
        }

        # ===== A. Statistical validity =====
        # Check probability sums
        pi_H_sums = np.sum(samples['pi_H'], axis=1)
        pi_R_sums = np.sum(samples['pi_R'], axis=1)

        pi_H_sum_ok = np.allclose(pi_H_sums, 1.0, atol=1e-5)
        pi_R_sum_ok = np.allclose(pi_R_sums, 1.0, atol=1e-5)

        report['checks'].append({
            'name': 'pi_H probability sum',
            'passed': pi_H_sum_ok,
            'mean': float(np.mean(pi_H_sums)),
            'min': float(np.min(pi_H_sums)),
            'max': float(np.max(pi_H_sums)),
        })

        report['checks'].append({
            'name': 'pi_R probability sum',
            'passed': pi_R_sum_ok,
            'mean': float(np.mean(pi_R_sums)),
            'min': float(np.min(pi_R_sums)),
            'max': float(np.max(pi_R_sums)),
        })

        # ===== B. Numerical invariants =====
        # Check isfinite for all arrays
        pi_H_finite = np.all(np.isfinite(samples['pi_H']))
        pi_R_finite = np.all(np.isfinite(samples['pi_R']))
        cov_finite = np.all(np.isfinite(samples['covariates']))

        report['checks'].append({
            'name': 'pi_H isfinite',
            'passed': pi_H_finite,
        })
        report['checks'].append({
            'name': 'pi_R isfinite',
            'passed': pi_R_finite,
        })
        report['checks'].append({
            'name': 'covariates isfinite',
            'passed': cov_finite,
        })

        # Check illegal action masking
        illegal_mask = ~samples['legal_masks'].astype(bool)
        illegal_probs_H = samples['pi_H'][illegal_mask]
        illegal_probs_R = samples['pi_R'][illegal_mask]

        # After smoothing, illegal probs should be ~epsilon/(1+K*eps)
        expected_illegal = self.config.smoothing_epsilon / (1 + 38 * self.config.smoothing_epsilon)
        illegal_H_ok = np.max(illegal_probs_H) < expected_illegal * 1.5 if len(illegal_probs_H) > 0 else True
        illegal_R_ok = np.max(illegal_probs_R) < expected_illegal * 1.5 if len(illegal_probs_R) > 0 else True

        report['checks'].append({
            'name': 'pi_H illegal action probs',
            'passed': illegal_H_ok,
            'max_illegal_prob': float(np.max(illegal_probs_H)) if len(illegal_probs_H) > 0 else 0,
            'expected_max': expected_illegal * 1.5,
        })

        report['checks'].append({
            'name': 'pi_R illegal action probs',
            'passed': illegal_R_ok,
            'max_illegal_prob': float(np.max(illegal_probs_R)) if len(illegal_probs_R) > 0 else 0,
            'expected_max': expected_illegal * 1.5,
        })

        # Check min prob (smoothing should ensure > 0)
        min_pi_H = float(np.min(samples['pi_H']))
        min_pi_R = float(np.min(samples['pi_R']))
        expected_min = self.config.smoothing_epsilon / (1 + 38 * self.config.smoothing_epsilon)
        min_ok = min_pi_H > 0 and min_pi_R > 0

        report['checks'].append({
            'name': 'min probability > 0 (smoothing)',
            'passed': min_ok,
            'min_pi_H': min_pi_H,
            'min_pi_R': min_pi_R,
            'expected_min': expected_min,
        })

        # Check covariate invariants
        covariates = samples['covariates']
        feature_names = sorted(self.extractor.get_feature_names())

        len_indices = [feature_names.index(f'len_{s}') for s in ['spade', 'heart', 'diamond', 'club']]
        len_sums = np.sum(covariates[:, len_indices], axis=1)
        len_sum_ok = np.allclose(len_sums, 13.0, atol=0.1)

        report['checks'].append({
            'name': 'suit lengths sum to 13',
            'passed': len_sum_ok,
            'mean': float(np.mean(len_sums)),
            'min': float(np.min(len_sums)),
            'max': float(np.max(len_sums)),
        })

        # ===== E. Hard-core checks (Critical) =====
        # E1. Pass always legal (REF_ACTION_IDX = 0)
        pass_legal = samples['legal_masks'][:, REF_ACTION_IDX]
        pass_always_legal = np.all(pass_legal)
        report['checks'].append({
            'name': 'Pass (ref_action) always legal [CRITICAL]',
            'passed': pass_always_legal,
            'pass_legal_rate': float(np.mean(pass_legal)),
            'n_illegal_pass': int(np.sum(~pass_legal)),
        })

        # E2. States per episode distribution
        if 'episode_ids' in samples:
            episode_ids = samples['episode_ids']
            states_per_ep = np.bincount(episode_ids)
            states_per_ep = states_per_ep[states_per_ep > 0]  # Remove zero counts

            states_min = int(np.min(states_per_ep))
            states_max = int(np.max(states_per_ep))
            states_median = float(np.median(states_per_ep))
            states_mean = float(np.mean(states_per_ep))

            # Bridge auctions typically have 8-20 states
            states_ok = states_min >= 1 and states_max <= 100  # Reasonable bounds

            report['checks'].append({
                'name': 'states per episode distribution',
                'passed': states_ok,
                'min': states_min,
                'median': states_median,
                'max': states_max,
                'mean': states_mean,
            })

            report['states_per_episode'] = {
                'min': states_min,
                'median': states_median,
                'max': states_max,
                'mean': states_mean,
            }

        # E3. Board no duplicates (if without-replacement)
        if 'board_ids' in samples:
            board_ids = samples['board_ids']
            episode_ids = samples['episode_ids']

            # Check unique boards per episode (each episode should have one board)
            unique_episodes = len(np.unique(episode_ids))
            unique_boards = len(np.unique(board_ids))

            boards_ok = unique_boards == unique_episodes  # 1:1 mapping
            report['checks'].append({
                'name': 'board no duplicates (without-replacement)',
                'passed': boards_ok,
                'unique_episodes': unique_episodes,
                'unique_boards': unique_boards,
            })

        # ===== Compute action legal rates =====
        action_legal_rates = np.mean(samples['legal_masks'].astype(float), axis=0)
        report['action_legal_rates'] = action_legal_rates.tolist()

        # Identify rare actions
        rare_actions = [i for i, rate in enumerate(action_legal_rates)
                        if rate < self.config.rare_action_threshold]
        report['rare_actions'] = rare_actions

        # Overall pass/fail
        all_passed = all(c['passed'] for c in report['checks'])
        report['all_passed'] = all_passed

        return report

    def save_samples(
        self,
        samples: Dict[str, np.ndarray],
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save samples to NPZ file with metadata.

        Args:
            samples: Sample dictionary
            metadata: Additional metadata to include (e.g., from verify_samples)

        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute statistics for metadata
        n_samples = len(samples['observations'])
        n_episodes = len(np.unique(samples.get('episode_ids', [])))

        # Compute action legal rates
        action_legal_rates = np.mean(samples['legal_masks'].astype(float), axis=0).tolist()

        # Identify rare actions
        rare_actions = [i for i, rate in enumerate(action_legal_rates)
                        if rate < self.config.rare_action_threshold]

        # Compute states per episode stats
        states_per_episode_stats = {}
        if 'episode_ids' in samples:
            episode_ids = samples['episode_ids']
            states_per_ep = np.bincount(episode_ids)
            states_per_ep = states_per_ep[states_per_ep > 0]
            states_per_episode_stats = {
                'min': int(np.min(states_per_ep)),
                'median': float(np.median(states_per_ep)),
                'max': int(np.max(states_per_ep)),
                'mean': float(np.mean(states_per_ep)),
            }

        # Build metadata
        full_metadata = {
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'n_episodes': n_episodes,
            'feature_names': sorted(self.extractor.get_feature_names()),
            # Action encoding info (critical for Step 4 ALR)
            'action_names': PGX_ACTION_NAMES,
            'ref_action': REF_ACTION,
            'ref_action_idx': REF_ACTION_IDX,
            # Statistics
            'action_legal_rates': action_legal_rates,
            'rare_actions': rare_actions,
            'states_per_episode': states_per_episode_stats,
            # Checkpoint paths
            'pi_H_checkpoint': str(self.pi_H.checkpoint_dir),
            'pi_R_checkpoint': str(self.pi_R.checkpoint_dir),
        }
        if metadata:
            full_metadata.update(metadata)

        # Save NPZ with correct dtypes
        npz_path = output_dir / f"{self.config.run_id}_policy_samples.npz"
        save_dict = {
            'observations': samples['observations'].astype(bool),  # bool, not float32!
            'covariates': samples['covariates'].astype(np.float32),
            'pi_H': samples['pi_H'].astype(np.float32),
            'pi_R': samples['pi_R'].astype(np.float32),
            'legal_masks': samples['legal_masks'].astype(bool),
        }

        # Add new fields if present
        if 'episode_ids' in samples:
            save_dict['episode_ids'] = samples['episode_ids'].astype(np.int32)
        if 'board_ids' in samples:
            save_dict['board_ids'] = samples['board_ids'].astype(np.int32)
        if 'timestep_in_episode' in samples:
            save_dict['timestep_in_episode'] = samples['timestep_in_episode'].astype(np.int16)

        np.savez_compressed(npz_path, **save_dict)

        # Save metadata JSON
        meta_path = output_dir / f"{self.config.run_id}_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2)

        # Print relative paths to avoid Unicode issues on Windows
        try:
            print(f"\nSamples saved to: {npz_path.name}")
            print(f"Metadata saved to: {meta_path.name}")
        except Exception:
            print("\nFiles saved successfully")

        # Print file sizes
        npz_size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"NPZ file size: {npz_size_mb:.1f} MB")

        # Print shape summary
        print(f"\nSaved arrays:")
        for key, arr in save_dict.items():
            print(f"  {key}: {arr.shape} ({arr.dtype})")

        return npz_path


def load_samples(path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load samples from NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        (samples_dict, metadata_dict)
    """
    path = Path(path)

    # Load NPZ
    with np.load(path) as data:
        samples = {key: data[key] for key in data.files}

    # Load metadata
    meta_path = path.with_suffix('').with_suffix('') / f"{path.stem.replace('_policy_samples', '')}_metadata.json"
    if not meta_path.exists():
        # Try alternative path
        meta_path = path.parent / f"{path.stem.replace('_policy_samples', '_metadata')}.json"

    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return samples, metadata
