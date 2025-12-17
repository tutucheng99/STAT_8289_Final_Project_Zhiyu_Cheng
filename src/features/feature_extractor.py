"""
Unified Bridge Feature Extractor

Integrates all feature extractors and provides:
- Unified extraction interface
- Feature subset selection
- Normalization (optional, stats-driven)
- Array conversion
- Metadata management

Key design decisions:
1. All features output as float32 for downstream compatibility
2. Normalization uses empirical statistics (not hardcoded ranges)
3. Feature metadata loaded from statistics JSON
"""
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from .hand_features import extract_hand_features
from .bidding_features import extract_bidding_features
from .context_features import extract_context_features


@dataclass
class FeatureMetadata:
    """Feature metadata."""
    name: str
    category: str  # 'hand', 'bidding', 'context'
    description: str
    # Filled from statistics
    empirical_min: Optional[float] = None
    empirical_max: Optional[float] = None
    empirical_mean: Optional[float] = None
    empirical_std: Optional[float] = None


# Feature definitions (ranges filled from Step 2.5 statistics)
FEATURE_DEFINITIONS = {
    # Hand features - HCP
    'hcp_total': FeatureMetadata('hcp_total', 'hand', 'Total HCP (A=4,K=3,Q=2,J=1)'),
    'hcp_spade': FeatureMetadata('hcp_spade', 'hand', 'Spade suit HCP'),
    'hcp_heart': FeatureMetadata('hcp_heart', 'hand', 'Heart suit HCP'),
    'hcp_diamond': FeatureMetadata('hcp_diamond', 'hand', 'Diamond suit HCP'),
    'hcp_club': FeatureMetadata('hcp_club', 'hand', 'Club suit HCP'),

    # Hand features - Distribution
    'len_spade': FeatureMetadata('len_spade', 'hand', 'Spade length'),
    'len_heart': FeatureMetadata('len_heart', 'hand', 'Heart length'),
    'len_diamond': FeatureMetadata('len_diamond', 'hand', 'Diamond length'),
    'len_club': FeatureMetadata('len_club', 'hand', 'Club length'),
    'longest_suit': FeatureMetadata('longest_suit', 'hand', 'Longest suit length'),
    'shortest_suit': FeatureMetadata('shortest_suit', 'hand', 'Shortest suit length'),
    'is_balanced': FeatureMetadata('is_balanced', 'hand', 'Balanced hand (4333/4432/5332)'),
    'n_singletons': FeatureMetadata('n_singletons', 'hand', 'Number of singletons'),
    'n_voids': FeatureMetadata('n_voids', 'hand', 'Number of voids'),

    # Hand features - Controls & Tricks
    'controls_total': FeatureMetadata('controls_total', 'hand', 'Total controls (A=2,K=1)'),
    'controls_spade': FeatureMetadata('controls_spade', 'hand', 'Spade controls'),
    'controls_heart': FeatureMetadata('controls_heart', 'hand', 'Heart controls'),
    'controls_diamond': FeatureMetadata('controls_diamond', 'hand', 'Diamond controls'),
    'controls_club': FeatureMetadata('controls_club', 'hand', 'Club controls'),
    'ltc': FeatureMetadata('ltc', 'hand', 'Losing Trick Count'),
    'quick_tricks': FeatureMetadata('quick_tricks', 'hand', 'Quick tricks'),

    # Hand features - Honors
    'n_aces': FeatureMetadata('n_aces', 'hand', 'Number of Aces'),
    'n_kings': FeatureMetadata('n_kings', 'hand', 'Number of Kings'),
    'n_queens': FeatureMetadata('n_queens', 'hand', 'Number of Queens'),
    'n_jacks': FeatureMetadata('n_jacks', 'hand', 'Number of Jacks'),
    'n_honors_in_long_suits': FeatureMetadata('n_honors_in_long_suits', 'hand', 'Honors in 4+ card suits'),
    'has_ak_in_any_suit': FeatureMetadata('has_ak_in_any_suit', 'hand', 'Has AK in any suit'),
    'n_suits_with_honors': FeatureMetadata('n_suits_with_honors', 'hand', 'Suits with AKQJ'),

    # Bidding features
    'n_contracts_bid': FeatureMetadata('n_contracts_bid', 'bidding', 'Number of contracts bid'),
    'auction_level': FeatureMetadata('auction_level', 'bidding', 'Current auction level (1-7, 0=none)'),
    # NOTE: contract_strain=-1 for passout. When using with len_{suit}/hcp_{suit},
    # use STRAIN_TO_SUIT_IDX mapping and mask with has_contract.
    'contract_strain': FeatureMetadata('contract_strain', 'bidding', 'Contract strain (0=C,1=D,2=H,3=S,4=NT,-1=none)'),
    'is_doubled': FeatureMetadata('is_doubled', 'bidding', 'Highest contract is doubled'),
    'is_redoubled': FeatureMetadata('is_redoubled', 'bidding', 'Highest contract is redoubled'),
    'is_competitive': FeatureMetadata('is_competitive', 'bidding', 'Both sides have bid'),
    'is_passout': FeatureMetadata('is_passout', 'bidding', 'No bids made'),
    'has_contract': FeatureMetadata('has_contract', 'bidding', 'At least one bid made'),
    'self_opened': FeatureMetadata('self_opened', 'bidding', 'Self made opening bid'),
    'partner_opened': FeatureMetadata('partner_opened', 'bidding', 'Partner made opening bid'),
    'rho_opened': FeatureMetadata('rho_opened', 'bidding', 'RHO made opening bid'),
    'lho_opened': FeatureMetadata('lho_opened', 'bidding', 'LHO made opening bid'),
    'double_status': FeatureMetadata('double_status', 'bidding', 'Double status (0=undoubled, 1=doubled, 2=redoubled)'),

    # Context features
    'we_vulnerable': FeatureMetadata('we_vulnerable', 'context', 'Our side vulnerable'),
    'they_vulnerable': FeatureMetadata('they_vulnerable', 'context', 'Their side vulnerable'),
    'both_vulnerable': FeatureMetadata('both_vulnerable', 'context', 'Both sides vulnerable'),
    'none_vulnerable': FeatureMetadata('none_vulnerable', 'context', 'Neither side vulnerable'),
    'favorable_vul': FeatureMetadata('favorable_vul', 'context', 'Favorable vulnerability'),
    'unfavorable_vul': FeatureMetadata('unfavorable_vul', 'context', 'Unfavorable vulnerability'),
    'n_passed_before_opening': FeatureMetadata('n_passed_before_opening', 'context', 'Passes before opening'),
}


class BridgeFeatureExtractor:
    """Unified bridge feature extractor."""

    def __init__(
        self,
        feature_subset: Optional[List[str]] = None,
        normalize: bool = False,
        stats_path: Optional[str] = None,
    ):
        """
        Initialize extractor.

        Args:
            feature_subset: Features to extract (None = all)
            normalize: Normalize to [0, 1] using empirical stats
            stats_path: Path to feature statistics JSON
        """
        self.feature_subset = feature_subset
        self.normalize = normalize
        self.metadata = {k: FeatureMetadata(**asdict(v)) for k, v in FEATURE_DEFINITIONS.items()}

        if stats_path and Path(stats_path).exists():
            self._load_statistics(stats_path)

    def _load_statistics(self, path: str):
        """Load feature statistics from JSON."""
        with open(path) as f:
            stats = json.load(f)

        for name, values in stats.items():
            if name in self.metadata:
                self.metadata[name].empirical_min = values.get('min')
                self.metadata[name].empirical_max = values.get('max')
                self.metadata[name].empirical_mean = values.get('mean')
                self.metadata[name].empirical_std = values.get('std')

    def extract(self, obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Extract features from observation.

        Args:
            obs: shape (480,) or (batch, 480)

        Returns:
            Dict of feature name -> values (all float32)
        """
        features = {}

        # Extract all feature categories
        features.update(extract_hand_features(obs))
        features.update(extract_bidding_features(obs))
        features.update(extract_context_features(obs))

        # Filter to subset if specified
        if self.feature_subset is not None:
            features = {k: v for k, v in features.items() if k in self.feature_subset}

        # Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)

        return features

    def _normalize_features(
        self,
        features: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Normalize features to [0, 1] using empirical stats."""
        normalized = {}
        for name, value in features.items():
            if name in self.metadata:
                meta = self.metadata[name]
                if meta.empirical_min is not None and meta.empirical_max is not None:
                    range_val = meta.empirical_max - meta.empirical_min
                    if range_val > 0:
                        normalized[name] = (value - meta.empirical_min) / range_val
                    else:
                        normalized[name] = jnp.zeros_like(value)
                else:
                    normalized[name] = value
            else:
                normalized[name] = value
        return normalized

    def to_array(
        self,
        features: Dict[str, jnp.ndarray],
        feature_order: Optional[List[str]] = None
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Convert feature dict to array.

        Args:
            features: Feature dictionary
            feature_order: Order of features in output (None = sorted keys)

        Returns:
            (array of shape (n_features,) or (batch, n_features), feature_names)
        """
        if feature_order is None:
            feature_order = sorted(features.keys())

        arrays = [features[name] for name in feature_order]
        stacked = jnp.stack(arrays, axis=-1)

        return stacked, feature_order

    def get_feature_names(self, category: Optional[str] = None) -> List[str]:
        """
        Get feature names.

        Args:
            category: Filter by category ('hand', 'bidding', 'context', None=all)

        Returns:
            List of feature names
        """
        if self.feature_subset:
            names = self.feature_subset
        else:
            names = list(self.metadata.keys())

        if category:
            names = [n for n in names if self.metadata.get(n, FeatureMetadata(n, '', '')).category == category]

        return names

    def save_metadata(self, path: str):
        """Save feature metadata to JSON."""
        data = {name: asdict(meta) for name, meta in self.metadata.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_feature_info(self, name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a specific feature."""
        return self.metadata.get(name)
