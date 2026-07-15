"""
Role Typer: Pseudo-Type Clustering Without Semantic Knowledge
============================================================
Clusters field contents by behavioral features, not meaning.
Features: length, entropy, positional freedom, co-occurrence patterns.
DBSCAN / hierarchical clustering → TYPE_A, TYPE_B, ... (no "god/man").

Based on:
- "Neural Network for Learning the Syntax of Network Protocols" (arXiv 2023)
- Behavioral type inference from raw packet streams

Usage:
    from blind.role_typer import RoleTyper, PseudoType
    typer = RoleTyper(sequences)
    types = typer.cluster()
    print(types[0].type_id)  # "TYPE_A", not "god"
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@dataclass
class PseudoType:
    """
    A behaviorally-inferred type — no semantic label.
    Only structural features and cluster membership.
    """
    type_id: str           # "TYPE_A", "TYPE_B", etc.
    members: List[str]     # Signs belonging to this type
    length_mean: float     # Mean token count
    length_std: float
    entropy: float         # Internal entropy (low = constrained = structured)
    positional_freedom: float  # How freely this type appears across positions
    transition_entropy: float  # H(next_type | this_type)
    co_occurring_types: Dict[str, float]  # type_id -> co-occurrence freq


class RoleTyper:
    """
    Cluster signs into pseudo-types based on behavioral features.
    NO semantic knowledge — purely structural clustering.
    """

    def __init__(self, sequences: List[List[str]], eps: float = 1.5, min_samples: int = 2):
        self.sequences = sequences
        self.eps = eps
        self.min_samples = min_samples
        self._sign_features: Dict[str, np.ndarray] = {}
        self._type_map: Dict[str, str] = {}  # sign -> type_id

    def cluster(self) -> List[PseudoType]:
        """
        Run clustering and return pseudo-type list.
        """
        features = self._compute_features()
        labels = self._cluster_dbscan(features)
        types = self._build_types(labels, features)
        return types

    def _compute_features(self) -> Dict[str, np.ndarray]:
        """
        Compute behavioral features for each sign.
        6 feature dimensions:
        1. log(frequency)
        2. positional entropy (high = freely positioned)
        3. length statistics (mean length of sequences containing this sign)
        4. initial ratio (fraction of sequences where this sign is at pos 0)
        5. final ratio (fraction of sequences where this sign is at final pos)
        6. unique neighbors ratio (fraction of unique neighbors / total neighbors)
        """
        flat = [s for seq in self.sequences for s in seq]
        sign_freq = Counter(flat)
        vocab = set(flat)

        # Position entropy per sign
        sign_positions = defaultdict(list)
        for i, sign in enumerate(flat):
            sign_positions[sign].append(i)

        # Length statistics per sign
        sign_lengths = defaultdict(list)
        for seq in self.sequences:
            for sign in set(seq):
                sign_lengths[sign].append(len(seq))

        # Initial/final ratio per sign
        initial_counts = Counter(seq[0] for seq in self.sequences if seq)
        final_counts = Counter(seq[-1] for seq in self.sequences if seq)

        # Neighbor entropy per sign
        bigrams = [(flat[i], flat[i+1]) for i in range(len(flat)-1)]
        sign_neighbors = defaultdict(set)
        sign_transition_counts = defaultdict(Counter)
        for prev, curr in bigrams:
            sign_neighbors[prev].add(curr)
            sign_transition_counts[prev][curr] += 1

        n_seq = len(self.sequences)
        features = {}

        for sign in vocab:
            freq = sign_freq.get(sign, 0)
            if freq < 2:
                continue

            positions = sign_positions[sign]
            # Positional entropy
            pos_counter = Counter(positions)
            pos_probs = np.array(list(pos_counter.values())) / len(positions)
            pos_entropy = -np.sum(pos_probs * np.log2(pos_probs + 1e-12))
            max_pos_entropy = np.log2(len(set(positions)) + 1)
            pos_entropy_norm = pos_entropy / max(max_pos_entropy, 1e-6)

            # Length mean/std
            lens = sign_lengths.get(sign, [0])
            len_mean = np.mean(lens)
            len_std = np.std(lens) if len(lens) > 1 else 0.0

            # Initial/final ratio
            init_ratio = initial_counts.get(sign, 0) / max(n_seq, 1)
            final_ratio = final_counts.get(sign, 0) / max(n_seq, 1)

            # Neighbor diversity
            neighbors = sign_neighbors.get(sign, set())
            n_transitions = sum(sign_transition_counts[sign].values())
            neighbor_entropy = 0.0
            if n_transitions > 0:
                probs = np.array(list(sign_transition_counts[sign].values())) / n_transitions
                neighbor_entropy = -np.sum(probs * np.log2(probs + 1e-12))

            self._sign_features[sign] = np.array([
                np.log1p(freq),
                pos_entropy_norm,
                len_mean / 50,
                len_std / 20,
                init_ratio,
                final_ratio,
                len(neighbors) / max(n_transitions, 1),
                neighbor_entropy / np.log2(max(len(neighbors), 2) + 1),
            ])
            features[sign] = self._sign_features[sign]

        return features

    def _cluster_dbscan(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Run DBSCAN on sign features. Falls back to frequency-based types on small vocab."""
        if len(features) < 3:
            # Too few signs — assign all to TYPE_A
            return {s: 0 for s in features}

        signs = sorted(features.keys())
        X = np.array([features[s] for s in signs])

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # DBSCAN with tighter eps for small feature spaces
        clustering = DBSCAN(eps=0.7, min_samples=2)
        labels = clustering.fit_predict(X_scaled)

        # If all are noise or all same cluster, use frequency-based fallback
        unique_labels = set(labels) - {-1}
        if len(unique_labels) <= 1:
            return self._frequency_based_clustering(features)

        return {signs[i]: int(labels[i]) for i in range(len(signs))}

    def _frequency_based_clustering(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Fallback: cluster by frequency tier (high/medium/low).
        Ensures we always get at least 2 types for meaningful analysis.
        """
        signs = sorted(features.keys(), key=lambda s: features[s][0], reverse=True)
        n = len(signs)
        # Top 30% = TYPE_A (frequent), rest = TYPE_B
        threshold = max(1, n // 3)
        labels = {}
        for i, s in enumerate(signs):
            labels[s] = 0 if i < threshold else 1
        return labels

    def _build_types(self, labels: Dict[str, int], features: Dict[str, np.ndarray]) -> List[PseudoType]:
        """Build PseudoType objects from cluster labels."""
        type_members = defaultdict(list)
        for sign, label in labels.items():
            type_members[label].append(sign)

        # Compute aggregate features per type
        transition_entropy_map = self._compute_transition_entropies(labels)

        types = []
        for label, members in sorted(type_members.items()):
            if label == -1:
                # Noise cluster — skip or label as TYPE_NOISE
                continue

            # Average features across members
            member_features = [features[m] for m in members if m in features]
            if not member_features:
                continue
            avg_features = np.mean(member_features, axis=0)

            # Length statistics
            flat = [s for seq in self.sequences for s in seq]
            member_lens = []
            for seq in self.sequences:
                if any(s in members for s in seq):
                    member_lens.append(len(seq))

            # Build co-occurring types
            cooc = self._compute_cooccurring_types(members)

            type_id = f"TYPE_{chr(65 + label)}"  # TYPE_A, TYPE_B, ...
            types.append(PseudoType(
                type_id=type_id,
                members=members,
                length_mean=float(avg_features[2]),
                length_std=float(avg_features[3]),
                entropy=float(avg_features[1]),
                positional_freedom=float(avg_features[1]),
                transition_entropy=transition_entropy_map.get(label, 0.0),
                co_occurring_types=cooc,
            ))
            self._type_map.update({m: type_id for m in members})

        return types

    def _compute_transition_entropies(self, labels: Dict[str, int]) -> Dict[int, float]:
        """Compute H(next_type | this_type) for each type."""
        flat = [s for seq in self.sequences for s in seq]
        bigrams = [(flat[i], flat[i+1]) for i in range(len(flat)-1)]

        label_map = labels
        type_transitions = defaultdict(Counter)
        type_counts = Counter()

        for prev, curr in bigrams:
            if prev in label_map and curr in label_map:
                t_prev = label_map[prev]
                t_curr = label_map[curr]
                type_transitions[t_prev][t_curr] += 1
                type_counts[t_prev] += 1

        entropies = {}
        for t_prev, next_types in type_transitions.items():
            total = type_counts[t_prev]
            probs = np.array([c / total for c in next_types.values()])
            entropies[t_prev] = -np.sum(probs * np.log2(probs + 1e-12))

        return entropies

    def _compute_cooccurring_types(self, members: List[str]) -> Dict[str, float]:
        """Compute co-occurrence frequency with other types."""
        flat = [s for seq in self.sequences for s in seq]
        bigrams = [(flat[i], flat[i+1]) for i in range(len(flat)-1)]

        cooc = Counter()
        member_set = set(members)
        for prev, curr in bigrams:
            if prev in member_set:
                cooc[curr] += 1
            if curr in member_set:
                cooc[prev] += 1

        total = sum(cooc.values())
        if total == 0:
            return {}
        return {s: c / total for s, c in cooc.most_common(10)}

    def type_of(self, sign: str) -> Optional[str]:
        """Get the pseudo-type for a sign."""
        return self._type_map.get(sign)
