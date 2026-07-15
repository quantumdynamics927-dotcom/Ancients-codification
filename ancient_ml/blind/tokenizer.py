"""
Tokenizer: Opaque Sign IDs with Optional Shape Clustering
==========================================================
Pure tokenization — no semantic knowledge. Signs are opaque tokens.
Optional shape-based clustering (visual similarity) for geometry channel.

Usage:
    from blind.tokenizer import Tokenizer
    tok = Tokenizer(sequences)
    ids = tok.tokenize()           # List[List[str]]
    shape_clusters = tok.shape_clusters()  # Optional
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple

# pylint: disable=unused-import
import numpy as np


@dataclass
class TokenizerConfig:
    """Configuration for tokenization behavior."""
    min_frequency: int = 1          # Minimum sign frequency to include in vocab
    use_shape_clusters: bool = False  # Enable shape-based clustering
    n_shape_clusters: int = 20      # Number of visual shape groups
    normalize: bool = True           # Normalize sign IDs


@dataclass
class TokenStats:
    """Per-sign statistics for downstream modules."""
    sign: str
    frequency: int
    positions: List[int]           # All positions in flat stream
    initial_count: int             # Times appears at position 0
    final_count: int               # Times appears at last position
    position_entropy: float         # H(positions) — spread vs fixed
    cooccurrence: Dict[str, int]    # Signs that follow this one
    first_position_mean: float      # Mean first occurrence position
    last_position_mean: float       # Mean last occurrence position


class Tokenizer:
    """
    Converts raw sign sequences to opaque token IDs.
    Optionally clusters signs by shape similarity for geometry channel.

    NO semantic knowledge used — purely structural.
    """

    def __init__(self, sequences: List[List[str]], config: Optional[TokenizerConfig] = None):
        self.sequences = sequences
        self.config = config or TokenizerConfig()
        self._vocab: Set[str] = set()
        self._flat: List[str] = []
        self._sign_stats: Dict[str, TokenStats] = {}
        self._shape_clusters: Optional[np.ndarray] = None

    def tokenize(self) -> List[List[str]]:
        """
        Return sequences as opaque token lists.
        All downstream modules work on these tokens only.
        """
        self._build_vocab()
        self._build_flat()
        self._build_sign_stats()
        return [[s for s in seq if s in self._vocab] for seq in self.sequences]

    def _build_vocab(self):
        """Build vocabulary from sequences, filtering by min_frequency."""
        freq = Counter(s for seq in self.sequences for s in seq)
        self._vocab = {s for s, c in freq.items() if c >= self.config.min_frequency}

    def _build_flat(self):
        """Flatten sequences for global statistics."""
        self._flat = [s for seq in self.sequences for s in seq if s in self._vocab]

    def _build_sign_stats(self):
        """Compute per-sign statistics for role_typer and field_segmenter."""
        sign_positions = defaultdict(list)
        for i, sign in enumerate(self._flat):
            sign_positions[sign].append(i)

        initial_counts = Counter(seq[0] for seq in self.sequences if seq)
        final_counts = Counter(seq[-1] for seq in self.sequences if seq)

        # Bigram co-occurrence
        bigrams = [(self._flat[i], self._flat[i+1]) for i in range(len(self._flat)-1)]
        cooccur = defaultdict(Counter)
        for prev, curr in bigrams:
            cooccur[prev][curr] += 1

        for sign in self._vocab:
            positions = sign_positions[sign]
            freq = len(positions)

            # Position entropy: uniform distribution = high entropy = freely positioned
            # Fixed position = low entropy
            unique_positions = len(set(positions))
            if unique_positions > 1:
                pos_counter = Counter(positions)
                probs = np.array(list(pos_counter.values())) / freq
                pos_entropy = -np.sum(probs * np.log2(probs + 1e-12))
            else:
                pos_entropy = 0.0

            self._sign_stats[sign] = TokenStats(
                sign=sign,
                frequency=freq,
                positions=positions,
                initial_count=initial_counts.get(sign, 0),
                final_count=final_counts.get(sign, 0),
                position_entropy=pos_entropy,
                cooccurrence=dict(cooccur[sign]),
                first_position_mean=np.mean(positions) if positions else 0,
                last_position_mean=np.mean(positions[-100:]) if positions else 0,
            )

    def sign_stats(self, sign: str) -> TokenStats:
        """Get statistics for a specific sign."""
        return self._sign_stats.get(sign)

    def vocab(self) -> Set[str]:
        """Return the vocabulary set."""
        return self._vocab

    def flat(self) -> List[str]:
        """Return flattened token stream."""
        return self._flat

    def n_signs(self) -> int:
        """Total number of tokens."""
        return len(self._flat)

    def n_vocab(self) -> int:
        """Vocabulary size."""
        return len(self._vocab)

    def shape_clusters(self) -> np.ndarray:
        """
        Optional: cluster signs by shape similarity.
        Returns array of cluster IDs (n_vocab,).

        Placeholder implementation: assigns each sign to a cluster
        based on its first character (heuristic for Gardiner sign categories).
        Real implementation would use image-based clustering.
        """
        if not self.config.use_shape_clusters:
            return np.array([])

        if self._shape_clusters is not None:
            return self._shape_clusters

        # Heuristic: group by first character of sign ID
        # G* = human/animal, D* = god, O* = building, N* = number, etc.
        cluster_map = {}
        for sign in self._vocab:
            prefix = sign[0] if sign else 'Z'
            cluster_map[sign] = hash(prefix) % self.config.n_shape_clusters

        self._shape_clusters = np.array([cluster_map.get(s, 0) for s in self._vocab])
        return self._shape_clusters

    def sign_to_cluster(self, sign: str) -> int:
        """Map a sign to its shape cluster ID."""
        clusters = self.shape_clusters()
        if clusters.size == 0:
            return -1
        vocab_list = sorted(self._vocab)
        if sign not in vocab_list:
            return -1
        idx = vocab_list.index(sign)
        return int(clusters[idx])
