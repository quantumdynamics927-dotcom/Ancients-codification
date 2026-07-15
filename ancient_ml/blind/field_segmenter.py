"""
Field Segmenter: Protocol Field Boundary Detection
=================================================
Segments aligned message sequences into protocol fields.
Uses vertical entropy and mutual information saturation at each position.
NO semantic knowledge — purely statistical boundary detection.

Based on:
- Lin et al. "Automatic Protocol Reverse Engineering" (ScienceDirect 2019)
- Change-point detection via entropy/MI voting

Usage:
    from blind.field_segmenter import FieldSegmenter, SegmentResult
    seg = FieldSegmenter(sequences)
    result = seg.segment()
    print(result.field_boundaries)
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class Field:
    """A detected protocol field."""
    field_id: int
    start_pos: int
    end_pos: int          # Inclusive
    width: int             # Number of sign positions
    signs: List[str]       # Signs observed in this field
    n_unique: int          # Vocabulary size within field
    entropy: float         # H(sign) within field
    stability: float       # Fraction of sequences that have this field
    position_entropy: float # H(position) — spread of field start positions


@dataclass
class SegmentResult:
    """Result of field segmentation."""
    fields: List[Field]
    boundaries: List[int]       # Boundary positions (between fields)
    n_fields_mean: float         # Mean number of fields per message
    field_stability_mean: float # Mean stability across fields
    alignment_score: float      # How well sequences align (0=random, 1=perfect)
    protocol_header: List[int]   # Positions that form stable header


class FieldSegmenter:
    """
    Detect field boundaries in aligned message sequences.
    Works on sequences of equal length (or padded to equal length).
    Uses vertical entropy: entropy of signs at each column/position.
    """

    def __init__(self, sequences: List[List[str]], min_length: int = 3):
        self.sequences = sequences
        self.min_length = min_length
        self._aligned: Optional[np.ndarray] = None
        self._n_seq: int = 0
        self._length: int = 0

    def segment(self, n_fields_hint: Optional[int] = None) -> SegmentResult:
        """
        Run field segmentation.
        n_fields_hint: if given, look for exactly this many fields.
        """
        self._align()
        boundaries = self._detect_boundaries(n_fields_hint)
        fields = self._build_fields(boundaries)
        header = self._detect_header(fields)
        return SegmentResult(
            fields=fields,
            boundaries=boundaries,
            n_fields_mean=np.mean([len(fields) for _ in self.sequences]),
            field_stability_mean=np.mean([f.stability for f in fields]),
            alignment_score=self._alignment_score(),
            protocol_header=header,
        )

    def _align(self):
        """Align sequences to common length (pad with None)."""
        self._n_seq = len(self.sequences)
        if self._n_seq == 0:
            self._length = 0
            self._aligned = np.array([])
            return
        self._length = max(len(s) for s in self.sequences)
        if self._length < self.min_length:
            self._length = self.min_length

        aligned = []
        for seq in self.sequences:
            padded = list(seq) + [None] * (self._length - len(seq))
            aligned.append(padded[:self._length])
        self._aligned = np.array(aligned, dtype=object)

    def _vertical_entropy(self, col: int) -> float:
        """Entropy of signs at position col across all sequences."""
        signs = self._aligned[:, col]
        signs = [s for s in signs if s is not None]
        if not signs:
            return 0.0
        counts = Counter(signs)
        total = len(signs)
        probs = np.array([c / total for c in counts.values()])
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _vertical_mi(self, col1: int, col2: int) -> float:
        """
        Mutual information between two adjacent columns.
        High MI = strong dependency = same field.
        Low MI = independent = possible boundary.
        """
        col1_signs = self._aligned[:, col1]
        col2_signs = self._aligned[:, col2]
        pairs = [(c1, c2) for c1, c2 in zip(col1_signs, col2_signs) if c1 is not None and c2 is not None]
        if len(pairs) < 2:
            return 0.0

        # MI = H(col1) + H(col2) - H(col1, col2)
        p1 = Counter(c1 for c1, _ in pairs)
        p2 = Counter(c2 for _, c2 in pairs)
        p12 = Counter(pairs)

        total = len(pairs)
        h1 = -sum((c/total) * np.log2(c/total + 1e-12) for c in p1.values())
        h2 = -sum((c/total) * np.log2(c/total + 1e-12) for c in p2.values())
        h12 = -sum((c/total) * np.log2(c/total + 1e-12) for c in p12.values())
        return h1 + h2 - h12

    def _detect_boundaries(self, n_fields_hint: Optional[int] = None) -> List[int]:
        """
        Detect field boundaries using entropy peaks and MI drops.
        Returns list of boundary positions (between positions).
        """
        if self._length < 2:
            return []

        # Compute entropy at each position
        entropies = np.array([self._vertical_entropy(c) for c in range(self._length)])

        # Compute MI between adjacent columns
        mis = np.array([self._vertical_mi(c, c+1) for c in range(self._length - 1)])

        # Score each gap: boundary = high entropy (new context) OR low MI (independent)
        # Normalize both to [0, 1]
        if entropies.max() > 0:
            ent_norm = entropies / entropies.max()
        else:
            ent_norm = np.zeros_like(entropies)

        if mis.max() > 0:
            mi_norm = 1 - (mis / mis.max())  # Invert: low MI = high score
        else:
            mi_norm = np.zeros_like(mis)

        # Combined boundary score
        boundary_scores = np.zeros(self._length - 1)
        for i in range(self._length - 1):
            boundary_scores[i] = 0.5 * ent_norm[i+1] + 0.5 * mi_norm[i]

        # Find peaks in boundary scores
        threshold = 0.5 * boundary_scores.max()
        if threshold < 0.1:
            threshold = 0.1

        peaks = []
        for i in range(1, len(boundary_scores) - 1):
            if boundary_scores[i] > boundary_scores[i-1] and boundary_scores[i] > boundary_scores[i+1]:
                if boundary_scores[i] >= threshold:
                    peaks.append((i, boundary_scores[i]))

        # Sort peaks by score
        peaks.sort(key=lambda x: x[1], reverse=True)

        # If n_fields_hint given, take top n-1 peaks
        if n_fields_hint is not None:
            n_boundaries = max(1, n_fields_hint - 1)
            boundaries = sorted([p[0] for p in peaks[:n_boundaries]])
        else:
            # Take all peaks above threshold
            boundaries = sorted([p[0] for p in peaks if p[1] >= threshold])

        return boundaries

    def _build_fields(self, boundaries: List[int]) -> List[Field]:
        """Build Field objects from boundary positions."""
        positions = [0] + boundaries + [self._length]
        fields = []
        for fid in range(len(positions) - 1):
            start, end = positions[fid], positions[fid+1] - 1
            width = end - start + 1

            # Collect signs in this field across all sequences
            field_signs = []
            for seq in self.sequences:
                if start < len(seq):
                    field_signs.extend([s for s in seq[start:min(end+1, len(seq))] if s is not None])

            unique_signs = set(field_signs)
            entropy = 0.0
            if field_signs:
                counts = Counter(field_signs)
                total = len(field_signs)
                probs = np.array([c/total for c in counts.values()])
                entropy = -np.sum(probs * np.log2(probs + 1e-12))

            # Stability: fraction of sequences that have content in this field
            n_present = sum(1 for seq in self.sequences if start < len(seq) and any(s is not None for s in seq[start:min(end+1, len(seq))]))
            stability = n_present / max(self._n_seq, 1)

            fields.append(Field(
                field_id=fid,
                start_pos=start,
                end_pos=end,
                width=width,
                signs=sorted(unique_signs),
                n_unique=len(unique_signs),
                entropy=entropy,
                stability=stability,
                position_entropy=0.0,  # Could add start-position spread here
            ))

        return fields

    def _detect_header(self, fields: List[Field]) -> List[int]:
        """Identify which fields form a stable protocol header."""
        # Header = fields with stability = 1.0
        header = [f.start_pos for f in fields if f.stability >= 0.95]
        return header

    def _alignment_score(self) -> float:
        """
        How well do sequences align? 0 = random, 1 = perfect alignment.
        Computed as 1 - (variance in length / mean length).
        """
        lengths = [len(s) for s in self.sequences]
        if not lengths:
            return 0.0
        mean_len = np.mean(lengths)
        if mean_len == 0:
            return 0.0
        variance = np.var(lengths)
        return max(0.0, 1.0 - variance / mean_len)
