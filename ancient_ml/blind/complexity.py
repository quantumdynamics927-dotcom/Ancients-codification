"""
Complexity Module: Kolmogorov Complexity Proxies
=============================================
Compression-based complexity measures for sign streams.
zlib, lzma, PPM-style compression ratios.
NCD (Normalized Compression Distance) between corpora.
NO semantic knowledge — purely algorithmic.

Based on:
- Kolmogorov Complexity (Li & Vitanyi 2008)
- "Compressor Complexity Features for Language Recognition" (CILI, IR 2020)
- NCD = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

Usage:
    from blind.complexity import ComplexityAnalyzer
    ca = ComplexityAnalyzer()
    result = ca.analyze(sequences)
    print(result.zlib_ratio)
    print(result.ncd_vs_random)
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple
import zlib
import random
import numpy as np


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    # Raw compression
    zlib_ratio: float       # compressed_size / original_size (lower = more structured)
    lzma_ratio: float       # LZMA compression ratio
    zlib_bits_per_token: float  # bits per token under zlib
    lzma_bits_per_token: float

    # NCD vs baselines
    ncd_vs_random: float   # NCD vs token-shuffled version
    ncd_vs_markov: float    # NCD vs Markov-model-generated
    ncd_vs_python: float    # vs Python source control

    # Grammar-based
    grammar_bits: float     # Bits from induced grammar
    residual_bits: float    # Uncompressed residual

    # Information density
    entropy_density: float  # H1 / bits_per_token
    mutual_info_density: float  # MI / bits_per_token

    # Verdict features
    is_compressible: bool   # zlib_ratio < 0.9
    is_high_structure: bool # zlib_ratio < 0.7 AND ncd_vs_random > 0.5


class ComplexityAnalyzer:
    """
    Measure algorithmic complexity of sign streams.
    Compression-based: zlib, LZMA.
    NCD vs null models: random shuffle, Markov, Python control.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def analyze(self, sequences: List[List[str]], python_control: bool = True) -> ComplexityResult:
        """
        Full complexity analysis.
        python_control: also compare to Python source code tokens.
        """
        flat = self._flatten(sequences)
        flat_str = " ".join(flat)

        # Compression ratios
        zlib_ratio = self._zlib_ratio(flat_str)
        lzma_ratio = self._lzma_ratio(flat_str)

        # Bits per token
        orig_size_bits = len(flat) * np.log2(max(len(set(flat)), 2))
        zlib_bits = zlib_ratio * orig_size_bits
        lzma_bits = lzma_ratio * orig_size_bits

        # NCD vs null models
        ncd_random = self._ncd(flat_str, self._shuffle_str(flat_str))
        ncd_markov = self._ncd(flat_str, self._markov_str(sequences))

        # Python control
        python_tokens = self._get_python_tokens()
        ncd_python = self._ncd(flat_str, python_tokens) if python_control else 0.0

        # Information density
        unigram_counts = Counter(flat)
        h1 = self._entropy_from_counts(unigram_counts)
        mutual_info = self._compute_mi(sequences)

        return ComplexityResult(
            zlib_ratio=zlib_ratio,
            lzma_ratio=lzma_ratio,
            zlib_bits_per_token=zlib_bits / max(len(flat), 1),
            lzma_bits_per_token=lzma_bits / max(len(flat), 1),
            ncd_vs_random=ncd_random,
            ncd_vs_markov=ncd_markov,
            ncd_vs_python=ncd_python,
            grammar_bits=0.0,  # Filled by grammar_inducer
            residual_bits=0.0,
            entropy_density=h1 / max(zlib_bits / max(len(flat), 1), 1e-6),
            mutual_info_density=mutual_info / max(zlib_bits / max(len(flat), 1), 1e-6),
            is_compressible=zlib_ratio < 0.9,
            is_high_structure=zlib_ratio < 0.7 and ncd_random > 0.5,
        )

    def _flatten(self, sequences: List[List[str]]) -> List[str]:
        return [s for seq in sequences for s in seq]

    def _entropy_from_counts(self, counts: Counter) -> float:
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def _zlib_ratio(self, s: str) -> float:
        """Compressed size / original size under zlib."""
        if not s:
            return 1.0
        original_bytes = s.encode("utf-8")
        compressed = zlib.compress(original_bytes, level=6)
        return len(compressed) / max(len(original_bytes), 1)

    def _lzma_ratio(self, s: str) -> float:
        """Compressed size / original size under LZMA."""
        try:
            import lzma
        except ImportError:
            return self._zlib_ratio(s)  # Fallback
        if not s:
            return 1.0
        original_bytes = s.encode("utf-8")
        compressed = lzma.compress(original_bytes)
        return len(compressed) / max(len(original_bytes), 1)

    def _shuffle_str(self, flat_str: str) -> str:
        """Generate random shuffle of tokens."""
        tokens = flat_str.split()
        rng = random.Random(self.random_seed)
        rng.shuffle(tokens)
        return " ".join(tokens)

    def _markov_str(self, sequences: List[List[str]]) -> str:
        """
        Generate a string using a Markov model (same bigram distribution).
        Order-1 Markov: preserve bigram frequencies.
        """
        flat = self._flatten(sequences)
        if len(flat) < 2:
            return flat_str

        # Build bigram transition probabilities
        bigram_counts = Counter(zip(flat[:-1], flat[1:]))

        rng = random.Random(self.random_seed)
        result = []
        # Pick start - weighted by first-sign frequency
        start_signs = [seq[0] for seq in sequences if seq]
        if start_signs:
            start_counter = Counter(start_signs)
            starts = list(start_counter.keys())
            weights = list(start_counter.values())
            current = rng.choices(starts, weights=weights, k=1)[0]
        else:
            current = flat[0]
        result.append(current)

        # Generate
        for _ in range(len(flat) - 1):
            next_options = [(nb, c) for (p, nb), c in bigram_counts.items() if p == current]
            if not next_options:
                break
            nexts, counts = zip(*next_options)
            total = sum(counts)
            probs = [c / total for c in counts]
            current = rng.choices(list(nexts), weights=probs, k=1)[0]
            result.append(current)

        return " ".join(result)

    def _get_python_tokens(self) -> str:
        """Get Python source code tokens as control corpus."""
        python_code = """
        def fibonacci n:
            if n less 2: return n
            return fibonacci n minus 2 plus fibonacci n minus 1
        for i in range 10:
            print fibonacci i
        class Calculator:
            def add self x y: return x plus y
        data equals list comprehension for x in range 100 if x mod 2 equals 0
        result equals map lambda x x times x data
        """.lower().split()
        return " ".join(python_code)

    def _ncd(self, s1: str, s2: str) -> float:
        """
        Normalized Compression Distance between two strings.
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        """
        c_x = len(zlib.compress(s1.encode("utf-8")))
        c_y = len(zlib.compress(s2.encode("utf-8")))
        c_xy = len(zlib.compress((s1 + " " + s2).encode("utf-8")))
        max_c = max(c_x, c_y)
        if max_c == 0:
            return 0.0
        return (c_xy - min(c_x, c_y)) / max_c

    def _compute_mi(self, sequences: List[List[str]]) -> float:
        """Compute mutual information from sequences."""
        flat = self._flatten(sequences)
        unigram_counts = Counter(flat)
        bigrams = Counter(zip(flat[:-1], flat[1:]))
        total = len(flat)

        # H(X)
        h1 = self._entropy_from_counts(unigram_counts)

        # H(X|X_prev)
        prev_counts = Counter(flat[:-1])
        h2 = 0.0
        for prev, c_prev in prev_counts.items():
            p_prev = c_prev / total
            # P(x | prev)
            h_given = 0.0
            for (p, curr), c_bigram in bigrams.items():
                if p == prev:
                    p_curr_given = c_bigram / c_prev
                    if p_curr_given > 0:
                        h_given -= p_curr_given * np.log2(p_curr_given)
            h2 += p_prev * h_given

        return h1 - h2
