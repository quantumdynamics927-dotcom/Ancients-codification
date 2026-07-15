"""
Null Models: Adversarial Controls for Structure Verification
========================================================
Generate shuffled/markov/random baselines to test whether structure
is real or could arise by chance. Mandatory for credibility.

Null models:
1. Token shuffle — preserve frequency distribution, destroy sequential structure
2. Markov match — preserve bigram distribution, random higher-order structure
3. Random alphabet — same vocab size, uniform random tokens
4. Python control — known code language baseline

Usage:
    from blind.null_models import NullModels
    nm = NullModels(sequences)
    shuffles = nm.shuffled_corpus(n=10)
    markovs = nm.markov_corpus(n=10)
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import random
import numpy as np


@dataclass
class NullMetrics:
    """Metrics for one null model."""
    name: str
    description: str
    # How does each fingerprint metric compare to real?
    h1_delta: float        # Real H1 - Null H1 (positive = real is more structured)
    h2_delta: float
    r1_delta: float        # Real R1 - Null R1
    mi_delta: float
    periodic_delta: float  # Real periodic strength - Null periodic strength
    grammar_gap: float     # Real holdout_accept - Null holdout_accept
    zlib_delta: float      # Real zlib_ratio - Null zlib_ratio (positive = real more compressible)


class NullModels:
    """
    Generate null model baselines and compute structure gaps.
    If real ≫ null on all metrics → strong evidence of designed structure.
    """

    def __init__(self, sequences: List[List[str]], seed: int = 42):
        self.sequences = sequences
        self.seed = seed
        self._flat = [s for seq in sequences for s in seq]
        self._bigram_counts = Counter(zip(self._flat[:-1], self._flat[1:]))
        self._unigram_counts = Counter(self._flat)
        self._n_seq = len(sequences)
        self._rng = random.Random(seed)

    # ── Null corpus generators ─────────────────────────────────────────

    def shuffled_corpus(self, n: int = None) -> List[List[str]]:
        """
        Shuffle each sequence independently.
        Preserves per-sequence length and sign frequency, destroys order.
        """
        n = n or self._n_seq
        result = []
        for seq in self.sequences[:n]:
            shuffled = seq.copy()
            local_rng = random.Random(self.seed)
            local_rng.shuffle(shuffled)
            result.append(shuffled)
        return result

    def markov_corpus(self, n: int = None) -> List[List[List[str]]]:
        """
        Generate sequences using learned bigram transition probabilities.
        Preserves bigram distribution, randomizes higher-order structure.
        """
        n = n or self._n_seq

        # Build transition probs
        transitions = defaultdict(Counter)
        for (prev, curr), count in self._bigram_counts.items():
            transitions[prev][curr] = count

        # Start distribution
        starts = Counter(seq[0] for seq in self.sequences if seq)

        result = []
        for i in range(n):
            seq_len = len(self.sequences[i % self._n_seq])
            seq = []
            # Pick start
            start_options = list(starts.keys())
            start_weights = list(starts.values())
            if start_options:
                current = self._rng.choices(start_options, weights=start_weights, k=1)[0]
            else:
                current = self._rng.choice(list(self._unigram_counts.keys()))
            seq.append(current)

            # Generate
            for _ in range(seq_len - 1):
                next_options = list(transitions[current].keys())
                next_weights = list(transitions[current].values())
                if next_options:
                    current = self._rng.choices(next_options, weights=next_weights, k=1)[0]
                else:
                    current = self._rng.choice(list(self._unigram_counts.keys()))
                seq.append(current)
            result.append(seq)

        return result

    def random_alphabet_corpus(self, n: int = None) -> List[List[str]]:
        """
        Generate sequences with random tokens from same alphabet size.
        Uniform distribution, no sequential structure.
        """
        n = n or self._n_seq
        vocab = list(self._unigram_counts.keys())
        vocab_size = len(vocab)

        result = []
        for i in range(n):
            seq_len = len(self.sequences[i % self._n_seq])
            seq = [self._rng.choice(vocab) for _ in range(seq_len)]
            result.append(seq)

        return result

    def python_control_corpus(self, n: int = None) -> List[List[str]]:
        """
        Python source code as positive control.
        Known structured language baseline.
        """
        n = n or self._n_seq
        keywords = ["def", "return", "if", "else", "for", "in", "import", "class", "self", "None"]
        identifiers = ["x", "y", "data", "result", "val", "item", "key", "n", "i"]
        operators = ["=", "+", "-", "*", "/", "==", "!=", "(", ")", ":", ",", "."]
        constants = ["0", "1", "2", "10", "True", "False"]

        result = []
        for i in range(n):
            template = i % 4
            if template == 0:
                seq = ["def", identifiers[i % len(identifiers)], "(",
                       identifiers[(i+1) % len(identifiers)], ")", ":",
                       "return", identifiers[i % len(identifiers)], "+", "1"]
            elif template == 1:
                seq = ["if", identifiers[i % len(identifiers)], "==",
                       constants[i % len(constants)], ":",
                       identifiers[(i+2) % len(identifiers)], "=", "1"]
            elif template == 2:
                seq = ["for", identifiers[i % len(identifiers)], "in",
                       "range", "(", constants[(i+1) % len(constants)], ")", ":",
                       "print", "(", identifiers[i % len(identifiers)], ")"]
            else:
                seq = ["class", identifiers[i % len(identifiers)], ":",
                       "def", "__init__", "(", "self", ")", ":", "self", ".",
                       identifiers[(i+1) % len(identifiers)], "=", "None"]
            result.append(seq)

        return result[:n]

    # ── Comparative metrics ─────────────────────────────────────────────

    def compute_null_metrics(
        self,
        real_fp,
        null_fp,
        real_grammar=None,
        null_grammar=None,
    ) -> NullMetrics:
        """
        Compare real fingerprint to null model fingerprint.
        real_fp: FingerprintVector from real corpus
        null_fp: FingerprintVector from null model corpus
        """
        return NullMetrics(
            name="null_comparison",
            description="Real vs null model differences",
            h1_delta=real_fp.h1 - null_fp.h1,
            h2_delta=real_fp.h2 - null_fp.h2,
            r1_delta=real_fp.r1 - null_fp.r1,
            mi_delta=real_fp.mutual_info - null_fp.mutual_info,
            periodic_delta=(
                sum(real_fp.periodic_signs.values()) / max(len(real_fp.periodic_signs), 1) -
                sum(null_fp.periodic_signs.values()) / max(len(null_fp.periodic_signs), 1)
            ),
            grammar_gap=(
                (real_grammar.acceptance_ratio - real_grammar.shuffle_ratio)
                if real_grammar and null_grammar else 0.0
            ),
            zlib_delta=real_fp.h1 - null_fp.h1,  # Placeholder: use complexity module
        )

    def full_null_comparison(self, real_fp, null_results: Dict[str, List]) -> Dict[str, NullMetrics]:
        """
        Compare real fingerprint against all null model results.
        null_results: dict of null_name -> list of fingerprint vectors
        """
        comparisons = {}
        for null_name, fps in null_results.items():
            if not fps:
                continue
            # Average across null runs
            avg_fp = self._average_fingerprints(fps)
            comparisons[null_name] = self.compute_null_metrics(real_fp, avg_fp)
        return comparisons

    def _average_fingerprints(self, fps: List) -> 'FingerprintVector':
        """Average multiple fingerprint vectors."""
        from blind.fingerprint import FingerprintVector
        avg = FingerprintVector()
        if not fps:
            return avg

        avg.n_signs = int(np.mean([fp.n_signs for fp in fps]))
        avg.n_sequences = int(np.mean([fp.n_sequences for fp in fps]))
        avg.vocab_size = int(np.mean([fp.vocab_size for fp in fps]))
        avg.h1 = float(np.mean([fp.h1 for fp in fps]))
        avg.h2 = float(np.mean([fp.h2 for fp in fps]))
        avg.h_conditional = float(np.mean([fp.h_conditional for fp in fps]))
        avg.mutual_info = float(np.mean([fp.mutual_info for fp in fps]))
        avg.r1 = float(np.mean([fp.r1 for fp in fps]))
        avg.r2 = float(np.mean([fp.r2 for fp in fps]))
        avg.zipf_alpha = float(np.mean([fp.zipf_alpha for fp in fps]))
        avg.repetition_ratio = float(np.mean([fp.repetition_ratio for fp in fps]))

        return avg


def null_summary(null_metrics: Dict[str, NullMetrics]) -> List[str]:
    """Human-readable summary of null model comparisons."""
    lines = []
    for name, m in null_metrics.items():
        lines.append(f"\n{name}:")
        lines.append(f"  H1_delta: {m.h1_delta:+.3f}  (real - null)")
        lines.append(f"  H2_delta: {m.h2_delta:+.3f}")
        lines.append(f"  R1_delta: {m.r1_delta:+.3f}")
        lines.append(f"  MI_delta: {m.mi_delta:+.3f}")
        lines.append(f"  periodic_delta: {m.periodic_delta:+.3f}")
        lines.append(f"  grammar_gap: {m.grammar_gap:+.3f}")
        # Verdict
        real_wins = sum([
            m.r1_delta > 0.05,
            m.mi_delta > 0.05,
            m.periodic_delta > 0.1,
            m.grammar_gap > 0.1,
        ])
        if real_wins >= 3:
            lines.append(f"  VERDICT: REAL STRUCTURE (null rejected on {real_wins}/4 metrics)")
        else:
            lines.append(f"  VERDICT: WEAK/NO STRUCTURE (null not rejected)")
    return lines
