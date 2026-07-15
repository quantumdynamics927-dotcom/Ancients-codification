"""
Fingerprint Module: L0 Structural Fingerprint
==========================================
Core statistical fingerprints from raw sign streams.
H1, H2, H3, redundancy, mutual information, Zipf slope.
NO semantic knowledge — purely statistical.

Based on:
- Shannon entropy (Shannon 1948)
- Mutual Information for dependency detection
- Zipf's law for natural vs designed language discrimination

Usage:
    from blind.fingerprint import compute_fingerprints, FingerprintVector
    fp = compute_fingerprints(sequences)
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import numpy as np


@dataclass
class FingerprintVector:
    """
    Complete statistical fingerprint of a sign stream.
    All values are derivable from raw signs without any dictionary.
    """
    # Counts
    n_signs: int = 0
    n_sequences: int = 0
    vocab_size: int = 0

    # Entropy measures
    h1: float = 0.0       # Unigram entropy H(X)
    h2: float = 0.0       # Bigram entropy H(X_2 | X_1)
    h3: float = 0.0       # Trigram entropy H(X_3 | X_1, X_2)
    h_conditional: float = 0.0  # H(X | X_prev) — same as H2
    mutual_info: float = 0.0    # I(X; X_prev) = H(X) - H(X|X_prev)

    # Redundancy
    r1: float = 0.0       # R1 = 1 - H1/log2(V)
    r2: float = 0.0       # R2 = 1 - H2/log2(V)

    # Zipf's law: log(rank) vs log(frequency) slope
    # Natural language: ~-1.0; random: ~0; structured code: steeper
    zipf_alpha: float = 0.0
    zipf_intercept: float = 0.0

    # Positional roles (induced)
    initial_signs: Set[str] = field(default_factory=set)
    final_signs: Set[str] = field(default_factory=set)
    fixed_signs: Set[str] = field(default_factory=set)
    header_signs: Set[str] = field(default_factory=set)  # Frequent at position 0
    footer_signs: Set[str] = field(default_factory=set)  # Frequent at final position

    # Periodic markers
    periodic_signs: Dict[str, float] = field(default_factory=dict)

    # Top patterns
    top_unigrams: List[Tuple[str, int]] = field(default_factory=list)
    top_bigrams: List[Tuple[str, int]] = field(default_factory=list)
    top_trigrams: List[Tuple[str, int]] = field(default_factory=list)

    # Repetition ratio
    repetition_ratio: float = 0.0  # fraction of n-grams appearing >1x

    # Mandatory transitions
    mandatory_pairs: List[Tuple[str, str]] = field(default_factory=list)
    forbidden_pairs: List[Tuple[str, str]] = field(default_factory=list)


def shannon_entropy(counts: Counter, smooth: float = 1e-10) -> float:
    """H = -sum(p * log2(p))"""
    if not counts:
        return 0.0
    total = sum(counts.values())
    probs = [(c + smooth) / (total + smooth * len(counts)) for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def compute_fingerprints(sequences: List[List[str]], smooth: float = 1e-10) -> FingerprintVector:
    """
    Compute the complete statistical fingerprint from raw sign sequences.
    NO semantic knowledge used — purely positional and statistical.
    """
    flat = [s for seq in sequences for s in seq]
    n_signs = len(flat)
    vocab = set(flat)
    vocab_size = len(vocab)

    fp = FingerprintVector()
    fp.n_signs = n_signs
    fp.n_sequences = len(sequences)
    fp.vocab_size = vocab_size

    # ── Unigram entropy H1 ──────────────────────────────────────────
    unigram_counts = Counter(flat)
    fp.h1 = shannon_entropy(unigram_counts, smooth)

    # ── Bigram entropy H2 = H(X | X_prev) ───────────────────────────
    bigrams = [tuple(flat[i:i+2]) for i in range(len(flat)-1)]
    bigram_counts = Counter(bigrams)
    fp.h2 = shannon_entropy(bigram_counts, smooth)

    # ── Trigram entropy H3 = H(X | X_prev, X_prev2) ─────────────────
    trigrams = [tuple(flat[i:i+3]) for i in range(len(flat)-2)]
    trigram_counts = Counter(trigrams)
    fp.h3 = shannon_entropy(trigram_counts, smooth)

    # ── Conditional entropy H(X|X_prev) via Bayes ───────────────────
    prev_counts = Counter(flat[:-1])
    total_prev = len(flat) - 1
    conditional_h = 0.0
    for prev, c_prev in prev_counts.items():
        p_prev = c_prev / total_prev
        # P(x | prev) for all x
        h_given_prev = 0.0
        for (p, curr), c_bigram in bigram_counts.items():
            if p == prev:
                p_curr_given_prev = c_bigram / c_prev
                if p_curr_given_prev > 0:
                    h_given_prev -= p_curr_given_prev * np.log2(p_curr_given_prev + smooth)
        conditional_h += p_prev * h_given_prev
    fp.h_conditional = conditional_h

    # ── Mutual information I(X; X_prev) ────────────────────────────
    fp.mutual_info = fp.h1 - conditional_h

    # ── Redundancy ─────────────────────────────────────────────────
    h_max = np.log2(max(vocab_size, 2))
    fp.r1 = 1.0 - (fp.h1 / h_max)
    fp.r2 = 1.0 - (fp.h2 / h_max)

    # ── Zipf's law: rank-frequency ───────────────────────────────────
    if unigram_counts:
        freqs = np.array(sorted(unigram_counts.values(), reverse=True))
        ranks = np.arange(1, len(freqs) + 1)
        # log(freq) = intercept + alpha * log(rank)
        log_freqs = np.log(freqs.astype(float) + 1)
        log_ranks = np.log(ranks.astype(float))
        # Linear regression
        valid = np.isfinite(log_freqs) & np.isfinite(log_ranks)
        if valid.sum() > 2:
            cov = np.cov(log_ranks[valid], log_freqs[valid])
            var = np.var(log_ranks[valid])
            if var > 1e-10:
                fp.zipf_alpha = cov[0, 1] / var
                fp.zipf_intercept = np.mean(log_freqs[valid]) - fp.zipf_alpha * np.mean(log_ranks[valid])
            else:
                fp.zipf_alpha = 0.0
                fp.zipf_intercept = np.mean(log_freqs)
        else:
            fp.zipf_alpha = 0.0
            fp.zipf_intercept = 0.0

    # ── Positional roles ────────────────────────────────────────────
    fp.initial_signs, fp.final_signs = set(), set()
    pos_counts = defaultdict(Counter)  # pos -> sign -> count

    for seq in sequences:
        if not seq:
            continue
        fp.initial_signs.add(seq[0])
        fp.final_signs.add(seq[-1])
        for pos, sign in enumerate(seq):
            pos_counts[pos][sign] += 1

    # Fixed-position signs: same sign at same position in all sequences
    for pos, sign_counts_pos in pos_counts.items():
        if len(sign_counts_pos) == 1:
            fp.fixed_signs.add(next(iter(sign_counts_pos)))

    # Header/footer signs: frequent at position 0 or last
    total_seqs = len(sequences)
    if 0 in pos_counts:
        fp.header_signs = {s for s, c in pos_counts[0].items() if c / total_seqs > 0.5}
    max_pos = max(pos_counts.keys()) if pos_counts else 0
    if max_pos > 0 and max_pos in pos_counts:
        fp.footer_signs = {s for s, c in pos_counts[max_pos].items() if c / total_seqs > 0.5}

    # ── Top patterns ─────────────────────────────────────────────────
    fp.top_unigrams = unigram_counts.most_common(20)
    fp.top_bigrams = bigram_counts.most_common(20)
    fp.top_trigrams = trigram_counts.most_common(20)

    # Repetition ratio: fraction of trigrams appearing more than once
    n_trigrams_total = len(trigrams)
    if n_trigrams_total > 0:
        n_repeated = sum(1 for c in trigram_counts.values() if c > 1)
        fp.repetition_ratio = n_repeated / len(trigram_counts)

    # ── Mandatory & forbidden pairs ──────────────────────────────────
    prev_to_next = defaultdict(Counter)
    all_transitions = set()
    for prev, curr in bigrams:
        all_transitions.add((prev, curr))
        prev_to_next[prev][curr] += 1

    for prev, next_signs in prev_to_next.items():
        total = sum(next_signs.values())
        for curr, count in next_signs.items():
            ratio = count / total
            if ratio > 0.9:
                fp.mandatory_pairs.append((prev, curr))

    # Forbidden: never observed despite both signs being common
    vocab_list = list(vocab)
    all_pairs = {(a, b) for a in vocab_list for b in vocab_list}
    observed_pairs = all_transitions
    never_observed = all_pairs - observed_pairs
    # Only keep pairs where both signs are frequent
    min_freq = 5
    freq_signs = {s for s, c in unigram_counts.items() if c >= min_freq}
    fp.forbidden_pairs = [
        (p, c) for p, c in sorted(never_observed)
        if p in freq_signs and c in freq_signs
    ][:100]  # Cap at 100

    # ── Periodic markers ─────────────────────────────────────────────
    fp.periodic_signs = _detect_periodic_markers(flat)

    return fp


def _detect_periodic_markers(flat: List[str], max_period: int = 20) -> Dict[str, float]:
    """Detect signs appearing at regular intervals — candidate frame markers."""
    sign_positions = defaultdict(list)
    for i, sign in enumerate(flat):
        sign_positions[sign].append(i)

    periodic_scores = {}
    for sign, positions in sign_positions.items():
        if len(positions) < 3:
            continue
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        if not intervals:
            continue
        interval_counts = Counter(intervals)
        most_common_interval, most_common_count = interval_counts.most_common(1)[0]
        score = most_common_count / len(intervals)
        if most_common_interval <= max_period:
            periodic_scores[sign] = score

    return periodic_scores


def fingerprint_to_array(fp: FingerprintVector) -> np.ndarray:
    """
    Convert fingerprint to feature array for family_manifold.
    All features normalized to [0, 1] range.
    """
    return np.array([
        fp.n_signs / 10000,
        fp.n_sequences / 1000,
        fp.vocab_size / 500,
        fp.h1 / 8,
        fp.h2 / 8,
        fp.h3 / 8,
        fp.h_conditional / 8,
        fp.mutual_info / 5,
        fp.r1,
        fp.r2,
        min(abs(fp.zipf_alpha), 3) / 3,
        len(fp.header_signs) / max(fp.vocab_size, 1),
        len(fp.footer_signs) / max(fp.vocab_size, 1),
        len(fp.fixed_signs) / max(fp.vocab_size, 1),
        sum(fp.periodic_signs.values()) / max(len(fp.periodic_signs), 1),
        fp.repetition_ratio,
        len(fp.mandatory_pairs) / 100,
        len(fp.forbidden_pairs) / 100,
    ])


def interpret_fingerprint(fp: FingerprintVector) -> List[str]:
    """Generate human-readable interpretation of fingerprint."""
    lines = []
    lines.append(f"n_signs={fp.n_signs}, vocab={fp.vocab_size}, n_seq={fp.n_sequences}")
    lines.append(f"H1={fp.h1:.3f}, H2={fp.h2:.3f}, H3={fp.h3:.3f}")
    lines.append(f"R1={fp.r1:.3f}, R2={fp.r2:.3f}, MI={fp.mutual_info:.3f}")
    lines.append(f"Zipf α={fp.zipf_alpha:.3f}")
    lines.append(f"header_signs={sorted(fp.header_signs)[:10]}")
    lines.append(f"footer_signs={sorted(fp.footer_signs)[:10]}")
    lines.append(f"fixed_signs={sorted(fp.fixed_signs)[:10]}")
    lines.append(f"periodic={dict(list(fp.periodic_signs.items())[:5])}")
    lines.append(f"mandatory_pairs={fp.mandatory_pairs[:10]}")
    lines.append(f"repetition_ratio={fp.repetition_ratio:.3f}")
    return lines
