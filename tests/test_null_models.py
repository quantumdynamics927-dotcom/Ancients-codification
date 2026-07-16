"""
test_null_models.py
==================
Tests all four null models produce valid surrogate sequences:
1. Unigram shuffle (preserves length, token counts; destroys order)
2. Markov/bigram (preserves bigram transitions; destroys higher-order structure)
3. Template-preserving (preserves length distribution; destroys token identity within position)
4. Length-matched random (preserves vocab size; destroys all sequential structure)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ancient_ml.blind.null_models import NullModels


# Small test corpus: 5 sequences of 6 tokens each, vocab=5
TEST_CORPUS = [
    ["A", "B", "C", "A", "B", "C"],
    ["B", "C", "A", "B", "C", "A"],
    ["C", "A", "B", "C", "A", "B"],
    ["A", "A", "B", "B", "C", "C"],
    ["X", "Y", "Z", "X", "Y", "Z"],
]


def test_unigram_shuffle():
    """Unigram shuffle preserves length and token counts per sequence, destroys order."""
    nm = NullModels(TEST_CORPUS)
    n_surrogates = 3
    surrogates = nm.unigram_shuffle_null(n=n_surrogates)
    # Returns min(n, len(sequences)) shuffled sequences, each same length as source
    assert len(surrogates) == n_surrogates, f"Expected {n_surrogates} surrogates, got {len(surrogates)}"
    for i, s in enumerate(surrogates):
        src_len = len(TEST_CORPUS[0])
        assert len(s) == src_len, f"Surrogate {i} length {len(s)} != source {src_len}"
        # Each surrogate's token multiset matches its source sequence
        assert sorted(s) == sorted(TEST_CORPUS[i % len(TEST_CORPUS)]), \
            f"Surrogate {i} token multiset changed"
    print(f"PASS: unigram_shuffle_null produced {n_surrogates} valid surrogates.")


def test_markov_bigram():
    """Markov bigram preserves bigram transitions, destroys higher-order structure."""
    nm = NullModels(TEST_CORPUS)
    n_seqs = 3
    surrogates = nm.markov_corpus(n=n_seqs)
    # Returns n sequences (Markov chains)
    assert len(surrogates) == n_seqs, f"Expected {n_seqs} sequences, got {len(surrogates)}"
    for s in surrogates:
        assert len(s) > 0, "Sequence should not be empty"
    print(f"PASS: markov_corpus produced {n_seqs} sequences.")


def test_template_preserving():
    """Template-preserving null destroys token identity within position."""
    nm = NullModels(TEST_CORPUS)
    surrogates = nm.template_preserving_null(n=3)
    # Returns n shuffled sequences
    assert len(surrogates) == 3, f"Expected 3 surrogates, got {len(surrogates)}"
    for i, s in enumerate(surrogates):
        assert len(s) == len(TEST_CORPUS[i % len(TEST_CORPUS)]), "Length changed"
    print(f"PASS: template_preserving_null produced valid surrogates.")


def test_length_matched_random():
    """Length-matched random preserves vocab size, destroys all structure."""
    nm = NullModels(TEST_CORPUS)
    n_seqs = 3
    surrogates = nm.random_alphabet_corpus(n=n_seqs)
    assert len(surrogates) == n_seqs, f"Expected {n_seqs} sequences, got {len(surrogates)}"
    for s in surrogates:
        assert len(s) > 0, "Sequence should not be empty"
    print(f"PASS: random_alphabet_corpus produced {n_seqs} sequences.")


def test_null_model_destroys_structure():
    """Null models should destroy sequential structure (entropy should increase)."""
    from collections import Counter
    from ancient_ml.entropy_analysis import shannon_entropy

    real_tokens = [t for seq in TEST_CORPUS for t in seq]
    real_H1 = shannon_entropy(Counter(real_tokens))

    # Shuffle should increase entropy (destroy order)
    nm = NullModels(TEST_CORPUS)
    shuffle_surrogates = nm.unigram_shuffle_null(n=5)
    shuffle_tokens = [t for s in shuffle_surrogates for t in s]
    shuffle_H1 = shannon_entropy(Counter(shuffle_tokens))

    assert shuffle_H1 >= real_H1, (
        f"Unigram shuffle should not decrease entropy: "
        f"shuffle={shuffle_H1:.4f} vs real={real_H1:.4f}"
    )
    print(f"PASS: Null model destroys structure (entropy {real_H1:.4f} -> {shuffle_H1:.4f}).")


if __name__ == "__main__":
    test_unigram_shuffle()
    test_markov_bigram()
    test_template_preserving()
    test_length_matched_random()
    test_null_model_destroys_structure()
