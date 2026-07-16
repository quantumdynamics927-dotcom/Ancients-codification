# Preregistered Metrics

**Version:** 1.0
**Date:** 2026-07-16
**Status:** All thresholds fixed before any primary data analysis.

This document records every metric, threshold, and decision rule used in the
validation pipeline — all fixed prior to examining primary corpus results.

---

## Data Sufficiency Gates

| Metric | Threshold | Source |
|-------|-----------|--------|
| n_tokens | >= 10,000 | Research design decision |
| n_containers/tablets | >= 100 | Research design decision |
| held-out split unit | tablet (never line) | Protocol requirement |
| vocabulary size | Any (reported, not gated) | — |

---

## L0 Fingerprint Metrics

These are computed on the full corpus (train+test combined) and reported
in the bootstrap CI.

| Metric | Formula | Bootstrap CI | Reported as |
|--------|---------|-------------|-------------|
| H1 | Shannon entropy of unigram distribution | Yes, 95% | bits |
| H2 | Shannon entropy of bigram distribution | Yes, 95% | bits |
| H3 | Trigram conditional entropy | No | bits |
| R1 | 1 - H1/log2(vocab) | Yes, 95% | redundancy ratio |
| R2 | 1 - H2/log2(vocab) | No | redundancy ratio |
| Mutual Information (MI) | H1 - H2(cond) | Yes, 95% | bits |
| Zipf alpha | MLE power-law exponent | No | slope |
| Repetition ratio | singleton tokens / total tokens | No | ratio |

---

## L1 Protocol Field Segmentation

| Metric | Threshold | Pass condition |
|--------|-----------|---------------|
| field_stability_mean | [0, 1] | Higher = more stable fields across sequences |
| generalization_delta_field_stability | < 0.20 | Train-test gap < 20pp |
| n_fields | Any | Reported; compared across train/test |
| fields_match | boolean | n_fields_train ≈ n_fields_test (within 0.5) |

---

## L2 Role Typing

| Metric | Threshold | Notes |
|--------|-----------|-------|
| n_types | >= 2 | Minimum to compute type-transition entropy |
| type_transition_entropy | [0, log(n_types)] | DBSCAN clustering, eps=0.7 |

---

## L3 Grammar Induction

| Metric | Threshold | Pass condition |
|--------|-----------|---------------|
| acceptance_ratio | [0, 1] | Fraction of sequences accepted by induced automaton |
| holdout_ratio | [0, 1] | Acceptance on held-out sequences |
| shuffle_ratio | [0, 1] | Acceptance on shuffled sequences |
| **grammar_accept_real_minus_shuffle** | **> 0.10** | **Real minus shuffled on TEST set. Primary grammar gate.** |
| generalization_delta_grammar | < 0.20 | Train-test gap < 20pp |
| n_rules | >= 1 | At least one PCFG rule induced |

**Why >0.10:** A shuffle destroys all sequential structure. If the real corpus
still has acceptance > 10pp above shuffle on the held-out test set, that gap
represents structure that survives token permutation. This is the primary test
for whether a formal grammar — not just repetition — is present.

**Preregistered:** Fixed before data inspection. Rationale: 0.10 is a conservative
threshold that corresponds to at least one in ten sequences being rejected by
shuffle but accepted by the real automaton.

---

## L4 Complexity

| Metric | Threshold | Notes |
|--------|-----------|-------|
| zlib_ratio | [0, 1] | Compressed size / original size |
| lzma_ratio | [0, 1] | LZMA compression ratio |
| NCD vs Python | [0, 2] | Normalized compression distance |
| NCD vs Random | [0, 2] | Real should be less than random |

---

## L5 Family Manifold

| Metric | Threshold | Notes |
|--------|-----------|-------|
| novelty_score | min_dist / median_dist | > 1.1 = far from all known families |
| Family self-classification | >= 80% (4/6) | **Calibration gate** |
| novelty_score_status | INTERPRETABLE / NON_INTERPRETABLE | Depends on calibration |

**Calibration gate:** The manifold must achieve >= 4/6 baseline controls
correctly self-classifying before novelty can be interpreted.
Until then: novelty = NON_INTERPRETABLE, ceiling = LEVEL_2.

**Baseline families:** Python, Rust, Java Bytecode, JSON, XML, SQL, TCP, DNA, English, Random.
Each family requires >= 10 independently sampled examples.
Tokenization must match (source code tokens vs sign IDs are NOT comparable).

---

## L6 Geometry Channel

| Metric | Threshold | Pass condition |
|--------|-----------|---------------|
| phi targets | 1.618, 0.618 | Fixed before data inspection |
| tolerance | ±5% | Fixed before data inspection |
| phi_rate | > 0.50 | At least 50% of signs with phi-proportional geometry |
| bonferroni_p | < 0.05 | Family-wise error rate controlled |
| fdr | < 0.05 | False discovery rate controlled |
| **geometry_status** | **NULL_ON_CURRENT_DATA** | Until primary image scan passes |

**Status rule:** Geometry begins as NULL_ON_CURRENT_DATA. It only becomes
an active claim when:
1. Primary images (not embedded approximations) are scanned
2. Preregistered targets/tolerance are fixed before scanning
3. Bonferroni + FDR both pass
4. Result replicates across independent image sets

---

## L7 Quantum Topology (Experimental)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| sierpinski_resonance | [0, 1] | Topological match to Sierpinski gasket |
| merkaba_resonance | [0, 1] | Topological match to Merkaba star |
| Status | HYPOTHESIS_GENERATING | Not a claim — exploratory only |

---

## Null Models

Four null models, all run at 20 iterations unless noted:

| Null | Preserves | Destroys | Purpose |
|------|-----------|----------|---------|
| Token shuffle (unigram) | Sequence length, token counts | Sequential order | First-order structure |
| Markov/bigram | Bigram transitions | Higher-order structure | Local transition model |
| Template-preserving | Sequence lengths | Token identity within position | Content vs position |
| Length-matched random | Vocab size | All structure | Corpus-size effects |

**Markov null threshold:** Real must beat Markov on MI by > 0.05 to claim
structure beyond first-order transitions.

---

## Verdict Ladder

```
LEVEL_0_NO_EVIDENCE
  <- null_wins < 2

LEVEL_1_STRUCTURED_SYMBOLIC
  <- null_wins >= 2
  <- grammar_real_minus_shuffle <= 0.10

LEVEL_2_FORMAL_GRAMMAR_CANDIDATE
  <- grammar_real_minus_shuffle > 0.10
  <- generalization_delta_grammar < 0.20
  <- (manifold UNCALIBRATED: ceiling LEVEL_2)

LEVEL_3_NOVEL_FORMAL_SYSTEM_CANDIDATE
  <- Level 2
  <- manifold CALIBRATED (>= 4/6)
  <- novelty > 1.1

LEVEL_4_DUAL_CHANNEL_ENCODING_CANDIDATE
  <- Level 3
  <- geometry passes Bonferroni + FDR + phi_rate > 0.50
```

**Ceiling rule:** When manifold is UNCALIBRATED, verdict ceiling is LEVEL_2.
LEVEL_3/4 are blocked until calibration passes.

---

## What Cannot Be Inferred

No verdict level permits these claims:
- Author identity
- Anunnaki origin or extraterrestrial authorship
- Historical purpose or intended meaning
- Communication protocol vs ritual notation
- Exact dating or cultural attribution

---

## Revision Log

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-07-16 | All thresholds preregistered before primary data analysis |

