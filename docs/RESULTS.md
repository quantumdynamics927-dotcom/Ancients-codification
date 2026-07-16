# Results Summary

**Last updated:** 2026-07-16
**Primary corpus:** ETCSRI (Sumerian Royal Inscriptions, ORACC CC0 1.0)
**Corpus stats:** 1,456 tablets, 68,083 signs, vocab=627
**Grouped experiment:** GROUP_A (n=10, divine-marker), GROUP_B (n=194, royal), GROUP_C (n=1,252, admin)
**Embedded samples:** hieroglyphs: 44 texts / 745 signs; cuneiform: 52 texts / 652 signs

**Pipeline status:** ETCSRI fully processed. RINAP download blocked (metadata-only zip from build-oracc;
oracc.iaas.upenn.edu API connection reset). See DATA_PROVENANCE.md for full details.

---

## Entropy Analysis

### Hieroglyphs (embedded corpus, n=745 signs, vocab=127)

| Metric | Value | English | Python | TCP | DNA |
|--------|-------|---------|--------|-----|-----|
| H1 | 3.903 bits | 4.816 | 4.850 | 7.990 | 2.000 |
| H2 | 6.446 bits | 5.167 | 6.070 | 9.190 | 3.990 |
| R1 | 0.226 | 0.019 | 0.054 | 0.062 | 0.000 |

**Result:** HIGHLY STRUCTURED — closest to TCP baseline (high redundancy)

### Cuneiform (embedded corpus, n=652 signs, vocab=89)

| Metric | Value | English | Python | TCP | DNA |
|--------|-------|---------|--------|-----|-----|
| H1 | 4.786 bits | 4.816 | 4.850 | 7.859 | 2.000 |
| H2 | 6.881 bits | 5.167 | 6.070 | 8.982 | 3.992 |
| R1 | 0.239 | 0.019 | 0.054 | 0.059 | 0.000 |

**Result:** HIGHLY STRUCTURED — closest to TCP baseline (high redundancy)

### Fair Comparison Checklist

- [x] Same unit (signs vs letters vs tokens): **signs**
- [x] Same R1 formula: `R = 1 - H / log2(V)`
- [x] n_signs and vocab size reported
- [x] Sproat caveat in caption

> **Sproat caveat (Sproat 2013):** Entropy and redundancy measures alone cannot distinguish linguistic from non-linguistic sequences. Information-theoretic measures are necessary but insufficient evidence for determining whether a writing system encodes language. These results are consistent with a constrained, classifier-rich writing system — not proof of engineered machine communication protocols.

### Scientific Wording

> "Under unigram/bigram sign models, both samples show substantially higher redundancy (R1 ≈ 0.23) than English (0.019) and Python (0.054) baselines, consistent with a constrained, classifier-rich writing system with heavy reuse of a closed sign inventory plus determinative-like structure. This is a descriptive result about statistical structure, not a claim that the scripts are network protocols."

---

## Determinative Graph

### Hieroglyphs (embedded corpus)

- **Graph:** 30 nodes, 13 edges
- **Determinative avg degree:** 1.62
- **Communities detected:** 20
- **Modularity:** 0.232
- **Semantic domains:** body, human, abstract, plant, element, object
- **Top hub determinatives:** D (body), A (human), K (object)

**Result:** Modularity 0.232 is a soft positive for domain clustering. Scale (30 nodes) is too small for strong claims. Run on ORACC corpus for hundreds of nodes.

### Cuneiform (embedded corpus)

- **Graph:** Larger embedded samples expanded to 30 tablets
- **Network type:** Determinative classifiers form domain-specific clusters

---

## Geometric Phi Scanner

### Idealized Gardiner Signs (n=30)

- **Phi matches:** 22/30 signs (73.3%)
- **Overall phi score:** 0.733
- **Quantum resonance (Sierpinski):** 0.467

### Image Controls (random baseline)

- **Random rectangles phi hit rate:** 3.9%
- **Gap vs idealized:** 73.3% - 3.9% = 69.4 percentage points

**Result:** Strong phi embedding in idealized geometry. Next step: scan real hieroglyph images and compare hit rate to 3.9% random control.

---

## CSA_OptMatcher Cognate Matching

### Benchmark Results

| Pair | Known | Correct | Accuracy | Notes |
|------|-------|---------|----------|-------|
| Ugaritic -> Hebrew | 8 | 8 | **100%** | Close Semitic family; noiseless |
| Sumerian -> Akkadian | 15 | 0 | **0%** | Different families; needs bilingual seeds |
| Egyptian -> Semitic | 9 | 1 | **11%** | Distant relation; high divergence |

### Interpretation

- **Ugaritic-Hebrew 100%:** CSA_OptMatcher handles close cognate pairs well (matches Tamburini 2025 results of 95.5% on same-script/different-language tasks)
- **Sumerian-Akkadian 0%:** Sumerian is a language isolate; Akkadian is East Semitic. Shared script (cuneiform) ≠ shared lexicon. CSA without bilingual seed pairs or logogram-sense alignment fails here.
- **Fix:** Use Oracc dual glosses / lexical lists for Sumerian-Akkadian; align on lemma/sense, not glyph identity.

### Scientific Wording

> "CSA_OptMatcher achieves 100% accuracy on 8 Ugaritic-Hebrew cognate pairs under noiseless conditions, consistent with published benchmarks. On Sumerian-Akkadian, accuracy drops to 0% because cognate matching requires shared lexical items across language families — shared script is insufficient without bilingual seed alignment."

---

## Blind Sophistication Stack (2026-07-16)

### Validation Results

#### ETCSRI Primary Corpus (Sumerian Royal Inscriptions, 80 tablets)

| Field | Value |
|-------|-------|
| n_signs | 14,257 |
| n_sequences / n_tablets | 80 |
| vocab_size | 401 |
| heldout_unit | tablet |
| field_stability_train | 0.323 |
| field_stability_test | 0.665 |
| generalization_delta (fields) | 0.342 |
| grammar_accept_train | 1.000 |
| grammar_accept_test | 1.000 |
| **grammar_accept_real_minus_shuffle_test** | **+0.000** (>0.10 required) |
| generalization_delta (grammar) | 0.000 |
| H1 95% CI | 6.06-6.97 |
| MI 95% CI | 2.58-3.89 |
| null_wins (shuffle/markov) | 2/4 |
| MI_delta vs shuffle | +1.228 |
| MI_delta vs markov | **-0.226** (real < markov) |
| R1_delta vs markov | +0.009 |
| baseline_correct | 1/6 (manifold UNCALIBRATED) |
| bonferroni_p | 0.0037 (passes) |

**Verdict: LEVEL_1_STRUCTURED_SYMBOLIC** (confidence: low, ceiling LEVEL_2)

**Interpretation:** ETCSRI data on 80 tablets shows:
- **MI_delta vs Markov = -0.226**: Real Sumerian sign sequences have *less* mutual information than a first-order Markov model trained on the same data. This means the sign stream is more predictable/patterned than a random sequence, but a simple bigram model captures more of the structure than the real data itself. This is consistent with a formulaic, template-heavy corpus (royal inscriptions follow conventions).
- **grammar_accept_real_minus_shuffle_test = +0.000**: The grammar does NOT survive token shuffle — formal grammar patterns are not distinguishable from what arises from repeating templates.
- **Markov null beaten on shuffle (MI_delta=+1.228) but not on Markov**: Structure exists beyond simple token shuffle, but is well-explained by a first-order bigram model.
- **manifold UNCALIBRATED**: novelty_score = NON_INTERPRETABLE
- **Geometry passes Bonferroni** (p=0.0037) but status=NULL_ON_CURRENT_DATA — embedded phi_rate not valid primary evidence without real image scanning.

**NOT SHOWN**: Machine language, Anunnaki authorship, ancient technology, phi encoding channel.

#### Hieroglyphs (embedded sample, n=335 signs, 40 sequences, vocab=11)

| Test | Result |
|------|--------|
| Held-out field stability (train) | 1.000 |
| Held-out field stability (test) | 1.000 |
| Generalization delta (fields) | 0.000 |
| Grammar accept real minus shuffle (test) | **pending re-run with new fields** |
| Null beats shuffle (MI_delta) | +0.913 — **PASS** |
| Null beats markov (MI_delta) | +0.116 — **PASS** |
| Baseline self-classification (6 controls) | 1/6 correct (Python only) |
| Bootstrap 95% CI on H1 | 3.267-3.420 |
| Bootstrap 95% CI on MI | 1.183-1.464 |
| Preregistered geometry (Bonferroni) | p=0.132 - does not pass |

**Verdict: LEVEL_2_FORMAL_GRAMMAR_CANDIDATE** (confidence: low)
- Held-out fields replicate on the embedded sample
- Null models beaten on MI - real sequences have structure beyond token shuffle
- Confidence low: baseline self-classification fails (manifold not discriminative for tiny synthetic corpora), no geometry signal
- **Revised interpretation:** The old "gap=0.000 PASS" reporting was misleading without the null-comparison threshold. The grammar_real_minus_shuffle metric is the correct test. Re-run required with updated validate.py to confirm.

#### Cuneiform (embedded sample, n=360 signs, 40 sequences, vocab=3)

| Test | Result |
|------|--------|
| Held-out field stability (train/test gap) | 0.000 - field inference unstable on 3-vocab |
| Null beats shuffle (MI_delta) | +0.031 - **FAIL** (margin) |
| Null beats markov (MI_delta) | +0.003 - **FAIL** |
| Preregistered geometry (Bonferroni) | p=0.444 - **FAIL** |

**Verdict: LEVEL_0_NO_EVIDENCE** (confidence: low)
- Tiny vocabulary (3 signs) makes the corpus a repeating pattern - trivial Markov chain
- Null models not robustly beaten
- Pipeline correctly rejects this corpus

---

## Anunnaki Target Experiment (GROUP_A/B/C Comparison)

**Date:** 2026-07-16
**Protocol:** BLIND_RESEARCH_PROTOCOL.md — groups assigned by curator, analyst receives only anonymized tokens+group_id
**Curation criteria:** GROUP_A: AN+KI co-occurrence (>=2 adjacent pairs); GROUP_B: >=50 signs, no divine markers; GROUP_C: <50 signs, no royal/divine markers
**Data sufficiency gate:** GROUP_A (n=10 tablets) is below the 100-tablet threshold — interpret with extreme caution.

### Group Statistics

| Group | n_tablets | n_signs | vocab | mean_len | Status |
|-------|-----------|---------|-------|----------|--------|
| GROUP_A (divine markers) | 10 | 15,488 | 429 | 1548.8 | Below threshold |
| GROUP_B (royal controls) | 194 | 24,349 | 508 | 125.5 | Sufficient |
| GROUP_C (administrative) | 1,252 | 28,246 | 411 | 22.6 | Sufficient |

### Blind Validation Results (held-out by tablet)

| Metric | GROUP_A | GROUP_B | GROUP_C | Notes |
|--------|---------|---------|---------|-------|
| Verdict | LEVEL_1 | LEVEL_1 | LEVEL_1 | All same level |
| grammar_real_minus_shuffle | +0.000 | +0.000 | +0.000 | None pass grammar gate |
| MI_delta vs Markov | -0.248 | -0.174 | **+0.124** | GROUP_C beats Markov |
| MI_delta vs shuffle | +1.137 | +1.481 | +2.139 | All beat shuffle |
| field_stability (test) | 0.638 | 0.629 | 0.722 | — |
| Bonferroni p | 0.0034 | 0.0052 | 0.0126 | All pass |
| manifold status | UNCALIBRATED | UNCALIBRATED | UNCALIBRATED | Ceiling LEVEL_2 |
| null_wins | 2/4 | 2/4 | 2/4 | — |

### Interpretation

- **All three groups return LEVEL_1_STRUCTURED_SYMBOLIC** — no group reaches formal grammar threshold.
- **GROUP_C (administrative) is the only group that beats the Markov null** (MI_delta=+0.124 > 0.05 threshold). This may reflect more consistent sequential structure in short formulaic administrative texts.
- **GROUP_A (n=10) is critically underpowered.** Ten tablets cannot support statistical inference. The 15,488 signs come from only 10 very long royal inscriptions.
- **No structural evidence for GROUP_A distinct from GROUP_B.** The divine-marker filter does not produce a measurable structural difference in the blind sign-stream analysis.
- **The manifold remains UNCALIBRATED across all groups** — novelty scores are NON_INTERPRETABLE until >=4/6 baseline self-classification passes.

### What This Does NOT Show

| Claim | Status |
|-------|--------|
| GROUP_A proves Anunnaki technology | **NOT SHOWN** — structural metrics do not distinguish GROUP_A from controls |
| Divine-marker texts have formal grammar | **NOT SHOWN** — grammar_real_minus_shuffle = 0.000 for all groups |
| GROUP_A is a distinct communication system | **NOT SHOWN** — only 10 tablets, manifold uncalibrated |

### Honest Conclusion
The blind pipeline finds no structural evidence that divine-marker texts (GROUP_A) differ from matched royal controls (GROUP_B) or administrative tablets (GROUP_C). The only statistically notable finding is that administrative tablets (GROUP_C) beat the Markov null — but this is explained by formulaic repetition, not formal grammar. All groups remain at LEVEL_1 with ceiling LEVEL_2.

---

## Validation Interpretation

### What the Results Show

The embedded hieroglyph sample (LEVEL_2, low confidence):
- **Real structure:** Field boundaries and grammar replicate on held-out tablets — genuine pattern, not overfitting
- **Beyond shuffle:** MI is higher in real than shuffled, meaning sign sequences are more predictable than random
- **Not yet generalizable:** The sample is small (335 signs, 11-vocab), hand-curated, and all sequences share a template. Baseline self-classification fails because the manifold can't distinguish 6 tiny synthetic corpora
- **Verdict wording:** "LEVEL_2_FORMAL_GRAMMAR_CANDIDATE means the pipeline finds regular structure in this specific sample — not that ancient Egyptian was a machine language"

The embedded cuneiform sample (LEVEL_0):
- **Too small to reach conclusions:** 3-sign vocabulary with 40 repeating sequences behaves like a formulaic list, not a formal language
- **Correct rejection:** The pipeline correctly does not claim structure for insufficient data

### What Is NOT Shown

| Claim | Status |
|-------|--------|
| Egyptian hieroglyphs are a machine language | **NOT SHOWN** — only LEVEL_2 on a small embedded sample |
| Anunnaki authored a formal communication protocol | **NOT SHOWN** — no author identity is inferable from sign structure |
| Geometry proves sacred encoding | **NOT SHOWN** — Bonferroni p=0.132, does not pass multiple-testing correction |
| This extends to real ORACC tablets | **NOT SHOWN** — requires validation on primary material |

### What Requires Next Steps

- [ ] Run on ORACC primary corpus (etcsri, rinap) with real sign IDs — the embedded samples are proof-of-concept only
- [ ] Obtain raw sign stream images for real phi measurements (not embedded approximations)
- [ ] Test on independent Sumerian and Egyptian corpora separately
- [ ] Improve baseline self-classification: the manifold needs larger control corpora for discriminative power
- [ ] Preregister geometry targets before running on primary image data

---

## Methodological Caveats

### Sproat (2013) — Entropy Cannot Distinguish Language
- Information-theoretic measures (H1, H2, MI, redundancy) are **necessary but insufficient**
- Cannot prove linguistic vs non-linguistic with entropy alone
- Must combine with: cognate matching, semantic plausibility, geometric analysis

### Selection Bias (Phi)
- Gardiner sign set was curated over decades by Egyptologists
- "73.3% phi embedding" may reflect sign selection that favored proportions
- Adversarial controls (random rectangles: 3.9%) partially address this
- Real image scan with bootstrap CI is the required next step

### Sample Size
- Current results use embedded samples (hieroglyphs: 745 signs, cuneiform: 652 signs)
- ORACC corpus target: 5,000–50,000+ signs
- All results are **preliminary** until run on full ORACC corpus

### CSA Limitations
- Degrades with noise; requires curated cognate seed lists
- Works on phonetically similar cognates (Ugaritic-Hebrew)
- Fails on divergent families without bilingual alignment

---

## Decision Rule

**Do not add another sophistication layer.** Freeze architecture, fix baseline calibration, acquire primary corpus data, then run the validation matrix unchanged.

The next meaningful milestone is:

```
CALIBRATION PASS:
  >= 80% baseline self-classification
  + primary dataset n >= 10,000 opaque sign tokens
  + cross-tablet held-out replication
  + result beats Markov null
```

Only after that should the pipeline be permitted to emit `LEVEL_2` or above.

---

## Honest Validation Status (Current Embedded Samples)

### Hieroglyph sample: LEVEL_2_FORMAL_GRAMMAR_CANDIDATE, low confidence
The result is limited by n=335 signs, vocabulary=11, incomplete baseline self-classification, and no surviving geometry signal.

### Cuneiform sample: LEVEL_0_NO_EVIDENCE
A 3-token, repeated-formula sample does not beat matched null models.

### Geometry: no statistically significant phi enrichment
No phi enrichment after preregistered +/-5% tolerance with Bonferroni and FDR correction.

### Conclusion
Current outputs do not support claims about machine-language authorship, Anunnaki origin, ancient technology, or a geometry channel.

### Next Steps (in priority order)
1. [x] ~~ETCSRI primary corpus~~ — Done: 1,456 tablets, 68,083 signs, LEVEL_1 result
2. [x] ~~Anunnaki GROUP_A/B/C experiment~~ — Done: all three groups LEVEL_1, GROUP_C beats Markov
3. [ ] Calibrate family manifold: >= 80% self-classification on control baselines (blocking LEVEL_3+)
4. [ ] Acquire RINAP corpus — network-enabled machine needed (curl -L https://build-oracc.museum.upenn.edu/json/rinap.zip)
5. [ ] Run on primary ORACC data with n >= 10,000 signs per group (GROUP_A currently n=10)
6. [ ] Confirm Markov null beaten before claiming formal grammar

---

## References
- Tamburini, F. (2025). CSA_OptMatcher. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2025.1581129
- Sproat, R. (2013). Fluency in Egyptian. *Journal of Egyptian History*. JSTOR: 24672182
- Xavier-de-Souza et al. (2010). Coupled Simulated Annealing. *IEEE Transactions on Systems*.
- ORACC. https://oracc.museum.upenn.edu/
- Veldhuis, N. ORACC-JSON. https://github.com/niekveldhuis/ORACC-JSON
- NDSS Symposium. "Auto Draft 342." https://www.ndss-symposium.org/ndss-paper/auto-draft-342/
- Berkeley EECS. "Automatic Protocol Reverse Engineering." https://people.eecs.berkeley.edu/~dawnsong/papers/2012%20Automatic%20Protocol%20Reverse%20Engineering.pdf
- PubMed. "Assessing chance in phi ratio studies." https://pubmed.ncbi.nlm.nih.gov/41587867/
- PMC. "Statistical validation of protocol inference." https://pmc.ncbi.nlm.nih.gov/articles/PMC6337927/
