# Results Summary

**Last updated:** 2026-07-15
**Corpus:** Embedded samples (hieroglyphs: 44 texts / 745 signs; cuneiform: 52 texts / 652 signs)
**Note:** Real conclusions require ORACC corpus. Run on your machine: `python run_pipeline.py --all --corpus oracc --project etcsri`

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
| Ugaritic → Hebrew | 8 | 8 | **100%** | Close Semitic family; noiseless |
| Sumerian → Akkadian | 15 | 0 | **0%** | Different families; needs bilingual seeds |
| Egyptian → Semitic | 9 | 1 | **11%** | Distant relation; high divergence |

### Interpretation

- **Ugaritic-Hebrew 100%:** CSA_OptMatcher handles close cognate pairs well (matches Tamburini 2025 results of 95.5% on same-script/different-language tasks)
- **Sumerian-Akkadian 0%:** Sumerian is a language isolate; Akkadian is East Semitic. Shared script (cuneiform) ≠ shared lexicon. CSA without bilingual seed pairs or logogram-sense alignment fails here.
- **Fix:** Use Oracc dual glosses / lexical lists for Sumerian-Akkadian; align on lemma/sense, not glyph identity.

### Scientific Wording

> "CSA_OptMatcher achieves 100% accuracy on 8 Ugaritic-Hebrew cognate pairs under noiseless conditions, consistent with published benchmarks. On Sumerian-Akkadian, accuracy drops to 0% because cognate matching requires shared lexical items across language families — shared script is insufficient without bilingual seed alignment."

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

## References

- Tamburini, F. (2025). CSA_OptMatcher. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2025.1581129
- Sproat, R. (2013). Fluency in Egyptian. *Journal of Egyptian History*. JSTOR: 24672182
- Xavier-de-Souza et al. (2010). Coupled Simulated Annealing. *IEEE Transactions on Systems*.
- ORACC. https://oracc.museum.upenn.edu/
- Veldhuis, N. ORACC-JSON. https://github.com/niekveldhuis/ORACC-JSON
