# Ancients Codification

## Project Overview
Research framework treating ancient writing systems (Egyptian hieroglyphs, Sumerian cuneiform) as **structured encoding systems** — analogous to Python, TCP/IP, or Bluetooth rather than natural organic languages.

**Framing (explicitly labeled as research design analogy, not historical claim):**
Ancient scripts exhibit design patterns consistent with structured communication protocols: logographic-phonetic hybrids, deterministic classifiers (determinatives), measurable geometry. This is a machine-language metaphor for hypothesis generation, not a claim that ancient peoples used Bluetooth.

**Key caveat (Sproat 2013):** Entropy alone cannot distinguish linguistic from non-linguistic sequences. Information-theoretic measures are necessary but insufficient evidence.

## Core Modules

### `ancient_ml/entropy_analysis.py`
Shannon entropy analysis (H1, H2, MI, redundancy) comparing ancient sign sequences against baselines: natural language (English), source code (Python), network protocols (TCP/IP), DNA sequences. Supports ORACC corpus via `--project etcsri` / `--project rinap`.

### `ancient_ml/determinative_graph.py`
Classifier/determinative co-occurrence network analysis with spectral methods, community detection, and centrality ranking. Tests whether determinatives form a designed taxonomy vs. organic categorization.

### `ancient_ml/geometric_phi_scanner.py`
Golden ratio (φ ≈ 1.618) detection in sign geometry and real hieroglyph images. Maps sign sequences to quantum circuit topologies (Sierpinski, Merkaba). Includes ImagePhiScanner with adversarial controls (random baseline, shuffled baseline, bootstrap CI).

### `ancient_ml/oracc_loader.py`
Downloads and parses ORACC JSON zip corpora (etcsri, rinap, cams/gkab). Extracts sign sequences, lemmas, determinative pairs. Caches under `data/oracc/` (gitignored).

### `ancient_ml/csa_optmatcher.py`
CSA_OptMatcher cognate matching — implements Tamburini (2025) Coupled Simulated Annealing optimization for ancient script decipherment. K-permutation encoding + edit distance with wildcards. 100% on Ugaritic-Hebrew benchmarks.

### `ancient_ml/run_pipeline.py`
Unified runner executing all modules. Supports `--project etcsri`, `--image-scan`.

## Running the Pipeline

```bash
cd ancient_ml

# Full pipeline with ORACC etcsri corpus
python run_pipeline.py --all --corpus oracc --project etcsri

# Hieroglyphs only
python run_pipeline.py --all --corpus hieroglyphs

# Image-based phi scanning (requires images in data/hieroglyphs/)
python run_pipeline.py --image-scan

# Individual modules
python entropy_analysis.py --corpus oracc --project etcsri --max-texts 100
python determinative_graph.py --corpus oracc
python geometric_phi_scanner.py --signs hieroglyphs

# CSA_OptMatcher cognate matching
python csa_optmatcher.py --benchmark
python csa_optmatcher.py --source sumerian --target akkadian
```

## ORACC Corpus Projects

Available projects (download from `http://oracc.iaas.upenn.edu/json/[PROJECT].zip`):
- `etcsri` — Electronic Text Corpus of the Sumerian Royal Inscriptions
- `rinap` — Royal Inscriptions of the Neo-Assyrian Period
- `cams/gkab` — Gudea cylinders
- `adsd/adart1`, `adsd/adart2` — Pre-Sargonic and Sargonic Texts

Cached under `ancient_ml/data/oracc/` (not committed to git).

## Known Results

| Vector | Finding |
|--------|---------|
| Entropy (hieroglyphs, sample) | H1=4.135, H2=5.777 — closer to Python than natural language |
| Phi Detection | 73.3% of Gardiner signs embed φ proportions in idealized geometry |
| Quantum Resonance | 0.467 phi-resonance score (Sierpinski topology) |
| CSA_OptMatcher | 100% on Ugaritic-Hebrew cognates (8/8), 0% Sumerian-Akkadian (needs larger corpus) |
| Determinative Graph | Classifiers form network topology — designed taxonomy hypothesis |

**Sample size warning:** Entropy and determinative graph results use small embedded samples (77 signs). Real conclusions require ORACC corpus (5k–50k signs). Run on your machine with network access.

## Research Framing

**Primary claim:** structured encoding — hybrid logographic/phonetic + classifiers + measurable geometry.
**Analogy (explicitly labeled):** protocol/machine-language metaphor for research design.
**Not claimed:** engineered Bluetooth-like comms as historical fact; entropy alone as language detector.

## Dependencies
- numpy, pandas, matplotlib, networkx, scipy, requests
- Optional: Pillow (image scanning), opencv-python (contour detection)
