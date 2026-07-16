# Ancients Codification

**Blind formal-language discovery stack for ancient sign streams.**
Treats ancient writing systems (Egyptian hieroglyphs, Sumerian cuneiform) as unknown
formal systems — analyzing statistical structure without semantic interpretation.

> **Research design analogy, not historical claim:** Ancient scripts exhibit design
> patterns consistent with structured communication protocols. This is a
> machine-language metaphor for hypothesis generation, not proof that ancient
> peoples used Bluetooth.

**Key caveat (Sproat 2013):** Entropy alone cannot distinguish linguistic from
non-linguistic sequences. Information-theoretic measures are necessary but
insufficient evidence.

---

## Quick Start

```powershell
# 1. Download ETCSRI corpus (Sumerian Royal Inscriptions, ORACC CC0 1.0)
python scripts/download_oracc.py --project etcsri --alt

# 2. Run full validation pipeline
python ancient_ml/blind_pattern.py --corpus local --path datasets/blind/etcsri_signs.jsonl --validate --bootstrap 100

# 3. Run tests
python tests/test_metadata_stripping.py
python tests/test_null_models.py
python tests/test_baseline_calibration.py
```

---

## Full Pipeline Commands

### Bash (Linux/macOS/Git Bash)

```bash
# Download ORACC corpus
python scripts/download_oracc.py --project etcsri --alt

# Run blind validation with bootstrap CI
python ancient_ml/blind_pattern.py \
    --corpus local \
    --path datasets/blind/etcsri_signs.jsonl \
    --validate \
    --bootstrap 200

# Run with 500 tablets max
python ancient_ml/blind_pattern.py \
    --corpus local \
    --path datasets/blind/etcsri_signs.jsonl \
    --validate \
    --bootstrap 200 \
    --sequences 500

# Run individual modules
python ancient_ml/entropy_analysis.py --corpus oracc --project etcsri --max-texts 200
python ancient_ml/determinative_graph.py --corpus oracc
python ancient_ml/geometric_phi_scanner.py --signs hieroglyphs

# CSA_OptMatcher cognate matching
python ancient_ml/csa_optmatcher.py --benchmark
python ancient_ml/csa_optmatcher.py --source sumerian --target akkadian

# Full pipeline
cd ancient_ml
python run_pipeline.py --all --corpus oracc --project etcsri
```

### PowerShell (Windows)

```powershell
# Download ORACC corpus
python scripts\download_oracc.py --project etcsri --alt

# Run blind validation with bootstrap CI
python ancient_ml\blind_pattern.py --corpus local --path datasets\blind\etcsri_signs.jsonl --validate --bootstrap 200

# Run with 500 tablets max
python ancient_ml\blind_pattern.py --corpus local --path datasets\blind\etcsri_signs.jsonl --validate --bootstrap 200 --sequences 500

# Run individual modules
python ancient_ml\entropy_analysis.py --corpus oracc --project etcsri --max-texts 200
python ancient_ml\determinative_graph.py --corpus oracc
python ancient_ml\geometric_phi_scanner.py --signs hieroglyphs

# CSA_OptMatcher cognate matching
python ancient_ml\csa_optmatcher.py --benchmark

# Full pipeline
cd ancient_ml
python run_pipeline.py --all --corpus oracc --project etcsri
```

---

## Anunnaki Target Experiment

```powershell
# Create GROUP_A (divine-marker), GROUP_B (royal controls), GROUP_C (administrative)
python scripts\anunnaki_experiment.py

# Validate each group separately
python ancient_ml\blind_pattern.py --corpus local --path datasets\blind\etcsri_grouped.jsonl --validate --bootstrap 100
```

The experiment creates three anonymized subgroups from ETCSRI:
- **GROUP_A:** Texts with AN+KI divine-name marker co-occurrence (n=10 tablets)
- **GROUP_B:** Royal inscription controls, matched length, no divine markers (n=194)
- **GROUP_C:** Administrative tablets, no royal/divine markers (n=1,252)

The curator manifest (full group assignments) is written to `datasets/manifests/`
and is **never** passed to the blind analyzer. The analyzer receives only
anonymized GROUP_A/B/C labels.

---

## Test Suite

```bash
# Run all tests
python tests/test_metadata_stripping.py
python tests/test_null_models.py
python tests/test_baseline_calibration.py
python tests/test_geometry_null_state.py
python tests/test_manifest_separation.py
python tests/test_heldout_split.py
```

---

## What This Pipeline Does

| What it does | What it does NOT do |
|---------------|-------------------|
| Treats sign sequences as opaque tokens (no lemmas/translations) | Does not interpret ancient languages |
| Measures Shannon entropy (H1, H2, MI), redundancy, field stability | Does not claim "machine language" |
| Compares against baselines (Python, TCP, DNA, English) | Does not identify authors or historical events |
| Tests formal grammar survival via shuffle controls | Does not claim Anunnaki/extraterrestrial origin |
| Maps sign geometry for phi (golden ratio) patterns | Does not prove sacred encoding |
| Validates held-out by tablet (not line) | Does not infer exact dating or cultural attribution |
| Issues LEVEL 0-4 verdicts with calibrated confidence | Does not skip validation gates |

---

## Architecture

```
datasets/
  blind/
    etcsri_signs.jsonl      # Primary blind corpus (1,456 tablets, 68,083 signs)
    etcsri_grouped.jsonl    # GROUP_A/B/C anonymized for Anunnaki experiment
    fixture_signs.jsonl     # Small synthetic test set
  manifests/
    anunnaki_manifest.json # Curator-only: full group assignments (NOT given to analyst)
    etcsri_manifest.json   # Corpus stats and provenance

ancient_ml/
  blind/
    tokenizer.py            # Opaque sign tokenization
    fingerprint.py          # L0: H1/H2, MI, redundancy, Zipf
    field_segmenter.py      # L1: protocol field boundary detection
    role_typer.py           # L2: behavioral pseudo-type clustering
    grammar_inducer.py      # L3: unsupervised CFG/automaton induction
    complexity.py           # L4: compression / Kolmogorov proxies
    family_manifold.py      # L5: multi-feature family distance + novelty
    geometry_channel.py     # L6: phi geometry (NULL_ON_CURRENT_DATA initially)
    quantum_map.py          # L7: optional TMT export
    null_models.py          # L8: shuffle/Markov/template/random controls
    validate.py             # Held-out validation, bootstrap CI, verdict ladder
    report.py               # Full scorecard report

docs/
  BLIND_RESEARCH_PROTOCOL.md   # Curator-analyst separation, data flow
  DATA_PROVENANCE.md          # Dataset origins, licensing, processing
  PREREGISTERED_METRICS.md    # All thresholds fixed before data analysis
  RESULTS.md                  # Validation results and honest conclusions
```

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
  <- manifold CALIBRATED (>= 4/6 baseline self-classification)
  <- novelty > 1.1

LEVEL_4_DUAL_CHANNEL_ENCODING_CANDIDATE
  <- Level 3
  <- geometry passes Bonferroni + FDR + phi_rate > 0.50
```

**Ceiling rule:** When manifold is UNCALIBRATED (baseline self-classification < 4/6),
verdict ceiling is LEVEL_2. LEVEL_3/4 are blocked until calibration passes.

---

## ORACC Corpus Projects

| Project | Description | License |
|---------|-------------|---------|
| etcsri | Electronic Text Corpus of Sumerian Royal Inscriptions | CC0 1.0 |
| rinap | Royal Inscriptions of the Neo-Assyrian Period | CC BY-SA 3.0 |
| cams/gkab | Gudea cylinders | CC0 1.0 |

Download from `http://oracc.iaas.upenn.edu/json/[PROJECT].zip`
or `https://build-oracc.museum.upenn.edu/json/[PROJECT].zip`

---

## Research Framing

**This project treats ancient sign sequences as unknown formal systems for
hypothesis generation. The protocol/machine-language analogy is a research
design tool, not a historical claim.**

What the pipeline can find:
- Statistical structure in sign sequences (LEVEL_1/2)
- Evidence of formal grammar surviving token shuffle (LEVEL_2+)
- Novel structural patterns far from known baselines (LEVEL_3+)
- Golden ratio geometry in sign shapes (LEVEL_4)

What it cannot find:
- Author identity or extraterrestrial origin
- Historical purpose or intended meaning
- Whether a script is "communication protocol vs ritual notation"

---

## Dependencies

```
numpy, pandas, matplotlib, networkx, scipy, requests
```
Optional: Pillow (image scanning), opencv-python (contour detection)

---

## License

Output reports and validation code: **CC0 1.0**

ETCSRI corpus: **CC0 1.0** (public domain)
RINAP corpus: **CC BY-SA 3.0**
