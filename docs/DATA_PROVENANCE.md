# Data Provenance

**Version:** 1.0
**Date:** 2026-07-16

Records the origin, licensing, and processing history of every dataset
used in blind analysis.

---

## Primary Datasets

### ETCSRI — Electronic Text Corpus of Sumerian Royal Inscriptions

| Field | Value |
|-------|-------|
| **Source** | ORACC (Open Richly Annotated Cuneiform Corpus) |
| **URL** | `http://oracc.iaas.upenn.edu/json/etcsri.zip` |
| **Alternate URL** | `https://build-oracc.museum.upenn.edu/json/etcsri.zip` |
| **License** | CC0 1.0 Universal (Public Domain) |
| **Project ID** | etcsri |
| **Period** | Early Dynastic to Old Babylonian (c. 2600–1700 BCE) |
| **Language** | Sumerian (with Akkadian scholarly material) |
| **Texts** | ~1,500 royal inscriptions |
| **Downloaded** | 2026-07-16 via `scripts/download_oracc.py --project etcsri --alt` |
| **Extracted** | `datasets/oracc/etcsri/` |
| **Exported** | `datasets/blind/etcsri_signs.jsonl` |
| **Manifest** | `datasets/manifests/etcsri_manifest.json` |
| **n_records** | 1,456 sequences |
| **n_tokens** | 68,083 sign instances |
| **vocab_size** | 627 unique sign types |
| **Blind mode** | True — only `gdl_sign` tokens extracted; no lemmas/glosses/translations |

**Processing pipeline:**
```
ORACC JSON (cdl/gdl tree)
  -> extract_sign_tokens() walks gdl_sign fields at all nesting levels
  -> One sequence per text record (one .json file = one tablet/inscription)
  -> Tokens: raw sign IDs from gdl_sign (e.g. "AN", "KI", "EN")
  -> No lemmatization, no translation, no normalized forms
  -> Exported as JSONL with blind_mode: true
```

**Curation applied:** None for the base corpus.
Anunnaki-selected subgroup created via `scripts/anunnaki_experiment.py` which
filters texts containing divine-name sign sequences (AN, KI with specific patterns).
Subgroup labels are GROUP_A/GROUP_B/GROUP_C in anonymized export.

---

### RINAP — Royal Inscriptions of the Neo-Assyrian Period

| Field | Value |
|-------|-------|
| **Source** | ORACC |
| **URL** | `http://oracc.iaas.upenn.edu/json/rinap.zip` |
| **Alternate URL** | `https://build-oracc.museum.upenn.edu/json/rinap.zip` |
| **License** | CC BY-SA 3.0 |
| **Project ID** | rinap |
| **Period** | Neo-Assyrian (744–609 BCE) |
| **Language** | Neo-Assyrian Akkadian |
| **Status** | **DOWNLOAD BLOCKED** — metadata-only zip in this environment |
| **Note** | Full corpus requires API access or different mirror |

RINAP is Neo-Assyrian/Akkadian cuneiform — NOT Egyptian hieroglyphs.
Available subprojects: rinap1 (Tiglath-pileser III), rinap2 (Shalmaneser V),
rinap3 (Sargon II), rinap4 (Sennacherib), rinap5 (Esarhaddon + Ashurbanipal).

---

## Synthetic/Control Datasets

### Hieroglyph embedded sample

| Field | Value |
|-------|-------|
| **Source** | Hardcoded in `blind_pattern.py:load_hieroglyph_sample()` |
| **n_sequences** | 44 |
| **n_tokens** | 745 |
| **vocab_size** | ~127 Gardiner sign IDs |
| **Purpose** | Proof-of-concept testing only |
| **Note** | Hand-curated; not suitable for research claims |

### Cuneiform embedded sample

| Field | Value |
|-------|-------|
| **Source** | Hardcoded in `blind_pattern.py:load_cuneiform_sample()` |
| **n_sequences** | 52 |
| **n_tokens** | 652 |
| **vocab_size** | 3 sign types (DI, KI, AN) |
| **Purpose** | Proof-of-concept testing; too small for meaningful inference |

### Python control corpus

| Field | Value |
|-------|-------|
| **Source** | Generated in `blind/null_models.py:_python_corpus()` |
| **Purpose** | Positive control — known structured language |
| **Tokens** | Python keywords, identifiers, operators |

---

## Data Licensing

| Dataset | License | Commercial use | Modification | Attribution required |
|---------|---------|---------------|-------------|---------------------|
| ETCSRI | CC0 1.0 | Yes | Yes | No |
| RINAP | CC BY-SA 3.0 | Yes | Yes | Yes |
| Embedded samples | None (synthetic) | N/A | N/A | N/A |
| Output reports | CC0 1.0 | Yes | Yes | No |

---

## Export Integrity

All exported JSONL files are verified to contain only:

- `record_id` (opaque string)
- `tokens` (list of sign ID strings)
- `group_id` (GROUP_A/B/C — anonymized)
- `blind_mode: true` (confirmation flag)
- Optional: `container_id`, `line_index`, `layout`

All of lemma, gloss, translation, reading, normalized form, deity name,
and historical interpretation are **explicitly excluded** from export.

Verification: `scripts/verify_blind_export.py` reads a JSONL file and
asserts no forbidden fields are present.

---

## Gitignore

Raw ORACC downloads and generated JSONL sign streams are gitignored:

```
datasets/oracc/          # Raw ORACC downloads (large)
datasets/blind/*.jsonl   # Generated sign streams
outputs/                 # Analysis outputs
datasets/manifests/      # Manifests (contain group labels)
```

Only `scripts/`, `datasets/blind/README.md`, `datasets/blind/fixture_signs.jsonl`,
and `docs/` are committed.

---

## Revision Log

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-07-16 | Initial provenance record |

