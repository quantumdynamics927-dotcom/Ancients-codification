# Blind Research Protocol
**Version:** 1.0
**Date:** 2026-07-16
**Status:** Active

---

## Purpose

This protocol defines how ancient sign-stream data enters the blind analysis pipeline
with zero semantic leakage. It governs data selection, provenance tracking,
metadata stripping, and the separation between curator knowledge and analyst knowledge.

---

## Data Selection

### Curator role (NOT accessible to blind analyst)

The curator examines source corpora (ORACC, CDLI, etc.) and applies selection criteria:

1. **GROUP_A (Anunnaki-selected):** Texts identified through source-catalogue searches
   for divine-name markers (AN, KI, d-en-lil, d-inanna, etc.) — these are provenance
   metadata, not semantic analysis by the blind system.
2. **GROUP_B (Royal inscription controls):** Matched non-divine royal inscriptions
   from the same corpus and period.
3. **GROUP_C (Administrative controls):** Non-royal, non-divine administrative tablets
   where available.

The curator records group membership in a **separate manifest file** that is
**never passed to the blind analyzer**.

### Analyst role (receives only opaque tokens)

The analyst receives:

```json
{
  "record_id": "opaque_identifier",
  "group_id": "GROUP_A",   // only group label, no semantic meaning
  "tokens": ["SIG_001", "SIG_042", ...],
  "blind_mode": true
}
```

The analyst does NOT receive:
- Translations
- Lemmas, glosses, readings
- Divine names in readable form
- Historical interpretations
- Sign-name mappings

---

## Metadata Flow

```
Source corpus (ORACC JSON)
    |
    v
[Curator selection + group labeling]
    |  group manifest (curator only)
    |  (never given to analyst)
    v
[Metadata stripping: export_blind_jsonl]
    |
    v
Blind JSONL  <--- analyst receives this
    |
    v
[Manifest: datasets/manifests/]
```

---

## Export Format

Each JSONL record contains ONLY structural data:

```json
{
  "record_id": "Q000376.1",
  "group_id": "GROUP_A",
  "container_id": "Q000376",
  "line_index": 1,
  "tokens": ["AN", "EN", "KI", "MU", "GIR"],
  "layout": {
    "column": 1,
    "row": 1
  },
  "blind_mode": true,
  "source_project": "etcsri"
}
```

Required fields: `record_id`, `tokens`, `blind_mode: true`
Optional fields: `group_id`, `container_id`, `line_index`, `layout`

Forbidden fields in export: `translation`, `lemma`, `gloss`, `reading`,
`normalized`, `sense`, `deity_name`, `historical_note`.

---

## Group Comparison Protocol

After all preprocessing thresholds and analysis parameters are **preregistered**:

1. Run full validation on GROUP_A, GROUP_B, GROUP_C independently.
2. Compare:
   - H1, H2, MI distributions across groups
   - Field stability replication
   - Grammar survival (real_minus_shuffle)
   - Null model win rates
3. Report ONLY distributional differences that achieve statistical significance
   under the preregistered thresholds.
4. Do NOT claim:
   - That GROUP_A proves Anunnaki technology
   - That differences indicate extraterrestrial authorship
   - That structural similarity implies design intention

---

## Reporting Requirements

Every publication-quality result must include:

1. The exact export command used (`python scripts/download_oracc.py --project etcsri`)
2. The manifest file path
3. The curation criteria applied (in the methods section, not the results)
4. The validation gates that passed/failed
5. The group comparison results with effect sizes and confidence intervals

---

## Provenance Metadata

| Field | Curator-accessible | Analyst-accessible |
|-------|-------------------|-------------------|
| Sign token IDs | Yes | Yes (opaque) |
| Sequence order | Yes | Yes |
| Tablet boundary | Yes | Yes |
| Line index | Yes | Yes |
| Group label | Yes | GROUP_A/B/C only |
| Lemma | Yes | **No** |
| Translation | Yes | **No** |
| Divine name | Yes | **No** |
| Historical note | Yes | **No** |

---

## Compliance

This protocol is verified by:
1. Unit test: `tests/test_metadata_stripping.py` — confirms no lemma/translation/gloss
   fields appear in exported JSONL.
2. Unit test: `tests/test_group_labels.py` — confirms group labels are anonymized
   (GROUP_A/B/C) in the JSONL.
3. Integration test: `tests/test_manifest_separation.py` — confirms manifest
   files are not accessible to the analyst module.

---

## Revision Log

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-07-16 | Initial protocol |

