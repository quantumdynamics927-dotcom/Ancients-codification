# Blind Dataset Setup

## Download Primary ORACC Corpora

### Step 1: Download ORACC JSON archives

From repository root, run:

```powershell
cd D:\Ancients-codification

python scripts\download_oracc.py --project etcsri
python scripts\download_oracc.py --project rinap
```

If the primary host is blocked, try the alternate:

```powershell
python scripts\download_oracc.py --project etcsri --alt
```

Available projects:
- `etcsri` — Electronic Text Corpus of the Sumerian Royal Inscriptions
- `rinap` — Royal Inscriptions of the Neo-Assyrian Period
- `cams/gkab` — Gudea cylinders

Manual download URLs:
- Primary: `http://oracc.iaas.upenn.edu/json/etcsri.zip`
- Alternate: `http://build-oracc.museum.upenn.edu/json/etcsri.zip`

### Step 2: Export blind sign streams

The download script automatically exports to `datasets/blind/{project}_signs.jsonl`.

Manual export:
```powershell
python scripts\download_oracc.py --project etcsri --output datasets\blind\etcsri_signs.jsonl
```

## File Format

Each line of the JSONL file is a JSON object:
```json
{"record_id": "oracc:0", "line_index": 0, "tokens": ["DI", "KI", "AN", "DI", "KI"], "blind_mode": true}
```

Fields:
- `record_id`: Unique text identifier from ORACC
- `line_index`: Sequence number in export
- `tokens`: List of sign IDs (opaque strings, no lemmas/translations)
- `blind_mode`: Always `true` — confirms blind analysis ready

## Running Analysis on ORACC Data

```powershell
cd ancient_ml

# Sophistication scorecard
python blind_pattern.py --corpus local --path ..\datasets\blind\etcsri_signs.jsonl --full

# Full validation (recommended before any claims)
python blind_pattern.py --corpus local --path ..\datasets\blind\etcsri_signs.jsonl --validate --bootstrap 1000 --heldout 0.25
```

## Anunnaki-Selected Set

To test Anunnaki-related texts specifically:
1. Download ORACC corpus as above
2. Filter texts containing Anunnaki divine names (sign sequences containing `AN`, `KI`, `d-en`, `d-inanna`, etc.)
3. Export as a separate JSONL with `_anunnaki` suffix
4. Run blind analysis on both the filtered and unfiltered sets
5. Compare: do Anunnaki-related texts show higher structural complexity?

This gives a controlled comparison within the same corpus/period.

## Gitignore

The following are NOT committed:
```
datasets/oracc/          # Raw ORACC downloads
datasets/blind/*.jsonl    # Generated sign streams
outputs/                 # Analysis outputs
```
