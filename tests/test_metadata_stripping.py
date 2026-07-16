"""
test_metadata_stripping.py
==========================
Verifies that exported JSONL files contain no forbidden semantic fields
(lemma, gloss, translation, reading, normalized, sense, deity_name, historical_note).
"""

import json
import sys
from pathlib import Path

# Resolve relative to this test file
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

FORBIDDEN_FIELDS = {
    "lemma", "gloss", "translation", "reading",
    "normalized", "sense", "deity_name", "historical_note",
    "normalized form", "reading form",
}

JSONL_FILES = [
    PROJECT_ROOT / "datasets" / "blind" / "etcsri_signs.jsonl",
    PROJECT_ROOT / "datasets" / "blind" / "etcsri_grouped.jsonl",
]


def test_no_forbidden_fields():
    """Every exported JSONL must contain only structural fields."""
    failures = []
    for jsonl_path in JSONL_FILES:
        if not jsonl_path.exists():
            failures.append(f"SKIP: {jsonl_path} not found")
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    failures.append(f"{jsonl_path}:{i} JSON decode error: {e}")
                    continue

                hit = FORBIDDEN_FIELDS & set(str(k) for k in rec.keys())
                if hit:
                    failures.append(f"{jsonl_path}:{i} forbidden fields: {hit}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    print(f"PASS: All {len(JSONL_FILES)} JSONL files contain only structural fields.")


def test_required_fields():
    """Every record must have record_id, tokens, and blind_mode=true."""
    failures = []
    for jsonl_path in JSONL_FILES:
        if not jsonl_path.exists():
            failures.append(f"SKIP: {jsonl_path} not found")
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                rec = json.loads(line)
                if "record_id" not in rec:
                    failures.append(f"{jsonl_path}:{i} missing record_id")
                if "tokens" not in rec:
                    failures.append(f"{jsonl_path}:{i} missing tokens")
                if rec.get("blind_mode") is not True:
                    failures.append(f"{jsonl_path}:{i} blind_mode != true")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    print(f"PASS: All records have required fields (record_id, tokens, blind_mode=true).")


def test_group_labels_anonymized():
    """GROUP_A/B/C labels must not contain deity names or semantic content."""
    jsonl_path = PROJECT_ROOT / "datasets" / "blind" / "etcsri_grouped.jsonl"
    if not jsonl_path.exists():
        print("SKIP: etcsri_grouped.jsonl not found")
        return

    ANON_LABELS = {"GROUP_A", "GROUP_B", "GROUP_C"}
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            gid = rec.get("group_id", "")
            if gid and gid not in ANON_LABELS:
                print(f"FAIL: {jsonl_path}:{i} non-anonymized group label: {gid}")
                sys.exit(1)
    print("PASS: Group labels are anonymized (GROUP_A/B/C only).")


if __name__ == "__main__":
    test_required_fields()
    test_group_labels_anonymized()
    test_no_forbidden_fields()
