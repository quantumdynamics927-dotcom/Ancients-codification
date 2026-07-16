"""
test_manifest_separation.py
===========================
Verifies that curator manifest files (containing full group assignments) are
NOT accessible to the blind analyzer module. The analyzer must receive only
the anonymized JSONL.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def test_manifest_not_in_package():
    """Manifest directory must not be importable by the blind module."""
    import os

    manifest_dir = PROJECT_ROOT / "datasets" / "manifests"
    blind_module_dir = PROJECT_ROOT / "ancient_ml"

    # The manifest directory should not be a subdirectory of ancient_ml
    try:
        manifest_dir.resolve().relative_to(blind_module_dir.resolve())
        is_subdir = True
    except ValueError:
        is_subdir = False

    if is_subdir:
        print("FAIL: Manifest directory is inside ancient_ml package")
        sys.exit(1)
    print("PASS: Manifest directory is outside ancient_ml package.")


def test_manifest_not_accessible_to_analyzer():
    """The blind_pattern.py module should not be able to import the manifest."""
    # Attempt to import manifest — should fail or return nothing
    manifest_path = PROJECT_ROOT / "datasets" / "manifests" / "anunnaki_manifest.json"
    if not manifest_path.exists():
        print("SKIP: anunnaki_manifest.json not found (not yet created)")
        return

    # Check that the manifest is NOT accessible via any sys.path under ancient_ml
    blind_module_dir = (PROJECT_ROOT / "ancient_ml").resolve()
    for p in sys.path:
        p_path = Path(p).resolve()
        try:
            manifest_path.resolve().relative_to(p_path)
            if p_path == blind_module_dir or blind_module_dir in p_path.parents:
                print("FAIL: Manifest path is inside a sys.path entry accessible to analyzer")
                sys.exit(1)
        except ValueError:
            pass  # Not inside this path — good

    print("PASS: Manifest is not accessible to blind analyzer via sys.path.")


def test_grouped_jsonl_has_no_semantic_fields():
    """The analyst-received JSONL (etcsri_grouped.jsonl) must not leak semantic data."""
    import json

    jsonl_path = PROJECT_ROOT / "datasets" / "blind" / "etcsri_grouped.jsonl"
    if not jsonl_path.exists():
        print("SKIP: etcsri_grouped.jsonl not found")
        return

    FORBIDDEN = {"lemma", "gloss", "translation", "reading", "normalized",
                 "sense", "deity_name", "historical_note", "normalized form"}

    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            hit = FORBIDDEN & set(rec.keys())
            if hit:
                print(f"FAIL: etcsri_grouped.jsonl:{i} leaks semantic fields: {hit}")
                sys.exit(1)
    print("PASS: etcsri_grouped.jsonl contains no semantic fields.")


if __name__ == "__main__":
    test_manifest_not_in_package()
    test_manifest_not_accessible_to_analyzer()
    test_grouped_jsonl_has_no_semantic_fields()
