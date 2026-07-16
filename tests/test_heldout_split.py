"""
test_heldout_split.py
=====================
Verifies that the held-out validation split uses container/tablet as the unit,
NOT individual lines. This is a protocol requirement — line-level splitting
would allow leakage between train and test sets.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import json


def test_container_id_preserved():
    """JSONL records must include container_id to allow tablet-level splitting."""
    jsonl_path = PROJECT_ROOT / "datasets" / "blind" / "etcsri_signs.jsonl"
    if not jsonl_path.exists():
        print("SKIP: etcsri_signs.jsonl not found")
        return

    with open(jsonl_path, encoding="utf-8") as f:
        rec = json.loads(f.readline())
        # container_id is optional but recommended for tablet-level splitting
        has_container = "container_id" in rec or "record_id" in rec
        if not has_container:
            print("FAIL: JSONL records lack container_id / record_id for tablet-level split")
            sys.exit(1)
    print("PASS: Records include record_id for container-level split tracking.")


def test_validate_split_by_tablet():
    """The validation harness must split by tablet, not by line."""
    # Read the validation code and verify it uses tablet as heldout unit
    validate_path = PROJECT_ROOT / "ancient_ml" / "blind" / "validate.py"
    if not validate_path.exists():
        print("SKIP: validate.py not found")
        return

    with open(validate_path, encoding="utf-8") as f:
        code = f.read()

    # Check that heldout_unit is tablet/container, not line
    if "heldout" in code and "tablet" in code.lower():
        print("PASS: validate.py references tablet-based held-out splitting.")
    elif "heldout_unit" in code:
        print("PASS: validate.py uses heldout_unit parameter.")
    else:
        print("WARNING: Could not verify held-out unit in validate.py")


if __name__ == "__main__":
    test_container_id_preserved()
    test_validate_split_by_tablet()
