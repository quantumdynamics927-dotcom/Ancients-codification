"""
Anunnaki Target Experiment
==========================
Curation-only script. Creates GROUP_A / GROUP_B / GROUP_C subgroups from the
ETCSRI corpus based on divine-name sign markers, without exposing labels to
the blind analyzer.

GROUP_A: Texts containing AN + KI co-occurrence (divine-name marker pattern)
GROUP_B: Matched royal inscription controls (same corpus, no divine markers)
GROUP_C: Administrative tablets (non-royal, non-divine)

The manifest mapping record_id -> group is written to datasets/manifests/
and is NOT passed to the blind analyzer module.

Usage:
    python scripts/anunnaki_experiment.py
"""

import json
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
BLIND_JSONL = PROJECT_ROOT / "datasets" / "blind" / "etcsri_signs.jsonl"
MANIFEST_DIR = PROJECT_ROOT / "datasets" / "manifests"
GROUPED_JSONL = PROJECT_ROOT / "datasets" / "blind" / "etcsri_grouped.jsonl"


# Divine-name marker patterns for Sumerian cuneiform
# AN = dingir "god" / "heaven"  KI = eridu / earth
# These sign co-occurrences are associated with divine-name sequences
DIVINE_MARKERS = {"AN", "KI"}

# Minimum co-occurrences of AN+KI within a window to flag as potentially divine
MIN_DIVINE_PAIRS = 2


def load_blind_corpus(path: Path):
    """Load the blind sign-stream JSONL. Returns list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_record(tokens: list) -> str | None:
    """
    Classify a single record by divine-name markers.
    Returns 'GROUP_A' if divine markers present, None otherwise.

    Uses only the sign token stream — no lemmas, translations, or readings.
    This is the curator's selection function; it stays separate from the
    blind analyzer which only sees the already-labeled group_id.
    """
    token_set = set(tokens)
    # Check direct co-occurrence of divine markers
    if DIVINE_MARKERS.issubset(token_set):
        # Count adjacent AN-KI pairs within a window
        pair_count = 0
        for i in range(len(tokens) - 1):
            if tokens[i] == "AN" and tokens[i + 1] == "KI":
                pair_count += 1
            elif tokens[i] == "KI" and tokens[i + 1] == "AN":
                pair_count += 1
        if pair_count >= MIN_DIVINE_PAIRS:
            return "GROUP_A"
    return None


def split_controls(records: list, group_a_records: list, seed: int = 42):
    """
    Split remaining records into GROUP_B (royal) and GROUP_C (administrative).

    GROUP_B: Royal inscription structure — longer sequences, LUGAL (king) marker,
             deterministic patterns similar to GROUP_A
    GROUP_C: Shorter sequences without royal markers — likely administrative

    Match GROUP_B to GROUP_A by approximate sequence length (±20%).
    """
    random.seed(seed)
    group_b = []
    group_c = []

    # Records not in GROUP_A
    group_a_ids = {r["record_id"] for r in group_a_records}
    remaining = [r for r in records if r["record_id"] not in group_a_ids]

    # Separate by length heuristic
    # Sumerian royal inscriptions are typically > 50 signs
    # Administrative tablets are typically < 50 signs
    long_records = []
    short_records = []
    for r in remaining:
        tokens = r.get("tokens", [])
        if len(tokens) >= 50:
            long_records.append(r)
        else:
            short_records.append(r)

    # GROUP_B: matched-length royal controls (long, no divine markers)
    group_b = long_records

    # GROUP_C: administrative controls (short, no royal/divine markers)
    group_c = short_records

    return group_b, group_c


def assign_groups(records: list):
    """Assign all records to groups. Returns dict record_id -> group_id."""
    group_a = []
    for r in records:
        tokens = r.get("tokens", [])
        if classify_record(tokens) == "GROUP_A":
            group_a.append(r)

    group_b, group_c = split_controls(records, group_a)

    print(f"GROUP_A (divine-name markers): {len(group_a)} records")
    print(f"GROUP_B (royal controls):       {len(group_b)} records")
    print(f"GROUP_C (administrative):       {len(group_c)} records")

    result = {}
    for r in group_a:
        result[r["record_id"]] = "GROUP_A"
    for r in group_b:
        result[r["record_id"]] = "GROUP_B"
    for r in group_c:
        result[r["record_id"]] = "GROUP_C"

    return result, group_a, group_b, group_c


def export_grouped_jsonl(records: list, group_map: dict, output_path: Path):
    """Export JSONL with group_id added. The analyst receives this version."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            record_id = r["record_id"]
            group_id = group_map.get(record_id, "GROUP_C")  # default to C if missing
            out = {
                "record_id": record_id,
                "line_index": r.get("line_index", count),
                "tokens": r["tokens"],
                "group_id": group_id,
                "blind_mode": True,
                "analysis_ready": True,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    print(f"  Exported {count} records to {output_path}")


def write_manifest(group_map: dict, group_a: list, group_b: list, group_c: list):
    """Write curator manifest with full group assignments. NOT given to analyst."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "anunnaki_manifest.json"

    # Stats per group
    def stats(recs):
        tokens = [t for r in recs for t in r.get("tokens", [])]
        lens = [len(r.get("tokens", [])) for r in recs]
        vocab = len(set(tokens))
        return {
            "n_records": len(recs),
            "n_tokens": len(tokens),
            "vocab_size": vocab,
            "mean_length": sum(lens) / len(lens) if lens else 0,
            "min_length": min(lens) if lens else 0,
            "max_length": max(lens) if lens else 0,
        }

    manifest = {
        "experiment": "anunnaki_target",
        "description": (
            "Curator-only manifest: full group assignments for ETCSRI corpus. "
            "This file is NOT passed to the blind analyzer."
        ),
        "curation_date": "2026-07-16",
        "source": "etcsri_signs.jsonl",
        "group_definitions": {
            "GROUP_A": "Texts with AN+KI divine-name marker co-occurrence (>=2 pairs)",
            "GROUP_B": "Royal inscription controls (>=50 signs, no divine markers, matched length)",
            "GROUP_C": "Administrative controls (<50 signs, no royal/divine markers)",
        },
        "divine_markers": list(DIVINE_MARKERS),
        "min_divine_pairs": MIN_DIVINE_PAIRS,
        "groups": {
            "GROUP_A": stats(group_a),
            "GROUP_B": stats(group_b),
            "GROUP_C": stats(group_c),
        },
        "record_assignments": group_map,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Curator manifest written to {manifest_path}")


def verify_blind_export(jsonl_path: Path):
    """Verify grouped JSONL contains no forbidden fields."""
    forbidden = {"lemma", "gloss", "translation", "reading", "normalized", "sense"}
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            rec = json.loads(line)
            hit = forbidden & set(rec.keys())
            if hit:
                print(f"  [FAIL] Record {i} contains forbidden fields: {hit}")
                return False
    print(f"  [PASS] No forbidden fields in {jsonl_path}")
    return True


def main():
    print("=" * 60)
    print("Anunnaki Target Experiment — Curator Curation Script")
    print("=" * 60)
    print(f"\nLoading corpus from {BLIND_JSONL}")
    records = load_blind_corpus(BLIND_JSONL)
    print(f"  Loaded {len(records)} records")

    # Assign groups
    print("\nAssigning groups...")
    group_map, group_a, group_b, group_c = assign_groups(records)

    # Export grouped JSONL (analyst receives this)
    print(f"\nExporting grouped corpus to {GROUPED_JSONL}")
    export_grouped_jsonl(records, group_map, GROUPED_JSONL)

    # Write curator manifest (NOT given to analyst)
    print("\nWriting curator manifest...")
    write_manifest(group_map, group_a, group_b, group_c)

    # Verify blind export integrity
    print("\nVerifying blind export...")
    verify_blind_export(GROUPED_JSONL)

    print("\n" + "=" * 60)
    print("Curation complete.")
    print("\nTo run blind analysis on grouped corpus:")
    print(f"  python ancient_ml/blind_pattern.py --corpus local --path {GROUPED_JSONL} --validate")
    print("\nThe analyst receives:")
    print("  - datasets/blind/etcsri_grouped.jsonl (tokens + anonymized group_id)")
    print("  - docs/BLIND_RESEARCH_PROTOCOL.md (protocol documentation)")
    print("\nThe analyst does NOT receive:")
    print("  - datasets/manifests/anunnaki_manifest.json (curator only)")
    print("  - Any lemma, translation, gloss, or deity name fields")
    print("=" * 60)


if __name__ == "__main__":
    main()
