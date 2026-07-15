"""
Blind Pattern Analysis — Thin CLI Orchestrator
============================================
CLI entry point for the full blind formal language discovery stack.
Delegates to blind.report (sophistication scorecard) and blind.validate
(validation harness with held-out splits, bootstrap CI, baseline self-classification).

Usage:
    # Sophistication scorecard
    python blind_pattern.py --corpus hieroglyphs
    python blind_pattern.py --corpus cuneiform --full

    # Validation (recommended before any claims)
    python blind_pattern.py --corpus hieroglyphs --validate --bootstrap 1000 --heldout 0.25

    # Local JSONL corpus
    python blind_pattern.py --corpus local --path "../datasets/blind/etcsri_signs.jsonl"
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent dir to path for imports from ancient_ml
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from blind.report import generate_report
from blind.validate import Validator, validation_summary


# ─────────────────────────────────────────────────────────────────────────────
# Sample corpora (no semantic meanings used in analysis)
# ─────────────────────────────────────────────────────────────────────────────

def load_hieroglyph_sample() -> list:
    """Raw hieroglyph sign sequences — Gardiner sign IDs as opaque tokens."""
    return [
        ["V30", "V30", "V30", "O1", "O1", "D4", "D21", "V30", "N35", "N1"],
        ["G1", "M17", "M17", "G43", "D4", "D21", "N35", "R12", "R12"],
        ["V30", "V30", "R12", "R12", "O1", "O1", "G1", "D4", "D21"],
        ["G1", "G1", "V30", "D4", "D21", "D26", "N35", "N1", "O1"],
        ["O1", "O1", "O1", "V30", "V30", "G43", "G43", "R12", "V30"],
        ["D4", "D21", "D26", "G1", "M17", "M17", "N35", "N1"],
        ["R12", "R12", "G1", "V30", "O1", "O1", "D4", "D21"],
        ["V30", "V30", "V30", "O1", "D4", "D21", "N35", "R12"],
        ["G43", "G43", "G1", "D4", "D21", "D26", "M17", "N1"],
        ["O1", "O1", "V30", "V30", "R12", "R12", "G1", "D4"],
        ["D26", "D21", "D4", "G1", "G43", "G43", "N35", "N1"],
        ["V30", "V30", "O1", "O1", "O1", "D4", "D21", "D26"],
        ["G1", "M17", "G43", "D4", "D21", "D26", "N35", "R12"],
        ["R12", "R12", "R12", "G1", "V30", "V30", "O1", "N1"],
        ["O1", "O1", "D4", "D26", "D21", "G1", "M17", "N35"],
        ["V30", "V30", "V30", "V30", "O1", "G1", "D4", "N1"],
        ["G43", "G43", "G43", "G1", "D4", "D21", "D26", "N35"],
        ["D21", "D26", "D4", "G1", "M17", "M17", "G43", "N1"],
        ["R12", "G1", "V30", "O1", "O1", "D4", "D21", "N35"],
        ["V30", "V30", "D4", "D21", "D26", "G1", "G43", "R12"],
        ["D4", "D21", "G1", "V30", "O1", "O1", "M17", "N1", "N35"],
        ["G43", "G43", "D4", "D21", "D26", "G1", "M17", "N35"],
        ["R12", "R12", "G1", "V30", "O1", "D4", "D21", "N1"],
        ["V30", "V30", "V30", "G1", "D4", "D21", "D26", "N35"],
        ["O1", "O1", "O1", "V30", "G43", "G43", "R12", "R12"],
        ["D4", "D21", "D26", "G1", "M17", "N35", "N1", "O1"],
        ["G1", "G1", "V30", "V30", "D4", "D21", "D26", "R12"],
        ["V30", "V30", "R12", "R12", "O1", "O1", "G43", "N1"],
        ["G43", "G43", "G1", "D4", "D21", "M17", "M17", "N35"],
        ["O1", "O1", "V30", "V30", "D4", "D26", "D21", "N1"],
        ["D26", "D21", "D4", "G1", "V30", "V30", "R12", "N35"],
        ["V30", "V30", "V30", "O1", "G1", "D4", "D21", "N1", "N35"],
        ["G1", "M17", "G43", "D4", "D21", "D26", "N35", "R12", "O1"],
        ["R12", "R12", "G1", "V30", "O1", "O1", "D4", "N1"],
        ["O1", "O1", "D4", "D26", "D21", "G1", "M17", "N35", "R12"],
        ["V30", "V30", "V30", "V30", "O1", "G1", "D4", "D21", "N35"],
        ["G43", "G43", "G43", "G1", "D4", "D21", "D26", "M17", "N1"],
        ["D21", "D26", "D4", "G1", "M17", "M17", "G43", "N35", "N1"],
        ["R12", "G1", "V30", "O1", "O1", "D4", "D21", "N35", "R12"],
        ["V30", "V30", "D4", "D21", "D26", "G1", "G43", "R12", "O1"],
        ["D4", "D21", "G1", "V30", "O1", "O1", "M17", "N1", "N35"],
        ["G43", "G43", "D4", "D21", "D26", "G1", "M17", "N35"],
        ["R12", "R12", "G1", "V30", "O1", "D4", "D21", "N1"],
        ["V30", "V30", "V30", "G1", "D4", "D21", "D26", "N35", "N1"],
    ]


def load_cuneiform_sample() -> list:
    """Raw cuneiform sign sequences — ORACC transliteration as opaque tokens."""
    return [
        ["DI", "KI", "AN", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["KI", "KI", "KI", "DI", "KI", "AN", "DI", "DI", "KI"],
        ["AN", "AN", "DI", "KI", "KI", "AN", "KI", "KI", "DI"],
        ["DI", "DI", "DI", "KI", "KI", "AN", "KI", "KI", "DI"],
        ["KI", "AN", "KI", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["DI", "KI", "DI", "KI", "DI", "KI", "DI", "KI", "AN"],
        ["AN", "AN", "AN", "KI", "KI", "DI", "DI", "KI", "AN"],
        ["KI", "KI", "DI", "DI", "KI", "KI", "DI", "KI", "AN"],
        ["DI", "DI", "DI", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["AN", "KI", "AN", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["KI", "KI", "KI", "DI", "KI", "DI", "KI", "KI", "AN"],
        ["DI", "AN", "DI", "KI", "AN", "DI", "KI", "KI", "DI"],
        ["AN", "AN", "KI", "KI", "DI", "DI", "KI", "KI", "AN"],
        ["KI", "DI", "KI", "DI", "KI", "KI", "DI", "DI", "KI"],
        ["DI", "DI", "AN", "AN", "KI", "KI", "DI", "DI", "KI"],
        ["AN", "KI", "KI", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["DI", "KI", "KI", "KI", "DI", "DI", "KI", "KI", "AN"],
        ["KI", "AN", "AN", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["DI", "DI", "KI", "KI", "AN", "AN", "KI", "KI", "DI"],
        ["AN", "DI", "DI", "KI", "KI", "AN", "KI", "KI", "DI"],
        ["DI", "KI", "DI", "KI", "DI", "KI", "AN", "KI", "KI"],
        ["KI", "KI", "DI", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["AN", "AN", "DI", "KI", "KI", "AN", "KI", "DI", "DI"],
        ["DI", "DI", "DI", "KI", "KI", "AN", "KI", "KI", "AN"],
        ["KI", "AN", "KI", "DI", "KI", "AN", "KI", "DI", "DI"],
        ["DI", "KI", "DI", "KI", "DI", "KI", "DI", "AN", "KI"],
        ["AN", "KI", "AN", "KI", "KI", "DI", "DI", "KI", "AN"],
        ["KI", "KI", "DI", "DI", "KI", "KI", "DI", "AN", "KI"],
        ["DI", "DI", "DI", "DI", "KI", "AN", "KI", "DI", "KI"],
        ["AN", "KI", "AN", "DI", "KI", "KI", "DI", "DI", "AN"],
        ["KI", "KI", "KI", "DI", "KI", "DI", "KI", "KI", "DI"],
        ["DI", "AN", "DI", "KI", "AN", "DI", "KI", "KI", "KI"],
        ["AN", "AN", "KI", "KI", "DI", "DI", "KI", "KI", "DI"],
        ["KI", "DI", "KI", "DI", "KI", "KI", "DI", "DI", "KI"],
        ["DI", "DI", "AN", "AN", "KI", "KI", "DI", "KI", "KI"],
        ["AN", "KI", "KI", "DI", "KI", "AN", "KI", "KI", "DI"],
        ["DI", "KI", "KI", "KI", "DI", "DI", "KI", "KI", "AN"],
        ["KI", "AN", "AN", "DI", "KI", "AN", "KI", "DI", "DI"],
        ["DI", "DI", "KI", "KI", "AN", "AN", "KI", "KI", "DI"],
        ["AN", "DI", "DI", "KI", "KI", "AN", "KI", "KI", "KI"],
        ["DI", "KI", "DI", "KI", "DI", "KI", "DI", "KI", "AN"],
    ]


def load_local_jsonl(path: str) -> list:
    """
    Load sequences from a JSONL file (one JSON object per line).
    Expected format: {"record_id": "...", "tokens": ["SIG1", "SIG2", ...]}
    or: ["SIG1", "SIG2", ...] per line
    """
    sequences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    sequences.append(obj)
                elif isinstance(obj, dict) and "tokens" in obj:
                    sequences.append(obj["tokens"])
                elif isinstance(obj, dict) and "sign_ids" in obj:
                    sequences.append(obj["sign_ids"])
            except json.JSONDecodeError:
                # Fallback: treat as space-separated tokens
                sequences.append(line.split())
    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Blind formal language analysis of ancient sign streams"
    )
    parser.add_argument(
        "--corpus",
        choices=["hieroglyphs", "cuneiform", "python", "local"],
        default="hieroglyphs",
        help="Corpus to analyze",
    )
    parser.add_argument(
        "--path",
        type=str, default=None,
        help="Path to local JSONL corpus (required with --corpus local)",
    )
    parser.add_argument(
        "--sequences", type=int, default=40,
        help="Number of sequences to analyze (default: 40)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full stack including null models",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for JSON report",
    )
    # Sophistication stack options
    parser.add_argument("--no-grammar", action="store_true", help="Skip grammar induction")
    parser.add_argument("--no-complexity", action="store_true", help="Skip complexity analysis")
    parser.add_argument("--no-geometry", action="store_true", help="Skip geometry channel")
    parser.add_argument("--no-nulls", action="store_true", help="Skip null model controls")
    # Validation options
    parser.add_argument("--validate", action="store_true",
                        help="Run validation harness (held-out splits, bootstrap CI, baseline self-classification)")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations for CI (default: 1000)")
    parser.add_argument("--heldout", type=float, default=0.25,
                        help="Fraction of tablets to hold out for validation (default: 0.25)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline self-classification in validation (faster)")

    args = parser.parse_args()

    # ── Load sequences ─────────────────────────────────────────────────
    if args.corpus == "local":
        if not args.path:
            print("[ERROR] --path required with --corpus local")
            return 1
        try:
            sequences = load_local_jsonl(args.path)
            corpus_name = Path(args.path).stem
            print(f"[Loaded] Local corpus: {len(sequences)} texts from {args.path}")
        except Exception as e:
            print(f"[ERROR] Could not load {args.path}: {e}")
            return 1
    elif args.corpus == "hieroglyphs":
        sequences = load_hieroglyph_sample()
        corpus_name = "hieroglyphs"
    elif args.corpus == "cuneiform":
        sequences = load_cuneiform_sample()
        corpus_name = "cuneiform"
    else:
        # Python control
        keywords = ["def", "return", "if", "else", "for", "in", "import"]
        identifiers = ["x", "y", "data", "result", "val"]
        sequences = []
        for i in range(40):
            if i % 3 == 0:
                sequences.append(
                    ["def", identifiers[i % 5], "(",
                     identifiers[(i+1) % 5], ")", ":",
                     "return", identifiers[i % 5], "+", "1"]
                )
            elif i % 3 == 1:
                sequences.append(
                    ["if", identifiers[i % 5], "==",
                     "0", ":", identifiers[(i+2) % 5], "=", "1"]
                )
            else:
                sequences.append(
                    ["for", identifiers[i % 5], "in",
                     identifiers[(i+1) % 5], ":",
                     "print", "(", identifiers[i % 5], ")"]
                )
        corpus_name = "python_control"

    sequences = sequences[: args.sequences]

    print(f"\n{'='*70}")
    print(f"BLIND FORMAL LANGUAGE DISCOVERY STACK")
    print(f"Corpus: {corpus_name}")
    print(f"Sequences: {len(sequences)}")
    print(f"Total signs: {sum(len(s) for s in sequences)}")
    print(f"{'='*70}")

    output_dir = Path(args.output) if args.output else None

    # ── Validation mode ────────────────────────────────────────────────
    if args.validate:
        print(f"\n{'='*70}")
        print(f"VALIDATION HARNESS")
        print(f"Bootstrap iterations: {args.bootstrap}")
        print(f"Held-out ratio: {args.heldout}")
        print(f"{'='*70}")

        validator = Validator(
            sequences=sequences,
            corpus_name=corpus_name,
            held_out_ratio=args.heldout,
            bootstrap_n=args.bootstrap,
            random_seed=42,
        )

        result = validator.run_full_validation()
        summary = validation_summary(result)
        print("\n" + summary)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"validation_{corpus_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"\n[Saved] Validation report to {out_file}")

        return 0

    # ── Sophistication scorecard mode ─────────────────────────────────
    report = generate_report(
        sequences=sequences,
        corpus_name=corpus_name,
        output_dir=output_dir,
        run_grammar=not args.no_grammar,
        run_complexity=not args.no_complexity,
        run_geometry=not args.no_geometry,
        run_nulls=(not args.no_nulls) and args.full,
    )

    print(report.to_text())

    # Print validation note
    print("""
NOTE: For credible claims, run with --validate first:
    python blind_pattern.py --corpus hieroglyphs --validate --bootstrap 1000

VERDICT LADDER (new):
  LEVEL_0: NO_EVIDENCE              - Indistinguishable from null
  LEVEL_1: STRUCTURED_SYMBOLIC      - Beats shuffle/Markov on key metrics
  LEVEL_2: FORMAL_GRAMMAR_CANDIDATE - Held-out fields/grammar replicate
  LEVEL_3: NOVEL_FORMAL_SYSTEM      - Far from all baselines, replicates across corpora
  LEVEL_4: DUAL_CHANNEL_ENCODING     - Geometry survives preregistered controls

NOT INFERABLE: author identity, Anunnaki origin, historical purpose, alien tech.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
