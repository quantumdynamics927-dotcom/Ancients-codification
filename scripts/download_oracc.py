"""
ORACC Downloader + Blind Sign Exporter
=====================================
Downloads ORACC JSON zip archives and exports sign-only streams.
Run from repo root:  python scripts/download_oracc.py --project etcsri
"""

import json
import sys
import zipfile
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ORACC_URLS = {
    "etcsri": "http://oracc.iaas.upenn.edu/json/etcsri.zip",
    "rinap": "http://oracc.iaas.upenn.edu/json/rinap.zip",
    "cams/gkab": "http://oracc.iaas.upenn.edu/json/cams_gkab.zip",
}
ORACC_ALT_URLS = {
    "etcsri": "http://build-oracc.museum.upenn.edu/json/etcsri.zip",
}


def download_project(project: str, output_dir: Path, timeout: int = 300):
    """Download an ORACC project JSON zip."""
    import urllib.request

    url = ORACC_URLS.get(project, ORACC_URLS["etcsri"])
    zip_path = output_dir / f"{project.replace('/', '_')}.zip"
    extract_dir = output_dir / project.replace('/', '_')

    print(f"Downloading {project} from {url}")
    print(f"  -> {zip_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(zip_path, "wb") as f:
        f.write(data)
    print(f"  Downloaded {len(data) / 1e6:.1f} MB")

    print(f"Extracting to {extract_dir}")
    if extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("  Done.")
    return extract_dir


def parse_oracc_signs(json_dir: Path, project: str):
    """
    Parse ORACC JSON files and extract raw sign sequences.
    Returns list of sign ID lists (opaque tokens).
    """
    sequences = []
    texts_seen = 0
    signs_seen = 0

    for json_file in json_dir.rglob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # ORACC JSON structure: list of text objects
        if not isinstance(data, list):
            continue

        for text in data[:500]:  # Cap per project
            # Extract sign tokens from the text
            tokens = extract_sign_tokens(text)
            if tokens:
                sequences.append(tokens)
                texts_seen += 1
                signs_seen += len(tokens)

    print(f"  Parsed {texts_seen} texts, {signs_seen} total signs")
    return sequences


def extract_sign_tokens(text: dict) -> list:
    """
    Extract raw sign IDs from an ORACC text object.
    Returns list of sign ID strings (opaque tokens, no lemmas/translations).
    """
    tokens = []

    # ORACC JSON: texts contain a "surface" or "lines" with "tokens"
    # Each token may have: id, form (transliteration), sign_id
    def walk(obj):
        if isinstance(obj, dict):
            # Check for sign_id field (the actual sign ID, not transliteration)
            if "sign_id" in obj:
                sid = obj["sign_id"]
                if isinstance(sid, str) and sid.strip():
                    tokens.append(sid.strip())
            # Check for sub_tokens (composite signs)
            if "sub_tokens" in obj and isinstance(obj["sub_tokens"], list):
                for st in obj["sub_tokens"]:
                    walk(st)
            # Check for tokens list
            if "tokens" in obj and isinstance(obj["tokens"], list):
                for tok in obj["tokens"]:
                    walk(tok)
            # Check for lines
            if "lines" in obj and isinstance(obj["lines"], list):
                for line in obj["lines"]:
                    walk(line)
            # Check for surfaces
            if "surfaces" in obj and isinstance(obj["surfaces"], list):
                for surf in obj["surfaces"]:
                    walk(surf)
            # Check for composition (some ORACC formats)
            if "composition" in obj and isinstance(obj["composition"], list):
                for item in obj["composition"]:
                    walk(item)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(text)
    return tokens


def export_blind_jsonl(sequences: list, output_path: Path):
    """Export sequences to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            record = {
                "record_id": f"oracc:{i}",
                "line_index": i,
                "tokens": seq,
                "blind_mode": True,
                "analysis_ready": True,
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Exported {len(sequences)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download ORACC corpus and export blind sign streams")
    parser.add_argument("--project", default="etcsri", choices=list(ORACC_URLS.keys()),
                        help="ORACC project name")
    parser.add_argument("--alt", action="store_true",
                        help="Use alternate build-oracc.museum.upenn.edu URL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: datasets/blind/{project}_signs.jsonl)")
    parser.add_argument("--max-texts", type=int, default=500,
                        help="Max texts per project (default: 500)")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "datasets"
    oracc_dir = dataset_dir / "oracc"
    blind_dir = dataset_dir / "blind"

    oracc_dir.mkdir(parents=True, exist_ok=True)
    blind_dir.mkdir(parents=True, exist_ok=True)

    # Download
    try:
        extract_dir = download_project(args.project, oracc_dir)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("Try with --alt or download manually from:")
        print(f"  {ORACC_URLS[args.project]}")
        print(f"  {ORACC_ALT_URLS.get(args.project, '')}")
        return 1

    # Parse and export
    print(f"\nParsing sign sequences from {extract_dir}")
    sequences = parse_oracc_signs(extract_dir, args.project)
    if not sequences:
        print("[ERROR] No sign sequences found. Check JSON structure.")
        return 1

    output_path = Path(args.output) if args.output else (
        blind_dir / f"{args.project.replace('/', '_')}_signs.jsonl"
    )
    export_blind_jsonl(sequences, output_path)

    print(f"\nTo run blind analysis:")
    print(f"  python ancient_ml/blind_pattern.py --corpus local --path {output_path} --validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
