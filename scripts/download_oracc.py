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
    "etcsri": "https://build-oracc.museum.upenn.edu/json/etcsri.zip",
    "rinap": "https://build-oracc.museum.upenn.edu/json/rinap.zip",
}


def download_project(project: str, output_dir: Path, use_alt: bool = False, timeout: int = 300):
    """Download an ORACC project JSON zip."""
    import ssl
    import urllib.request

    if use_alt and project in ORACC_ALT_URLS:
        url = ORACC_ALT_URLS[project]
    else:
        url = ORACC_URLS.get(project, ORACC_URLS["etcsri"])
    zip_path = output_dir / f"{project.replace('/', '_')}.zip"
    extract_dir = output_dir / project.replace('/', '_')

    print(f"Downloading {project} from {url}")
    print(f"  -> {zip_path}")

    # Bypass SSL verification (ORACC cert chain issues on some hosts)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
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
    Handles the cdl/gdl nested structure (ORACC build-oracc format).
    Each .json file is one text record. Returns list of sign ID lists.
    """
    sequences = []
    texts_seen = 0
    signs_seen = 0

    for json_file in json_dir.rglob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        # ORACC JSON: each file is one text object (not a list)
        # Structure: {"type":"cdl","textid":"Q000376", "cdl":[...]}
        tokens = extract_sign_tokens(data)
        if tokens:
            sequences.append(tokens)
            texts_seen += 1
            signs_seen += len(tokens)

    print(f"  Parsed {texts_seen} texts, {signs_seen} total signs")
    return sequences


def extract_sign_tokens(text: dict) -> list:
    """
    Extract raw sign IDs from an ORACC text object.
    Handles the cdl/gdl nested structure (ORACC build-oracc format).
    Returns list of sign ID strings (opaque tokens, no lemmas/translations).

    ORACC structure:
      text.cdl[].cdl[].cdl[].cdl[].f.gdl[].seq[].gdl_sign  <- sign inside seq
      text.cdl[].cdl[].cdl[].cdl[].f.gdl[].gdl_sign         <- sign directly on gdl
    """
    tokens = []

    def walk(obj):
        if isinstance(obj, dict):
            # Always check for gdl_sign at this level
            if "gdl_sign" in obj:
                sid = obj["gdl_sign"]
                if isinstance(sid, str) and sid.strip():
                    tokens.append(sid.strip())
            # Walk into seq[] arrays (sign tokens inside form)
            if "seq" in obj and isinstance(obj["seq"], list):
                for item in obj["seq"]:
                    walk(item)
            # Walk into gdl[] arrays (form-level containers)
            if "gdl" in obj and isinstance(obj["gdl"], list):
                for item in obj["gdl"]:
                    walk(item)
            # Walk into cdl[] (ORACC node tree)
            if "cdl" in obj and isinstance(obj["cdl"], list):
                for child in obj["cdl"]:
                    walk(child)
            # Walk into f{} (form container)
            if "f" in obj and isinstance(obj["f"], dict):
                walk(obj["f"])
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
        extract_dir = download_project(args.project, oracc_dir, use_alt=args.alt)
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
