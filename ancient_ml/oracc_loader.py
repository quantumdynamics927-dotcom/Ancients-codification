"""
Oracc JSON Corpus Loader
=======================
Downloads and parses ORACC (Open Richly Annotated Cuneiform Texts) JSON corpora.

ORACC JSON format:
    http://oracc.iaas.upenn.edu/json/[PROJECT].zip

Supported projects:
    - etcsri: Electronic Text Corpus of the Sumerian Royal Inscriptions
    - cams/gkab: Gudea
    - rinap: Royal Inscriptions of the Neo-Assyrian Period
    - adart: Archi for Pre-Sargonic and Sargonic Texts

API:
    load_sign_sequences(project, max_texts=None) -> list[list[str]]
    load_determinative_pairs(project, max_texts=None) -> list[tuple[str, str]]

Usage:
    python oracc_loader.py --project etcsri --max-texts 100
    python oracc_loader.py --project etcsri --download
"""

import json
import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Third-party
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# =============================================================================
# CONFIGURATION
# =============================================================================

ORACC_JSON_BASE = "http://oracc.iaas.upenn.edu/json"
CACHE_DIR = Path(__file__).parent / "data" / "oracc"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Projects available as zip downloads
ORACC_PROJECTS = {
    "etcsri": "etcsri.zip",
    "cams/gkab": "cams/gkab.zip",
    "rinap": "rinap.zip",
    "adsd/adart1": "adsd/adart1.zip",
    "adsd/adart2": "adsd/adart2.zip",
    "cams/nimrud": "cams/nimrud.zip",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OraccText:
    """Single ORACC text with parsed fields."""
    id: str
    project: str
    sequence: List[str]  # Sign list
    lemmas: List[Tuple[str, str]]  # (reading, pos) tuples
    determinatives: List[str]  # Determinative sign list


@dataclass
class CorpusStats:
    """Statistics about loaded corpus."""
    n_texts: int
    n_signs: int
    n_unique_signs: int
    n_determinative_pairs: int
    projects: List[str]


# =============================================================================
# ZIP DOWNLOAD
# =============================================================================

def download_project_zip(project: str, cache_dir: Path = None) -> Path:
    """
    Download ORACC project zip file.

    Args:
        project: ORACC project name (e.g. 'etcsri')
        cache_dir: Local cache directory

    Returns:
        Path to downloaded zip file
    """
    if not HAS_REQUESTS:
        raise ImportError("requests library required: pip install requests")

    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_name = ORACC_PROJECTS.get(project, f"{project}.zip")
    zip_path = cache_dir / zip_name.replace("/", "_")

    # Already cached?
    if zip_path.exists():
        print(f"[Cache hit] {zip_path.name}")
        return zip_path

    url = f"{ORACC_JSON_BASE}/{zip_name}"
    print(f"[Downloading] {url}")
    print(f"[Target] {zip_path}")

    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[Saved] {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return zip_path


# =============================================================================
# JSON PARSING
# =============================================================================

class OraccTextWalker:
    """
    Recursive walker for ORACC JSON node trees.
    Defined as class to avoid pylint closure-in-loop warnings.
    """
    __slots__ = ("seq_acc", "lem_acc", "det_acc")

    def __init__(self):
        self.seq_acc: List[str] = []
        self.lem_acc: List[Tuple[str, str]] = []
        self.det_acc: List[str] = []

    def walk(self, node: Dict) -> None:
        """Recursively extract signs from node tree into acc lists."""
        node_type = node.get("type", "")

        # Determinative marker
        if node_type == "d":
            field_f = node.get("f", {})
            if isinstance(field_f, dict):
                for sign_node in field_f.get("cdl", []):
                    if isinstance(sign_node, dict) and sign_node.get("f"):
                        self.det_acc.append(sign_node["f"])
            # Recurse for children of d node
            for child_key in ("nodes", "children"):
                for child in node.get(child_key, []):
                    if isinstance(child, dict):
                        self.walk(child)
            return

        # Field node with cdl (sign list)
        field_f = node.get("f", {})
        if isinstance(field_f, dict):
            for sign_node in field_f.get("cdl", []):
                if isinstance(sign_node, dict):
                    if sign_node.get("f"):
                        self.seq_acc.append(sign_node["f"])
                    for lemma in sign_node.get("lemmas", []):
                        if isinstance(lemma, list) and len(lemma) >= 2:
                            self.lem_acc.append((lemma[0], lemma[1]))

        # Recurse into children
        for child_key in ("nodes", "children"):
            for child in node.get(child_key, []):
                if isinstance(child, dict):
                    self.walk(child)


def parse_oracc_json(blob: Dict[str, Any]) -> List[OraccText]:
    """
    Parse ORACC JSON blob into list of OraccText objects.

    ORACC JSON structure:
    {
      "project": "... ",
      "texts": [
        {
          "id": "pecsig001",
          "type": "document",
          "nodes": [
            {"type": "surface", ...},
            {"id": "...", "f": {"cdl": [...], "f": "...", "g": "..."},
             "lemmas": [["gi", "GN"], ...],
             "type": "line", ...},
            ...
          ]
        },
        ...
      ]
    }

    Sign extraction: walk nodes, collect 'f' field from field 'cdl'
    Determinative: nodes with 'type': 'd' (determinative)
    """
    texts = []
    project = blob.get("project", "unknown")

    for text_node in blob.get("texts", []):
        text_id = text_node.get("id", "unknown")

        walker = OraccTextWalker()
        walker.walk(text_node)

        if walker.seq_acc:  # Only store texts with actual signs
            texts.append(OraccText(
                id=text_id,
                project=project,
                sequence=walker.seq_acc,
                lemmas=walker.lem_acc,
                determinatives=walker.det_acc,
            ))

    return texts


def parse_zip( zip_path: Path) -> List[OraccText]:
    """
    Parse ORACC zip file into list of OraccText objects.

    Args:
        zip_path: Path to downloaded zip file

    Returns:
        List of parsed OraccText objects
    """
    all_texts = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find all .json files (one per text or one corpus blob)
        json_files = [f for f in zf.namelist() if f.endswith(".json")]

        # Check if it's a single corpus blob
        # ORACC zips contain: project/member.json or just *.json
        for json_file in json_files:
            with zf.open(json_file) as jf:
                try:
                    blob = json.load(jf)
                    # Single blob structure with 'texts' key
                    if isinstance(blob, dict) and "texts" in blob:
                        texts = parse_oracc_json(blob)
                        all_texts.extend(texts)
                    # List of text objects
                    elif isinstance(blob, list):
                        for item in blob:
                            if isinstance(item, dict) and "nodes" in item:
                                # Single text without project wrapper
                                single_blob = {"project": "unknown", "texts": [item]}
                                texts = parse_oracc_json(single_blob)
                                all_texts.extend(texts)
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping invalid JSON: {json_file}")
                    continue

    return all_texts


# =============================================================================
# MAIN API
# =============================================================================

def load_sign_sequences(
    project: str,
    max_texts: Optional[int] = None,
    use_cache: bool = True
) -> List[List[str]]:
    """
    Load sign sequences from ORACC project.

    Args:
        project: ORACC project name (e.g. 'etcsri')
        max_texts: Limit number of texts (None = all)
        use_cache: Use cached zip if available

    Returns:
        List of sign sequences, each sequence is a list of sign strings
    """
    # Download or load from cache
    if use_cache:
        zip_name = ORACC_PROJECTS.get(project, f"{project}.zip")
        cached_zip = CACHE_DIR / zip_name.replace("/", "_")
        if cached_zip.exists():
            zip_path = cached_zip
        else:
            zip_path = download_project_zip(project, CACHE_DIR)
    else:
        zip_path = download_project_zip(project, CACHE_DIR)

    # Parse
    texts = parse_zip(zip_path)

    if max_texts:
        texts = texts[:max_texts]

    return [text.sequence for text in texts]


def load_determinative_pairs(
    project: str,
    max_texts: Optional[int] = None,
    use_cache: bool = True
) -> List[Tuple[str, str]]:
    """
    Load (sign, determinative) pairs from ORACC project.

    Args:
        project: ORACC project name (e.g. 'etcsri')
        max_texts: Limit number of texts (None = all)
        use_cache: Use cached zip if available

    Returns:
        List of (sign, determinative) tuples
    """
    # Download or load from cache
    if use_cache:
        zip_name = ORACC_PROJECTS.get(project, f"{project}.zip")
        cached_zip = CACHE_DIR / zip_name.replace("/", "_")
        if cached_zip.exists():
            zip_path = cached_zip
        else:
            zip_path = download_project_zip(project, CACHE_DIR)
    else:
        zip_path = download_project_zip(project, CACHE_DIR)

    # Parse
    texts = parse_zip(zip_path)

    if max_texts:
        texts = texts[:max_texts]

    pairs = []
    for text in texts:
        for det in text.determinatives:
            # Determinative precedes the word it classifies
            # Find surrounding signs
            idx = text.sequence.index(det) if det in text.sequence else -1
            if idx > 0:
                pairs.append((text.sequence[idx - 1], det))
            elif idx == 0 and len(text.sequence) > 1:
                pairs.append((text.sequence[1], det))
            else:
                # Determinative not in sequence, skip
                pass

    return pairs


def get_corpus_stats(project: str, use_cache: bool = True) -> CorpusStats:
    """Get statistics about a corpus without loading all sequences."""
    if use_cache:
        zip_name = ORACC_PROJECTS.get(project, f"{project}.zip")
        cached_zip = CACHE_DIR / zip_name.replace("/", "_")
        if cached_zip.exists():
            zip_path = cached_zip
        else:
            zip_path = download_project_zip(project, CACHE_DIR)
    else:
        zip_path = download_project_zip(project, CACHE_DIR)

    texts = parse_zip(zip_path)
    all_signs = []
    all_pairs = []

    for text in texts:
        all_signs.extend(text.sequence)
        for det in text.determinatives:
            if det in text.sequence:
                idx = text.sequence.index(det)
                if idx > 0:
                    all_pairs.append((text.sequence[idx - 1], det))

    unique_signs = set(all_signs)

    return CorpusStats(
        n_texts=len(texts),
        n_signs=len(all_signs),
        n_unique_signs=len(unique_signs),
        n_determinative_pairs=len(all_pairs),
        projects=[project],
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ORACC JSON Corpus Loader")
    parser.add_argument("--project", default="etcsri",
                        help="ORACC project name (default: etcsri)")
    parser.add_argument("--download", action="store_true",
                        help="Force re-download even if cached")
    parser.add_argument("--max-texts", type=int, default=None,
                        help="Limit number of texts to load")
    parser.add_argument("--stats", action="store_true",
                        help="Show corpus statistics only")
    parser.add_argument("--list-projects", action="store_true",
                        help="List available ORACC projects")
    parser.add_argument("--determinatives", action="store_true",
                        help="Load determinative pairs instead of sequences")
    parser.add_argument("--output", default=None,
                        help="Save sequences to JSON file")

    args = parser.parse_args()

    if args.list_projects:
        print("Available ORACC projects:")
        for proj, zip_name in ORACC_PROJECTS.items():
            print(f"  {proj:20s} -> {zip_name}")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"ORACC Loader: {args.project}")
    print(f"{'='*60}")

    try:
        # Stats only
        if args.stats:
            stats = get_corpus_stats(args.project, use_cache=not args.download)
            print("\nCorpus Statistics:")
            print("  Texts:", stats.n_texts)
            print("  Total signs:", stats.n_signs)
            print("  Unique signs:", stats.n_unique_signs)
            print("  Determinative pairs:", stats.n_determinative_pairs)
        else:
            # Load sequences or pairs
            if args.determinatives:
                data = load_determinative_pairs(
                    args.project,
                    max_texts=args.max_texts,
                    use_cache=not args.download
                )
                print(f"\nLoaded {len(data)} (sign, determinative) pairs")
                print("\nFirst 10 pairs:")
                for i, (sign, det) in enumerate(data[:10]):
                    print(f"  {sign} <- {det}")
            else:
                data = load_sign_sequences(
                    args.project,
                    max_texts=args.max_texts,
                    use_cache=not args.download
                )
                print(f"\nLoaded {len(data)} texts")
                total_signs = sum(len(seq) for seq in data)
                unique_signs = set(s for seq in data for s in seq)
                print(f"Total signs: {total_signs}")
                print(f"Unique signs: {len(unique_signs)}")

                if data:
                    print("\nFirst text (first 20 signs):")
                    print(f"  {' '.join(data[0][:20])}")

            # Save to file
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"\n[Saved] {output_path}")

    except ImportError as e:
        print(f"\n[ERROR] {e}")
        print("Install required: pip install requests")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
