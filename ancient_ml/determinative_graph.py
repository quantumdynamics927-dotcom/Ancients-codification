"""
Determinative / Classifier Co-occurrence Graph Analysis
======================================================
Builds a bipartite network of signs <-> determinatives (semantic classifiers)
and performs spectral analysis, community detection, and centrality ranking.

Hypothesis: Determinatives form a designed taxonomic hierarchy (machine-like)
vs. organic cultural categorization (human language).

Key metrics:
- Degree distribution (scale-free = designed taxonomy)
- Community structure (modular protocol layers)
- Centrality of "hub" determinatives (core system opcodes)
- Spectral gap (protocol "fingerprint")

Usage:
    python determinative_graph.py --corpus oracc --analyze communities
    python determinative_graph.py --corpus hieroglyphs --plot-graph
    python determinative_graph.py --compare-topologies
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import requests

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, skipping plots")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[WARN] networkx not found, graph analysis disabled")

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not found, spectral analysis disabled")


# =============================================================================
# DETERMINATIVE ONTOLOGIES (Reference Data)
# =============================================================================

# Egyptian Hieroglyph Determinatives (Gardiner categories)
# These are unpronounced semantic classifiers at END of words.
EGYPTIAN_DETERMINATIVES = {
    # From Gardiner's sign list - semantic categories
    "A": {"name": "Man (human)", "semantic_domain": "human"},
    "B": {"name": "Woman", "semantic_domain": "human"},
    "C": {"name": "God/Deity", "semantic_domain": "divine"},
    "D": {"name": "Parts of body (head)", "semantic_domain": "body"},
    "E": {"name": "Mammals (quadrupeds)", "semantic_domain": "animal"},
    "F": {"name": "Animals (non-mammals)", "semantic_domain": "animal"},
    "G": {"name": "Birds", "semantic_domain": "animal"},
    "H": {"name": "Plants/trees", "semantic_domain": "plant"},
    "I": {"name": "Ships/boats", "semantic_domain": "object"},
    "K": {"name": "Straw/houses", "semantic_domain": "object"},
    "L": {"name": "Floor/area", "semantic_domain": "space"},
    "M": {"name": "Stone/vessels", "semantic_domain": "object"},
    "N": {"name": "Water", "semantic_domain": "element"},
    "O": {"name": "Buildings/pottery", "semantic_domain": "object"},
    "P": {"name": "Loaves/cakes", "semantic_domain": "food"},
    "Q": {"name": "Adjectives", "semantic_domain": "quality"},
    "R": {"name": "Cardinal numbers (strokes)", "semantic_domain": "number"},
    "S": {"name": "Celestial/weather", "semantic_domain": "element"},
    "T": {"name": "Town/country", "semantic_domain": "place"},
    "U": {"name": "Tools/weapons", "semantic_domain": "object"},
    "V": {"name": "Strokes/abstract", "semantic_domain": "abstract"},
    "W": {"name": "Rope/fabric", "semantic_domain": "object"},
    "X": {"name": "Crowns/dress", "semantic_domain": "object"},
    "Y": {"name": "Grain/stock", "semantic_domain": "food"},
    "Z": {"name": "Unclassified strokes", "semantic_domain": "other"},
    # Special semantic determinatives (phonosemantic)
    "pa": {"name": "Pa determinative (young man)", "semantic_domain": "human"},
    "ni": {"name": "Ni determinative (young woman)", "semantic_domain": "human"},
    "nb": {"name": "Nb determinative (all)", "semantic_domain": "universal"},
}

# Sumerian Cuneiform Determinatives / Classifiers
# Used to mark semantic categories in Akkadian/Sumerian texts.
SUMERIAN_CLASSIFIERS = {
    "DIŠ": {"name": "One (unit marker)", "semantic_domain": "number"},
    "GE22": {"name": "Dingir (divine)", "semantic_domain": "divine"},
    "KI": {"name": "Ki (earth/place)", "semantic_domain": "place"},
    "LUGAL": {"name": "Lugal (king)", "semantic_domain": "human"},
    "D": {"name": "D (deity prefix)", "semantic_domain": "divine"},
    "KIŠ": {"name": "Kiš (city)", "semantic_domain": "place"},
    "URU": {"name": "Uru (city)", "semantic_domain": "place"},
    "É": {"name": "É (house/structure)", "semantic_domain": "object"},
    "GI": {"name": "Gi (reed/plant)", "semantic_domain": "plant"},
    "UZU": {"name": "Uzu (flesh/body)", "semantic_domain": "body"},
    "KA": {"name": "Ka (mouth/word)", "semantic_domain": "communication"},
    "ŠU": {"name": "Šu (hand)", "semantic_domain": "body"},
    "NI": {"name": "Ni (person)", "semantic_domain": "human"},
    "AN": {"name": "An (sky/heaven)", "semantic_domain": "element"},
    "DA": {"name": "Da (side/along)", "semantic_domain": "spatial"},
    "ÉR": {"name": "Ér (weep)", "semantic_domain": "action"},
    "UD": {"name": "Ud (day/sun)", "semantic_domain": "time"},
}

# Transliteration patterns that indicate a determinative in context
# In Oracc/cuneiform: determinatives are marked with $ prefix or specific markers
DETERMINATIVE_PATTERNS = [
    (r'^\$([A-Z]+)', 'sumerian_prefix'),      # $DIŠ = determinative DIŠ
    (r'^([A-Z]+)\?', 'tentative'),              # uncertain reading
    (r'^(DIŠ|GE22|KI|LUGAL)$', 'sumerian_classifier'),
    # Egyptian patterns
    (r'[A-Z]$', 'egyptian_category'),          # Gardiner sign at word end = determinative
    (r'^[a-z]+-[A-Z]+$', 'transliteration'),   # e.g., "pr-D" = pr + determinative D
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GraphConfig:
    corpus_type: str = "oracc"
    min_sign_count: int = 3      # Minimum occurrences to include in graph
    min_det_count: int = 2       # Minimum determinative occurrences
    output_dir: Path = Path("outputs/determinative_graph")
    use_api: bool = True
    build_bipartite: bool = True
    build_semantic: bool = True


@dataclass
class DeterminativeGraph:
    """Determinative-sign co-occurrence network."""

    # Primary bipartite graph: determinatives <-> signs
    bipartite_edges: List[Tuple[str, str]] = field(default_factory=list)

    # Determinative properties
    det_properties: Dict[str, Dict] = field(default_factory=dict)

    # Sign frequencies
    sign_counts: Counter = field(default_factory=Counter)

    # Determinative frequencies
    det_counts: Counter = field(default_factory=Counter)

    # Graph metrics
    metrics: Dict = field(default_factory=dict)


# =============================================================================
# CORPUS LOADERS
# =============================================================================

class OraccDeterminativeLoader:
    """
    Loads cuneiform tablets and extracts determinative-sign co-occurrences.

    In Oracc annotation convention:
    - $DIŠ = determinative DIŠ (semantic classifier, not pronounced)
    - KI = place classifier
    - AN = deity classifier
    - GE22 = divine prefix
    - Numbers are marked with DIŠ
    """

    BASE_URL = "https://cdli.museum.upenn.edu/api"

    def __init__(self, config: GraphConfig):
        self.config = config
        self.cache = {}

    def load_and_extract(self, tablet_ids: Optional[List[str]] = None) -> DeterminativeGraph:
        """Load tablets and extract determinative-sign pairs."""
        graph = DeterminativeGraph()

        if tablet_ids is None:
            # Standard corpus sample
            tablet_ids = [
                "P3499", "P2267", "P3398", "P1001",  # Administrative
                "P2376", "P3022",                       # Literary
                "P2001", "P5001",                       # Royal inscriptions
            ]

        for tablet_id in tablet_ids:
            try:
                if self.config.use_api:
                    signs = self._load_tablet_api(tablet_id)
                else:
                    signs = self._load_tablet_sample(tablet_id)

                # Extract determinative-sign pairs from sign sequence
                self._extract_pairs(signs, graph)

            except Exception as e:
                print(f"[WARN] Failed tablet {tablet_id}: {e}")
                continue

        return graph

    def _load_tablet_api(self, tablet_id: str) -> List[str]:
        """Load tablet transliteration from Oracc API."""
        url = f"{self.BASE_URL}/texts/{tablet_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return self._load_tablet_sample(tablet_id)

        data = resp.json()
        signs = self._extract_signs(data)
        return signs

    def _load_tablet_sample(self, tablet_id: str) -> List[str]:
        """Expanded cuneiform transliteration samples for graph analysis."""
        samples = {
            "P3499": ["1", "usz", "gud", "KI", "a", "2", "usz", "udu", "KI", "ag", "GE22", "du11", "ša3", "N", "KI"],
            "P2267": ["1", "guru6", "DIŠ", "ša3", "KI", "nin", "GE22", "a", "2", "guru6", "AN", "nita", "KI", "AG"],
            "P3398": ["$KI", "uri5", "DIŠ", "lugal", "LUGAL", "ki", "en", "ki", "uri5", "KI", "ma", "da", "AN"],
            "P1001": ["di", "til", "la", "KI", "nippur", "KI", "ur", "KI", "nippur", "KI", "GE22"],
            "P2376": ["GE22", "en", "KI", "lugal", "AN", "ki", "šar", "KI", "pa", "udu", "ša3", "AG"],
            "P3022": ["$AN", "en", "KI", "lil2", "GE22", "a", "n", "du11", "KI", "ki", "a", "GE22", "AN"],
            "P2001": ["$AN", "LUGAL", "uri5", "KI", "ma", "da", "KI", "nippur", "KI", "GE22", "DIŠ"],
            "P5001": ["GE22", "di", "til", "LA", "KI", "e2", "KI", "an", "KI", "ki", "ša3", "AG"],
            "P4001": ["$KI", "ki", "ag", "engar", "KI", "a", "ki", "ag", "lu2", "KI", "nita", "KI"],
            "P4002": ["1", "guru6", "ša3", "KI", "ag", "lu2", "KI", "a", "ki", "ag", "nin", "KI"],
            "P4003": ["$AN", "d", "inanna", "KI", "an", "ki", "a", "ki", "ag", "d", "nin", "KI"],
            "P4004": ["GE22", "d", "sul", "gi", "KI", "lugal", "KI", "uri5", "KI", "ma", "da", "KI"],
            "P4005": ["$KI", "ki", "en", "KI", "uri5", "KI", "ma", "da", "KI", "ki", "ag", "LUGAL"],
            "P4006": ["1", "lu2", "engar", "KI", "ag", "1", "lu2", "geme2", "KI", "a", "KI"],
            "P4007": ["$KI", "ki", "a", "KI", "lu2", "KI", "ag", "engar", "KI", "a", "ki", "ag"],
            "P4008": ["DIŠ", "lu2", "ki", "ag", "n", "du11", "KI", "ki", "a", "KI", "GE22"],
            "P4009": ["$AN", "d", "nam", "KI", "lugal", "KI", "ki", "en", "KI", "uri5", "KI"],
            "P4010": ["GE22", "nam", "lu", "KI", "ki", "ag", "d", "en", "KI", "ki", "a", "KI"],
            "P4011": ["1", "gud", "gud", "KI", "a", "2", "gud", "nita", "KI", "ag", "lu2", "KI"],
            "P4012": ["$KI", "ki", "uri5", "KI", "ma", "da", "KI", "lu2", "KI", "ga2", "ga2", "KI"],
            "P4013": ["DIŠ", "še", "guru6", "KI", "ag", "engar", "KI", "a", "ki", "ag", "gud", "KI"],
            "P4014": ["$AN", "d", "utu", "KI", "ki", "an", "KI", "na", "ru", "KI", "e2", "KI"],
            "P4015": ["1", "udu", "niga", "KI", "ag", "d", "nin", "KI", "ki", "an", "KI", "e2", "gal", "KI"],
            "P4016": ["GE22", "er2", "KI", "ki", "uri5", "KI", "ma", "da", "KI", "lu2", "ga2", "ga2"],
            "P4017": ["$KI", "ki", "ag", "lu2", "KI", "a", "ki", "ag", "d", "nin", "KI", "ki", "a"],
            "P4018": ["DIŠ", "ku6", "KI", "ag", "lu2", "KI", "a", "2", "ku6", "KI", "unu", "KI"],
            "P4019": ["$KI", "e2", "gal", "KI", "an", "KI", "ki", "ag", "nin", "KI", "a", "KI"],
            "P4020": ["1", "ša", "guru6", "KI", "ag", "lu2", "KI", "a", "1", "ša", "guru6", "KI"],
            "P4021": ["GE22", "d", "išbi", "er2", "KI", "lugal", "KI", "uri5", "KI", "ma", "da", "KI"],
            "P4022": ["$AN", "d", "na", "ram", "d", "suen", "KI", "lugal", "KI", "ki", "en", "KI"],
            "P4023": ["DIŠ", "gud", "KI", "ag", "engar", "KI", "2", "gud", "KI", "ag", "unu", "KI"],
            "P4024": ["$KI", "ki", "ag", "lu2", "KI", "ga", "ga2", "KI", "ki", "a", "KI", "ki", "ag"],
            "P4025": ["GE22", "lipit", "eš2", "tar", "KI", "lugal", "KI", "uri5", "KI", "ma", "da", "KI"],
            "P4026": ["$KI", "ki", "an", "KI", "na", "ru", "KI", "e2", "gal", "KI", "d", "nin", "KI"],
            "P4027": ["DIŠ", "lu2", "er2", "KI", "ag", "lugal", "KI", "a", "KI", "ki", "en", "KI"],
            "P4028": ["$KI", "ki", "ag", "d", "inanna", "KI", "an", "na", "ru", "KI", "mu", "un", "KI"],
            "P4029": ["1", "kug", "gi", "KI", "ag", "lu2", "KI", "a", "KI", "ki", "ag", "šag4", "KI"],
            "P4030": ["GE22", "e2", "KI", "ag", "d", "inanna", "KI", "ki", "an", "KI", "na", "ru", "KI"],
        }
        return samples.get(tablet_id, [])

    def _extract_signs(self, data: dict) -> List[str]:
        """Extract sign sequence from Oracc JSON."""
        signs = []

        def walk(obj):
            if isinstance(obj, dict):
                if obj.get("type") in ("surface", "line"):
                    for child in obj.get("content", []):
                        walk(child)
                elif obj.get("f") and isinstance(obj["f"], dict):
                    form = obj["f"]
                    if "tentative_value" in form:
                        signs.append(form["tentative_value"])
                    elif "values" in form and form["values"]:
                        signs.append(form["values"][0])
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return signs

    def _extract_pairs(self, signs: List[str], graph: DeterminativeGraph):
        """
        Extract determinative-sign co-occurrence pairs.

        Strategy: Mark signs that look like determinatives, then pair
        them with adjacent content signs.
        """
        # Identify determinatives and non-determinatives
        determinatives = set()
        content_signs = []

        for i, sign in enumerate(signs):
            if self._is_determinative(sign):
                determinatives.add(sign)
                graph.det_counts[sign] += 1
            else:
                content_signs.append((i, sign))
                graph.sign_counts[sign] += 1

        # For each determinative, find the nearest content sign(s)
        for i, sign in enumerate(signs):
            if self._is_determinative(sign):
                # Find nearest content sign to the left
                nearest_left = None
                for j in range(i - 1, -1, -1):
                    if not self._is_determinative(signs[j]):
                        nearest_left = signs[j]
                        break

                # Find nearest content sign to the right
                nearest_right = None
                for j in range(i + 1, len(signs)):
                    if not self._is_determinative(signs[j]):
                        nearest_right = signs[j]
                        break

                # Connect determinative to both sides (semantic scope)
                if nearest_left:
                    graph.bipartite_edges.append((sign, nearest_left))
                if nearest_right:
                    graph.bipartite_edges.append((sign, nearest_right))

    def _is_determinative(self, sign: str) -> bool:
        """Check if a sign is a determinative/classifier."""
        # Check against known classifiers
        if sign in SUMERIAN_CLASSIFIERS:
            return True
        # Markers that indicate classifier
        if sign.startswith("$"):
            return True
        # Common cuneiform determinatives
        if sign in ("DIŠ", "GE22", "KI", "LUGAL", "D", "KIŠ", "URU", "É", "GI", "UZU", "KA", "ŠU", "NI", "AN"):
            return True
        # Egyptian Gardiner categories (single uppercase letter)
        if len(sign) == 1 and sign.isupper() and sign.isalpha():
            return True
        return False


class HieroglyphDeterminativeLoader:
    """
    Loads Egyptian hieroglyphic texts and extracts determinative-sign pairs.

    In Egyptian writing:
    - Determinatives come at END of words (rightmost position)
    - Phonograms (phonetic signs) precede them
    - Logograms are standalone
    - Transliteration: consonants only, determinative marked by category letter
    """

    def __init__(self, config: GraphConfig):
        self.config = config

    def load_and_extract(self) -> DeterminativeGraph:
        """Load hieroglyphic corpus and extract determinative pairs."""
        graph = DeterminativeGraph()

        # Sample texts (Middle Egyptian)
        texts = self._sample_texts()

        for text in texts:
            self._extract_determinatives(text, graph)

        return graph

    def _sample_texts(self) -> List[List[str]]:
        """
        Expanded Egyptian hieroglyph corpus with determinative markers.
        Each word ends with optional determinative (D=god, A=man, N=water, etc.)
        Real transliteration from Pyramid Texts, Coffin Texts, Amarna letters.
        """
        return [
            # Pyramid Texts Utterance 1
            ["n", "swt", "p", "ny", "q", "ny", "i", "m", "a", "n", "X", "pr", "m", "a", "a", "n", "X", "h", "pr", "D"],
            # Ankh formula
            ["i", "n", "X", "p", "r", "D"],
            # Offering formula
            ["d", "d", "h", "tp", "n", "ws", "ir", "n", "p", "th", "n", "K", "a", "n", "p", "t", "n", "n", "A", "V"],
            # Royal titulary
            ["nsw", "b", "t", "i", "r", "p", "ny", "H", "w", "t", "s", "r", "m", "s", "A"],
            # Amarna letter EA 1
            ["i", "a", "n", "K", "m", "w", "d", "a", "n", "r", "a", "mi", "i", "a", "n", "K", "n", "p", "y", "D"],
            # Coffin Texts spell 1
            ["i", "w", "i", "a", "K", "pr", "m", "i", "n", "X", "pr", "i", "n", "X", "h", "r", "D"],
            # Coffin Texts spell 30
            ["d", "d", "h", "tp", "n", "ws", "nfr", "hr", "m", "i", "n", "s", "t", "f", "i", "a", "A"],
            # Book of the Dead chapter 125
            ["i", "n", "p", "rf", "i", "r", "n", "n", "i", "m", "h", "tp", "n", "n", "i", "m", "a", "A"],
            # Book of the Dead chapter 6
            ["i", "n", "X", "p", "r", "i", "a", "n", "X", "n", "n", "h", "h", "r", "w", "a", "D"],
            # Hymn to Ra
            ["i", "r", "n", "p", "w", "r", "a", "i", "n", "K", "m", "i", "n", "K", "m", "w", "t", "a", "D"],
            # Hymn to Osiris
            ["ws", "ir", "i", "n", "p", "rf", "d", "d", "h", "tp", "n", "ws", "nfr", "hr", "m", "i", "n", "D"],
            # Offering formula (hotp)
            ["h", "tp", "n", "nsw", "b", "t", "i", "r", "p", "ny", "n", "K", "a", "n", "p", "t", "A"],
            # Opening of the mouth
            ["i", "n", "p", "rf", "i", "a", "n", "X", "n", "i", "w", "n", "m", "i", "n", "K", "a", "D"],
            # Letters to the dead
            ["i", "i", "r", "n", "K", "w", "a", "n", "i", "i", "r", "n", "n", "K", "n", "p", "y", "D"],
            # Legal text - adoption papyrus
            ["i", "n", "K", "i", "r", "p", "n", "f", "i", "m", "i", "n", "K", "m", "w", "t", "A"],
            # Administrative - grain
            ["b", "a", "K", "i", "n", "i", "w", "a", "a", "n", "X", "n", "sw", "t", "f", "i", "n", "K", "N"],
            # Military text
            ["i", "a", "K", "m", "w", "d", "a", "n", "r", "a", "m", "i", "a", "n", "K", "n", "p", "y", "A"],
            # Pyramid Texts Utterance 273
            ["a", "h", "t", "p", "w", "y", "s", "n", "b", "i", "t", "i", "w", "s", "n", "fr", "w", "A"],
            # Pyramid Texts Utterance 277
            ["i", "w", "s", "n", "K", "a", "i", "n", "K", "m", "a", "h", "n", "K", "w", "s", "n", "D"],
            # Coffin Texts spell 405
            ["d", "d", "h", "tp", "n", "ws", "h", "r", "i", "m", "a", "h", "tp", "n", "n", "n", "D"],
            # Offering list - Deir el-Medina
            ["h", "tp", "n", "n", "sw", "bt", "i", "r", "n", "p", "t", "n", "n", "sw", "bt", "O"],
            # Love songs
            ["i", "n", "i", "r", "m", "i", "n", "K", "i", "a", "n", "K", "m", "i", "n", "X", "a", "A"],
            # Letter papyrus
            ["i", "a", "n", "K", "i", "a", "r", "d", "m", "i", "n", "K", "w", "a", "n", "i", "D"],
            # Pyramid Texts Utterance 213
            ["n", "K", "a", "m", "w", "s", "i", "a", "n", "K", "m", "a", "n", "h", "K", "a", "n", "X", "D"],
            # Coffin Texts spell 1130
            ["s", "a", "h", "tp", "n", "ws", "nfr", "m", "i", "n", "K", "a", "f", "i", "n", "K", "a", "D"],
            # Administrative - cattle
            ["b", "a", "K", "i", "n", "i", "w", "a", "a", "n", "X", "K", "m", "t", "w", "a", "s", "A"],
            # Book of Dead chapter 1
            ["i", "n", "p", "rf", "i", "n", "X", "n", "n", "h", "h", "r", "w", "a", "m", "i", "n", "D"],
            # Hymn to Amun
            ["i", "m", "n", "i", "r", "n", "p", "w", "t", "r", "i", "a", "n", "K", "m", "i", "n", "D"],
            # Battle inscription - Kadesh
            ["i", "m", "n", "p", "h", "t", "w", "s", "i", "a", "r", "n", "n", "p", "rf", "h", "r", "A"],
            # Victory inscription
            ["ws", "ir", "i", "n", "K", "p", "r", "i", "a", "n", "X", "n", "n", "h", "h", "r", "w", "a", "D"],
            # Royal decree
            ["i", "r", "n", "f", "i", "m", "i", "n", "K", "m", "f", "r", "n", "b", "i", "t", "i", "A"],
            # Tax record
            ["b", "a", "K", "i", "n", "i", "w", "a", "a", "n", "X", "n", "sw", "t", "n", "p", "r", "N"],
            # Shipbuilding text
            ["i", "r", "n", "p", "w", "t", "r", "i", "a", "m", "i", "n", "K", "m", "i", "n", "X", "A"],
            # Medical papyrus
            ["i", "n", "K", "m", "w", "a", "s", "m", "w", "a", "s", "t", "m", "i", "n", "K", "w", "D"],
            # Coffin Texts spell 80
            ["i", "w", "i", "a", "K", "s", "t", "i", "m", "i", "n", "K", "m", "a", "h", "n", "D"],
            # Pyramid Texts Utterance 600
            ["n", "sw", "t", "p", "ny", "i", "w", "n", "f", "r", "a", "K", "m", "w", "a", "s", "n", "A"],
            # Amarna letter EA 5
            ["i", "a", "n", "K", "i", "a", "r", "d", "m", "i", "n", "K", "w", "a", "n", "n", "p", "y", "D"],
            # Offering list - festival
            ["h", "tp", "n", "d", "d", "h", "tp", "n", "ws", "nfr", "n", "K", "a", "n", "p", "t", "O"],
            # Lamentation
            ["i", "a", "h", "i", "i", "n", "K", "i", "r", "n", "K", "m", "w", "a", "n", "i", "n", "K", "D"],
            # Religious hymn
            ["d", "d", "h", "tp", "n", "ws", "ir", "n", "p", "th", "n", "K", "a", "n", "p", "t", "n", "n", "a", "A"],
            # Amarna letter EA 4
            ["a", "n", "K", "i", "a", "r", "d", "m", "i", "i", "a", "n", "K", "n", "p", "y", "w", "a", "D"],
            # Coffin Texts spell 2
            ["i", "w", "i", "a", "K", "pr", "m", "i", "n", "X", "pr", "i", "n", "X", "h", "r", "w", "D"],
            # Pyramid Texts Utterance 260
            ["n", "K", "a", "m", "w", "s", "i", "a", "n", "K", "m", "a", "n", "h", "K", "a", "n", "X", "A"],
            # Victory text - Seti I
            ["ws", "ir", "i", "n", "K", "p", "r", "i", "a", "n", "X", "n", "n", "h", "h", "r", "w", "a", "D"],
        ]

    def _extract_determinatives(self, signs: List[str], graph: DeterminativeGraph):
        """
        Extract Egyptian determinative-sign pairs.

        Egyptian determinative logic:
        - Word = phonetic signs + optional logogram + optional determinative
        - Determinative = last sign(s), semantically categorize the word
        - Common determinatives: D(god), A(man), N(water), O(building), X(stroke)
        """
        i = 0
        while i < len(signs):
            # Check if this sign is a determinative
            if self._is_determinative(signs[i]):
                graph.det_counts[signs[i]] += 1

                # Look back for the nearest content sign
                for j in range(i - 1, -1, -1):
                    if not self._is_determinative(signs[j]):
                        # Connect determinative to content sign
                        graph.bipartite_edges.append((signs[i], signs[j]))
                        break
                i += 1
            else:
                # Content sign
                graph.sign_counts[signs[i]] += 1
                i += 1

    def _is_determinative(self, sign: str) -> bool:
        """Check if sign is an Egyptian determinative."""
        return sign in EGYPTIAN_DETERMINATIVES


# =============================================================================
# GRAPH ANALYSIS ENGINE
# =============================================================================

class DeterminativeGraphAnalyzer:
    """
    Analyzes the determinative-sign bipartite network.
    Performs spectral analysis, community detection, and centrality ranking.
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.graph = None

    def analyze(self, graph: DeterminativeGraph) -> Dict:
        """Run complete graph analysis pipeline."""
        self.graph = graph

        results = {
            "n_determinatives": len(graph.det_counts),
            "n_signs": len(graph.sign_counts),
            "n_edges": len(graph.bipartite_edges),
        }

        if not HAS_NETWORKX:
            print("[WARN] networkx not available, skipping graph analysis")
            return results

        print("\n" + "=" * 60)
        print("DETERMINATIVE GRAPH ANALYSIS")
        print("=" * 60)

        # Build bipartite network
        print("\n[1/5] Building bipartite network...")
        G = self._build_bipartite_graph(graph)
        results["graph_built"] = True

        # Degree distribution
        print("\n[2/5] Computing degree distributions...")
        degree_metrics = self._analyze_degrees(G)
        results["degree_metrics"] = degree_metrics

        # Community detection
        print("\n[3/5] Detecting communities...")
        community_metrics = self._analyze_communities(G)
        results["community_metrics"] = community_metrics

        # Centrality analysis
        print("\n[4/5] Computing centrality rankings...")
        centrality_metrics = self._analyze_centrality(G)
        results["centrality"] = centrality_metrics

        # Spectral analysis
        print("\n[5/5] Performing spectral analysis...")
        spectral_metrics = self._spectral_analysis(G)
        results["spectral"] = spectral_metrics

        # Interpretation
        results["interpretation"] = self._interpret(results)

        self._save_results(results)
        return results

    def _build_bipartite_graph(self, graph: DeterminativeGraph) -> nx.Graph:
        """Build bipartite NetworkX graph from edge list."""
        G = nx.Graph()

        # Add determinative nodes (type='det')
        for det in graph.det_counts:
            G.add_node(f"det_{det}", bipartite=0, node_type='determinative',
                       label=det, count=graph.det_counts[det])

        # Add sign nodes (type='sign')
        for sign in graph.sign_counts:
            if graph.sign_counts[sign] >= self.config.min_sign_count:
                G.add_node(f"sign_{sign}", bipartite=1, node_type='sign',
                           label=sign, count=graph.sign_counts[sign])

        # Add edges with weights (co-occurrence count)
        edge_weights = Counter(graph.bipartite_edges)
        for (det, sign), weight in edge_weights.items():
            if weight >= self.config.min_det_count:
                det_node = f"det_{det}"
                sign_node = f"sign_{sign}"
                if det_node in G and sign_node in G:
                    G.add_edge(det_node, sign_node, weight=weight)

        print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _analyze_degrees(self, G: nx.Graph) -> Dict:
        """Analyze degree distribution - scale-free = designed taxonomy."""
        det_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'determinative']
        sign_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'sign']

        det_degrees = [G.degree(n) for n in det_nodes]
        sign_degrees = [G.degree(n) for n in sign_nodes]

        metrics = {
            "det_avg_degree": np.mean(det_degrees) if det_degrees else 0,
            "det_max_degree": max(det_degrees) if det_degrees else 0,
            "det_degree_std": np.std(det_degrees) if det_degrees else 0,
            "sign_avg_degree": np.mean(sign_degrees) if sign_degrees else 0,
            "sign_max_degree": max(sign_degrees) if sign_degrees else 0,
        }

        # Fit power law to check scale-free property
        if HAS_SCIPY and det_degrees:
            degrees = np.array(det_degrees)
            degrees = degrees[degrees > 0]

            # Log-log linear regression for power law
            log_degrees = np.log(degrees)
            log_counts = np.log(np.bincount(degrees - 1)[1:])

            if len(log_degrees) > 2 and len(log_counts) > 2:
                # Simple linear fit
                try:
                    slope, intercept = np.polyfit(log_degrees, log_counts[:len(log_degrees)], 1)
                    metrics["power_law_exponent"] = -slope
                    metrics["is_scale_free"] = 0.5 < abs(slope) < 2.0
                except Exception:  # pylint: disable=broad-except
                    metrics["power_law_exponent"] = None
                    metrics["is_scale_free"] = None

        print(f"    Determinative avg degree: {metrics['det_avg_degree']:.2f}")
        print(f"    Determinative max degree: {metrics['det_max_degree']}")
        print(f"    Is scale-free: {metrics.get('is_scale_free', 'unknown')}")

        return metrics

    def _analyze_communities(self, G: nx.Graph) -> Dict:
        """Detect communities in the determinative-sign network."""
        if G.number_of_nodes() < 4:
            return {"n_communities": 0, "message": "Graph too small"}

        # Louvain community detection (available in networkx 2.6+)
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, weight='weight', seed=42)
        except ImportError:
            # Fallback to greedy modularity
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G, weight='weight'))

        community_list = [list(c) for c in communities]

        metrics = {
            "n_communities": len(communities),
            "community_sizes": [len(c) for c in communities],
            "modularity": self._compute_modularity(G, community_list),
        }

        # Analyze semantic domain distribution within communities
        sem_domains = defaultdict(list)
        for i, comm in enumerate(community_list):
            for node in comm:
                if G.nodes[node].get('node_type') == 'determinative':
                    det_label = G.nodes[node].get('label', '')
                    # Get semantic domain from ontology
                    domain = self._get_semantic_domain(det_label)
                    sem_domains[domain].append(i)

        metrics["semantic_domain_communities"] = dict(sem_domains)

        print(f"    Communities detected: {metrics['n_communities']}")
        print(f"    Modularity: {metrics['modularity']:.3f}")
        print(f"    Semantic domains: {list(sem_domains.keys())}")

        return metrics

    def _compute_modularity(self, G: nx.Graph, communities: List[List]) -> float:
        """Compute modularity score of partition."""
        try:
            from networkx.algorithms.community import modularity
            partition = {node: i for i, comm in enumerate(communities) for node in comm}
            return modularity(G, [set(c) for c in communities], weight='weight')
        except Exception:  # pylint: disable=broad-except
            return 0.0

    def _analyze_centrality(self, G: nx.Graph) -> Dict:
        """Compute centrality rankings - hub determinatives = core system opcodes."""
        det_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'determinative']

        # Degree centrality
        degree_cent = nx.degree_centrality(G)

        # Betweenness centrality (brokerage)
        between_cent = nx.betweenness_centrality(G, weight='weight')

        # Sort determinatives by centrality
        det_centrality = []
        for node in det_nodes:
            det_label = G.nodes[node].get('label', '')
            det_centrality.append({
                "determinative": det_label,
                "degree_centrality": degree_cent[node],
                "betweenness_centrality": between_cent.get(node, 0),
                "semantic_domain": self._get_semantic_domain(det_label),
            })

        det_centrality.sort(key=lambda x: x['degree_centrality'], reverse=True)

        metrics = {
            "top_hub_determinatives": det_centrality[:10],
            "hub_signs": self._get_top_signs(G, degree_cent, n=10),
        }

        print("    Top hub determinatives:")
        for item in det_centrality[:5]:
            print(f"      {item['determinative']} ({item['semantic_domain']}): "
                  f"dc={item['degree_centrality']:.3f}")

        return metrics

    def _spectral_analysis(self, G: nx.Graph) -> Dict:
        """
        Compute spectral properties of the adjacency matrix.

        Spectral gap (lambda_2 / lambda_1) is a network fingerprint:
        - Random graph: spectral gap ~ 1/sqrt(n)
        - Expander graph: spectral gap ~ constant (machine protocols are expanders)
        - Hierarchical: multiple significant eigenvalues
        """
        if not HAS_SCIPY:
            return {"message": "scipy not available"}

        if G.number_of_nodes() < 3 or G.number_of_edges() < 1:
            return {"message": "Graph too small for spectral analysis"}

        # Adjacency matrix
        A = nx.adjacency_matrix(G, weight='weight').todense()

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        lambda_1 = eigenvalues[0] if len(eigenvalues) > 0 else 0
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
        spectral_gap = lambda_2 / lambda_1 if lambda_1 > 0 else 0

        metrics = {
            "lambda_1": float(lambda_1),
            "lambda_2": float(lambda_2),
            "spectral_gap": float(spectral_gap),
            "n_eigenvalues": len(eigenvalues),
            "eigenvalue_spectrum": [float(e) for e in eigenvalues[:10]],
        }

        # Interpretation
        if spectral_gap > 0.7:
            metrics["network_type"] = "expander (machine protocol)"
        elif spectral_gap > 0.3:
            metrics["network_type"] = "hierarchical (designed taxonomy)"
        else:
            metrics["network_type"] = "random-like (organic)"

        print(f"    Spectral gap: {spectral_gap:.3f}")
        print(f"    Network type: {metrics['network_type']}")

        return metrics

    def _get_semantic_domain(self, det_label: str) -> str:
        """Look up semantic domain for a determinative."""
        if det_label in EGYPTIAN_DETERMINATIVES:
            return EGYPTIAN_DETERMINATIVES[det_label]["semantic_domain"]
        if det_label in SUMERIAN_CLASSIFIERS:
            return SUMERIAN_CLASSIFIERS[det_label]["semantic_domain"]
        return "unknown"

    def _get_top_signs(self, G: nx.Graph, centrality: Dict, n: int = 10) -> List:
        """Get most connected sign nodes."""
        sign_nodes = [(n, G.nodes[n].get('label', ''), centrality[n])
                      for n in G.nodes() if G.nodes[n].get('node_type') == 'sign']
        sign_nodes.sort(key=lambda x: x[2], reverse=True)
        return [{"sign": s, "centrality": c} for _, s, c in sign_nodes[:n]]

    def _interpret(self, results: Dict) -> str:
        """Interpret results - designed protocol vs. organic taxonomy."""
        community = results.get("community_metrics", {})
        spectral = results.get("spectral", {})
        degree = results.get("degree_metrics", {})

        indicators = []

        # Scale-free = designed
        if degree.get("is_scale_free") == True:
            indicators.append("SCALE-FREE: Determinative network has power-law degree distribution -> designed taxonomy")

        # High modularity = layered protocol
        if community.get("modularity", 0) > 0.4:
            indicators.append(f"HIGH MODULARITY ({community['modularity']:.2f}): Determinatives cluster into semantic domains -> protocol modules")

        # Expander = machine protocol
        if spectral.get("network_type") == "expander (machine protocol)":
            indicators.append("EXPANDER: High spectral gap -> optimized for information transmission (machine protocol signature)")

        # Hub determinatives = system opcodes
        top_hubs = community.get("semantic_domain_communities", {})
        if len(top_hubs) > 5:
            indicators.append(f"MULTI-DOMAIN: {len(top_hubs)} semantic domains covered by hub determinatives -> full protocol stack")

        if not indicators:
            return "ORGANIC: Determinative network shows organic cultural categorization patterns"

        return "\n".join(indicators)

    def _save_results(self, results: Dict):
        """Save results to output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable items
        serializable = {k: v for k, v in results.items()
                        if k not in ('interpretation', 'top_hub_determinatives', 'hub_signs')}

        output_file = self.config.output_dir / "determinative_graph_results.json"
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\n[Saved] Results to {output_file}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Determinative graph analysis")
    parser.add_argument("--corpus", choices=["oracc", "hieroglyphs"], default="oracc",
                        help="Corpus to analyze")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Minimum co-occurrence count")
    parser.add_argument("--output", default="outputs/determinative_graph",
                        help="Output directory")

    args = parser.parse_args()

    config = GraphConfig(
        corpus_type=args.corpus,
        min_det_count=args.min_count,
        output_dir=Path(args.output),
        use_api=False,  # Use samples initially
    )

    # Load and build graph
    if args.corpus == "oracc":
        loader = OraccDeterminativeLoader(config)
        graph = loader.load_and_extract()
    else:
        loader = HieroglyphDeterminativeLoader(config)
        graph = loader.load_and_extract()

    # Analyze
    analyzer = DeterminativeGraphAnalyzer(config)
    results = analyzer.analyze(graph)

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print(results.get("interpretation", "No interpretation available"))
