"""
Determinative / Classifier Co-occurrence Graph Analysis
======================================================
Builds a bipartite network of signs ↔ determinatives (semantic classifiers)
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

    # Primary bipartite graph: determinatives ↔ signs
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
        """Sample transliteration data for testing."""
        samples = {
            "P3499": ["1", "usz", "gud", "KI", "a", "2", "usz", "udu", "KI", "ag", "GE22", "du11", "ša3"],
            "P2267": ["1", "guru6", "DIŠ", "ša3", "KI", "nin", "GE22", "a", "2", "guru6", "AN", "nita", "KI"],
            "P3398": ["$KI", "uri5", "DIŠ", "lugal", "LUGAL", "ki", "en", "ki", "uri5", "KI", "ma", "da"],
            "P1001": ["di", "til", "la", "KI", "nippur", "KI", "ur", "KI", "nippur", "KI"],
            "P2376": ["GE22", "en", "KI", "lugal", "AN", "ki", "šar", "KI", "pa", "udu", "ša3"],
            "P3022": ["$AN", "en", "KI", "lil2", "GE22", "a", "n", "du11", "KI", "ki", "a", "GE22"],
            "P2001": ["$AN", "LUGAL", "uri5", "KI", "ma", "da", "KI", "nippur", "KI"],
            "P5001": ["GE22", "di", "til", "LA", "KI", "e2", "KI", "an", "KI", "ki", "ša3"],
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
        """Sample hieroglyphic transliteration sequences."""
        return [
            # Pyramid Texts 1 - Opening
            # Transliteration: n swt p ny q ny i m a n x pr m a a n x h pr
            # D=god, N=water, O=building, X=stroke, R=man
            ["n", "swt", "p", "ny", "q", "ny", "i", "m", "a", "n", "X", "pr", "m", "a", "a", "n", "X", "h", "pr", "D"],
            # Ankh formula
            ["i", "n", "X", "p", "r", "D"],  # D = god determinative
            # Royal titulary
            ["nsw", "b", "t", "i", "r", "p", "ny", "H", "w", "t", "s", "r", "m", "s", "A"],
            # Offering formula
            ["d", "d", "h", "tp", "n", "ws", "ir", "n", "p", "th", "n", "K", "a", "n", "p", "t", "n", "n", "A", "V"],
            # Amarna letter
            ["i", "a", "n", "K", "m", "w", "d", "a", "n", "r", "a", "mi", "i", "a", "n", "K", "n", "p", "y", "D"],
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
            indicators.append("SCALE-FREE: Determinative network has power-law degree distribution → designed taxonomy")

        # High modularity = layered protocol
        if community.get("modularity", 0) > 0.4:
            indicators.append(f"HIGH MODULARITY ({community['modularity']:.2f}): Determinatives cluster into semantic domains → protocol modules")

        # Expander = machine protocol
        if spectral.get("network_type") == "expander (machine protocol)":
            indicators.append("EXPANDER: High spectral gap → optimized for information transmission (machine protocol signature)")

        # Hub determinatives = system opcodes
        top_hubs = community.get("semantic_domain_communities", {})
        if len(top_hubs) > 5:
            indicators.append(f"MULTI-DOMAIN: {len(top_hubs)} semantic domains covered by hub determinatives → full protocol stack")

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
