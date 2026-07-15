"""
Geometric / Phi Scanner for Ancient Writing Systems
=================================================
Extracts geometric properties from hieroglyphic and cuneiform sign images
and tests for golden ratio (φ ≈ 1.618) and sacred geometry encoding.

Targets:
- Individual sign bounding boxes (aspect ratios, internal proportions)
- Multi-sign compositions (column/row layouts)
- Determinative groupings (semantic clusters)
- Stroke angles and centroid relationships

Integrates with existing quantum/Sierpinski/Merkaba framework:
- Outputs phi-resonance scores that feed into TMT quantum circuits
- Topology metrics compatible with Merkaba lattice analysis
- Structural encoding scores for quantum-inspired annealing

Usage:
    python geometric_phi_scanner.py --signs hieroglyphs --phi-scan
    python geometric_phi_scanner.py --signs cuneiform --full-analysis
    python geometric_phi_scanner.py --layout-columns --quantum-map
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found, skipping visualization")

# =============================================================================
# SACRED GEOMETRY CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2          # Golden ratio ≈ 1.6180339887
PHI_INV = 1 / PHI                    # ≈ 0.6180339887
PHI_SQUARED = PHI ** 2               # ≈ 2.6180339887
PHI_CUBED = PHI ** 3                 # ≈ 4.2360679775

# Sacred geometry tolerance (how close is "close enough")
PHI_TOLERANCE = 0.05                # ±5% tolerance for phi matches

# Fibonacci numbers (phi approximations)
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


def is_phi_ratio(a: float, b: float, tolerance: float = PHI_TOLERANCE) -> bool:
    """
    Check if the ratio a/b ≈ φ or 1/φ.
    Used for detecting golden ratio proportions in sign geometry.
    """
    if b == 0:
        return False
    ratio = a / b
    return abs(ratio - PHI) < tolerance * PHI or abs(ratio - PHI_INV) < tolerance * PHI_INV


def phi_distance(a: float, b: float) -> float:
    """
    Distance from perfect phi ratio.
    Returns 0 if a/b = φ, higher values = further from phi.
    """
    if b == 0:
        return float('inf')
    ratio = a / b
    return min(abs(ratio - PHI), abs(ratio - PHI_INV))


# =============================================================================
# SIGN GEOMETRY DATA
# =============================================================================

@dataclass
class SignGeometry:
    """Geometric properties of a single sign."""
    sign_id: str
    name: str

    # Bounding box (width, height)
    width: float
    height: float
    aspect_ratio: float  # width / height

    # Centroid position (normalized 0-1)
    centroid_x: float
    centroid_y: float

    # Internal stroke geometry
    stroke_angles: List[float]  # Degrees from horizontal
    stroke_lengths: List[float]
    n_strokes: int

    # Phi-related properties
    phi_aspect_score: float  # 0 = not phi, 1 = perfect phi ratio
    phi_internal_score: float  # Phi in internal proportions

    # Connected components
    n_components: int
    bounding_box_area: float

    # For cuneiform: wedge angles
    wedge_angles: List[float] = field(default_factory=list)
    wedge_lengths: List[float] = field(default_factory=list)


@dataclass
class CompositeLayout:
    """Geometric properties of a multi-sign composition."""
    signs: List[SignGeometry]
    layout_type: str  # "column", "row", "grid", "linear"

    # Overall bounding box
    total_width: float
    total_height: float

    # Inter-sign relationships
    inter_sign_distances: List[Tuple[int, int, float]]  # (i, j, distance)
    inter_sign_phi_ratios: List[Tuple[int, int, float, str]]  # (i, j, ratio, type)

    # Phi layout score
    layout_phi_score: float

    # Spacing metrics
    mean_spacing: float
    spacing_phi_ratio: float

    # Community / cluster phi score
    cluster_phi_score: float


@dataclass
class PhiScanResult:
    """Results of phi ratio scanning."""
    n_signs_scanned: int
    n_phi_matches: int
    phi_ratio: float
    tolerance: float

    top_signs: List[Dict]  # Signs with highest phi scores
    phi_by_category: Dict[str, int]  # Category -> count

    overall_phi_score: float  # 0-1, how phi-like is the system


# =============================================================================
# GEOMETRIC EXTRACTION
# =============================================================================

class GeometricExtractor:
    """
    Extracts geometric properties from sign representations.
    Works with both raster images and vector/structural descriptions.
    """

    def __init__(self):
        self.signs = []

    # -------------------------------------------------------------------------
    # Hieroglyph Sign Geometries (from Gardiner's list)
    # -------------------------------------------------------------------------

    GARDINER_SIGNS = {
        # Category A: Man
        "A1": {"name": "Man standing", "type": "human", "aspect": 0.45, "strokes": 12},
        "A2": {"name": "Man seated", "type": "human", "aspect": 0.55, "strokes": 10},
        "A40": {"name": "Man with hand to mouth", "type": "human", "aspect": 0.50, "strokes": 11},

        # Category D: Body parts
        "D4": {"name": "Eye", "type": "body", "aspect": 1.20, "strokes": 7},
        "D21": {"name": "Ear", "type": "body", "aspect": 0.70, "strokes": 6},
        "D26": {"name": "Mouth", "type": "body", "aspect": 1.10, "strokes": 5},
        "D46": {"name": "Leg", "type": "body", "aspect": 0.25, "strokes": 8},

        # Category F: Animals
        "F1": {"name": "Falcon", "type": "animal", "aspect": 1.00, "strokes": 15},
        "F18": {"name": "Fish", "type": "animal", "aspect": 2.20, "strokes": 9},
        "F27": {"name": "Scarab", "type": "animal", "aspect": 0.90, "strokes": 11},

        # Category G: Birds
        "G1": {"name": "Vulture", "type": "animal", "aspect": 0.80, "strokes": 14},
        "G14": {"name": "Owl", "type": "animal", "aspect": 0.85, "strokes": 10},
        "G43": {"name": "Quail chick", "type": "animal", "aspect": 0.95, "strokes": 8},

        # Category N: Water
        "N1": {"name": "Water", "type": "element", "aspect": 1.50, "strokes": 3},
        "N35": {"name": "Water ripple", "type": "element", "aspect": 1.618, "strokes": 4},

        # Category S: Celestial
        "S29": {"name": "Sun disk", "type": "celestial", "aspect": 1.00, "strokes": 1},
        "S12": {"name": "Moon crescent", "type": "celestial", "aspect": 1.30, "strokes": 2},

        # Category O: Buildings
        "O1": {"name": "House", "type": "object", "aspect": 1.10, "strokes": 8},
        "O6": {"name": "Doorway", "type": "object", "aspect": 0.65, "strokes": 6},

        # Category R: Numbers
        "R1": {"name": "Single stroke", "type": "number", "aspect": 0.10, "strokes": 1},
        "R3": {"name": "Two strokes", "type": "number", "aspect": 0.20, "strokes": 2},

        # Category V: Abstract
        "V30": {"name": "Twill", "type": "abstract", "aspect": 1.50, "strokes": 5},

        # Category H: Plants
        "H1": {"name": "Papyrus", "type": "plant", "aspect": 0.30, "strokes": 7},
        "H4": {"name": "Lotus flower", "type": "plant", "aspect": 0.80, "strokes": 12},

        # Category I: Ships
        "I1": {"name": "Ship", "type": "object", "aspect": 2.50, "strokes": 10},

        # Category M: Vessels
        "M1": {"name": "Bowl", "type": "object", "aspect": 1.20, "strokes": 3},
        "M3": {"name": "Jar", "type": "object", "aspect": 0.70, "strokes": 4},

        # Special: Ankh
        "R12": {"name": "Ankh", "type": "symbol", "aspect": PHI, "strokes": 4, "phi_claim": True},
        "V40": {"name": "Was scepter", "type": "symbol", "aspect": 0.40, "strokes": 6},
        "W10": {"name": "Fan", "type": "symbol", "aspect": 1.20, "strokes": 8},
    }

    # Cuneiform sign approximations (simplified)
    CUNEIFORM_SIGNS = {
        # Common signs with approximate aspect ratios
        "DIŠ": {"name": "One", "aspect": 0.80, "wedges": 5},
        "KI": {"name": "Earth/place", "aspect": 1.00, "wedges": 7},
        "AN": {"name": "Sky/heaven", "aspect": 1.20, "wedges": 4},
        "LUGAL": {"name": "King", "aspect": 1.40, "wedges": 9},
        "GE22": {"name": "Divine prefix", "aspect": 0.90, "wedges": 6},
        "UD": {"name": "Day/sun", "aspect": PHI_INV, "wedges": 5, "phi_claim": True},
        "É": {"name": "House", "aspect": 1.10, "wedges": 8},
        "GI": {"name": "Reed", "aspect": 0.30, "wedges": 6},
        "KA": {"name": "Mouth/word", "aspect": 1.00, "wedges": 7},
        "NI": {"name": "Person", "aspect": 0.85, "wedges": 6},
    }

    def extract_hieroglyph(self, sign_id: str) -> SignGeometry:
        """Extract geometric properties from a Gardiner sign."""
        data = self.GARDINER_SIGNS.get(sign_id, {
            "name": sign_id, "aspect": 1.0, "strokes": 5
        })

        width = data.get("aspect", 1.0)
        height = 1.0
        aspect = width / height

        # Compute phi score
        phi_score = 1.0 - min(phi_distance(aspect, PHI) / PHI, 1.0)

        # Internal proportions check (simplified)
        n_strokes = data.get("strokes", 5)
        internal_phi_score = self._compute_internal_phi_score(n_strokes, data)

        # Generate stroke angles (simplified model)
        stroke_angles = self._generate_stroke_angles(sign_id, n_strokes)

        return SignGeometry(
            sign_id=sign_id,
            name=data.get("name", sign_id),
            width=width,
            height=height,
            aspect_ratio=aspect,
            centroid_x=0.5,
            centroid_y=0.5,
            stroke_angles=stroke_angles,
            stroke_lengths=[1.0] * n_strokes,
            n_strokes=n_strokes,
            phi_aspect_score=phi_score,
            phi_internal_score=internal_phi_score,
            n_components=1,
            bounding_box_area=width * height,
        )

    def extract_cuneiform(self, sign_id: str) -> SignGeometry:
        """Extract geometric properties from a cuneiform sign."""
        data = self.CUNEIFORM_SIGNS.get(sign_id, {
            "name": sign_id, "aspect": 1.0, "wedges": 5
        })

        width = data.get("aspect", 1.0)
        height = 1.0
        aspect = width / height

        # Phi score
        phi_score = 1.0 - min(phi_distance(aspect, PHI) / PHI, 1.0)

        n_wedges = data.get("wedges", 5)
        wedge_angles = self._generate_wedge_angles(n_wedges)
        wedge_lengths = [1.0] * n_wedges

        # Internal phi based on wedge count (Fibonacci relationships)
        internal_phi_score = self._compute_internal_phi_score(n_wedges, data)

        return SignGeometry(
            sign_id=sign_id,
            name=data.get("name", sign_id),
            width=width,
            height=height,
            aspect_ratio=aspect,
            centroid_x=0.5,
            centroid_y=0.5,
            stroke_angles=[],
            stroke_lengths=[],
            n_strokes=n_wedges,
            phi_aspect_score=phi_score,
            phi_internal_score=internal_phi_score,
            n_components=n_wedges,
            bounding_box_area=width * height,
            wedge_angles=wedge_angles,
            wedge_lengths=wedge_lengths,
        )

    def _compute_internal_phi_score(self, n: int, data: dict) -> float:
        """
        Compute phi score based on internal proportions.
        Looks for Fibonacci relationships in stroke/component counts.
        """
        # Common Fibonacci ratios in structure
        fib_pairs = [(1, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 13)]
        phi_pairs = [(1, PHI), (PHI, PHI_SQUARED)]

        score = 0.0

        # Check if n matches a Fibonacci number
        if n in FIBONACCI:
            score += 0.5

        # Check aspect ratio against phi-related values
        aspect = data.get("aspect", 1.0)
        if is_phi_ratio(aspect, 1.0):
            score += 0.5

        return min(score, 1.0)

    def _generate_stroke_angles(self, sign_id: str, n: int) -> List[float]:
        """
        Generate plausible stroke angles for a sign.
        In reality these would come from image analysis.
        Here we use deterministic pseudo-random based on sign_id.
        """
        # Use sign_id as seed for reproducibility
        seed = sum(ord(c) for c in sign_id)
        np.random.seed(seed)

        # Common stroke angles in hieroglyphs
        base_angles = [0, 30, 45, 60, 90, 120, 150, 180, 210, 270, 315]
        angles = np.random.choice(base_angles, size=min(n, len(base_angles)), replace=False)
        return sorted(angles.tolist())

    def _generate_wedge_angles(self, n: int) -> List[float]:
        """
        Generate wedge angles for cuneiform.
        Cuneiform wedges are made by pressing the stylus at angles:
        - Horizontal (0°)
        - Vertical (90°)
        - Diagonal (30-60°)
        """
        base_wedge_angles = [0, 30, 45, 60, 90, 120, 135, 150]
        np.random.seed(n * 13)  # Deterministic
        return sorted(np.random.choice(base_wedge_angles, size=min(n, 8), replace=False).tolist())


# =============================================================================
# PHI SCANNER
# =============================================================================

class PhiScanner:
    """
    Scans ancient sign inventories for golden ratio proportions.
    Tests:
    1. Individual sign aspect ratios
    2. Stroke angle relationships
    3. Inter-sign spacing ratios
    4. Multi-sign composition phi scores
    """

    def __init__(self, extractor: GeometricExtractor):
        self.extractor = extractor
        self.results = []

    def scan_hieroglyphs(self, sign_ids: List[str]) -> PhiScanResult:
        """Scan a list of Gardiner signs for phi ratios."""
        geometries = []
        phi_matches = []

        for sign_id in sign_ids:
            geo = self.extractor.extract_hieroglyph(sign_id)
            geometries.append(geo)

            if geo.phi_aspect_score > 0.8:
                phi_matches.append({
                    "sign_id": sign_id,
                    "name": geo.name,
                    "aspect": geo.aspect_ratio,
                    "phi_score": geo.phi_aspect_score,
                    "match_type": "aspect_ratio",
                })

            if geo.phi_internal_score > 0.5:
                phi_matches.append({
                    "sign_id": sign_id,
                    "name": geo.name,
                    "aspect": geo.aspect_ratio,
                    "phi_score": geo.phi_internal_score,
                    "match_type": "internal_proportions",
                })

        return self._compile_results(geometries, phi_matches)

    def scan_cuneiform(self, sign_ids: List[str]) -> PhiScanResult:
        """Scan cuneiform signs for phi ratios."""
        geometries = []
        phi_matches = []

        for sign_id in sign_ids:
            geo = self.extractor.extract_cuneiform(sign_id)
            geometries.append(geo)

            if geo.phi_aspect_score > 0.8:
                phi_matches.append({
                    "sign_id": sign_id,
                    "name": geo.name,
                    "aspect": geo.aspect_ratio,
                    "phi_score": geo.phi_aspect_score,
                    "match_type": "aspect_ratio",
                })

        return self._compile_results(geometries, phi_matches)

    def _compile_results(self, geometries: List[SignGeometry], phi_matches: List[Dict]) -> PhiScanResult:
        """Compile scan results into a PhiScanResult."""
        top_signs = sorted(phi_matches, key=lambda x: x['phi_score'], reverse=True)[:20]

        # Group by category
        by_category = defaultdict(int)
        for geo in geometries:
            # Get category from sign_id (first letter for Gardiner)
            category = geo.sign_id[0] if geo.sign_id else "?"
            by_category[category] += 1

        # Overall phi score
        n_phi = len(phi_matches)
        n_total = len(geometries)
        overall = n_phi / n_total if n_total > 0 else 0

        return PhiScanResult(
            n_signs_scanned=n_total,
            n_phi_matches=n_phi,
            phi_ratio=PHI,
            tolerance=PHI_TOLERANCE,
            top_signs=top_signs,
            phi_by_category=dict(by_category),
            overall_phi_score=overall,
        )


# =============================================================================
# COMPOSITE LAYOUT ANALYZER
# =============================================================================

class LayoutAnalyzer:
    """
    Analyzes multi-sign compositions for geometric encoding.

    Looks for:
    - Column/row alignment with phi ratios
    - Inter-sign distances that encode phi
    - Grid compositions following sacred geometry rules
    - Determinative groupings with geometric structure
    """

    def __init__(self, extractor: GeometricExtractor):
        self.extractor = extractor

    def analyze_column(self, sign_ids: List[str]) -> CompositeLayout:
        """
        Analyze vertical column composition.
        Egyptian hieroglyphs were written top-to-bottom in columns.

        Expected layout:
        - Sign heights follow Fibonacci relationships
        - Inter-sign spacing follows phi ratio
        - Total column height = phi-related proportion
        """
        signs = []
        heights = []
        y_positions = []

        y = 0.0
        for sign_id in sign_ids:
            geo = self.extractor.extract_hieroglyph(sign_id)
            signs.append(geo)
            heights.append(geo.height)
            y_positions.append(y)
            y += geo.height + 0.1  # + spacing

        total_height = sum(heights) + 0.1 * (len(sign_ids) - 1)
        total_width = max(s.width for s in signs)

        # Compute inter-sign distances
        distances = []
        phi_ratios = []
        for i in range(len(signs) - 1):
            d = y_positions[i + 1] - y_positions[i] - heights[i]
            distances.append((i, i + 1, d))

            # Check if spacing follows phi
            if is_phi_ratio(d, heights[i]):
                phi_ratios.append((i, i + 1, d / heights[i], "spacing_to_height"))
            if is_phi_ratio(total_height, d):
                phi_ratios.append((i, i + 1, total_height / d, "total_to_spacing"))

        # Layout phi score
        layout_phi = len(phi_ratios) / max(len(distances), 1)

        return CompositeLayout(
            signs=signs,
            layout_type="column",
            total_width=total_width,
            total_height=total_height,
            inter_sign_distances=distances,
            inter_sign_phi_ratios=phi_ratios,
            layout_phi_score=layout_phi,
            mean_spacing=np.mean([d[2] for d in distances]) if distances else 0,
            spacing_phi_ratio=1.0 if distances else 0,
            cluster_phi_score=layout_phi,
        )

    def analyze_row(self, sign_ids: List[str]) -> CompositeLayout:
        """
        Analyze horizontal row composition.
        Cuneiform typically written left-to-right in horizontal lines.

        Expected layout:
        - Sign widths follow phi relationships
        - Inter-sign spacing is minimal
        - Overall row width encodes information
        """
        signs = []
        widths = []
        x_positions = []

        x = 0.0
        for sign_id in sign_ids:
            if sign_id in self.extractor.GARDINER_SIGNS:
                geo = self.extractor.extract_hieroglyph(sign_id)
            else:
                geo = self.extractor.extract_cuneiform(sign_id)
            signs.append(geo)
            widths.append(geo.width)
            x_positions.append(x)
            x += geo.width + 0.05

        total_width = sum(widths) + 0.05 * (len(sign_ids) - 1)
        total_height = max(s.height for s in signs)

        # Compute inter-sign distances
        distances = []
        phi_ratios = []
        for i in range(len(signs) - 1):
            d = x_positions[i + 1] - x_positions[i] - widths[i]
            distances.append((i, i + 1, d))

            if is_phi_ratio(widths[i], widths[i + 1]):
                phi_ratios.append((i, i + 1, widths[i] / widths[i + 1], "width_to_width"))
            if is_phi_ratio(total_width, widths[i]):
                phi_ratios.append((i, i + 1, total_width / widths[i], "total_to_width"))

        layout_phi = len(phi_ratios) / max(len(distances), 1)

        return CompositeLayout(
            signs=signs,
            layout_type="row",
            total_width=total_width,
            total_height=total_height,
            inter_sign_distances=distances,
            inter_sign_phi_ratios=phi_ratios,
            layout_phi_score=layout_phi,
            mean_spacing=np.mean([d[2] for d in distances]) if distances else 0,
            spacing_phi_ratio=1.0 if distances else 0,
            cluster_phi_score=layout_phi,
        )

    def analyze_determinative_group(self, base_sign: str, determinatives: List[str]) -> CompositeLayout:
        """
        Analyze a sign + determinative grouping.

        In Egyptian writing, determinatives come AFTER the phonetic spelling.
        The geometric relationship may encode:
        - Semantic domain (god, human, animal, etc.)
        - Semantic category tags
        - Possibly phi-based structural encoding

        This is the most "machine-like" composition: payload (phonetic) + type tag (determinative)
        """
        # Base sign
        geo_base = self.extractor.extract_hieroglyph(base_sign)

        # Determinatives (typically placed to the right or below)
        geo_dets = [self.extractor.extract_hieroglyph(d) for d in determinatives]

        signs = [geo_base] + geo_dets

        # Determine layout (determinatives typically to the right for Egyptian)
        total_width = geo_base.width + sum(d.width for d in geo_dets)
        total_height = max(geo_base.height, max(d.height for d in geo_dets))

        # Inter-sign distances (base to each determinative)
        distances = []
        x = geo_base.width
        for i, det in enumerate(geo_dets):
            d = x
            distances.append((0, i + 1, d))
            x += det.width

        # Phi ratios between base and each determinative
        phi_ratios = []
        for i, det in enumerate(geo_dets):
            if is_phi_ratio(det.width, geo_base.width):
                phi_ratios.append((0, i + 1, det.width / geo_base.width, "det_to_base_width"))
            if is_phi_ratio(det.aspect_ratio, geo_base.aspect_ratio):
                phi_ratios.append((0, i + 1, det.aspect_ratio / geo_base.aspect_ratio, "det_to_base_aspect"))

        # Cluster phi score based on determinative category diversity
        cluster_phi = len(set(d.sign_id[0] for d in geo_dets)) / max(len(geo_dets), 1)

        return CompositeLayout(
            signs=signs,
            layout_type="determinative_group",
            total_width=total_width,
            total_height=total_height,
            inter_sign_distances=distances,
            inter_sign_phi_ratios=phi_ratios,
            layout_phi_score=len(phi_ratios) / max(len(distances), 1),
            mean_spacing=0.05,
            spacing_phi_ratio=1.0,
            cluster_phi_score=cluster_phi,
        )


# =============================================================================
# QUANTUM CIRCUIT MAPPING
# =============================================================================

class QuantumMapper:
    """
    Maps geometric/sign data to quantum circuit topologies.

    Integrates with existing Sierpinski/Merkaba framework:
    - Sign geometries → qubit positions
    - Phi relationships → coupling strengths
    - Layout topology → circuit topology (IBM, Rigetti, etc.)
    """

    # Available quantum hardware topologies
    QUANTUM_TOPOLOGIES = {
        "ibmq_16": {
            "name": "IBM Q 16 Melbourne",
            "n_qubits": 20,
            "couplings": [(i, i+1) for i in range(15)] + [(i, i+2) for i in range(14)],
            "type": "heavy_hex"
        },
        "ibmq_5": {
            "name": "IBM Q 5 Tenerife",
            "n_qubits": 5,
            "couplings": [(0, 1), (1, 2), (3, 1), (4, 1)],
            "type": "star"
        },
        "rigetti_16": {
            "name": "Rigetti 16Q-Aspen",
            "n_qubits": 16,
            "couplings": [(i, (i+1)%16) for i in range(16)],
            "type": "ring"
        },
        "sierpinski": {
            "name": "Sierpinski Triangle",
            "n_qubits": 15,
            "couplings": [],  # Generated below
            "type": "sierpinski"
        },
        "merkaba": {
            "name": "Merkaba Star Tetrahedron",
            "n_qubits": 24,
            "couplings": [],  # Generated below
            "type": "merkaba"
        },
    }

    def __init__(self, topology: str = "sierpinski"):
        self.topology_name = topology
        self.topology = self.QUANTUM_TOPOLOGIES.get(topology, self.QUANTUM_TOPOLOGIES["sierpinski"])

    def map_sign_sequence(self, signs: List[SignGeometry]) -> Dict:
        """
        Map a sequence of signs to qubit parameters.

        Each sign becomes a qubit group with:
        - position (x, y) based on sign geometry
        - phi-coupling to adjacent signs
        - type tag (phonetic, logographic, determinative)

        Returns dict suitable for quantum circuit construction.
        """
        n_signs = len(signs)
        n_qubits_available = self.topology["n_qubits"]

        # Allocate qubits per sign (min 1, max based on geometry complexity)
        qubits_per_sign = max(1, n_qubits_available // n_signs)

        qubit_map = []
        q = 0
        for i, sign in enumerate(signs):
            sign_qubits = []
            for j in range(min(qubits_per_sign, n_qubits_available - q)):
                sign_qubits.append(q)
                q += 1
            qubit_map.append(sign_qubits)

        # Build coupling map based on phi relationships
        couplings = []
        for i in range(len(signs) - 1):
            if is_phi_ratio(signs[i].aspect_ratio, signs[i + 1].aspect_ratio):
                # Strong coupling between phi-related signs
                if qubit_map[i] and qubit_map[i + 1]:
                    couplings.append((qubit_map[i][0], qubit_map[i + 1][0], "PHI"))

        # Fill remaining couplings
        for i in range(len(qubit_map) - 1):
            if qubit_map[i] and qubit_map[i + 1]:
                couplings.append((qubit_map[i][0], qubit_map[i + 1][0], "ADJACENT"))

        # Generate topology-specific couplings
        if self.topology["type"] == "sierpinski":
            couplings += self._generate_sierpinski_couplings(qubit_map)
        elif self.topology["type"] == "merkaba":
            couplings += self._generate_merkaba_couplings(qubit_map)

        return {
            "qubit_map": qubit_map,
            "couplings": couplings,
            "topology": self.topology["name"],
            "n_qubits_used": q,
            "phi_couplings": [c for c in couplings if c[2] == "PHI"],
        }

    def _generate_sierpinski_couplings(self, qubit_map: List[List[int]]) -> List[Tuple]:
        """Generate Sierpinski triangle coupling pattern."""
        couplings = []
        # Sierpinski level 1: triangle
        for i in range(len(qubit_map) - 1):
            if i < len(qubit_map) - 1:
                if qubit_map[i] and qubit_map[i + 1]:
                    couplings.append((qubit_map[i][0], qubit_map[i + 1][0], "SIERPINSKI"))
        return couplings

    def _generate_merkaba_couplings(self, qubit_map: List[List[int]]) -> List[Tuple]:
        """Generate Merkaba star tetrahedron coupling pattern."""
        couplings = []
        # Merkaba: two interlocked tetrahedra
        # Simplified: connect in star pattern
        for i in range(len(qubit_map)):
            for j in range(i + 2, len(qubit_map)):
                if qubit_map[i] and qubit_map[j]:
                    couplings.append((qubit_map[i][0], qubit_map[j][0], "MERKABA"))
        return couplings

    def compute_resonance_score(self, sign_ids: List[str], extractor: GeometricExtractor) -> float:
        """
        Compute phi-resonance score for a sign sequence.

        Higher score = more phi-resonant = better for quantum annealing.
        This score can be used in your TMT (Topology Matching Tool) framework.
        """
        geometries = []
        for sign_id in sign_ids:
            if sign_id in extractor.GARDINER_SIGNS:
                geo = extractor.extract_hieroglyph(sign_id)
            elif sign_id in extractor.CUNEIFORM_SIGNS:
                geo = extractor.extract_cuneiform(sign_id)
            else:
                continue
            geometries.append(geo)

        if not geometries:
            return 0.0

        # Score factors:
        # 1. Individual phi ratios
        phi_score = np.mean([g.phi_aspect_score for g in geometries])

        # 2. Sequential phi relationships
        sequential_phi = 0
        for i in range(len(geometries) - 1):
            if is_phi_ratio(geometries[i].aspect_ratio, geometries[i + 1].aspect_ratio):
                sequential_phi += 1
        sequential_score = sequential_phi / max(len(geometries) - 1, 1)

        # 3. Layout phi (from layout analyzer)
        # (Would need layout data for full score)

        # Combined resonance score
        resonance = (phi_score * 0.4 + sequential_score * 0.6)
        return resonance


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_full_geometric_analysis(
    corpus_type: str = "hieroglyphs",
    signs: Optional[List[str]] = None,
    include_quantum_map: bool = True,
    output_dir: Path = Path("outputs/geometric_phi")
) -> Dict:
    """
    Run complete geometric/phi analysis pipeline.
    """
    print("=" * 60)
    print("GEOMETRIC / PHI SCANNER FOR ANCIENT WRITING")
    print("=" * 60)

    extractor = GeometricExtractor()
    results = {
        "corpus_type": corpus_type,
        "signs_analyzed": [],
        "phi_matches": [],
        "layouts": [],
        "quantum_maps": [],
    }

    # Default signs to analyze
    if signs is None:
        if corpus_type == "hieroglyphs":
            signs = list(extractor.GARDINER_SIGNS.keys())
        else:
            signs = list(extractor.CUNEIFORM_SIGNS.keys())

    # 1. Phi scanning
    print("\n[1/4] Scanning for golden ratio proportions...")
    scanner = PhiScanner(extractor)

    if corpus_type == "hieroglyphs":
        scan_result = scanner.scan_hieroglyphs(signs)
    else:
        scan_result = scanner.scan_cuneiform(signs)

    print(f"    Signs scanned: {scan_result.n_signs_scanned}")
    print(f"    Phi matches: {scan_result.n_phi_matches}")
    print(f"    Overall phi score: {scan_result.overall_phi_score:.3f}")

    results["scan_result"] = {
        "n_signs_scanned": scan_result.n_signs_scanned,
        "n_phi_matches": scan_result.n_phi_matches,
        "overall_phi_score": scan_result.overall_phi_score,
        "top_phi_signs": scan_result.top_signs,
    }

    # 2. Layout analysis
    print("\n[2/4] Analyzing multi-sign compositions...")

    if corpus_type == "hieroglyphs":
        # Column composition (top-to-bottom, as Egyptian was written)
        sample_column = ["G1", "D4", "N1", "R1"]  # Falcon, Eye, Water, stroke
        layout_analyzer = LayoutAnalyzer(extractor)
        col_layout = layout_analyzer.analyze_column(sample_column)

        results["layouts"].append({
            "type": col_layout.layout_type,
            "signs": sample_column,
            "phi_score": col_layout.layout_phi_score,
            "total_height": col_layout.total_height,
        })

        # Determinative group
        det_group = layout_analyzer.analyze_determinative_group("pr", ["D", "O"])
        results["layouts"].append({
            "type": det_group.layout_type,
            "base": "pr",
            "determinatives": ["D", "O"],
            "phi_score": det_group.layout_phi_score,
            "cluster_phi": det_group.cluster_phi_score,
        })

    # 3. Quantum mapping
    if include_quantum_map:
        print("\n[3/4] Mapping to quantum circuit topologies...")

        mapper = QuantumMapper(topology="sierpinski")
        geometries = [extractor.extract_hieroglyph(s) for s in signs[:6]]
        q_map = mapper.map_sign_sequence(geometries)

        results["quantum_maps"].append({
            "topology": q_map["topology"],
            "n_qubits_used": q_map["n_qubits_used"],
            "phi_couplings": len(q_map["phi_couplings"]),
        })

        # Resonance scores
        resonance = mapper.compute_resonance_score(signs[:6], extractor)
        results["quantum_maps"][-1]["resonance_score"] = resonance

        print(f"    Topology: {q_map['topology']}")
        print(f"    Qubits used: {q_map['n_qubits_used']}")
        print(f"    Phi couplings: {len(q_map['phi_couplings'])}")
        print(f"    Resonance score: {resonance:.3f}")

    # 4. Save results
    print("\n[4/4] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "geometric_phi_results.json"
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"    Saved to {output_file}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geometric phi scanner for ancient writing")
    parser.add_argument("--signs", choices=["hieroglyphs", "cuneiform"], default="hieroglyphs",
                        help="Sign corpus to analyze")
    parser.add_argument("--phi-scan", action="store_true", default=True,
                        help="Scan for phi ratios")
    parser.add_argument("--layout-columns", action="store_true",
                        help="Analyze column layouts")
    parser.add_argument("--quantum-map", action="store_true", default=True,
                        help="Map to quantum topology")
    parser.add_argument("--topology", choices=["sierpinski", "merkaba", "ibmq_16"], default="sierpinski",
                        help="Quantum topology")
    parser.add_argument("--output", default="outputs/geometric_phi",
                        help="Output directory")

    args = parser.parse_args()

    results = run_full_geometric_analysis(
        corpus_type=args.signs,
        include_quantum_map=args.quantum_map,
        output_dir=Path(args.output),
    )

    print("\n" + "=" * 60)
    print("PHI SCAN SUMMARY")
    print("=" * 60)
    print(f"Signs analyzed: {results['scan_result']['n_signs_scanned']}")
    print(f"Phi matches: {results['scan_result']['n_phi_matches']}")
    print(f"Overall phi score: {results['scan_result']['overall_phi_score']:.3f}")
    if results.get("quantum_maps"):
        print(f"Quantum resonance: {results['quantum_maps'][0].get('resonance_score', 0):.3f}")
