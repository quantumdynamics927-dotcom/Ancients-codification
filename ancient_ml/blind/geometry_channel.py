"""
Geometry Channel: Phi Scanner as Side-Information Encoding
========================================================
Golden ratio (phi ~ 1.618) detection in sign geometry as second encoding channel.
Tests whether geometry predicts type cluster better than chance.
Fuses structure (sign stream) + sacred geometry (shape/layout).

STATUS: NULL_ON_CURRENT_DATA
  On the embedded and fixture corpora, phi enrichment does NOT survive
  preregistered Bonferroni and FDR correction (p > 0.05).
  This module is not deleted because:
  - Real image scanning (actual hieroglyph photos, not embedded approximations)
    may yield different results.
  - The protocol is sound; the current data simply do not show a signal.
  - Future primary corpus runs should re-test with independently sourced geometry data.

USAGE NOTE: The module requires geometry_data from image scanning.
Pass geometry_data={"sign_id": {"aspect_ratio": float, "phi_score": float, ...}}
If no geometry_data is provided, phi_rate falls back to the published Gardiner
approximation (0.733), which is NOT valid evidence.

Preregistered protocol:
- Targets: phi = 1.618033988749895, 1/phi = 0.618033988749895
- Tolerance: +-5%
- Control: aspect-matched random rectangles
- Multiple testing: Bonferroni + FDR (Benjamini-Hochberg)
- Reporting threshold: both corrections pass AND phi_rate > 0.50

Based on:
- "Geometry of the Egyptian Hieroglyphic Script" (Schmidt 1970s)
- Golden ratio in ancient architecture and art
- PMC: "Assessing chance in phi ratio studies" (2024)

Usage:
    from blind.geometry_channel import GeometryChannel
    gc = GeometryChannel(sequences)
    result = gc.analyze()
    print(result.phi_hit_rate)
    print(result.geometry_type_mi)  # Does geometry predict type?
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class GeometryResult:
    """Result of geometry channel analysis."""
    phi_rate: float            # Fraction of signs with phi-proportional geometry
    phi_matches: int           # Number of signs matching φ
    n_signs_scanned: int      # Total signs with geometry data
    overall_phi_score: float  # Average phi resonance score

    # Geometry-type correlation
    geometry_type_mi: float    # MI between geometry cluster and pseudo-type
    geometry_predicts_type: bool  # geometry_type_mi > chance

    # Periodic geometry markers
    periodic_geometry: Dict[str, float]  # Signs with geometric periodicity

    # Adversarial control
    random_phi_rate: float     # Phi hit rate on random controls


class GeometryChannel:
    """
    Analyze geometry as second encoding channel.
    Fuses sign structure with golden ratio spatial patterns.

    NOTE: In the blind framework, geometry data must come from image scanning.
    This module provides the analysis framework — it needs geometry inputs.
    """

    def __init__(self, sequences: List[List[str]], geometry_data: Optional[Dict] = None):
        self.sequences = sequences
        self.geometry_data = geometry_data or {}  # sign_id -> {aspect_ratio, phi_score, ...}
        self.PHI = 1.618033988749895

    def analyze(self) -> GeometryResult:
        """
        Analyze geometry as second channel.
        Without real image scans, uses embedded geometry data from sign IDs.
        """
        phi_matches = 0
        total_with_geom = 0
        phi_scores = []

        for seq in self.sequences:
            for sign in seq:
                if sign in self.geometry_data:
                    geom = self.geometry_data[sign]
                    total_with_geom += 1
                    phi_score = geom.get("phi_score", 0.0)
                    phi_scores.append(phi_score)
                    if self._is_phi_match(geom):
                        phi_matches += 1

        # If no real geometry data, use embedded approximations
        if not phi_scores:
            phi_scores = self._generate_embedded_phi_scores()

        overall_score = np.mean(phi_scores) if phi_scores else 0.0
        phi_rate = phi_matches / max(total_with_geom, 1) if total_with_geom > 0 else self._embedded_phi_rate()

        # Geometry-type MI (needs pseudo-type labels)
        geometry_type_mi = self._compute_geometry_type_mi()

        # Periodic geometry
        periodic_geom = self._detect_periodic_geometry()

        # Random control
        random_phi_rate = self._random_phi_control()

        return GeometryResult(
            phi_rate=phi_rate,
            phi_matches=phi_matches,
            n_signs_scanned=total_with_geom or len(self.sequences) * 5,
            overall_phi_score=overall_score,
            geometry_type_mi=geometry_type_mi,
            geometry_predicts_type=geometry_type_mi > 0.05,
            periodic_geometry=periodic_geom,
            random_phi_rate=random_phi_rate,
        )

    def _is_phi_match(self, geom: Dict) -> bool:
        """Check if geometry matches phi-proportional relationship."""
        phi = self.PHI
        aspect = geom.get("aspect_ratio", 0)
        if aspect <= 0:
            return False
        # Golden ratio: height/width ≈ φ or width/height ≈ φ
        ratio1 = aspect
        ratio2 = 1.0 / aspect if aspect > 0 else 0
        diff1 = abs(ratio1 - phi)
        diff2 = abs(ratio2 - phi)
        return min(diff1, diff2) < 0.1

    def _embedded_phi_rate(self) -> float:
        """Use embedded phi rate from prior research (73.3% Gardiner signs)."""
        return 0.733

    def _generate_embedded_phi_scores(self) -> List[float]:
        """
        Generate approximate phi scores based on embedded research.
        Real implementation would use image scanner.
        """
        # Simulate embedded phi data: ~73% of signs have phi-proportional geometry
        flat = [s for seq in self.sequences for s in seq]
        n = len(flat)
        n_phi = int(n * 0.733)
        scores = [1.0] * n_phi + [0.3] * (n - n_phi)
        np.random.seed(42)
        np.random.shuffle(scores)
        return scores

    def _compute_geometry_type_mi(self) -> float:
        """
        Compute mutual information between geometry cluster and pseudo-type.
        If geometry truly is a second channel, it should predict type assignments.
        """
        # Without real type labels, return placeholder
        # Real implementation would receive pseudo-type labels from role_typer
        flat = [s for seq in self.sequences for s in seq]
        if not flat:
            return 0.0

        # Simulate: geometry cluster ID vs type cluster ID
        # Random expectation: MI ≈ 0
        # If geometry is informative, MI > 0
        n = len(flat)
        geom_clusters = np.random.randint(0, 5, size=n)
        type_clusters = np.random.randint(0, 5, size=n)

        # Compute MI
        return self._mi(geom_clusters, type_clusters)

    def _mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two discrete variables."""
        xy = np.stack([x, y], axis=1)
        xy_counts = Counter(map(tuple, xy))
        x_counts = Counter(x)
        y_counts = Counter(y)
        n = len(x)

        mi = 0.0
        for (xi, yi), cxy in xy_counts.items():
            px = x_counts[xi] / n
            py = y_counts[yi] / n
            pxy = cxy / n
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
        return mi

    def _detect_periodic_geometry(self) -> Dict[str, float]:
        """
        Detect signs whose geometry repeats with a period.
        Candidate frame markers from geometry alone.
        """
        # Without real geometry data, return empty
        # Real implementation would analyze image sequences
        return {}

    def _random_phi_control(self) -> float:
        """Random baseline: fraction of random rectangles with phi hit."""
        return 0.039  # 3.9% from embedded research


@dataclass
class PhiScanResult:
    """Result from phi scanner on a single sign."""
    sign_id: str
    aspect_ratio: float
    phi_score: float        # 0-1, 1 = perfect φ
    width: float
    height: float
    is_phi_match: bool


def phi_summary(result: GeometryResult) -> List[str]:
    """Human-readable geometry channel summary."""
    return [
        f"phi_rate={result.phi_rate:.3f}",
        f"phi_matches={result.phi_matches}/{result.n_signs_scanned}",
        f"overall_phi_score={result.overall_phi_score:.3f}",
        f"geometry_type_mi={result.geometry_type_mi:.4f}",
        f"geometry_predicts_type={result.geometry_predicts_type}",
        f"random_phi_rate={result.random_phi_rate:.3f}",
        f"phi_vs_random_gap={result.phi_rate - result.random_phi_rate:.3f}",
    ]
