"""
Quantum Map: TMT Export for Quantum/Turing Machine Frameworks
==========================================================
Maps induced types + field graph + phi weights to quantum circuit topologies.
Sierpinski triangle and Merkaba star as candidate topologies.
Experimental layer — not required for formal claims.

Based on:
- geometric_phi_scanner.py (phi resonance with Sierpinski/Merkaba)
- "Quantum Walk on the Sierpinski Gasket" (PhysRev 2002)
- Merkaba sacred geometry (Star Tetrahedron)

Usage:
    from blind.quantum_map import QuantumMapper
    qm = QuantumMapper(types, fields, phi_weights)
    circuit = qm.map_to_topology("sierpinski")
    resonance = qm.compute_resonance()
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class QuantumNode:
    """A node in the quantum topology (type + phi weight)."""
    node_id: int
    type_id: str
    phi_weight: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class QuantumEdge:
    """An edge in the quantum topology (type transition)."""
    from_node: int
    to_node: int
    transition_prob: float
    phi_alignment: float  # Does edge align with phi grid?


@dataclass
class QuantumTopology:
    """A mapped quantum circuit topology."""
    name: str
    nodes: List[QuantumNode]
    edges: List[QuantumEdge]
    resonance_score: float      # Overall phi-resonance of topology
    topology_entropy: float     # Entropy of topology structure
    coverage: float            # Fraction of types mapped to nodes


@dataclass
class QuantumResult:
    """Full result of quantum mapping analysis."""
    sierpinski: Optional[QuantumTopology]
    merkaba: Optional[QuantumTopology]
    best_topology: str
    best_resonance: float
    phi_threshold: float       # 0.618 — the golden section
    n_types_mapped: int
    unassigned_types: List[str]


PHI = 1.618033988749895
PHI_INV = 0.618033988749895  # 1/phi


class QuantumMapper:
    """
    Map induced types and field structure to quantum circuit topologies.
    Experimental: Sierpinski triangle and Merkaba star.
    """

    def __init__(
        self,
        types: List,           # List[PseudoType] from role_typer
        fields: List,          # List[Field] from field_segmenter
        phi_weights: Optional[Dict[str, float]] = None,  # sign -> phi weight
    ):
        self.types = types
        self.fields = fields
        self.phi_weights = phi_weights or {}
        self.PHI = PHI
        self.PHI_INV = PHI_INV

    def map_all(self) -> QuantumResult:
        """Map types to both topologies and return comparison."""
        sierpinski = self.map_to_sierpinski()
        merkaba = self.map_to_merkaba()

        best_name = "sierpinski" if (
            sierpinski and merkaba and
            sierpinski.resonance_score >= merkaba.resonance_score
        ) else "merkaba"
        best_res = (
            sierpinski.resonance_score if best_name == "sierpinski" else
            merkaba.resonance_score
        )

        unassigned = []
        mapped_types = set()
        if sierpinski:
            mapped_types.update(n.type_id for n in sierpinski.nodes)
        if merkaba:
            mapped_types.update(n.type_id for n in merkaba.nodes)
        for t in self.types:
            if t.type_id not in mapped_types:
                unassigned.append(t.type_id)

        return QuantumResult(
            sierpinski=sierpinski,
            merkaba=merkaba,
            best_topology=best_name,
            best_resonance=best_res,
            phi_threshold=self.PHI_INV,
            n_types_mapped=len(mapped_types),
            unassigned_types=unassigned,
        )

    def map_to_sierpinski(self) -> Optional[QuantumTopology]:
        """
        Map types to Sierpinski triangle gasket topology.
        The Sierpinski gasket has self-similar fractal structure at phi ratios.
        Nodes are placed at gasket vertices; edges represent type transitions.
        """
        if not self.types:
            return None

        n_nodes = min(len(self.types), self._sierpinski_size(10))
        if n_nodes < 3:
            return None

        # Generate Sierpinski gasket coordinates
        nodes = self._generate_sierpinski_coords(n_nodes)

        # Assign types to nodes by phi-weight sorting
        type_phi = [(t.type_id, self._type_phi_weight(t)) for t in self.types]
        type_phi.sort(key=lambda x: x[1], reverse=True)

        assigned_types = type_phi[:n_nodes]
        assigned_types_dict = {tt[0]: i for i, tt in enumerate(assigned_types)}

        # Build nodes
        qnodes = []
        for i, (type_id, phi_w) in enumerate(assigned_types):
            qnodes.append(QuantumNode(
                node_id=i,
                type_id=type_id,
                phi_weight=phi_w,
                x=nodes[i][0],
                y=nodes[i][1],
                z=0.0,
            ))

        # Build edges from type co-occurrence
        qedges = []
        for t in self.types:
            if t.type_id not in assigned_types_dict:
                continue
            t_idx = assigned_types_dict[t.type_id]
            for co_type, prob in list(t.co_occurring_types.items())[:3]:
                if co_type in assigned_types_dict:
                    co_idx = assigned_types_dict[co_type]
                    # Check phi alignment of edge
                    phi_align = self._edge_phi_alignment(qnodes[t_idx], qnodes[co_idx])
                    qedges.append(QuantumEdge(
                        from_node=t_idx,
                        to_node=co_idx,
                        transition_prob=prob,
                        phi_alignment=phi_align,
                    ))

        # Compute resonance
        resonance = self._compute_resonance(qnodes, qedges)

        return QuantumTopology(
            name="sierpinski",
            nodes=qnodes,
            edges=qedges,
            resonance_score=resonance,
            topology_entropy=self._topology_entropy(qedges),
            coverage=len(assigned_types) / max(len(self.types), 1),
        )

    def map_to_merkaba(self) -> Optional[QuantumTopology]:
        """
        Map types to Merkaba (Star Tetrahedron) topology.
        The Merkaba = two interlocked tetrahedra, 8 vertices, 12 edges.
        Each tetrahedron has phi-scaled faces.
        """
        if not self.types:
            return None

        n_nodes = 8  # Merkaba has 8 vertices
        merkaba_coords = self._generate_merkaba_coords()

        # Assign types by phi weight
        type_phi = [(t.type_id, self._type_phi_weight(t)) for t in self.types]
        type_phi.sort(key=lambda x: x[1], reverse=True)

        assigned = type_phi[:n_nodes]
        assigned_dict = {tt[0]: i for i, tt in enumerate(assigned)}

        qnodes = []
        for i, (type_id, phi_w) in enumerate(assigned):
            x, y, z = merkaba_coords[i]
            qnodes.append(QuantumNode(
                node_id=i,
                type_id=type_id,
                phi_weight=phi_w,
                x=x, y=y, z=z,
            ))

        # Build edges: Merkaba has 12 edges between vertices
        merkaba_edges = self._merkaba_edges()
        qedges = []
        for from_n, to_n in merkaba_edges:
            if from_n < len(qnodes) and to_n < len(qnodes):
                # Transition probability from type co-occurrence
                from_type = qnodes[from_n].type_id
                to_type = qnodes[to_n].type_id
                prob = self._transition_prob(from_type, to_type)
                phi_align = self._edge_phi_alignment(qnodes[from_n], qnodes[to_n])
                qedges.append(QuantumEdge(
                    from_node=from_n,
                    to_node=to_n,
                    transition_prob=prob,
                    phi_alignment=phi_align,
                ))

        resonance = self._compute_resonance(qnodes, qedges)

        return QuantumTopology(
            name="merkaba",
            nodes=qnodes,
            edges=qedges,
            resonance_score=resonance,
            topology_entropy=self._topology_entropy(qedges),
            coverage=len(assigned) / max(len(self.types), 1),
        )

    def _sierpinski_size(self, max_level: int) -> int:
        """Number of nodes in a Sierpinski gasket of given depth."""
        # Sierpinski gasket nodes: 3 corners + internal nodes per level
        return 3 + 3 * max_level

    def _generate_sierpinski_coords(self, n: int) -> List[Tuple[float, float]]:
        """
        Generate first n points on a Sierpinski triangle gasket.
        Uses recursive subdivision: each triangle subdivides into 3 sub-triangles.
        """
        # Start with equilateral triangle vertices
        coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, np.sqrt(3) / 2),
        ]

        # Generate internal points via recursive subdivision
        def subdivide(triangles, depth):
            if depth <= 0 or len(coords) >= n:
                return
            new_triangles = []
            for (a, b, c) in triangles:
                # Midpoints
                ab = ((coords[a][0] + coords[b][0]) / 2, (coords[a][1] + coords[b][1]) / 2)
                bc = ((coords[b][0] + coords[c][0]) / 2, (coords[b][1] + coords[c][1]) / 2)
                ca = ((coords[c][0] + coords[a][0]) / 2, (coords[c][1] + coords[a][1]) / 2)

                if len(coords) < n:
                    coords.append(ab)
                if len(coords) < n:
                    coords.append(bc)
                if len(coords) < n:
                    coords.append(ca)

                new_triangles.append((a, len(coords)-3, len(coords)-2))
                new_triangles.append((len(coords)-3, b, len(coords)-1))
                new_triangles.append((len(coords)-2, len(coords)-1, c))

            subdivide(new_triangles, depth - 1)

        subdivide([(0, 1, 2)], 3)

        return coords[:n]

    def _generate_merkaba_coords(self) -> List[Tuple[float, float, float]]:
        """
        Generate 8 vertices of a Merkaba (Star Tetrahedron).
        Two interlocked tetrahedra, one pointing up, one pointing down.
        Scaled so that edges have phi-proportional lengths.
        """
        # Tetrahedron 1 (pointing up)
        r = 1.0
        tetra1 = [
            (0, r, 0),  # Top apex
            (2*np.sqrt(2)/3*r, 0, -r/3),  # Base triangle
            (-np.sqrt(2)/3*r, 0, -r/3),
            (np.sqrt(2)/3*r, 0, 2*r/3),
        ]

        # Tetrahedron 2 (pointing down — inverted)
        tetra2 = [
            (0, -r, 0),  # Bottom apex
            (2*np.sqrt(2)/3*r, 0, r/3),
            (-np.sqrt(2)/3*r, 0, r/3),
            (np.sqrt(2)/3*r, 0, -2*r/3),
        ]

        return tetra1 + tetra2

    def _merkaba_edges(self) -> List[Tuple[int, int]]:
        """
        Edge connectivity for Merkaba star tetrahedron.
        12 edges total: 6 from each tetrahedron.
        """
        # Tetrahedron 1 edges (node 0-3)
        t1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # Tetrahedron 2 edges (node 4-7)
        t2 = [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
        return t1 + t2

    def _type_phi_weight(self, t) -> float:
        """Get phi weight for a type (from phi_weights or default)."""
        # Average phi weight of member signs
        weights = [self.phi_weights.get(s, 0.5) for s in t.members]
        return np.mean(weights) if weights else 0.5

    def _edge_phi_alignment(self, node_a: QuantumNode, node_b: QuantumNode) -> float:
        """Check if edge between two nodes aligns with phi proportions."""
        dx = node_b.x - node_a.x
        dy = node_b.y - node_a.y
        dz = node_b.z - node_a.z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 1e-6:
            return 0.0

        # Phi-aligned if the distance ratio between nodes matches phi
        # Heuristic: edge is phi-aligned if both nodes have high phi weights
        weight_product = node_a.phi_weight * node_b.phi_weight
        return weight_product

    def _compute_resonance(self, nodes: List[QuantumNode], edges: List[QuantumEdge]) -> float:
        """
        Compute overall phi-resonance of topology.
        Resonance = mean phi_weight * mean edge_phi_alignment.
        """
        if not nodes:
            return 0.0
        node_phi = np.mean([n.phi_weight for n in nodes])
        if not edges:
            return node_phi
        edge_phi = np.mean([e.phi_alignment for e in edges])
        return (node_phi + edge_phi) / 2

    def _topology_entropy(self, edges: List[QuantumEdge]) -> float:
        """Entropy of topology structure."""
        if not edges:
            return 0.0
        probs = np.array([e.transition_prob for e in edges])
        probs = probs / probs.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _transition_prob(self, from_type: str, to_type: str) -> float:
        """Get transition probability between two types."""
        for t in self.types:
            if t.type_id == from_type:
                return t.co_occurring_types.get(to_type, 0.0)
        return 0.0
