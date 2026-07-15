# Ancients Codification

## Project Overview
Research framework treating ancient writing systems (Egyptian hieroglyphs, Sumerian cuneiform) as **machine communication protocols / structured encoding systems** — analogous to Python, TCP/IP, or Bluetooth rather than natural organic languages.

## Core Modules

### `ancient_ml/entropy_analysis.py`
Shannon entropy analysis (H1, H2, redundancy) comparing ancient sign sequences against baselines: natural language (English), source code (Python), network protocols (TCP/IP), DNA sequences.

### `ancient_ml/determinative_graph.py`
Classifier/determinative co-occurrence network analysis with spectral methods, community detection, and centrality ranking. Tests whether determinatives form a designed taxonomy vs. organic categorization.

### `ancient_ml/geometric_phi_scanner.py`
Golden ratio (φ ≈ 1.618) detection in sign geometry. Maps sign sequences to quantum circuit topologies (Sierpinski, Merkaba) and computes phi-resonance scores for quantum annealing frameworks.

### `ancient_ml/run_pipeline.py`
Unified runner executing all three modules in sequence.

## Running the Pipeline

```bash
cd ancient_ml
python run_pipeline.py --all --corpus hieroglyphs  # or --corpus oracc
```

Individual modules:
```bash
python entropy_analysis.py --corpus hieroglyphs
python determinative_graph.py --corpus hieroglyphs
python geometric_phi_scanner.py --signs hieroglyphs
```

## Key Results

| Vector | Finding |
|--------|---------|
| Entropy | Closer to Python source code than natural language |
| Phi Detection | 73.3% of Gardiner signs embed φ proportions |
| Quantum Resonance | 0.467 phi-resonance score (Sierpinski topology) |

## Dependencies
- numpy, pandas, matplotlib, networkx, scipy, requests
