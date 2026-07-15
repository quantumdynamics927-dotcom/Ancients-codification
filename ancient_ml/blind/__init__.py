"""
Blind Package: Blind Formal Language Discovery Stack
================================================
Full sophistication pipeline for blind protocol reverse engineering
of ancient sign streams. All modules use raw sign IDs only — no semantic knowledge.

Modules:
    tokenizer.py     - Opaque tokenization + optional shape clusters
    fingerprint.py   - L0: H1/H2, R, MI, Zipf, positional roles
    field_segmenter - L1: protocol field boundary detection
    role_typer.py   - L2: behavioral pseudo-type clustering
    grammar_inducer - L3: unsupervised CFG/automaton induction
    complexity.py   - L4: compression / Kolmogorov proxies
    family_manifold - L5: multi-feature family distance + novelty
    geometry_channel - L6: phi geometry as second encoding channel
    quantum_map.py  - L7: optional TMT export
    null_models.py  - L8: adversarial shuffle/Markov/random controls
    validate.py     - Validation harness: held-out splits, bootstrap CI,
                      baseline self-classification, matched nulls,
                      preregistered geometry, LEVEL 0-4 verdict ladder
    report.py       - Full sophistication scorecard

Usage:
    from blind.report import generate_report
    report = generate_report(sequences, corpus_name="etcsri")

    from blind.validate import Validator
    validator = Validator(sequences, corpus_name="etcsri")
    result = validator.run_full_validation()
    print(result.verdict_ladder)
"""

from blind.report import generate_report, FullReport
from blind.tokenizer import Tokenizer, TokenizerConfig
from blind.fingerprint import compute_fingerprints, FingerprintVector
from blind.field_segmenter import FieldSegmenter, SegmentResult
from blind.role_typer import RoleTyper, PseudoType
from blind.grammar_inducer import GrammarInducer, InducedGrammar
from blind.complexity import ComplexityAnalyzer, ComplexityResult
from blind.family_manifold import FamilyManifold, FeatureVector, ManifoldResult
from blind.geometry_channel import GeometryChannel, GeometryResult
from blind.null_models import NullModels, NullMetrics
from blind.validate import Validator, ValidationResult, validation_summary

__all__ = [
    "generate_report",
    "FullReport",
    "Validator",
    "ValidationResult",
    "validation_summary",
    "Tokenizer",
    "TokenizerConfig",
    "compute_fingerprints",
    "FingerprintVector",
    "FieldSegmenter",
    "SegmentResult",
    "RoleTyper",
    "PseudoType",
    "GrammarInducer",
    "InducedGrammar",
    "ComplexityAnalyzer",
    "ComplexityResult",
    "FamilyManifold",
    "FeatureVector",
    "ManifoldResult",
    "GeometryChannel",
    "GeometryResult",
    "NullModels",
    "NullMetrics",
]
