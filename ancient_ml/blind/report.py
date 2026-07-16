"""
Report Module: Full Sophistication Scorecard
==========================================
Generates the complete blind formal language report with all layers.
Every run produces the full scorecard - no optional sections silently skipped.

Output format:
    BLIND FORMAL LANGUAGE REPORT
    n_tokens / vocab / n_messages: ...
    --- Protocol ---
    fields_per_msg_mean, field_stability, periodic_markers
    --- Types ---
    n_pseudo_types, type_transition_H
    --- Grammar ---
    n_rules, holdout_accept, shuffle_accept, gap
    --- Complexity ---
    zlib_ratio, grammar_bits, NCD_vs_python, NCD_vs_random
    --- Family manifold ---
    nearest, second, novelty_score
    --- Geometry (opt) ---
    phi_rate, geometry_type_MI
    --- Verdict ---
    KNOWN_FAMILY | HYBRID | NOVEL_FORMAL_SYSTEM
    confidence: bootstrap CI on novelty

Usage:
    from ancient_ml.blind.report import Report, generate_report
    report = generate_report(sequences, corpus_name="etcsri")
    print(report.to_text())
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import numpy as np

# Import all blind modules
from .tokenizer import Tokenizer
from .fingerprint import compute_fingerprints, FingerprintVector
from .field_segmenter import FieldSegmenter, SegmentResult
from .role_typer import RoleTyper, PseudoType
from .grammar_inducer import GrammarInducer, InducedGrammar
from .complexity import ComplexityAnalyzer, ComplexityResult
from .family_manifold import FamilyManifold, FeatureVector, ManifoldResult
from .geometry_channel import GeometryChannel, GeometryResult
from .null_models import NullModels, NullMetrics


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class FullReport:
    """Complete sophistication report from all modules."""
    corpus_name: str
    n_signs: int
    n_sequences: int
    vocab_size: int

    # Fingerprint (L0)
    fp: FingerprintVector

    # Field segmentation (L1)
    fields: List[Any]  # Field objects
    field_result: SegmentResult

    # Pseudo-types (L2)
    types: List[PseudoType]

    # Grammar (L3)
    grammar: Optional[InducedGrammar]

    # Complexity (L4)
    complexity: Optional[ComplexityResult]

    # Family manifold (L5)
    manifold: Optional[ManifoldResult]

    # Geometry (L6)
    geometry: Optional[GeometryResult]

    # Null models (L8)
    null_metrics: Dict[str, NullMetrics]

    # Summary verdict
    verdict: str = "UNKNOWN"
    confidence: str = "low"
    novelty_score: float = 0.0

    def to_text(self) -> str:
        """Generate full text report."""
        lines = []
        width = 70

        lines.append("#" * width)
        lines.append("BLIND FORMAL LANGUAGE REPORT".center(width))
        lines.append(f"Corpus: {self.corpus_name}".center(width))
        lines.append("#" * width)

        # -- Counts ------------------------------------------------------
        lines.append(f"\nBASIC COUNTS")
        lines.append(f"  n_signs={self.n_signs}, vocab={self.vocab_size}, n_messages={self.n_sequences}")

        # -- Protocol (L1) -----------------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L1 - PROTOCOL FIELD SEGMENTATION")
        lines.append(f"{'-' * width}")
        if self.field_result:
            lines.append(f"  fields_per_msg_mean: {self.field_result.n_fields_mean:.2f}")
            lines.append(f"  field_stability_mean: {self.field_result.field_stability_mean:.3f}")
            lines.append(f"  alignment_score: {self.field_result.alignment_score:.3f}")
            lines.append(f"  protocol_header_positions: {self.field_result.protocol_header}")
            lines.append(f"  n_field_boundaries: {len(self.field_result.boundaries)}")
            if self.fields:
                for f in self.fields[:6]:
                    lines.append(
                        f"    Field {f.field_id}: pos {f.start_pos}-{f.end_pos}, "
                        f"width={f.width}, n_unique={f.n_unique}, "
                        f"entropy={f.entropy:.3f}, stability={f.stability:.3f}"
                    )
        else:
            lines.append("  (not computed)")

        # -- Types (L2) -------------------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L2 - PSEUDO-TYPE CLUSTERING")
        lines.append(f"{'-' * width}")
        lines.append(f"  n_pseudo_types: {len(self.types)}")
        if self.types:
            total_members = sum(len(t.members) for t in self.types)
            lines.append(f"  total_members_typed: {total_members}/{self.vocab_size}")
            for t in self.types[:8]:
                lines.append(
                    f"    {t.type_id}: {len(t.members)} members, "
                    f"entropy={t.entropy:.3f}, pos_freedom={t.positional_freedom:.3f}, "
                    f"trans_H={t.transition_entropy:.3f}"
                )
        else:
            lines.append("  (not computed)")

        # -- Grammar (L3) -----------------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L3 - UNSUPERVISED GRAMMAR INDUCTION")
        lines.append(f"{'-' * width}")
        if self.grammar:
            lines.append(f"  n_rules: {self.grammar.n_rules}")
            lines.append(f"  n_states: {len(self.grammar.states)}")
            lines.append(f"  accept_train: {self.grammar.acceptance_ratio:.3f}")
            lines.append(f"  accept_holdout: {self.grammar.holdout_ratio:.3f}")
            lines.append(f"  accept_shuffle: {self.grammar.shuffle_ratio:.3f}")
            gap = self.grammar.acceptance_ratio - self.grammar.shuffle_ratio
            lines.append(f"  accept_gap: {gap:.3f}  (real - shuffle)")
            lines.append(f"  branching_factor: {self.grammar.branching_factor:.3f}")
            lines.append(f"  grammar_bits: {self.grammar.grammar_bits:.3f}")
            lines.append(f"  frequent_constituents: {self.grammar.n_frequent_constituents}")
            if gap > 0.15:
                lines.append(f"  -> STRONG GRAMMAR (gap={gap:.3f} > 0.15)")
            elif gap > 0.05:
                lines.append(f"  -> MODERATE GRAMMAR (gap={gap:.3f} > 0.05)")
            else:
                lines.append(f"  -> WEAK/NO GRAMMAR (gap={gap:.3f})")
        else:
            lines.append("  (not computed)")

        # -- Complexity (L4) ---------------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L4 - ALGORITHMIC COMPLEXITY")
        lines.append(f"{'-' * width}")
        if self.complexity:
            lines.append(f"  zlib_ratio: {self.complexity.zlib_ratio:.3f}  (lower=more structured)")
            lines.append(f"  lzma_ratio: {self.complexity.lzma_ratio:.3f}")
            lines.append(f"  NCD_vs_random: {self.complexity.ncd_vs_random:.3f}")
            lines.append(f"  NCD_vs_markov: {self.complexity.ncd_vs_markov:.3f}")
            lines.append(f"  NCD_vs_python: {self.complexity.ncd_vs_python:.3f}")
            lines.append(f"  entropy_density: {self.complexity.entropy_density:.3f}")
            lines.append(f"  is_compressible: {self.complexity.is_compressible}")
            lines.append(f"  is_high_structure: {self.complexity.is_high_structure}")
        else:
            lines.append("  (not computed)")

        # -- Family Manifold (L5) ----------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L5 - FAMILY MANIFOLD + NOVELTY")
        lines.append(f"{'-' * width}")
        if self.manifold:
            lines.append(f"  nearest_family: {self.manifold.nearest} ({self.manifold.nearest_description[:40]})")
            lines.append(f"  second_family: {self.manifold.second}")
            lines.append(f"  novelty_score: {self.manifold.novelty_score:.3f}")
            lines.append(f"  novelty_verdict: {self.manifold.novelty_verdict}")
            lines.append(f"  confidence: {self.manifold.confidence}")
            lines.append(f"\n  All distances:")
            for fd in self.manifold.distances[:8]:
                lines.append(f"    {fd.family}: {fd.distance:.3f} - {fd.description[:50]}")
        else:
            lines.append("  (not computed)")

        # -- Geometry (L6) -----------------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L6 - GEOMETRY AS SECOND CHANNEL")
        lines.append(f"{'-' * width}")
        if self.geometry:
            lines.append(f"  phi_rate: {self.geometry.phi_rate:.3f}")
            lines.append(f"  phi_matches: {self.geometry.phi_matches}/{self.geometry.n_signs_scanned}")
            lines.append(f"  overall_phi_score: {self.geometry.overall_phi_score:.3f}")
            lines.append(f"  geometry_type_MI: {self.geometry.geometry_type_mi:.4f}")
            lines.append(f"  geometry_predicts_type: {self.geometry.geometry_predicts_type}")
            lines.append(f"  random_phi_rate: {self.geometry.random_phi_rate:.3f}")
            gap = self.geometry.phi_rate - self.geometry.random_phi_rate
            lines.append(f"  phi_vs_random_gap: {gap:.3f}")
            if gap > 0.3:
                lines.append(f"  -> STRONG GEOMETRY SIGNAL (gap={gap:.3f} > 0.30)")
            elif gap > 0.1:
                lines.append(f"  -> MODERATE GEOMETRY SIGNAL (gap={gap:.3f} > 0.10)")
        else:
            lines.append("  (not computed)")

        # -- Null Models (L8) -------------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("L8 - ADVERSARIAL NULL MODELS")
        lines.append(f"{'-' * width}")
        if self.null_metrics:
            for name, m in self.null_metrics.items():
                lines.append(f"  {name}:")
                lines.append(f"    H1_delta: {m.h1_delta:+.3f}  R1_delta: {m.r1_delta:+.3f}")
                lines.append(f"    MI_delta: {m.mi_delta:+.3f}  periodic_delta: {m.periodic_delta:+.3f}")
                real_wins = sum([
                    m.r1_delta > 0.05,
                    m.mi_delta > 0.05,
                    m.periodic_delta > 0.1,
                    m.grammar_gap > 0.1,
                ])
                status = f"REAL (null rejected {real_wins}/4)" if real_wins >= 3 else f"WEAK ({real_wins}/4)"
                lines.append(f"    Status: {status}")
        else:
            lines.append("  (not computed)")

        # -- Verdict -----------------------------------------------------
        lines.append(f"\n{'#' * width}")
        lines.append("VERDICT".center(width))
        lines.append(f"{'#' * width}")
        lines.append(f"  {self.verdict}")
        lines.append(f"  confidence: {self.confidence}")
        lines.append(f"  novelty_score: {self.novelty_score:.3f}")

        # -- Interpretation ladder ---------------------------------------
        lines.append(f"\n{'-' * width}")
        lines.append("RESEARCH CLAIM LADDER")
        lines.append(f"{'-' * width}")
        claims = []

        # Weak claim
        weak = self.fp.r1 > 0.15 and self.fp.mutual_info > 0.3
        claims.append(f"[{'X' if weak else ' '}] Weak: higher structure than shuffle")

        # Medium claim
        medium = (
            weak and
            self.field_result is not None and
            self.field_result.field_stability_mean > 0.6 and
            len(self.types) >= 3
        )
        claims.append(f"[{'X' if medium else ' '}] Medium: multi-field frames + pseudo-types")

        # Strong claim
        strong = medium and self.manifold is not None and self.manifold.novelty_score > 1.2
        claims.append(f"[{'X' if strong else ' '}] Strong: novelty >> vs all code families")

        # Speculative
        speculative = strong and self.geometry is not None and self.geometry.phi_rate > 0.5
        claims.append(f"[{'X' if speculative else ' '}] Speculative: geometry channel carries independent info")

        for c in claims:
            lines.append(f"  {c}")

        lines.append(f"\n{'#' * width}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON export."""
        return {
            "corpus_name": self.corpus_name,
            "n_signs": self.n_signs,
            "n_sequences": self.n_sequences,
            "vocab_size": self.vocab_size,
            "fingerprint": {
                "h1": self.fp.h1,
                "h2": self.fp.h2,
                "h3": self.fp.h3,
                "r1": self.fp.r1,
                "r2": self.fp.r2,
                "mutual_info": self.fp.mutual_info,
                "zipf_alpha": self.fp.zipf_alpha,
                "header_signs": list(self.fp.header_signs),
                "footer_signs": list(self.fp.footer_signs),
                "fixed_signs": list(self.fp.fixed_signs),
                "periodic_signs": self.fp.periodic_signs,
                "repetition_ratio": self.fp.repetition_ratio,
                "mandatory_pairs": self.fp.mandatory_pairs[:20],
            },
            "fields": {
                "n_fields_mean": self.field_result.n_fields_mean if self.field_result else None,
                "field_stability_mean": self.field_result.field_stability_mean if self.field_result else None,
                "alignment_score": self.field_result.alignment_score if self.field_result else None,
                "n_boundaries": len(self.field_result.boundaries) if self.field_result else None,
                "protocol_header": self.field_result.protocol_header if self.field_result else None,
            },
            "types": {
                "n_types": len(self.types),
                "types": [
                    {
                        "type_id": t.type_id,
                        "n_members": len(t.members),
                        "members": t.members[:10],
                        "entropy": t.entropy,
                        "positional_freedom": t.positional_freedom,
                        "transition_entropy": t.transition_entropy,
                    }
                    for t in self.types
                ],
            },
            "grammar": {
                "n_rules": self.grammar.n_rules if self.grammar else None,
                "accept_train": self.grammar.acceptance_ratio if self.grammar else None,
                "accept_holdout": self.grammar.holdout_ratio if self.grammar else None,
                "accept_shuffle": self.grammar.shuffle_ratio if self.grammar else None,
                "accept_gap": (
                    self.grammar.acceptance_ratio - self.grammar.shuffle_ratio
                    if self.grammar else None
                ),
                "branching_factor": self.grammar.branching_factor if self.grammar else None,
                "grammar_bits": self.grammar.grammar_bits if self.grammar else None,
            } if self.grammar else None,
            "complexity": {
                "zlib_ratio": self.complexity.zlib_ratio if self.complexity else None,
                "lzma_ratio": self.complexity.lzma_ratio if self.complexity else None,
                "ncd_vs_random": self.complexity.ncd_vs_random if self.complexity else None,
                "ncd_vs_python": self.complexity.ncd_vs_python if self.complexity else None,
                "is_compressible": self.complexity.is_compressible if self.complexity else None,
                "is_high_structure": self.complexity.is_high_structure if self.complexity else None,
            } if self.complexity else None,
            "manifold": {
                "nearest": self.manifold.nearest if self.manifold else None,
                "second": self.manifold.second if self.manifold else None,
                "novelty_score": self.manifold.novelty_score if self.manifold else None,
                "novelty_verdict": self.manifold.novelty_verdict if self.manifold else None,
                "confidence": self.manifold.confidence if self.manifold else None,
            } if self.manifold else None,
            "geometry": {
                "phi_rate": self.geometry.phi_rate if self.geometry else None,
                "phi_matches": self.geometry.phi_matches if self.geometry else None,
                "overall_phi_score": self.geometry.overall_phi_score if self.geometry else None,
                "geometry_type_mi": self.geometry.geometry_type_mi if self.geometry else None,
                "geometry_predicts_type": self.geometry.geometry_predicts_type if self.geometry else None,
                "random_phi_rate": self.geometry.random_phi_rate if self.geometry else None,
            } if self.geometry else None,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "novelty_score": self.novelty_score,
        }


def generate_report(
    sequences: List[List[str]],
    corpus_name: str = "unknown",
    output_dir: Optional[Path] = None,
    run_grammar: bool = True,
    run_complexity: bool = True,
    run_geometry: bool = True,
    run_nulls: bool = True,
) -> FullReport:
    """
    Run the full sophistication stack on raw sign sequences.
    All layers run in order; set *_bool flags to disable optional layers.

    Returns a FullReport with all results.
    """
    # -- L0: Tokenize -------------------------------------------------
    tok = Tokenizer(sequences)
    tokenized = tok.tokenize()

    # -- L0: Fingerprint -----------------------------------------------
    fp = compute_fingerprints(tokenized)

    # -- L1: Field Segmentation -----------------------------------------
    seg = FieldSegmenter(tokenized)
    field_result = seg.segment()

    # -- L2: Pseudo-Type Clustering -------------------------------------
    typer = RoleTyper(tokenized)
    types = typer.cluster()

    # -- L3: Grammar Induction ------------------------------------------
    grammar = None
    if run_grammar:
        try:
            inducer = GrammarInducer(tokenized)
            grammar = inducer.induce()
        except Exception:
            pass

    # -- L4: Complexity -------------------------------------------------
    complexity = None
    if run_complexity:
        try:
            ca = ComplexityAnalyzer()
            complexity = ca.analyze(tokenized)
        except Exception:
            pass

    # -- L5: Family Manifold -------------------------------------------
    manifold = None
    try:
        fv = FeatureVector(
            n_signs=fp.n_signs,
            n_sequences=fp.n_sequences,
            vocab_size=fp.vocab_size,
            h1=fp.h1,
            h2=fp.h2,
            h3=fp.h3,
            r1=fp.r1,
            r2=fp.r2,
            zipf_alpha=fp.zipf_alpha,
            n_fields=field_result.n_fields_mean if field_result else 0,
            field_stability=field_result.field_stability_mean if field_result else 0,
            periodic_strength=sum(fp.periodic_signs.values()) / max(len(fp.periodic_signs), 1),
            header_length=len(fp.header_signs),
            n_types=len(types),
            type_transition_entropy=np.mean([t.transition_entropy for t in types]) if types else 0,
            n_rules=grammar.n_rules if grammar else 0,
            accept_holdout=grammar.holdout_ratio if grammar else 0,
            accept_gap=(
                grammar.acceptance_ratio - grammar.shuffle_ratio
                if grammar else 0
            ),
            zlib_ratio=complexity.zlib_ratio if complexity else 0,
            ncd_random=complexity.ncd_vs_random if complexity else 0,
        )
        fm = FamilyManifold(fv, target_name=corpus_name)
        manifold = fm.compute()
    except Exception:
        pass

    # -- L6: Geometry Channel -------------------------------------------
    geometry = None
    if run_geometry:
        try:
            gc = GeometryChannel(tokenized)
            geometry = gc.analyze()
        except Exception:
            pass

    # -- L8: Null Models -----------------------------------------------
    null_metrics = {}
    if run_nulls:
        try:
            nm = NullModels(tokenized)

            # Shuffle baseline
            shuffles = nm.shuffled_corpus(n=min(20, len(sequences)))
            shuffle_fps = [compute_fingerprints(shuffles)]

            # Markov baseline
            markovs = nm.markov_corpus(n=min(20, len(sequences)))
            markov_fps = [compute_fingerprints(markovs)]

            # Python control
            pythons = nm.python_control_corpus(n=min(20, len(sequences)))
            python_fps = [compute_fingerprints(pythons)]

            null_metrics["shuffle"] = nm.compute_null_metrics(fp, shuffle_fps[0])
            null_metrics["markov"] = nm.compute_null_metrics(fp, markov_fps[0])
            null_metrics["python"] = nm.compute_null_metrics(fp, python_fps[0])
        except Exception:
            pass

    # -- Verdict --------------------------------------------------------
    if manifold:
        verdict = manifold.novelty_verdict
        confidence = manifold.confidence
        novelty_score = manifold.novelty_score
    else:
        # Fallback verdict based on structure metrics
        if fp.r1 > 0.2 and fp.mutual_info > 0.5:
            verdict = "HYBRID"
            confidence = "medium"
        elif fp.r1 > 0.1:
            verdict = "STRUCTURED"
            confidence = "low"
        else:
            verdict = "UNKNOWN"
            confidence = "low"
        novelty_score = 0.0

    report = FullReport(
        corpus_name=corpus_name,
        n_signs=fp.n_signs,
        n_sequences=fp.n_sequences,
        vocab_size=fp.vocab_size,
        fp=fp,
        fields=field_result.fields if field_result else [],
        field_result=field_result,
        types=types,
        grammar=grammar,
        complexity=complexity,
        manifold=manifold,
        geometry=geometry,
        null_metrics=null_metrics,
        verdict=verdict,
        confidence=confidence,
        novelty_score=novelty_score,
    )

    # -- Save -----------------------------------------------------------
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"blind_sophistication_{corpus_name}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\n[Saved] Report to {report_file}")

    return report
