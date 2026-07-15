"""
Grammar Inducer: Unsupervised Formal Grammar from Raw Sign Streams
================================================================
Induces a formal grammar (bigram automaton → PCFG) from raw sequences.
No semantic knowledge — purely structural pattern discovery.

Based on:
- "Unsupervised Grammar Induction" (ACL Anthology Q18-1016)
- "深度自动化语法 Induction" (Stanford NLP group UP-GI)
- Bigram automaton + frequent constituent mining

Usage:
    from blind.grammar_inducer import GrammarInducer, InducedGrammar
    inducer = GrammarInducer(sequences)
    grammar = inducer.induce()
    print(grammar.n_rules)
    print(grammar.acceptance_ratio)  # real vs shuffle
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, FrozenSet
import numpy as np


@dataclass
class ProductionRule:
    """A single context-free grammar production: A -> alpha"""
    lhs: str           # Left-hand side non-terminal
    rhs: Tuple         # Right-hand side sequence of symbols
    support: int       # Number of times observed
    confidence: float  # P(rhs | lhs)


@dataclass
class InducedGrammar:
    """Result of grammar induction."""
    # Rules
    rules: List[ProductionRule]
    n_rules: int

    # Bigram automaton
    states: Set[str]
    accepting_states: Set[str]
    start_state: str
    transitions: Dict[Tuple[str, str], str]  # (state, symbol) -> next_state

    # Coverage
    acceptance_ratio: float    # Fraction of training sequences accepted
    holdout_ratio: float      # Fraction of held-out sequences accepted
    shuffle_ratio: float      # Fraction of shuffled sequences accepted

    # Grammar statistics
    avg_rule_length: float
    branching_factor: float   # Average outgoing transitions per state
    grammar_bits: float       # Log2 of number of possible sequences

    # Rule quality
    n_frequent_constituents: int  # N-grams that form valid constituents


class GrammarInducer:
    """
    Induce a formal grammar from raw sign sequences.
    Pipeline:
    1. Build bigram automaton
    2. Mine frequent constituents
    3. Extract PCFG rules
    4. Evaluate on real vs null models
    """

    def __init__(self, sequences: List[List[str]], max_rule_len: int = 4, min_support: int = 2):
        self.sequences = sequences
        self.max_rule_len = max_rule_len
        self.min_support = min_support
        self._n_train: int = 0
        self._n_holdout: int = 0

    def induce(self) -> InducedGrammar:
        """Run full grammar induction pipeline."""
        # Split train/holdout
        n = len(self.sequences)
        split = max(2, int(0.8 * n))
        train_seqs = self.sequences[:split]
        holdout_seqs = self.sequences[split:]
        self._n_train = len(train_seqs)
        self._n_holdout = len(holdout_seqs)

        # 1. Bigram automaton
        automaton = self._build_automaton(train_seqs)

        # 2. Frequent constituents
        constituents = self._mine_constituents(train_seqs)

        # 3. Extract rules
        rules = self._extract_rules(train_seqs, constituents)

        # 4. Evaluate
        accept_train = self._evaluate(train_seqs, automaton)
        accept_holdout = self._evaluate(holdout_seqs, automaton)
        accept_shuffle = self._evaluate(self._shuffle_sequences(train_seqs), automaton)

        avg_rule_len = np.mean([len(r.rhs) for r in rules]) if rules else 0.0
        branching = self._branching_factor(automaton)
        grammar_bits = self._grammar_bits(rules, automaton)

        return InducedGrammar(
            rules=rules,
            n_rules=len(rules),
            states=set(automaton.keys()),
            accepting_states=self._accepting_states(automaton),
            start_state="_START_",
            transitions=self._build_transition_map(automaton),
            acceptance_ratio=accept_train,
            holdout_ratio=accept_holdout,
            shuffle_ratio=accept_shuffle,
            avg_rule_length=avg_rule_len,
            branching_factor=branching,
            grammar_bits=grammar_bits,
            n_frequent_constituents=len(constituents),
        )

    def _build_automaton(self, sequences: List[List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Build a deterministic bigram automaton.
        State = last observed symbol (or _START_).
        Transition = (state, symbol) -> next_state.
        Counts how many times each transition is used.
        """
        automaton = defaultdict(lambda: defaultdict(int))
        # Initialize with start state
        for seq in sequences:
            if not seq:
                continue
            automaton["_START_"][seq[0]] += 1
            for i in range(len(seq) - 1):
                automaton[seq[i]][seq[i+1]] += 1
            automaton[seq[-1]]["_END_"] += 1

        return dict(automaton)

    def _build_transition_map(self, automaton: Dict) -> Dict[Tuple[str, str], str]:
        """Build deterministic transition map (mode transition per state)."""
        trans = {}
        for state, next_counts in automaton.items():
            if not next_counts:
                continue
            # Mode transition
            mode_next = max(next_counts, key=next_counts.get)
            if mode_next != "_END_":
                trans[(state, mode_next)] = mode_next
        return trans

    def _accepting_states(self, automaton: Dict) -> Set[str]:
        """States that have _END_ transitions."""
        return {s for s, nexts in automaton.items() if "_END_" in nexts}

    def _mine_constituents(self, sequences: List[List[str]]) -> List[Tuple]:
        """
        Mine frequent n-grams that form valid constituents.
        A constituent is an n-gram that appears more than min_support times.
        """
        all_ngrams = []
        for seq in sequences:
            for n in range(2, self.max_rule_len + 1):
                for i in range(len(seq) - n + 1):
                    all_ngrams.append(tuple(seq[i:i+n]))

        ngram_counts = Counter(all_ngrams)
        constituents = [(ng, c) for ng, c in ngram_counts.items() if c >= self.min_support]
        constituents.sort(key=lambda x: x[1], reverse=True)
        return [ng for ng, _ in constituents]

    def _extract_rules(self, sequences: List[List[str]], constituents: List[Tuple]) -> List[ProductionRule]:
        """Extract PCFG rules from sequences and constituents."""
        rules = []
        rule_counts = Counter()

        for seq in sequences:
            # Replace frequent constituents with non-terminal placeholders
            # Simple approach: collect bigram and trigram rules
            if not seq:
                continue
            # Start rule: _START_ -> first_symbol
            rule_counts[("_START_", (seq[0],))] += 1

            for i in range(len(seq) - 1):
                rule_counts[(seq[i], (seq[i+1],))] += 1
                # Also check if this bigram is a constituent
                bigram = (seq[i], seq[i+1])
                if bigram in constituents:
                    rule_counts[(f"N_{len(bigram)}", bigram)] += 1

            for n in range(2, min(self.max_rule_len, len(seq))):
                for i in range(len(seq) - n + 1):
                    ngram = tuple(seq[i:i+n])
                    if ngram in constituents:
                        # Constituent rule: N -> constituent
                        rule_counts[(f"N_{n}", ngram)] += 1

            # End rule
            rule_counts[(seq[-1], ("_END_",))] += 1

        # Build rules with confidence
        # For each LHS, compute P(RHS | LHS)
        lhs_totals = Counter(lhs for (lhs, _) in rule_counts)
        for (lhs, rhs), count in rule_counts.items():
            confidence = count / lhs_totals[lhs]
            rules.append(ProductionRule(
                lhs=lhs,
                rhs=rhs,
                support=count,
                confidence=confidence,
            ))

        rules.sort(key=lambda r: r.support, reverse=True)
        return rules

    def _evaluate(self, sequences: List[List[str]], automaton: Dict) -> float:
        """
        Evaluate how many sequences are accepted by the automaton.
        A sequence is accepted if all its bigram transitions exist in the automaton.
        """
        if not sequences:
            return 0.0
        accepted = 0
        for seq in sequences:
            if self._accept_sequence(seq, automaton):
                accepted += 1
        return accepted / len(sequences)

    def _accept_sequence(self, seq: List[str], automaton: Dict) -> bool:
        """Check if all bigram transitions in seq exist in automaton."""
        if not seq:
            return True
        # Check start
        if seq[0] not in automaton.get("_START_", {}):
            return False
        for i in range(len(seq) - 1):
            if seq[i+1] not in automaton.get(seq[i], {}):
                return False
        return True

    def _shuffle_sequences(self, sequences: List[List[str]]) -> List[List[str]]:
        """Generate shuffle null: preserve bigram distribution, shuffle word order."""
        import random
        result = []
        for seq in sequences:
            shuffled = seq.copy()
            random.seed(42)
            random.shuffle(shuffled)
            result.append(shuffled)
        return result

    def _branching_factor(self, automaton: Dict) -> float:
        """Average number of outgoing transitions per state."""
        if not automaton:
            return 0.0
        total = sum(len(nexts) for nexts in automaton.values())
        return total / len(automaton)

    def _grammar_bits(self, rules: List[ProductionRule], automaton: Dict) -> float:
        """
        Approximate bits needed to encode sequences with this grammar.
        Based on rule support counts.
        """
        if not rules:
            return 0.0
        total_support = sum(r.support for r in rules)
        if total_support == 0:
            return 0.0
        # Bits = -sum P(rule) * log2 P(rule)
        bits = 0.0
        for r in rules:
            p = r.support / total_support
            if p > 0:
                bits -= p * np.log2(p)
        return bits


def grammar_summary(grammar: InducedGrammar) -> List[str]:
    """Generate human-readable summary of induced grammar."""
    lines = []
    lines.append(f"n_rules={grammar.n_rules}")
    lines.append(f"states={len(grammar.states)}")
    lines.append(f"accepting_states={len(grammar.accepting_states)}")
    lines.append(f"branching_factor={grammar.branching_factor:.3f}")
    lines.append(f"acceptance_train={grammar.acceptance_ratio:.3f}")
    lines.append(f"acceptance_holdout={grammar.holdout_ratio:.3f}")
    lines.append(f"acceptance_shuffle={grammar.shuffle_ratio:.3f}")
    gap = grammar.acceptance_ratio - grammar.shuffle_ratio
    lines.append(f"accept_gap={gap:.3f}  (real - shuffle, higher = stronger grammar)")
    lines.append(f"grammar_bits={grammar.grammar_bits:.3f}")
    lines.append(f"frequent_constituents={grammar.n_frequent_constituents}")
    return lines
