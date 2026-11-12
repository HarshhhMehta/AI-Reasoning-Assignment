"""
representation.py
=================
Defines the internal logical representation for the AI Reasoning System.

Supports:
- Atomic predicates with constants and variables
- Horn-clause facts and rules
- Conjunctive antecedents
- Negation as Failure (NAF) for literals
- Knowledge Base construction
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re


# ---------------------------------------------------------------------------
# Naming conventions
# ---------------------------------------------------------------------------
# Variables: strings starting with uppercase letter OR '?' prefix (e.g. X, Y, ?x)
# Constants: strings starting with lowercase letter or digits (e.g. bob, alice, 42)
# Predicates: any string used as a functor (case-insensitive normalised to Title case)
# ---------------------------------------------------------------------------


def is_variable(term: str) -> bool:
    """Return True if the term is a logical variable.

    Variables begin with an uppercase letter or start with '?'.
    """
    if not isinstance(term, str) or not term:
        return False
    return term[0].isupper() or term.startswith('?')


def is_constant(term: str) -> bool:
    """Return True if the term is a constant (not a variable)."""
    return not is_variable(term)


def normalize_predicate(name: str) -> str:
    """Normalise a predicate name to a canonical form (stripped, lower-case)."""
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Literal:
    """A single atomic literal, optionally negated.

    Attributes
    ----------
    predicate : str
        The predicate / relation name (normalised to lower-case).
    args : tuple[str, ...]
        Ordered sequence of term arguments (constants or variables).
    negated : bool
        True when the literal carries negation-as-failure (NAF).
    """
    predicate: str
    args: tuple[str, ...]
    negated: bool = False

    def __post_init__(self) -> None:
        # Enforce immutable tuple
        object.__setattr__(self, 'args', tuple(self.args))
        object.__setattr__(self, 'predicate', normalize_predicate(self.predicate))

    @property
    def arity(self) -> int:
        return len(self.args)

    def __repr__(self) -> str:
        sign = "¬" if self.negated else ""
        args_str = ", ".join(self.args)
        return f"{sign}{self.predicate}({args_str})"

    def negate(self) -> 'Literal':
        """Return the logical negation of this literal."""
        return Literal(self.predicate, self.args, not self.negated)

    def apply_substitution(self, subst: dict[str, str]) -> 'Literal':
        """Apply a variable substitution and return a new Literal."""
        new_args = tuple(subst.get(a, a) if is_variable(a) else a for a in self.args)
        return Literal(self.predicate, new_args, self.negated)

    def is_ground(self) -> bool:
        """Return True if all arguments are constants (no variables)."""
        return all(is_constant(a) for a in self.args)


@dataclass
class Clause:
    """A Horn clause: head :- body.

    If ``body`` is empty, the clause is a *fact*.
    If ``head`` is None, the clause is a *goal* (integrity constraint).

    Attributes
    ----------
    head : Optional[Literal]
        The consequent literal (None for a constraint / goal).
    body : list[Literal]
        The antecedent literals (possibly empty for facts).
    label : str
        Human-readable identifier for tracing.
    """
    head: Optional[Literal]
    body: list[Literal] = field(default_factory=list)
    label: str = ""

    @property
    def is_fact(self) -> bool:
        return len(self.body) == 0 and self.head is not None

    @property
    def is_rule(self) -> bool:
        return len(self.body) > 0 and self.head is not None

    def __repr__(self) -> str:
        if self.is_fact:
            return f"{self.head}."
        body_str = " ∧ ".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}."

    def apply_substitution(self, subst: dict[str, str]) -> 'Clause':
        new_head = self.head.apply_substitution(subst) if self.head else None
        new_body = [lit.apply_substitution(subst) for lit in self.body]
        return Clause(new_head, new_body, self.label)


@dataclass
class KnowledgeBase:
    """Container for facts, rules, and a structured index.

    Attributes
    ----------
    facts : list[Literal]
        All ground facts.
    rules : list[Clause]
        All rules (head :- body).
    _fact_index : dict
        Maps (predicate, arity) → list of ground Literals for O(1) lookup.
    _rule_index : dict
        Maps (predicate, arity) → list of Clauses whose head matches.
    """
    facts: list[Literal] = field(default_factory=list)
    rules: list[Clause] = field(default_factory=list)
    _fact_index: dict = field(default_factory=dict, repr=False)
    _rule_index: dict = field(default_factory=dict, repr=False)
    _fact_set: set = field(default_factory=set, repr=False)

    def add_fact(self, literal: Literal) -> bool:
        """Add a ground fact; return True if it was new."""
        if not literal.is_ground():
            raise ValueError(f"Facts must be ground. Got: {literal}")
        key = (literal.predicate, literal.arity, literal.args, literal.negated)
        if key in self._fact_set:
            return False  # duplicate
        self._fact_set.add(key)
        self.facts.append(literal)
        idx_key = (literal.predicate, literal.arity)
        self._fact_index.setdefault(idx_key, []).append(literal)
        return True

    def add_rule(self, clause: Clause) -> None:
        """Add a rule to the knowledge base and index it."""
        self.rules.append(clause)
        if clause.head is not None:
            idx_key = (clause.head.predicate, clause.head.arity)
            self._rule_index.setdefault(idx_key, []).append(clause)

    def get_facts(self, predicate: str, arity: int) -> list[Literal]:
        """Retrieve all facts matching a (predicate, arity) pair."""
        return self._fact_index.get((normalize_predicate(predicate), arity), [])

    def get_rules(self, predicate: str, arity: int) -> list[Clause]:
        """Retrieve all rules whose head matches a (predicate, arity) pair."""
        return self._rule_index.get((normalize_predicate(predicate), arity), [])

    def contains_fact(self, literal: Literal) -> bool:
        """Check if a ground literal is in the fact base."""
        key = (literal.predicate, literal.arity, literal.args, literal.negated)
        return key in self._fact_set

    def all_predicates(self) -> set[str]:
        """Return all predicate names mentioned in facts and rules."""
        preds = set()
        for f in self.facts:
            preds.add(f.predicate)
        for r in self.rules:
            if r.head:
                preds.add(r.head.predicate)
            for b in r.body:
                preds.add(b.predicate)
        return preds

    def summary(self) -> str:
        """Return a short textual summary of the KB."""
        return (f"KnowledgeBase: {len(self.facts)} facts, "
                f"{len(self.rules)} rules, "
                f"{len(self.all_predicates())} predicates.")


@dataclass
class Example:
    """Normalised example from any of the three datasets.

    Attributes
    ----------
    id : str
        Unique identifier for the example.
    context : str
        Raw natural-language knowledge base text.
    query : str
        The natural-language query.
    label : Optional[bool]
        Ground-truth label (True/False/None if unknown).
    dataset : str
        Source dataset name (ruletaker | counterbench | folio).
    split : str
        Dataset split (train | validation | test).
    raw : dict
        Original fields from the dataset for reference.
    """
    id: str
    context: str
    query: str
    label: Optional[bool]
    dataset: str
    split: str
    raw: dict = field(default_factory=dict, repr=False)
