"""
folio_fol_parser.py
===================
Parser for FOLIO's formal FOL (first-order logic) annotations.

FOLIO provides `premises-FOL` and `conclusion-FOL` fields with
Unicode FOL notation. This parser converts them into our internal
Literal/Clause representation.

Supported FOL constructs in FOLIO:
- Universal quantification: ∀x (P(x) → Q(x))
- Existential quantification: ∃x (P(x) ∧ Q(x))
- Conjunction: ∧
- Disjunction: ∨ (partially supported in rules)
- Negation: ¬
- Implication: →
- Biconditional: ↔

Limitations (documented):
- Disjunctions in rule heads → converted to separate rules (approximation)
- Existential quantification → treated as a ground constant placeholder
- Nested quantifiers → limited support
- Arithmetic and counting → not supported
"""
from __future__ import annotations
import re
import logging
from typing import Optional
from src.representation import Literal, Clause, KnowledgeBase, normalize_predicate

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9_]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or 'unknown'


def _parse_atom(atom_str: str) -> Optional[Literal]:
    """Parse a single atomic formula like Pred(x, y) or ¬Pred(x)."""
    atom_str = atom_str.strip()
    negated = False
    if atom_str.startswith('¬') or atom_str.startswith('~'):
        negated = True
        atom_str = atom_str[1:].strip()

    # Match Predicate(arg1, arg2, ...)
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*$', atom_str)
    if m:
        pred = _slugify(m.group(1))
        args_raw = m.group(2).split(',')
        args = tuple(_slugify(a.strip()) for a in args_raw if a.strip())
        return Literal(pred, args, negated=negated)

    # Bare constant/predicate with no parentheses (propositional)
    m2 = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*$', atom_str)
    if m2:
        pred = _slugify(m2.group(1))
        return Literal(pred, (), negated=negated)

    return None


def _split_conjunction(formula: str) -> list[str]:
    """Split a formula on top-level ∧ (not inside parentheses)."""
    parts = []
    depth = 0
    current = []
    for ch in formula:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '∧' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def _split_disjunction(formula: str) -> list[str]:
    """Split a formula on top-level ∨."""
    parts = []
    depth = 0
    current = []
    for ch in formula:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '∨' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def _strip_quantifier(formula: str) -> tuple[str, str]:
    """Strip leading universal or existential quantifier.

    Returns (quantifier_var, inner_formula).
    """
    formula = formula.strip()
    # ∀x or ∃x possibly followed by space and optional comma
    m = re.match(r'^[∀∃]([A-Za-z_][A-Za-z0-9_]*)\s*[,]?\s*(.+)$', formula, re.DOTALL)
    if m:
        var = m.group(1)
        inner = m.group(2).strip()
        # Remove wrapping parentheses
        if inner.startswith('(') and inner.endswith(')'):
            inner = inner[1:-1].strip()
        return var, inner
    return '', formula


def parse_fol_formula(formula: str, label: str = "") -> list[Clause]:
    """Parse a single FOL formula string into Clause(s).

    Handles:
    - Universal rules: ∀x (P(x) → Q(x))
    - Conjunctive antecedents: ∀x (P(x) ∧ R(x) → Q(x))
    - Disjunctive heads: ∀x (P(x) → Q(x) ∨ R(x))  → two separate rules
    - Ground facts: P(alice)
    - Negative facts: ¬P(alice)
    - Existential → treated as skolem constant

    Returns a list of Clause objects.
    """
    formula = formula.strip()
    clauses: list[Clause] = []

    # Skip empty or unparseable
    if not formula or formula in ('?', ''):
        return clauses

    # Strip outermost parentheses
    while formula.startswith('(') and formula.endswith(')'):
        inner = formula[1:-1].strip()
        # Only strip if parens are balanced
        depth = 0
        for ch in inner:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                break
        else:
            formula = inner
            continue
        break

    # Handle quantifiers
    if formula.startswith('∀') or formula.startswith('∃'):
        _var, inner = _strip_quantifier(formula)
        # Recurse on inner
        inner_clauses = parse_fol_formula(inner, label)
        clauses.extend(inner_clauses)
        return clauses

    # Handle implication: A → B
    # Find top-level →
    depth = 0
    arrow_pos = -1
    for i, ch in enumerate(formula):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '→' and depth == 0:
            arrow_pos = i
            break

    if arrow_pos >= 0:
        ant_str = formula[:arrow_pos].strip()
        cons_str = formula[arrow_pos+1:].strip()

        # Parse antecedent (conjunction of literals)
        ant_lits_raw = _split_conjunction(ant_str)
        body_lits: list[Literal] = []
        for a in ant_lits_raw:
            a = a.strip()
            # Remove wrapping parens
            if a.startswith('(') and a.endswith(')'):
                a = a[1:-1].strip()
            # May start with quantifier
            if a.startswith('∀') or a.startswith('∃'):
                _, a = _strip_quantifier(a)
            lit = _parse_atom(a)
            if lit:
                body_lits.append(lit)
            else:
                logger.debug(f"  FOL parser: could not parse antecedent atom: '{a}'")

        if not body_lits:
            logger.warning(f"  FOL: no body parsed from '{ant_str}' in '{formula}'")

        # Parse consequent (may be disjunction)
        cons_parts = _split_disjunction(cons_str)
        for cp in cons_parts:
            cp = cp.strip()
            if cp.startswith('(') and cp.endswith(')'):
                cp = cp[1:-1].strip()
            if cp.startswith('∀') or cp.startswith('∃'):
                _, cp = _strip_quantifier(cp)
            head_lit = _parse_atom(cp)
            if head_lit:
                clauses.append(Clause(head_lit, list(body_lits), label=label))
            else:
                logger.debug(f"  FOL: could not parse consequent: '{cp}'")

        return clauses

    # Handle biconditional: A ↔ B → A→B and B→A
    if '↔' in formula:
        parts = formula.split('↔', 1)
        l_clauses = parse_fol_formula(parts[0].strip() + '→' + parts[1].strip(), label)
        r_clauses = parse_fol_formula(parts[1].strip() + '→' + parts[0].strip(), label)
        return l_clauses + r_clauses

    # Handle conjunction at top level (multiple facts)
    conj_parts = _split_conjunction(formula)
    if len(conj_parts) > 1:
        for p in conj_parts:
            clauses.extend(parse_fol_formula(p.strip(), label))
        return clauses

    # Handle disjunction at top level (multiple disjunctive facts — skip, unsupported)
    disj_parts = _split_disjunction(formula)
    if len(disj_parts) > 1:
        logger.debug(f"  FOL: top-level disjunction not supported as fact: '{formula}'")
        return clauses

    # Atomic formula (fact or negated fact)
    lit = _parse_atom(formula)
    if lit is not None:
        if lit.is_ground():
            clauses.append(Clause(lit, [], label=label))
        else:
            # Variable atom without rule → skip as isolated term
            logger.debug(f"  FOL: variable atom with no rule: '{formula}'")
    else:
        logger.debug(f"  FOL: could not parse atom: '{formula}'")

    return clauses


def parse_folio_fol(premises_fol: str, conclusion_fol: str) -> tuple[KnowledgeBase, Optional[Literal]]:
    """Parse FOLIO's formal FOL fields into a KB and query literal.

    Parameters
    ----------
    premises_fol : str
        The `premises-FOL` field (newline-separated FOL formulas).
    conclusion_fol : str
        The `conclusion-FOL` field (single FOL formula).

    Returns
    -------
    (kb, query_literal)
    """
    kb = KnowledgeBase()

    # Parse each premise
    for line in premises_fol.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        clauses = parse_fol_formula(line, label=line[:50])
        for clause in clauses:
            if clause.is_fact and clause.head.is_ground():
                kb.add_fact(clause.head)
            elif clause.is_rule:
                kb.add_rule(clause)
            # Facts with variables are rules without body — skip or handle as open facts

    # Parse conclusion
    query_lit: Optional[Literal] = None
    if conclusion_fol.strip():
        clauses = parse_fol_formula(conclusion_fol.strip(), label='query')
        if clauses:
            cl = clauses[0]
            if cl.head is not None:
                query_lit = cl.head

    return kb, query_lit
