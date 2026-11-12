"""
unification.py
==============
Robinson's Unification Algorithm for the AI Reasoning System.

Supports:
- Variable-to-constant substitutions
- Variable-to-variable substitutions
- Repeated variables
- Incompatible predicates or arities
- Conflicting substitutions
- Consistent substitution propagation
- Occurs check (optional, disabled by default for Horn-clause efficiency)
"""
from __future__ import annotations
from typing import Optional
from src.representation import Literal, is_variable, normalize_predicate


# ---------------------------------------------------------------------------
# Type alias: a substitution maps variable names to term strings
# ---------------------------------------------------------------------------
Substitution = dict[str, str]


class UnificationError(Exception):
    """Raised when two terms cannot be unified."""
    pass


def apply_subst(term: str, subst: Substitution) -> str:
    """Chase variable bindings to their ultimate ground value or unbound var.

    Handles chains: X→Y, Y→alice  →  X→alice.
    """
    visited: set[str] = set()
    while is_variable(term) and term in subst:
        if term in visited:
            raise UnificationError(f"Circular substitution detected for variable '{term}'")
        visited.add(term)
        term = subst[term]
    return term


def apply_subst_literal(lit: Literal, subst: Substitution) -> Literal:
    """Apply a substitution to every argument of a Literal."""
    new_args = tuple(apply_subst(a, subst) for a in lit.args)
    return Literal(lit.predicate, new_args, lit.negated)


def compose_subst(s1: Substitution, s2: Substitution) -> Substitution:
    """Return the composition of two substitutions s1 ∘ s2.

    The result maps each variable to its value after applying s1 then s2.
    """
    composed: Substitution = {}
    for var, val in s1.items():
        new_val = apply_subst(val, s2)
        if new_val != var:
            composed[var] = new_val
    for var, val in s2.items():
        if var not in composed:
            composed[var] = val
    return composed


def occurs_check(var: str, term: str, subst: Substitution) -> bool:
    """Return True if var occurs in term under substitution (for occurs-check).

    Prevents constructing circular terms.
    """
    term = apply_subst(term, subst)
    if var == term:
        return True
    # For simple term language (no nested functors), variable cannot occur in a constant
    return False


def unify_terms(t1: str, t2: str, subst: Substitution,
                use_occurs_check: bool = False) -> Substitution:
    """Unify two terms under the given substitution.

    Modifies and returns the substitution on success.
    Raises UnificationError on failure.

    Parameters
    ----------
    t1, t2 : str
        Terms to unify (variable names or constant strings).
    subst : Substitution
        Current substitution (will be extended).
    use_occurs_check : bool
        If True, perform occurs check before binding a variable.
    """
    # Chase bindings
    t1 = apply_subst(t1, subst)
    t2 = apply_subst(t2, subst)

    # Identical → already unified
    if t1 == t2:
        return subst

    # t1 is a variable → bind it
    if is_variable(t1):
        if use_occurs_check and occurs_check(t1, t2, subst):
            raise UnificationError(f"Occurs check failed: '{t1}' occurs in '{t2}'")
        subst[t1] = t2
        return subst

    # t2 is a variable → bind it
    if is_variable(t2):
        if use_occurs_check and occurs_check(t2, t1, subst):
            raise UnificationError(f"Occurs check failed: '{t2}' occurs in '{t1}'")
        subst[t2] = t1
        return subst

    # Both are constants and they differ → failure
    raise UnificationError(f"Constants '{t1}' and '{t2}' do not unify")


def unify_literals(lit1: Literal, lit2: Literal,
                   subst: Optional[Substitution] = None,
                   use_occurs_check: bool = False) -> Optional[Substitution]:
    """Attempt to unify two literals.

    Returns the most-general-unifier (as a substitution dict) on success,
    or None on failure.

    Parameters
    ----------
    lit1, lit2 : Literal
        Literals to unify.
    subst : dict or None
        Existing substitution to extend (defaults to empty).
    use_occurs_check : bool
        Whether to perform the occurs check.
    """
    if subst is None:
        subst = {}
    else:
        subst = dict(subst)  # work on a copy

    # Predicates must match (already normalised)
    if lit1.predicate != lit2.predicate:
        return None

    # Arity must match
    if lit1.arity != lit2.arity:
        return None

    # Negation must match
    if lit1.negated != lit2.negated:
        return None

    try:
        for a1, a2 in zip(lit1.args, lit2.args):
            subst = unify_terms(a1, a2, subst, use_occurs_check)
        return subst
    except UnificationError:
        return None


def rename_variables(clause_head: Literal, clause_body: list[Literal],
                     suffix: int) -> tuple[Literal, list[Literal]]:
    """Rename all variables in a rule by appending a unique suffix.

    This avoids variable name collisions when the same rule is used
    multiple times in a proof.

    Parameters
    ----------
    clause_head : Literal
        Head literal of the clause.
    clause_body : list[Literal]
        Body literals of the clause.
    suffix : int
        Integer suffix to append to each variable name.

    Returns
    -------
    (renamed_head, renamed_body)
    """
    from src.representation import is_variable

    def rename_term(t: str) -> str:
        return f"{t}_{suffix}" if is_variable(t) else t

    def rename_lit(lit: Literal) -> Literal:
        return Literal(lit.predicate,
                       tuple(rename_term(a) for a in lit.args),
                       lit.negated)

    new_head = rename_lit(clause_head)
    new_body = [rename_lit(b) for b in clause_body]
    return new_head, new_body
