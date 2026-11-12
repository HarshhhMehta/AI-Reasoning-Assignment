"""
backward_chaining.py
====================
Goal-Directed Backward-Chaining (SLD Resolution) Inference Engine.

Algorithm
---------
1. Start with the query literal as the goal.
2. If the goal is already a known fact → succeed.
3. Find all rules whose head unifies with the goal.
4. For each such rule:
   a. Rename variables to avoid clashes.
   b. Unify the goal with the rule head to obtain a substitution.
   c. Replace the goal with the (substituted) body literals as sub-goals.
   d. Recursively prove each sub-goal.
5. If no rule applies and the goal is not a fact → fail.

Supports:
- Variable substitution and unification (via unification.py)
- Cycle detection via a visited-goals set
- Negation-as-Failure (NAF) for negated goals
- Inference trace for explainability
- Bounded depth to prevent infinite recursion
"""
from __future__ import annotations
import time
from typing import Optional
from src.representation import KnowledgeBase, Literal, Clause, is_variable
from src.unification import (unify_literals, apply_subst_literal,
                              apply_subst, rename_variables)


# Type alias
Substitution = dict[str, str]

# Counter for unique variable renaming across recursive calls
_rename_counter = 0


def _fresh_rename_counter() -> int:
    global _rename_counter
    _rename_counter += 1
    return _rename_counter


def _prove(goal: Literal, kb: KnowledgeBase,
           subst: Substitution, visited: frozenset,
           depth: int, max_depth: int,
           trace: list[str], verbose: bool) -> Optional[Substitution]:
    """Attempt to prove a single goal literal using backward chaining (SLD).

    Returns the most-general unifier (substitution) on success, or None on failure.

    Parameters
    ----------
    goal : Literal
        The current goal literal (may contain variables).
    kb : KnowledgeBase
        The knowledge base.
    subst : Substitution
        Current variable bindings.
    visited : frozenset
        Set of (goal_string, subst_frozen) pairs to detect cycles.
    depth : int
        Current recursion depth.
    max_depth : int
        Maximum allowed recursion depth.
    trace : list[str]
        Accumulates human-readable proof steps.
    verbose : bool
        If True, print steps to stdout.

    Returns
    -------
    Substitution or None
    """
    indent = "  " * depth

    def log(msg: str) -> None:
        trace.append(msg)
        if verbose:
            print(msg)

    if depth > max_depth:
        log(f"{indent}[DEPTH LIMIT] Reached max depth {max_depth} for goal {goal}")
        return None

    # Apply current substitution to goal
    goal = apply_subst_literal(goal, subst)

    # Cycle detection: encode goal + current bindings as a hashable key
    goal_key = (str(goal), tuple(sorted(subst.items())))
    if goal_key in visited:
        log(f"{indent}[CYCLE] Detected cycle on goal {goal} — failing")
        return None
    visited = visited | {goal_key}

    log(f"{indent}Goal: {goal}")

    # ---- Negation-as-Failure ----
    if goal.negated:
        positive_goal = Literal(goal.predicate, goal.args, negated=False)
        result = _prove(positive_goal, kb, dict(subst), visited,
                        depth + 1, max_depth, trace, verbose)
        if result is None:
            log(f"{indent}NAF: {goal} succeeds (positive goal unprovable) → ¬{positive_goal} holds")
            return subst
        else:
            log(f"{indent}NAF: {goal} fails (positive goal provable)")
            return None

    # ---- Check known facts ----
    for fact in kb.get_facts(goal.predicate, goal.arity):
        new_subst = unify_literals(goal, fact, dict(subst))
        if new_subst is not None:
            log(f"{indent}✓ Fact match: {goal} ← {fact}  (subst: {new_subst})")
            return new_subst

    # ---- Try rules ----
    for rule in kb.get_rules(goal.predicate, goal.arity):
        # Rename rule variables to avoid clashes
        suffix = _fresh_rename_counter()
        renamed_head, renamed_body = rename_variables(
            rule.head, rule.body, suffix)

        # Unify goal with rule head
        new_subst = unify_literals(goal, renamed_head, dict(subst))
        if new_subst is None:
            continue

        log(f"{indent}↓ Rule: {rule}  [renamed suffix={suffix}]")
        log(f"{indent}  Head unified: {renamed_head} → subst {new_subst}")

        # Prove each body literal in sequence
        current_subst = new_subst
        all_proved = True
        for body_lit in renamed_body:
            result = _prove(body_lit, kb, dict(current_subst), visited,
                            depth + 1, max_depth, trace, verbose)
            if result is None:
                all_proved = False
                log(f"{indent}  ✗ Sub-goal {body_lit} failed")
                break
            current_subst = result

        if all_proved:
            log(f"{indent}✓ Rule proved: {rule}  (final subst: {current_subst})")
            return current_subst

    log(f"{indent}✗ Goal {goal} FAILED (no fact or rule applies)")
    return None


def backward_chain(kb: KnowledgeBase, query: Literal,
                   max_depth: int = 50,
                   verbose: bool = False) -> tuple[bool, list[str], float]:
    """Run backward chaining to determine if query is entailed by kb.

    Parameters
    ----------
    kb : KnowledgeBase
        The knowledge base with facts and rules.
    query : Literal
        The query literal to prove.
    max_depth : int
        Maximum recursion depth (prevents infinite loops).
    verbose : bool
        If True, print the proof trace.

    Returns
    -------
    (entailed, trace, elapsed_seconds)
    - entailed : bool  — True if the query was proved.
    - trace : list[str]  — Human-readable proof steps.
    - elapsed_seconds : float  — Wall-clock inference time.
    """
    global _rename_counter
    _rename_counter = 0  # reset for reproducibility

    t_start = time.perf_counter()
    trace: list[str] = []

    trace.append(f"Backward Chaining — Query: {query}")
    trace.append(f"KB: {len(kb.facts)} facts, {len(kb.rules)} rules")

    result = _prove(query, kb, {}, frozenset(), 0, max_depth, trace, verbose)
    elapsed = time.perf_counter() - t_start

    if result is not None:
        trace.append(f"\n✓ ENTAILED: {query} is provable (elapsed: {elapsed:.6f}s)")
        return True, trace, elapsed
    else:
        trace.append(f"\n✗ NOT ENTAILED: {query} has no proof (elapsed: {elapsed:.6f}s)")
        return False, trace, elapsed
