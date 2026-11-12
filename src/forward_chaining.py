"""
forward_chaining.py
===================
Agenda-based Forward-Chaining (Rete-style) Inference Engine.

Algorithm
---------
1. Initialise the working memory with all facts in the KB.
2. Build an agenda (work-list) of rules whose bodies are fully matched.
3. For each triggered rule:
   a. Compute all variable substitutions that satisfy all body literals.
   b. Derive the head literal under those substitutions.
   c. If the derived literal is new, add it to the KB and re-trigger rules.
4. Continue until the query is derived or no new facts can be added.

Supports:
- Variable substitution and unification in rule bodies
- Negation-as-Failure (NAF) for negated body literals
- Cycle prevention via a closed set of derived facts
- Indexed lookup for efficient rule matching
- Inference trace for explainability
"""
from __future__ import annotations
import time
from typing import Optional
from src.representation import KnowledgeBase, Literal, Clause, is_variable
from src.unification import unify_literals, apply_subst_literal, apply_subst


# ---------------------------------------------------------------------------
# Substitution type alias
# ---------------------------------------------------------------------------
Substitution = dict[str, str]


def _match_body(body: list[Literal], kb: KnowledgeBase,
                subst: Substitution) -> list[Substitution]:
    """Recursively find all substitutions that satisfy the body literals.

    Handles both positive and NAF-negated literals.

    Parameters
    ----------
    body : list[Literal]
        Remaining body literals to satisfy.
    kb : KnowledgeBase
        Current knowledge base (may include newly derived facts).
    subst : Substitution
        Current variable bindings.

    Returns
    -------
    list[Substitution]
        All extending substitutions that satisfy the entire body.
    """
    if not body:
        return [dict(subst)]

    lit, *rest = body

    # Apply current substitution to the literal
    grounded_lit = apply_subst_literal(lit, subst)

    # --- Negation-as-Failure ---
    if grounded_lit.negated:
        positive = Literal(grounded_lit.predicate, grounded_lit.args, negated=False)
        # NAF succeeds if we CANNOT prove the positive literal
        can_prove = _literal_entailed(positive, kb)
        if can_prove:
            return []  # NAF fails
        else:
            return _match_body(rest, kb, subst)

    # --- Positive literal: find matching facts ---
    results: list[Substitution] = []
    candidate_facts = kb.get_facts(grounded_lit.predicate, grounded_lit.arity)
    for fact in candidate_facts:
        new_subst = unify_literals(grounded_lit, fact, dict(subst))
        if new_subst is not None:
            extended = _match_body(rest, kb, new_subst)
            results.extend(extended)
    return results


def _literal_entailed(lit: Literal, kb: KnowledgeBase) -> bool:
    """Quick membership check: is the ground literal in the KB fact base?"""
    return kb.contains_fact(lit)


def forward_chain(kb: KnowledgeBase, query: Literal,
                  max_iterations: int = 1000,
                  verbose: bool = False) -> tuple[bool, list[str], float]:
    """Run forward chaining until the query is derived or fixpoint is reached.

    Parameters
    ----------
    kb : KnowledgeBase
        The knowledge base (facts + rules). New facts are added in place.
    query : Literal
        The ground query literal to prove.
    max_iterations : int
        Safety cap on the number of rule-firing iterations.
    verbose : bool
        If True, print inference trace to stdout.

    Returns
    -------
    (entailed, trace, elapsed_seconds)
    - entailed : bool  — True if the query was derived.
    - trace : list[str]  — Human-readable inference steps.
    - elapsed_seconds : float  — Wall-clock time for the inference.
    """
    t_start = time.perf_counter()
    trace: list[str] = []
    step = 0

    def log(msg: str) -> None:
        trace.append(msg)
        if verbose:
            print(msg)

    log(f"Forward Chaining — Query: {query}")
    log(f"Initial facts ({len(kb.facts)}): {[str(f) for f in kb.facts]}")

    # Check if query is already a known fact
    if query.is_ground() and kb.contains_fact(query):
        log(f"Step 0: Query {query} is already a known fact → True")
        return True, trace, time.perf_counter() - t_start

    # Fixpoint iteration
    iteration = 0
    changed = True
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        log(f"\n--- Iteration {iteration} ---")

        for rule in kb.rules:
            # Find all substitutions satisfying the rule body
            solutions = _match_body(rule.body, kb, {})
            for subst in solutions:
                # Derive the head literal
                derived = apply_subst_literal(rule.head, subst)
                if not derived.is_ground():
                    continue  # skip non-ground conclusions
                if not kb.contains_fact(derived):
                    step += 1
                    kb.add_fact(derived)
                    changed = True
                    log(f"  Step {step}: Rule [{rule}] + subst {subst} → derived {derived}")

                    # Check query immediately
                    if query.is_ground() and kb.contains_fact(query):
                        log(f"  ✓ Query {query} derived at step {step} → True")
                        return True, trace, time.perf_counter() - t_start

    elapsed = time.perf_counter() - t_start

    # Final query check
    if query.is_ground() and kb.contains_fact(query):
        log(f"✓ Query {query} found in facts after fixpoint → True")
        return True, trace, elapsed

    log(f"✗ Query {query} not derivable (fixpoint after {iteration} iterations) → False")
    return False, trace, elapsed
