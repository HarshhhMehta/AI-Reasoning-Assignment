"""
test_reasoning_engine.py
========================
Comprehensive unit tests for the AI Reasoning System.

Tests cover:
1.  Query directly present as a fact.
2.  Query follows from one rule.
3.  Query follows from multiple chained rules.
4.  Query requiring two premises in one rule.
5.  Query involving variable substitution.
6.  Query involving repeated variables.
7.  Unprovable query.
8.  Rule cycle that must terminate.
9.  Conflicting substitution (should fail).
10. Query with incorrect predicate or arity.
11. Knowledge base with duplicate facts.
12. Negation-as-Failure (NAF) if supported.
13. Multi-step derivation (Grandparent example).
14. Unknown entity.
15. Empty knowledge base.
16. Malformed input (graceful handling).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.representation import Literal, Clause, KnowledgeBase
from src.unification import unify_literals, rename_variables, apply_subst
from src.forward_chaining import forward_chain
from src.backward_chaining import backward_chain
from src.reasoning_engine import predict_single


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kb(facts: list[Literal], rules: list[Clause] = None) -> KnowledgeBase:
    kb = KnowledgeBase()
    for f in facts:
        kb.add_fact(f)
    if rules:
        for r in rules:
            kb.add_rule(r)
    return kb


def L(pred, *args, neg=False) -> Literal:
    return Literal(pred, args, negated=neg)


def R(head: Literal, *body: Literal, label: str = "") -> Clause:
    return Clause(head, list(body), label=label)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
ERRORS = []


def run_test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  FAIL {name}: {e}")
        FAIL += 1
        ERRORS.append((name, str(e)))
    except Exception as e:
        print(f"  ERROR {name}: EXCEPTION - {e}")
        FAIL += 1
        ERRORS.append((name, f"EXCEPTION: {e}"))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_01_query_is_fact():
    """Query directly present as a fact → True."""
    kb = make_kb([L('kind', 'bob')])
    result, _, _ = backward_chain(kb, L('kind', 'bob'))
    assert result is True, "Expected True (fact directly present)"


def test_02_one_rule():
    """Query follows from one rule: young(bob) → smart(bob)."""
    kb = make_kb(
        [L('young', 'bob')],
        [R(L('smart', 'X'), L('young', 'X'))]
    )
    result, _, _ = backward_chain(kb, L('smart', 'bob'))
    assert result is True, "Expected True via one-rule deduction"


def test_03_multi_rule_chain():
    """Query requires chained rules: young→smart, smart→successful."""
    kb = make_kb(
        [L('young', 'bob'), L('kind', 'bob')],
        [
            R(L('smart', 'X'), L('young', 'X')),
            R(L('successful', 'X'), L('smart', 'X'), L('kind', 'X'))
        ]
    )
    result, trace, _ = backward_chain(kb, L('successful', 'bob'))
    assert result is True, f"Expected True (multi-rule chain). Trace: {trace[-5:]}"


def test_04_two_premises():
    """Rule requires two premises: parent(a,b) ∧ parent(b,c) → grandparent(a,c)."""
    kb = make_kb(
        [L('parent', 'alice', 'bob'), L('parent', 'bob', 'carol')],
        [R(L('grandparent', 'X', 'Z'),
           L('parent', 'X', 'Y'), L('parent', 'Y', 'Z'))]
    )
    result, trace, _ = backward_chain(kb, L('grandparent', 'alice', 'carol'))
    assert result is True, f"Expected True (two-premise rule). Trace: {trace[-5:]}"


def test_05_variable_substitution():
    """Forward chaining derives likes(alice, chess) via variable substitution."""
    kb = make_kb(
        [L('plays', 'alice', 'chess')],
        [R(L('likes', 'X', 'Y'), L('plays', 'X', 'Y'))]
    )
    result, _, _ = forward_chain(kb, L('likes', 'alice', 'chess'))
    assert result is True, "Expected True (variable substitution)"


def test_06_repeated_variables():
    """Rule with repeated variable: reflexive → same(X,X)."""
    kb = make_kb(
        [L('person', 'alice')],
        [R(L('same', 'X', 'X'), L('person', 'X'))]
    )
    result, _, _ = backward_chain(kb, L('same', 'alice', 'alice'))
    assert result is True, "Expected True (repeated variable)"


def test_07_unprovable_query():
    """Query that cannot be derived → False."""
    kb = make_kb(
        [L('kind', 'bob')],
        [R(L('smart', 'X'), L('young', 'X'))]
    )
    result_bc, _, _ = backward_chain(kb, L('smart', 'bob'))
    result_fc, _, _ = forward_chain(kb, L('smart', 'bob'))
    assert result_bc is False, "BC: Expected False (unprovable)"
    assert result_fc is False, "FC: Expected False (unprovable)"


def test_08_cycle_terminates():
    """Cyclic rule: a :- b, b :- a. Must terminate."""
    kb = make_kb(
        [],
        [
            R(L('a', 'x'), L('b', 'x')),
            R(L('b', 'x'), L('a', 'x'))
        ]
    )
    # Both BC and FC should terminate without looping
    t0 = time.perf_counter()
    result_bc, _, _ = backward_chain(kb, L('a', 'x'), max_depth=10)
    result_fc, _, _ = forward_chain(kb, L('a', 'x'), max_iterations=20)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Cycle test took too long: {elapsed:.2f}s"
    assert result_bc is False, "BC: Expected False (no base fact, cyclic)"
    assert result_fc is False, "FC: Expected False (no base fact, cyclic)"


def test_09_conflicting_substitution():
    """Unification with conflicting substitution → None."""
    lit1 = L('foo', 'X', 'X')
    lit2 = L('foo', 'alice', 'bob')
    result = unify_literals(lit1, lit2)
    assert result is None, f"Expected None (conflicting subst), got {result}"


def test_10_wrong_predicate():
    """Query with wrong predicate → False."""
    kb = make_kb([L('kind', 'bob')])
    result, _, _ = backward_chain(kb, L('smart', 'bob'))
    assert result is False, "Expected False (predicate mismatch)"


def test_10b_wrong_arity():
    """Query with correct predicate but wrong arity → False (no unification)."""
    kb = make_kb([L('parent', 'alice', 'bob')])
    result, _, _ = backward_chain(kb, L('parent', 'alice'))
    assert result is False, "Expected False (arity mismatch)"


def test_11_duplicate_facts():
    """KB with duplicate facts should not grow unboundedly."""
    kb = KnowledgeBase()
    lit = L('kind', 'bob')
    added1 = kb.add_fact(lit)
    added2 = kb.add_fact(lit)
    added3 = kb.add_fact(lit)
    assert added1 is True, "First add should succeed"
    assert added2 is False, "Duplicate add should return False"
    assert added3 is False, "Duplicate add should return False"
    assert len(kb.facts) == 1, f"KB should have 1 fact, got {len(kb.facts)}"


def test_12_naf_negation():
    """Negation-as-Failure: not_bird(X) :- ¬bird(X)."""
    kb = make_kb(
        [L('animal', 'bob')],
        [R(L('not_bird', 'X'), L('animal', 'X'), L('bird', 'X', neg=True))]
    )
    result, trace, _ = backward_chain(kb, L('not_bird', 'bob'))
    assert result is True, f"Expected True (NAF: bob is not a bird). Trace: {trace[-5:]}"


def test_13_grandparent_example():
    """Full grandparent example from the assignment spec."""
    facts = [
        L('parent', 'alice', 'bob'),
        L('parent', 'bob', 'carol'),
    ]
    rules = [
        R(L('grandparent', 'X', 'Z'),
          L('parent', 'X', 'Y'), L('parent', 'Y', 'Z'),
          label="grandparent rule")
    ]
    kb = make_kb(facts, rules)
    result, trace, _ = backward_chain(kb, L('grandparent', 'alice', 'carol'))
    assert result is True, f"Expected True (grandparent). Trace: {trace[-5:]}"

    # Also verify False for wrong query
    result2, _, _ = backward_chain(kb, L('grandparent', 'carol', 'alice'))
    assert result2 is False, "Expected False (carol is not alice's grandparent)"


def test_14_unknown_entity():
    """Query about an entity not in the KB → False."""
    kb = make_kb([L('kind', 'alice')])
    result, _, _ = backward_chain(kb, L('kind', 'david'))
    assert result is False, "Expected False (unknown entity)"


def test_15_empty_kb():
    """Empty KB → any query False."""
    kb = KnowledgeBase()
    result_bc, _, _ = backward_chain(kb, L('kind', 'bob'))
    result_fc, _, _ = forward_chain(kb, L('kind', 'bob'))
    assert result_bc is False
    assert result_fc is False


def test_16_malformed_input_graceful():
    """Malformed NL input should not crash the system."""
    ir = predict_single(
        context="!@#$%^&*() completely broken text !!!",
        query="Is Bob smart?"
    )
    # Should return a bool, not raise
    assert isinstance(ir.prediction, bool)


def test_17_assignment_example():
    """Verify the assignment specification example works end-to-end."""
    facts = [L('young', 'bob'), L('kind', 'bob')]
    rules = [
        R(L('smart', 'X'), L('young', 'X')),
        R(L('successful', 'X'), L('smart', 'X'), L('kind', 'X')),
    ]
    kb = make_kb(facts, rules)
    result, trace, _ = backward_chain(kb, L('successful', 'bob'))
    assert result is True, f"Assignment example failed. Trace: {trace}"


def test_18_unification_variable_to_constant():
    """Unification: X unifies with 'alice'."""
    from src.unification import unify_terms
    subst = {}
    result = unify_terms('X', 'alice', subst)
    assert result.get('X') == 'alice', f"Expected X→alice, got {result}"


def test_19_unification_variable_to_variable():
    """Unification: X unifies with Y (both variables)."""
    from src.unification import unify_terms
    subst = {}
    result = unify_terms('X', 'Y', subst)
    # One of them should be bound
    assert 'X' in result or 'Y' in result, f"Expected variable binding, got {result}"


def test_20_forward_chain_assignment_example():
    """Forward chaining also proves the assignment example."""
    import copy
    facts = [L('young', 'bob'), L('kind', 'bob')]
    rules = [
        R(L('smart', 'X'), L('young', 'X')),
        R(L('successful', 'X'), L('smart', 'X'), L('kind', 'X')),
    ]
    kb = make_kb(facts, rules)
    result, trace, _ = forward_chain(kb, L('successful', 'bob'))
    assert result is True, f"FC: Assignment example failed. Trace: {trace}"


def test_21_counterfactual_basic_chain():
    """Counterfactual reasoning: causal chain with intervention -> False."""
    from src.reasoning_engine import predict_single
    ctx = "We know that Ziklo causes Blaf, Blaf causes Trune, and Trune causes Lumbo."
    q = "Would Lumbo occur if not Ziklo instead of Ziklo?"
    res = predict_single(ctx, q)
    assert res.prediction is False, f"Expected False (Lumbo prevented by do(not Ziklo)), got {res.prediction}"


def test_22_counterfactual_negation():
    """Counterfactual reasoning: causal negation reversal -> True."""
    from src.reasoning_engine import predict_single
    ctx = "We know that Ziklo causes not Blaf, Blaf causes Lumbo."
    q = "Would Lumbo occur if not Ziklo instead of Ziklo?"
    res = predict_single(ctx, q)
    assert res.prediction is True, f"Expected True (not Ziklo -> Blaf -> Lumbo), got {res.prediction}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_tests() -> tuple[int, int]:
    global PASS, FAIL, ERRORS
    PASS = 0
    FAIL = 0
    ERRORS = []

    tests = [
        ("T01 Query is fact", test_01_query_is_fact),
        ("T02 One rule", test_02_one_rule),
        ("T03 Multi-rule chain", test_03_multi_rule_chain),
        ("T04 Two premises", test_04_two_premises),
        ("T05 Variable substitution (FC)", test_05_variable_substitution),
        ("T06 Repeated variables", test_06_repeated_variables),
        ("T07 Unprovable query", test_07_unprovable_query),
        ("T08 Cycle terminates", test_08_cycle_terminates),
        ("T09 Conflicting substitution", test_09_conflicting_substitution),
        ("T10 Wrong predicate", test_10_wrong_predicate),
        ("T10b Wrong arity", test_10b_wrong_arity),
        ("T11 Duplicate facts", test_11_duplicate_facts),
        ("T12 NAF negation", test_12_naf_negation),
        ("T13 Grandparent example", test_13_grandparent_example),
        ("T14 Unknown entity", test_14_unknown_entity),
        ("T15 Empty KB", test_15_empty_kb),
        ("T16 Malformed input", test_16_malformed_input_graceful),
        ("T17 Assignment example (BC)", test_17_assignment_example),
        ("T18 Unify var->constant", test_18_unification_variable_to_constant),
        ("T19 Unify var->var", test_19_unification_variable_to_variable),
        ("T20 Assignment example (FC)", test_20_forward_chain_assignment_example),
        ("T21 Counterfactual basic chain", test_21_counterfactual_basic_chain),
        ("T22 Counterfactual negation reversal", test_22_counterfactual_negation),
    ]

    print("\n" + "="*60)
    print("REASONING ENGINE TEST SUITE")
    print("="*60)

    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'='*60}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    if ERRORS:
        print("\nFailed tests:")
        for name, msg in ERRORS:
            print(f"  ✗ {name}: {msg}")
    print("="*60 + "\n")

    return PASS, FAIL


if __name__ == '__main__':
    p, f = run_all_tests()
    sys.exit(0 if f == 0 else 1)
