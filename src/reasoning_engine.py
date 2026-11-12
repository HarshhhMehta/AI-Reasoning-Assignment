"""
reasoning_engine.py
===================
High-level Reasoning Engine combining forward and backward chaining.

This module:
1. Selects the appropriate inference strategy.
2. Orchestrates parsing → KB construction → inference.
3. Returns a binary prediction (True/False) and an explanation trace.

Design
------
- For queries that are directly answerable, backward chaining is tried first
  (it is goal-directed and terminates quickly for shallow proofs).
- Forward chaining is run on the full KB to derive new facts, then the
  query is checked against the enriched KB.
- Both strategies are tried; if either confirms entailment, True is returned.

Negation semantics
------------------
Under Closed World Assumption (CWA) / Negation-as-Failure:
  - If the query cannot be proved, the label is False.
  - This does NOT mean the negation has been proved classically.
  - For NAF negated queries (¬P(x)), the system proves P(x) fails.
"""
from __future__ import annotations
import sys
import os
import time
import copy
import logging
from typing import Optional

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.representation import KnowledgeBase, Literal, Example
from src.parser import parse_example, ParseResult, parse_context, parse_query
from src.forward_chaining import forward_chain
from src.backward_chaining import backward_chain

logger = logging.getLogger(__name__)


class InferenceResult:
    """Container for a single inference result.

    Attributes
    ----------
    prediction : bool
        The binary prediction (True = entailed, False = not entailed).
    method : str
        Which strategy produced the prediction ('backward', 'forward', 'default').
    fc_time : float
        Time used by forward chaining (seconds).
    bc_time : float
        Time used by backward chaining (seconds).
    total_time : float
        Total inference time (seconds).
    trace : list[str]
        Full inference trace for debugging and reporting.
    parse_result : ParseResult
        Parsing output (facts, rules, failures).
    parse_time : float
        Time used by the parser (seconds).
    query_lit : Optional[Literal]
        The parsed query literal.
    n_derived_facts : int
        Number of new facts derived by forward chaining.
    """

    def __init__(self) -> None:
        self.prediction: bool = False
        self.method: str = "default"
        self.fc_time: float = 0.0
        self.bc_time: float = 0.0
        self.total_time: float = 0.0
        self.trace: list[str] = []
        self.parse_result: Optional[ParseResult] = None
        self.parse_time: float = 0.0
        self.query_lit: Optional[Literal] = None
        self.n_derived_facts: int = 0

    def summary(self) -> str:
        return (
            f"Prediction: {self.prediction} | Method: {self.method} | "
            f"Parse: {self.parse_time*1000:.2f}ms | "
            f"BC: {self.bc_time*1000:.2f}ms | "
            f"FC: {self.fc_time*1000:.2f}ms | "
            f"Total: {self.total_time*1000:.2f}ms | "
            f"Derived: {self.n_derived_facts} facts"
        )


class ReasoningEngine:
    """Orchestrates parsing and classical inference for the reasoning system.

    Parameters
    ----------
    use_forward : bool
        Whether to run forward chaining (default True).
    use_backward : bool
        Whether to run backward chaining (default True).
    fc_max_iterations : int
        Maximum fixpoint iterations for forward chaining.
    bc_max_depth : int
        Maximum recursion depth for backward chaining.
    verbose : bool
        If True, print inference traces to stdout.
    """

    def __init__(self,
                 use_forward: bool = True,
                 use_backward: bool = True,
                 fc_max_iterations: int = 200,
                 bc_max_depth: int = 50,
                 verbose: bool = False) -> None:
        self.use_forward = use_forward
        self.use_backward = use_backward
        self.fc_max_iterations = fc_max_iterations
        self.bc_max_depth = bc_max_depth
        self.verbose = verbose

    def predict(self, context: str, query: str,
                example_id: str = "unknown",
                example: Optional['Example'] = None) -> InferenceResult:
        """Run the full reasoning pipeline for a single context-query pair.

        Parameters
        ----------
        context : str
            Natural-language knowledge base.
        query : str
            Natural-language query.
        example_id : str
            Identifier for logging.
        example : Optional[Example]
            If provided, uses parse_example which can leverage structured annotations.

        Returns
        -------
        InferenceResult
        """
        result = InferenceResult()
        t_total_start = time.perf_counter()

        # ---- Phase 1: Parse ----
        # Check for Counterfactual / Causal reasoning (e.g. CounterBench)
        is_counterfactual = (example is not None and example.dataset == 'counterbench') or (
            'causes' in context.lower() and 'would ' in query.lower() and 'occur' in query.lower()
        )
        if is_counterfactual:
            from src.counterfactual_engine import evaluate_counterfactual, parse_counterfactual_query
            tgt, _, _ = parse_counterfactual_query(query)
            if tgt:
                result.query_lit = Literal(tgt, ())
            cf_pred, cf_trace, cf_time = evaluate_counterfactual(context, query)
            result.prediction = cf_pred
            result.method = "counterfactual_scm"
            result.fc_time = cf_time
            result.trace.extend(cf_trace)
            result.total_time = time.perf_counter() - t_total_start
            return result

        t_parse_start = time.perf_counter()
        if example is not None:
            kb, query_lit, pr = parse_example(example)
        else:
            pr = parse_context(context)
            kb = pr.to_kb()
            query_lit = parse_query(query)
        result.parse_time = time.perf_counter() - t_parse_start
        result.parse_result = pr
        result.query_lit = query_lit

        result.trace.append(f"[{example_id}] Context parsed: "
                            f"{len(kb.facts)} facts, {len(kb.rules)} rules, "
                            f"{len(pr.failures)} parse failures")
        result.trace.append(f"  Query literal: {query_lit}")

        if query_lit is None:
            result.trace.append("  ✗ Query could not be parsed → defaulting to False")
            result.prediction = False
            result.method = "parse_failure"
            result.total_time = time.perf_counter() - t_total_start
            return result

        # ---- Phase 2: Backward Chaining ----
        if self.use_backward:
            # Work on a copy so KB is not modified by forward chaining yet
            bc_kb = copy.deepcopy(kb)
            bc_entailed, bc_trace, bc_time = backward_chain(
                bc_kb, query_lit,
                max_depth=self.bc_max_depth,
                verbose=self.verbose
            )
            result.bc_time = bc_time
            result.trace.extend(["--- Backward Chaining ---"] + bc_trace)

            if bc_entailed:
                result.prediction = True
                result.method = "backward"
                result.total_time = time.perf_counter() - t_total_start
                return result

        # ---- Phase 3: Forward Chaining (enriches KB, then re-checks) ----
        if self.use_forward:
            fc_kb = copy.deepcopy(kb)
            n_before = len(fc_kb.facts)
            fc_entailed, fc_trace, fc_time = forward_chain(
                fc_kb, query_lit,
                max_iterations=self.fc_max_iterations,
                verbose=self.verbose
            )
            result.fc_time = fc_time
            result.n_derived_facts = len(fc_kb.facts) - n_before
            result.trace.extend(["--- Forward Chaining ---"] + fc_trace)

            if fc_entailed:
                result.prediction = True
                result.method = "forward"
                result.total_time = time.perf_counter() - t_total_start
                return result

        # ---- Not entailed under CWA ----
        result.prediction = False
        result.method = "not_entailed_cwa"
        result.total_time = time.perf_counter() - t_total_start
        result.trace.append(
            f"✗ {query_lit} is NOT entailed under CWA "
            "(no proof found by backward or forward chaining)"
        )
        return result

    def predict_example(self, example: 'Example') -> InferenceResult:
        """Convenience wrapper: predict from an Example object, using structured annotations if available."""
        return self.predict(example.context, example.query, example.id, example=example)


def predict_single(context: str, query: str,
                   use_forward: bool = True,
                   use_backward: bool = True,
                   verbose: bool = False) -> InferenceResult:
    """Standalone function to run inference on a single context-query pair.

    This is the primary API for the command-line and notebook usage.
    """
    engine = ReasoningEngine(use_forward=use_forward, use_backward=use_backward,
                             verbose=verbose)
    return engine.predict(context, query)
