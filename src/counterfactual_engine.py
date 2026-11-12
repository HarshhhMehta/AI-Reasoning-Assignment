"""
counterfactual_engine.py
========================
Classical Counterfactual and Causal Reasoning Engine.

Implements deterministic evaluation of counterfactual queries over Structural
Causal Models (SCMs) following Judea Pearl's three-step counterfactual logic:
  1. Abduction: Condition on observed factual background information.
  2. Action / Graph Surgery: Apply counterfactual interventions do(X = x).
  3. Deduction: Deterministic forward inference across causal equations to evaluate
     the counterfactual query literal.

Supported constructs:
  - Deterministic causal implications: X causes Y, X causes not Y, not X causes Y
  - Conjunctions in antecedents: A and B together cause C
  - Disjunctions in antecedents: A or B causes C
  - Multi-consequent causal rules: X causes B and C, X causes B, C, and D
  - Interventions: "if not X instead of X", "if not X and not Y", "Assume not X, suppose not Y"
  - Factual observations: "We observed T"

Unsupported constructs (documented):
  - Probabilistic distributions (e.g. Bern(p) in stochastic backdoor adjustment)
  - Cyclic causal feedback loops
"""
from __future__ import annotations
import re
import time
from typing import Optional, Tuple, List, Dict


def parse_counterfactual_query(question: str) -> Tuple[Optional[str], Dict[str, bool], List[str]]:
    """Extract the query target and interventions from a natural-language counterfactual question.

    Returns:
      (target_var, interventions_dict, trace)
    """
    trace = []
    # 1. Target variable
    m_tgt = re.search(r'Would\s+([A-Za-z0-9_]+)\s+occur', question, re.I)
    if not m_tgt:
        trace.append(f"Could not extract target variable from question: {question}")
        return None, {}, trace

    target = m_tgt.group(1).lower()
    trace.append(f"Target variable: {target}")

    # 2. Interventions
    interventions = {}
    # 'not Var' -> Var = False
    for m in re.finditer(r'(?:if|and|suppose|assume)\s+not\s+([A-Za-z0-9_]+)', question, re.I):
        var = m.group(1).lower()
        interventions[var] = False
        trace.append(f"Intervention: do({var} = False)")

    # positive interventions
    for m in re.finditer(r'(?:if|and|suppose|assume)\s+([A-Za-z0-9_]+)(?:\s+instead|\s*\?)', question, re.I):
        var = m.group(1).lower()
        if var not in ('not', 'instead', 'based', 'this', 'that') and var not in interventions:
            interventions[var] = True
            trace.append(f"Intervention: do({var} = True)")

    return target, interventions, trace


def parse_causal_rules(given_info: str) -> Tuple[List[Tuple[str, List[str], str, bool]], Dict[str, bool], List[str]]:
    """Parse causal relations and factual observations from given_info text.

    Returns:
      (rules, observations, trace)
      where rule is (op, parents, child, neg_consequent)
    """
    trace = []
    obs = {}

    # Extract factual observations
    m_obs = re.search(r'We observed\s+([^.]+)', given_info, re.I)
    if m_obs:
        for w in re.findall(r'[A-Za-z0-9_]+', m_obs.group(1)):
            var = w.lower()
            obs[var] = True
            trace.append(f"Factual observation: {var} = True")

    # Clean text
    text = re.sub(r'We know that\s+', '', given_info, flags=re.I)
    text = re.sub(r'We observed\s+[^.]*\.?', '', text, flags=re.I)
    text = re.sub(r'[A-Za-z0-9_]+\s*~\s*Bern\([^)]*\)\.?', '', text, flags=re.I)
    text = text.strip().rstrip('.')

    # Split into clauses
    text = re.sub(r',\s*and\s+', ';', text)
    text = text.replace(',', ';')
    clauses = [c.strip() for c in text.split(';') if c.strip()]

    rules = []
    for c in clauses:
        # A and B together cause C
        m_together = re.match(r'(.+?)\s+together\s+causes?\s+(not\s+)?(.+)', c, re.I)
        m_causes = re.match(r'(.+?)\s+causes?\s+(not\s+)?(.+)', c, re.I)

        if m_together:
            parents_str = m_together.group(1)
            neg = bool(m_together.group(2))
            children_str = m_together.group(3)
            parents = [p.strip().lower() for p in re.split(r'\s+and\s+', parents_str)]
            children = [ch.strip().lower() for ch in re.split(r'\s+and\s+|,\s*', children_str) if ch.strip()]
            for ch in children:
                rules.append(('and', parents, ch, neg))
                trace.append(f"Causal rule (AND): {' ∧ '.join(parents)} -> {'¬' if neg else ''}{ch}")
        elif m_causes:
            parents_str = m_causes.group(1)
            neg = bool(m_causes.group(2))
            children_str = m_causes.group(3)
            children = [ch.strip().lower() for ch in re.split(r'\s+and\s+|,\s*', children_str) if ch.strip()]
            if ' or ' in parents_str.lower():
                parents = [p.strip().lower() for p in re.split(r'\s+or\s+', parents_str, flags=re.I)]
                for ch in children:
                    rules.append(('or', parents, ch, neg))
                    trace.append(f"Causal rule (OR): {' ∨ '.join(parents)} -> {'¬' if neg else ''}{ch}")
            else:
                parents = [parents_str.strip().lower()]
                for ch in children:
                    rules.append(('single', parents, ch, neg))
                    trace.append(f"Causal rule: {parents[0]} -> {'¬' if neg else ''}{ch}")

    return rules, obs, trace


def evaluate_counterfactual(given_info: str, question: str, max_steps: int = 30) -> Tuple[bool, List[str], float]:
    """Run Pearl's 3-step counterfactual reasoning algorithm deterministically.

    Returns:
      (prediction, trace, elapsed_seconds)
    """
    t_start = time.perf_counter()
    full_trace = ["=== Counterfactual Reasoning Engine (Pearl SCM) ==="]

    target, interventions, q_trace = parse_counterfactual_query(question)
    full_trace.extend(q_trace)

    if not target:
        elapsed = time.perf_counter() - t_start
        full_trace.append("Failed to parse target variable -> prediction: False")
        return False, full_trace, elapsed

    rules, obs, r_trace = parse_causal_rules(given_info)
    full_trace.extend(r_trace)

    # State tracking: mapping variable -> bool
    state: Dict[str, bool] = dict(interventions)
    for o_var, o_val in obs.items():
        if o_var not in state:
            state[o_var] = o_val
            full_trace.append(f"Step 1 (Abduction): Carried over observation {o_var}={o_val}")

    full_trace.append(f"Step 2 (Graph Surgery): Applied interventions {interventions}")

    # Helper to evaluate parent literal
    def eval_parent(p: str, st: Dict[str, bool]) -> Optional[bool]:
        if p.startswith('not '):
            base = p[4:].strip()
            return (not st[base]) if base in st else None
        return st.get(p, None)

    # Step 3 (Deduction): Forward propagate across mutilated causal graph
    full_trace.append("Step 3 (Deduction): Forward propagating across causal rules...")
    for step in range(max_steps):
        changed = False
        for op, parents, child, neg in rules:
            if child in interventions:
                continue  # Intervention cuts incoming causes (graph surgery)

            p_vals = [eval_parent(p, state) for p in parents]
            if None in p_vals:
                continue  # Not all parent values resolved yet

            if op == 'single':
                cond = p_vals[0]
            elif op == 'and':
                cond = all(p_vals)
            elif op == 'or':
                cond = any(p_vals)
            else:
                cond = False

            child_val = (not cond) if neg else cond
            if state.get(child) != child_val:
                state[child] = child_val
                changed = True
                full_trace.append(f"  Step 3.{step+1}: Derived {child} = {child_val} from {op}({parents})")

        if not changed:
            break

    target_result = state.get(target, False)
    elapsed = time.perf_counter() - t_start
    full_trace.append(f"Result for target '{target}': {target_result} (elapsed: {elapsed*1000:.3f} ms)")

    return bool(target_result), full_trace, elapsed
