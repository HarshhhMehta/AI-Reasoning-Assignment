"""
parser.py
=========
Natural-Language to Logic Parser for the AI Reasoning System.

Converts natural-language context (knowledge base text) and queries
into the internal representation (Literal, Clause, KnowledgeBase).

Parsing Strategy
----------------
1. Split into sentences.
2. For each sentence, try rule patterns first (more specific).
3. Then try atomic fact patterns.
4. Sentences that cannot be parsed are flagged as failures.

Key patterns handled:
- "X is [a/an] Y."           → Y(x)
- "X is not [a/an] Y."       → ¬Y(x)  (NAF)
- "X is Y and Z."            → Y(x), Z(x)  (conjunction)
- "All X are Y."             → ∀v: Y(v) :- X(v)
- "If X is Y, X is Z."      → Z(V) :- Y(V)
- "If X [and Y], then Z."   → Z :- X, Y
- "Something/someone/anyone is P." → ∃x: P(x)  (skolemised)
- "X chases/likes/eats Y."  → chases(x, y)
- "they are Y" in rules     → uses pronoun variable V
- Comma-separated conjunctions: "Kind, young things are ..."
"""
from __future__ import annotations
import re
import logging
from typing import Optional
from src.representation import (
    Literal, Clause, KnowledgeBase, Example, normalize_predicate, is_variable
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to a valid lower-case identifier."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def _make_constant(name: str) -> str:
    return _slugify(name)


def _make_predicate(name: str) -> str:
    return _slugify(name)


# Pronouns that should be replaced by a variable
_PRONOUNS = {'they', 'it', 'he', 'she', 'this', 'that', 'someone', 'something',
             'something', 'anyone', 'everything', 'everyone', 'whoever', 'whatever'}

# Common auxiliary-only phrases to skip
_SKIP_PATTERNS = [
    re.compile(r'^\s*$'),
    re.compile(r'^[,\.\!\?]+$'),
]

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split paragraph into individual sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence-ending punctuation followed by space or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sentences:
        s = s.strip().rstrip('.').strip()
        if s:
            cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# Sentence-level pattern matching
# ---------------------------------------------------------------------------

# ---- Compiled patterns ----

# "X is not a/an Y" or "X is not Y"
_PAT_IS_NOT = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+is\s+not\s+(?:an?\s+)?(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "X is a/an Y"
_PAT_ISA = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+is\s+an?\s+(?P<obj>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "X is Y" (property/adjective)
_PAT_IS_PROP = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+is\s+(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "X is Y and Z and ..." (multi-property)
_PAT_IS_MULTI = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+is\s+(?P<preds>[A-Za-z][A-Za-z0-9 ,_ -]*?)$',
    re.IGNORECASE
)

# "All X are Y" → rule
_PAT_ALL_ARE = re.compile(
    r'^[Aa]ll\s+(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+are\s+(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "All X are not Y" → NAF rule
_PAT_ALL_NOT = re.compile(
    r'^[Aa]ll\s+(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+are\s+not\s+(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "If [antecedent], [consequent]" or "If [antecedent] then [consequent]"
_PAT_IF_THEN = re.compile(
    r'^[Ii]f\s+(?P<antecedent>.+?)(?:\s+then\s+|\s*,\s*)(?P<consequent>.+)$',
    re.IGNORECASE
)

# "X does not Y Z" / "X does not like Y"
_PAT_NEG_DOES = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+does\s+not\s+(?P<verb>[a-z]+(?:\s+[a-z]+)?)\s+(?P<obj>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "X verbs Y" → binary predicate
_PAT_TRANS_VERB = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+(?P<verb>chases?|likes?|eats?|sees?|helps?|visits?|knows?|loves?|hates?|needs?|wants?|has|have|owns?)\s+(?P<obj>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "X, Y things/people/animals are Z" → multi-antecedent rule
_PAT_COMMA_ALL = re.compile(
    r'^(?P<preds>[A-Za-z][A-Za-z0-9 ,_-]*?)\s+things?\s+are\s+(?:not\s+)?(?P<cons>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# Pattern: "X are Y" (subjects in plural, treat as rule variable)
_PAT_PLURAL_ARE = re.compile(
    r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+are\s+(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# --- Pronoun / anaphoric patterns for rule bodies ---
# "they are Y" → Y(V)
_PAT_THEY_ARE = re.compile(
    r'^(?:they|it|someone|something)\s+are?\s+(?:not\s+)?(?:an?\s+)?(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)
_PAT_THEY_ARE_NOT = re.compile(
    r'^(?:they|it)\s+are?\s+not\s+(?:an?\s+)?(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "they verb Y" → verb(V, Y)
_PAT_THEY_VERB = re.compile(
    r'^(?:they|it)\s+(?P<verb>chase?s?|like?s?|eat?s?|see?s?|help?s?|visit?s?|know?s?|love?s?|hate?s?|need?s?|want?s?|own?s?)\s+(?P<obj>[A-Za-z][A-Za-z0-9 _-]*)$',
    re.IGNORECASE
)

# "the X" where X is a noun — treat as constant with article
_ARTICLE_RE = re.compile(r'^(?:the|a|an)\s+', re.IGNORECASE)


def _strip_article(text: str) -> str:
    return _ARTICLE_RE.sub('', text).strip()


def _is_pronoun(text: str) -> bool:
    return text.strip().lower() in _PRONOUNS


# ---------------------------------------------------------------------------
# Atomic literal parser
# ---------------------------------------------------------------------------

def _parse_atomic(sentence: str, var: str = "V") -> Optional[Literal]:
    """Try to parse a sentence as an atomic predicate literal.

    Returns a Literal on success or None if no pattern matches.
    The var parameter is used for pronouns/anaphoric references.
    """
    sentence = sentence.strip().rstrip('.')

    # --- Negation first ---
    m = _PAT_IS_NOT.match(sentence)
    if m:
        subj = _make_constant(_strip_article(m.group('subj')))
        pred = _make_predicate(m.group('pred'))
        if _is_pronoun(m.group('subj')):
            return Literal(pred, (var,), negated=True)
        return Literal(pred, (subj,), negated=True)

    # --- "they are not Y" ---
    m = _PAT_THEY_ARE_NOT.match(sentence)
    if m:
        pred = _make_predicate(m.group('pred'))
        return Literal(pred, (var,), negated=True)

    # --- "they are Y" ---
    m = _PAT_THEY_ARE.match(sentence)
    if m:
        pred = _make_predicate(m.group('pred'))
        neg = 'not' in sentence.lower().split('are')[1] if ' are ' in sentence.lower() else False
        return Literal(pred, (var,), negated=neg)

    # --- "they verb Y" ---
    m = _PAT_THEY_VERB.match(sentence)
    if m:
        pred = _make_predicate(m.group('verb'))
        obj = _make_constant(_strip_article(m.group('obj')))
        return Literal(pred, (var, obj))

    # --- "does not verb Y" ---
    m = _PAT_NEG_DOES.match(sentence)
    if m:
        subj = _make_constant(_strip_article(m.group('subj')))
        pred = _make_predicate(m.group('verb').split()[0])  # first word of verb phrase
        obj = _make_constant(_strip_article(m.group('obj')))
        if _is_pronoun(m.group('subj')):
            return Literal(pred, (var, obj), negated=True)
        return Literal(pred, (subj, obj), negated=True)

    # --- "X is a/an Y" ---
    m = _PAT_ISA.match(sentence)
    if m:
        subj = _make_constant(_strip_article(m.group('subj')))
        obj = _make_predicate(m.group('obj'))
        if _is_pronoun(m.group('subj')):
            return Literal(obj, (var,))
        return Literal(obj, (subj,))

    # --- "X is Y" (single property) ---
    m = _PAT_IS_PROP.match(sentence)
    if m:
        subj_raw = m.group('subj')
        subj = _make_constant(_strip_article(subj_raw))
        pred = _make_predicate(m.group('pred'))
        if _is_pronoun(subj_raw):
            return Literal(pred, (var,))
        return Literal(pred, (subj,))

    # --- Transitive verb "X chases Y" ---
    m = _PAT_TRANS_VERB.match(sentence)
    if m:
        subj_raw = m.group('subj')
        subj = _make_constant(_strip_article(subj_raw))
        pred = _make_predicate(m.group('verb').lower().rstrip('es').rstrip('s') or m.group('verb').lower())
        obj = _make_constant(_strip_article(m.group('obj')))
        if _is_pronoun(subj_raw):
            return Literal(pred, (var, obj))
        return Literal(pred, (subj, obj))

    return None


# ---------------------------------------------------------------------------
# Multi-literal atomic parser (handles "X is Y and Z")
# ---------------------------------------------------------------------------

def _parse_multi_property(sentence: str) -> list[Literal]:
    """Parse 'X is Y and Z and ...' into multiple literals."""
    sentence = sentence.strip().rstrip('.')
    m = re.match(
        r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+is\s+(?P<props>.+)$',
        sentence, re.IGNORECASE
    )
    if not m:
        return []
    subj_raw = m.group('subj').strip()
    subj = _make_constant(_strip_article(subj_raw))
    use_var = _is_pronoun(subj_raw)
    props_str = m.group('props')
    # Split on " and " and ","
    parts = re.split(r'\s+and\s+|,\s*', props_str)
    lits = []
    for p in parts:
        p = p.strip().lstrip('not ').strip()
        neg = p.startswith('not ')
        p = p.lstrip('not ').lstrip('a ').lstrip('an ').strip()
        if not p:
            continue
        pred = _make_predicate(p)
        arg = 'V' if use_var else subj
        lits.append(Literal(pred, (arg,), negated=neg))
    return lits


# ---------------------------------------------------------------------------
# Antecedent clause parser
# ---------------------------------------------------------------------------

def _parse_antecedent_clause(phrase: str, var: str = "V") -> list[Optional[Literal]]:
    """Parse a (possibly conjunctive) antecedent phrase into body literals."""
    phrase = phrase.strip().rstrip('.')

    # Check for comma-separated predicates: "kind and young and ..."
    # e.g. "someone is kind and young"
    m_isprops = re.match(
        r'^(?:someone|something|anything|a person|an animal|an object|an entity)\s+is\s+(?P<props>.+)$',
        phrase, re.IGNORECASE
    )
    if m_isprops:
        props = re.split(r'\s+and\s+|,\s*', m_isprops.group('props'))
        lits = []
        for p in props:
            p = p.strip()
            neg = False
            if p.lower().startswith('not '):
                neg = True
                p = p[4:].strip()
            p = re.sub(r'^an?\s+', '', p)
            if p:
                lits.append(Literal(_make_predicate(p), (var,), negated=neg))
        return lits

    # Split on top-level " and "
    parts = re.split(r'\s+and\s+', phrase, flags=re.IGNORECASE)
    lits: list[Optional[Literal]] = []
    for part in parts:
        part = part.strip().rstrip('.')
        lit = _parse_atomic(part, var=var)
        if lit:
            lits.append(lit)
            continue
        # Try "is Y" → Y(var)
        m_prop = re.match(
            r'^(?:is\s+not\s+)?(?:is\s+)?(?:an?\s+)?(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
            part, re.IGNORECASE
        )
        if m_prop:
            pred = _make_predicate(m_prop.group('pred'))
            neg = 'not' in part.lower()[:10]
            lits.append(Literal(pred, (var,), negated=neg))
            continue
        # Could not parse this conjunct
        logger.debug(f"  Could not parse antecedent conjunct: '{part}'")
        lits.append(None)
    return lits


# ---------------------------------------------------------------------------
# Rule parser
# ---------------------------------------------------------------------------

def _parse_rule(sentence: str) -> Optional[list[Clause]]:
    """Try to parse a sentence as one or more Horn-clause rules.

    Returns a list of Clause objects on success, or None if not a rule.
    """
    sentence = sentence.strip().rstrip('.')

    # --- "All X are not Y" → NAF rule ---
    m = _PAT_ALL_NOT.match(sentence)
    if m:
        subj_pred = _make_predicate(m.group('subj'))
        obj_pred = _make_predicate(m.group('pred'))
        var = "V"
        head = Literal(obj_pred, (var,), negated=True)
        body = [Literal(subj_pred, (var,))]
        return [Clause(head, body, label=sentence)]

    # --- "All X are Y" → rule ---
    m = _PAT_ALL_ARE.match(sentence)
    if m:
        subj_pred = _make_predicate(m.group('subj'))
        obj_pred = _make_predicate(m.group('pred'))
        var = "V"
        head = Literal(obj_pred, (var,))
        body = [Literal(subj_pred, (var,))]
        return [Clause(head, body, label=sentence)]

    # --- "X, Y things are [not] Z" (comma-separated multi-antecedent) ---
    m = _PAT_COMMA_ALL.match(sentence)
    if m:
        preds_raw = m.group('preds')
        cons_raw = m.group('cons')
        neg_cons = 'not' in sentence.lower().split('are')[1].split(cons_raw)[0]
        preds = re.split(r',\s*|\s+and\s+', preds_raw)
        var = "V"
        body = [Literal(_make_predicate(p.strip()), (var,)) for p in preds if p.strip()]
        head = Literal(_make_predicate(cons_raw), (var,), negated=neg_cons)
        return [Clause(head, body, label=sentence)]

    # --- "If [antecedent] [then] [consequent]" ---
    m = _PAT_IF_THEN.match(sentence)
    if m:
        ant_phrase = m.group('antecedent').strip()
        cons_phrase = m.group('consequent').strip().rstrip('.')

        var = "V"
        body_lits = _parse_antecedent_clause(ant_phrase, var=var)
        valid_body = [l for l in body_lits if l is not None]
        if not valid_body:
            logger.debug(f"  No valid body literals from: '{ant_phrase}'")
            return None

        # Parse consequent — may be "is Y" or full sentence
        cons_lit = _parse_atomic(cons_phrase, var=var)
        if cons_lit is None:
            # Try "is Y" → Y(var)
            m2 = re.match(
                r'^(?:is\s+not\s+)?(?:is\s+)?(?:an?\s+)?(?P<pred>[A-Za-z][A-Za-z0-9 _-]*)$',
                cons_phrase, re.IGNORECASE
            )
            if m2:
                pred = _make_predicate(m2.group('pred'))
                neg = 'not' in cons_phrase.lower()[:10]
                cons_lit = Literal(pred, (var,), negated=neg)
            else:
                logger.debug(f"  Could not parse consequent: '{cons_phrase}'")
                return None

        return [Clause(cons_lit, valid_body, label=sentence)]

    return None


# ---------------------------------------------------------------------------
# Main ParseResult class and entry points
# ---------------------------------------------------------------------------

class ParseResult:
    """Output of the natural-language parser."""

    def __init__(self) -> None:
        self.facts: list[Literal] = []
        self.rules: list[Clause] = []
        self.query: Optional[Literal] = None
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def to_kb(self) -> KnowledgeBase:
        kb = KnowledgeBase()
        for f in self.facts:
            if f.is_ground():
                kb.add_fact(f)
            else:
                self.warnings.append(f"Non-ground fact skipped: {f}")
        for r in self.rules:
            kb.add_rule(r)
        return kb

    def debug_str(self) -> str:
        lines = ["=== ParseResult ==="]
        lines.append(f"Facts ({len(self.facts)}): " + ", ".join(str(f) for f in self.facts))
        lines.append(f"Rules ({len(self.rules)}):")
        for r in self.rules:
            lines.append(f"  {r}")
        lines.append(f"Query: {self.query}")
        if self.failures:
            lines.append(f"Unparsed ({len(self.failures)}): " + "; ".join(self.failures))
        if self.warnings:
            lines.append(f"Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


def parse_context(context: str) -> ParseResult:
    """Parse a natural-language knowledge base into facts and rules.

    Parameters
    ----------
    context : str
        Raw natural-language KB text.

    Returns
    -------
    ParseResult
    """
    result = ParseResult()
    sentences = split_sentences(context)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Skip obvious non-parseable fragments
        if any(p.match(sent) for p in _SKIP_PATTERNS):
            continue

        # --- Try rule patterns first (more specific) ---
        rules = _parse_rule(sent)
        if rules is not None:
            result.rules.extend(rules)
            continue

        # --- Try multi-property: "X is Y and Z" ---
        multi_lits = _parse_multi_property(sent)
        if multi_lits:
            for lit in multi_lits:
                if lit.is_ground():
                    result.facts.append(lit)
                else:
                    result.warnings.append(f"Non-ground from multi-prop: {lit}")
            continue

        # --- Try single atomic fact ---
        lit = _parse_atomic(sent)
        if lit is not None:
            if lit.is_ground():
                result.facts.append(lit)
            else:
                result.warnings.append(f"Non-ground literal: {lit}")
            continue

        # --- Could not parse ---
        result.failures.append(sent)
        logger.debug(f"Unparsed sentence: '{sent}'")

    return result


def parse_query(query_text: str) -> Optional[Literal]:
    """Parse a natural-language query into a Literal.

    Parameters
    ----------
    query_text : str
        The natural-language query.

    Returns
    -------
    Literal or None.
    """
    query_text = query_text.strip().rstrip('?').rstrip('.').strip()

    # Multi-property query: "X is Y and Z" → use the first predicate as query
    multi = _parse_multi_property(query_text)
    if multi:
        return multi[0]  # Primary property to prove

    lit = _parse_atomic(query_text)
    if lit is not None:
        return lit

    # Try with "is Y" form
    m = re.match(
        r'^(?P<subj>[A-Za-z][A-Za-z0-9 _-]*?)\s+(?P<pred>is\s+[A-Za-z].*)$',
        query_text, re.IGNORECASE
    )
    if m:
        return _parse_atomic(f"{m.group('subj')} {m.group('pred')}")

    logger.warning(f"Could not parse query: '{query_text}'")
    return None


def parse_example(example: 'Example') -> tuple[KnowledgeBase, Optional[Literal], ParseResult]:
    """Parse an Example into (KnowledgeBase, query Literal, ParseResult).

    Automatically uses FOLIO's FOL annotations when available.
    """
    # Check if FOLIO example with FOL annotations
    if example.dataset == 'folio' and example.raw:
        premises_fol = example.raw.get('premises-FOL', '')
        conclusion_fol = example.raw.get('conclusion-FOL', '')
        if premises_fol and conclusion_fol:
            try:
                from src.folio_fol_parser import parse_folio_fol
                kb, query_lit = parse_folio_fol(premises_fol, conclusion_fol)
                pr = ParseResult()
                pr.query = query_lit
                return kb, query_lit, pr
            except Exception as e:
                logger.warning(f"FOLIO FOL parsing failed: {e}; falling back to NL parser")

    pr = parse_context(example.context)
    kb = pr.to_kb()
    query_lit = parse_query(example.query)
    pr.query = query_lit
    return kb, query_lit, pr
