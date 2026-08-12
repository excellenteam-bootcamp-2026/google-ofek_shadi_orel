"""Alignment of a query against a single sentence, allowing at most one edit.

This is the *verify* half of filter-and-verify. The index has already thrown
away almost the whole corpus; what arrives here is a short list of plausible
sentences that still have to be checked properly and turned into a score.
"""

from __future__ import annotations

import re
from functools import lru_cache

from autocomplete.domain.ports import MatchKind, MatchResult, Scorer

# How many distinct queries' compiled plans to keep. Generous: even a long
# typing session touches a few hundred distinct queries at most, and each
# plan is tiny (a handful of compiled patterns).
_PLAN_CACHE_SIZE = 512


class OneEditMatcher:
    """Finds the best-scoring alignment of a query inside a sentence.

    The query has to appear as a substring of the sentence, either exactly or
    after at most one edit. Any position in the sentence counts — the match
    does not have to start at the beginning.

    The matcher takes a :class:`~autocomplete.domain.ports.Scorer` rather than
    knowing the penalty tables itself. It has to compare alignments to pick
    the best one, and duplicating the scoring rules here would give the
    project two sources of truth for the same formula.

    **Approach.** An earlier version walked every start position in a
    sentence character by character in Python. Correct, but on a candidate
    set of ~150,000 sentences that meant tens of millions of Python-level
    function calls for one query — profiling showed the interpreter's
    per-call overhead, not the character comparisons themselves, was the cost.

    This version turns the question around: instead of asking a sentence
    "what matches you at this position?" once per position, it asks the
    query "what would each possible alignment look like?" up front — there
    are only ``3 * query_length + 1`` of them (exact, plus a substitution, a
    deletion and an insertion at every character position). Each is scored
    *before* checking whether it exists, so they can be tried in true
    best-to-worst order; the first one confirmed present in a sentence, via
    a single regex or substring search covering the whole sentence at once,
    is that sentence's answer.

    :meth:`plan_for` exposes this ordered, pre-scored, pre-compiled list
    directly (cached per query — see below) so a caller checking *many*
    sentences against the same query, like the engine's one-edit search,
    can walk it tier by tier across the whole candidate set instead of
    re-deriving it once per sentence, and stop as soon as it has enough
    results from the tiers already checked — nothing left in a worse tier
    could outrank them.
    """

    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer
        self._plan_for = lru_cache(maxsize=_PLAN_CACHE_SIZE)(self._build_plan)

    def match(
        self, normalized_query: str, normalized_sentence: str
    ) -> MatchResult | None:
        """Return the highest-scoring alignment, or None if there isn't one.

        None means the query needs more than one edit to fit anywhere in this
        sentence, which per the spec is not a match at all — not a low score.
        """
        if not normalized_query:
            return None
        for candidate, occurs in self.plan_for(normalized_query):
            if occurs(normalized_sentence):
                return candidate
        return None

    def plan_for(
        self, normalized_query: str
    ) -> tuple[tuple[MatchResult, "re.Pattern[str].search | object"], ...]:
        """Every alignment for ``normalized_query`` — exact, then every
        one-edit alternative — best-scoring first, each paired with a
        ready-to-call ``occurs(sentence) -> bool`` check.

        Built once per distinct query and cached: the plan depends only on
        the query, never on which sentence is being checked against it.
        """
        if not normalized_query:
            return ()
        return self._plan_for(normalized_query)

    def _build_plan(self, query: str) -> tuple:
        query_length = len(query)
        candidates = [MatchResult(MatchKind.EXACT, 0, query_length)]
        for i in range(query_length):
            candidates.append(MatchResult(MatchKind.SUBSTITUTION, i + 1, query_length))
            candidates.append(MatchResult(MatchKind.DELETION, i + 1, query_length))
            candidates.append(MatchResult(MatchKind.INSERTION, i + 1, query_length - 1))

        # Sorted by actual score, not assumed type priority. "Substitution
        # beats deletion beats insertion" only holds when comparing them at
        # the *same* position — the penalty tables are what actually decide
        # it, and a substitution at a bad position can score below a
        # deletion at a good one elsewhere in the same sentence. Trying
        # candidates in a fixed type-then-position order instead of true
        # score order was tried and cross-checked against ~300,000 random
        # cases against the original position-walking algorithm: it produced
        # the wrong (lower-scoring) answer in cases exactly like that.
        candidates.sort(key=self._scorer.score, reverse=True)

        plan = []
        for candidate in candidates:
            i = candidate.error_position - 1
            if candidate.kind is MatchKind.EXACT:
                plan.append((candidate, _contains(query)))
            elif candidate.kind is MatchKind.SUBSTITUTION:
                # query with the character at i replaced by anything.
                pattern = re.compile(re.escape(query[:i]) + "." + re.escape(query[i + 1 :]))
                plan.append((candidate, pattern.search))
            elif candidate.kind is MatchKind.DELETION:
                # query with one extra, arbitrary sentence character inserted
                # at i -- the sentence has a character here the query doesn't.
                pattern = re.compile(re.escape(query[:i]) + "." + re.escape(query[i:]))
                plan.append((candidate, pattern.search))
            else:
                # INSERTION: query with its character at i removed -- the
                # query has a character here the sentence doesn't, so what's
                # left must appear verbatim.
                variant = query[:i] + query[i + 1 :]
                plan.append((candidate, _contains(variant) if variant else _never))
        return tuple(plan)


def _contains(variant: str):
    return lambda sentence: variant in sentence


def _never(_sentence: str) -> bool:
    return False
