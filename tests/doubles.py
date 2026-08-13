"""Test doubles shared by the matcher and index tests.

The real scorer lives in ``domain/scorer.py`` and is not this track's file.
The matcher needs *a* scorer to rank the alignments it finds, so the tests
supply their own minimal one built straight from the appendix tables. If this
ever disagrees with the real scorer, the appendix is the tie-breaker.
"""

from __future__ import annotations

from autocomplete.domain.ports import MatchKind, MatchResult

SUBSTITUTION_PENALTIES = (5, 4, 3, 2)
SUBSTITUTION_TAIL_PENALTY = 1

INDEL_PENALTIES = (10, 8, 6, 4)
INDEL_TAIL_PENALTY = 2


class AppendixScorer:
    """Scores an alignment exactly as the English appendix describes."""

    def score(self, result: MatchResult) -> int:
        base = 2 * result.matched_chars
        if result.kind is MatchKind.EXACT:
            return base

        if result.kind is MatchKind.SUBSTITUTION:
            table, tail = SUBSTITUTION_PENALTIES, SUBSTITUTION_TAIL_PENALTY
        else:
            table, tail = INDEL_PENALTIES, INDEL_TAIL_PENALTY

        position = result.error_position
        penalty = table[position - 1] if position <= len(table) else tail
        return base - penalty
