"""Scorer adapter for the frozen `Scorer` Protocol in `domain/ports.py`.

Pure arithmetic, no search: `OneEditMatcher` (domain/matcher.py) already
decided whether a match exists, what kind it is, and where the edit sits.
This file only turns that `MatchResult` into the score the assignment's
formula defines.
"""
from __future__ import annotations

from autocomplete.domain.ports import MatchKind, MatchResult

# Keyed by the 1-based error_position. Positions at or beyond the largest
# listed key fall back to the default, which is what the spec's "5+" row
# means. EXACT carries no entry -- it is always penalty-free.
_SUBSTITUTION_PENALTIES = {1: 5, 2: 4, 3: 3, 4: 2}
_SUBSTITUTION_DEFAULT_PENALTY = 1
_INDEL_PENALTIES = {1: 10, 2: 8, 3: 6, 4: 4}
_INDEL_DEFAULT_PENALTY = 2


class RuleBasedScorer:
    """Implements `ports.Scorer`: score = 2*matched_chars - penalty.

    `matched_chars` already carries the kind-specific alignment count
    (e.g. it excludes the inserted character for INSERTION), so the same
    `base = 2 * matched_chars` applies uniformly -- only the penalty
    lookup depends on `kind`.
    """

    def score(self, match: MatchResult) -> int:
        base = 2 * match.matched_chars
        if match.kind is MatchKind.EXACT:
            penalty = 0
        elif match.kind is MatchKind.SUBSTITUTION:
            penalty = _SUBSTITUTION_PENALTIES.get(
                match.error_position, _SUBSTITUTION_DEFAULT_PENALTY
            )
        else:  # INSERTION or DELETION share one penalty table
            penalty = _INDEL_PENALTIES.get(
                match.error_position, _INDEL_DEFAULT_PENALTY
            )
        return base - penalty
