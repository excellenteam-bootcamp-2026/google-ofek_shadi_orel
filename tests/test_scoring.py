"""Tests for domain/scorer.py.

Rewritten against the now-frozen `Scorer` contract in `domain/ports.py`:
scoring is pure arithmetic over a `MatchResult`. Finding *whether* and
*where* a match exists is Ofek's `OneEditMatcher` (domain/matcher.py); this
file only exercises the score() half.
"""
import pytest

from autocomplete.domain.ports import MatchKind, MatchResult
from autocomplete.domain.scorer import RuleBasedScorer

scorer = RuleBasedScorer()


@pytest.mark.parametrize(
    "match, expected_score",
    [
        (MatchResult(MatchKind.EXACT, 0, 5), 10),          # base 10, no penalty
        (MatchResult(MatchKind.EXACT, 0, 6), 12),           # base 12, no penalty
        (MatchResult(MatchKind.EXACT, 0, 7), 14),           # base 14, no penalty
        (MatchResult(MatchKind.SUBSTITUTION, 1, 5), 5),     # base 10, -5
        (MatchResult(MatchKind.SUBSTITUTION, 4, 5), 8),     # base 10, -2
        (MatchResult(MatchKind.INSERTION, 4, 6), 8),        # base 12, -4
    ],
)
def test_worked_examples(match, expected_score):
    assert scorer.score(match) == expected_score


def test_deletion_inferred_by_symmetry_with_insertion():
    """No worked example for DELETION appears in the assignment appendix.

    This case is inferred by symmetry with INSERTION (they share the same
    insert/delete penalty table): base = 2*4 = 8, penalty for position 3
    on that table is 6, so expected score is 2. Flagged as unverified
    against a staff-provided example -- see the agent report.
    """
    match = MatchResult(MatchKind.DELETION, 3, 4)
    assert scorer.score(match) == 2
