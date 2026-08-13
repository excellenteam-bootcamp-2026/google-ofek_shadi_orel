"""Tests for OneEditMatcher.

Three layers: the appendix examples, which pin the scores the graders will
check; structural tests on MatchResult, which pin the alignment the matcher
reports so the scorer can be trusted downstream; and the tie-break rules,
which pin *which* reading wins when a typo can be explained more than one way.
"""

from __future__ import annotations

import pytest

from autocomplete.domain.matcher import OneEditMatcher
from autocomplete.domain.ports import MatchKind
from tests.doubles import AppendixScorer

APPENDIX_SENTENCE = "to be or not to be that is the question"
CAT_SENTENCE = "the cat sat on the mat"


@pytest.fixture
def matcher() -> OneEditMatcher:
    return OneEditMatcher(AppendixScorer())


@pytest.fixture
def scorer() -> AppendixScorer:
    return AppendixScorer()


# --- the appendix examples ------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_score"),
    [
        ("to be", 10),
        ("or not", 12),
        ("be that", 14),
        ("2o be", 5),
        ("to pe", 8),
        ("or knot", 8),
    ],
)
def test_appendix_examples(matcher, scorer, query, expected_score):
    result = matcher.match(query, APPENDIX_SENTENCE)
    assert result is not None
    assert scorer.score(result) == expected_score


def test_not_be_needs_more_than_one_edit(matcher):
    """The appendix says this one is not a match at all, not a low score."""
    assert matcher.match("not be", APPENDIX_SENTENCE) is None


# --- exact matches, at every position -------------------------------------


@pytest.mark.parametrize("query", ["the cat", "cat sat", "the mat"])
def test_exact_match_at_start_middle_and_end(matcher, query):
    result = matcher.match(query, CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.EXACT
    assert result.error_position == 0
    assert result.matched_chars == len(query)


# --- substitution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_position"),
    [
        ("xat", 1),      # wrong at the start
        ("the cxt", 6),  # wrong in the middle
        ("cax", 3),      # wrong at the end
    ],
)
def test_substitution_reports_position_and_full_length(
    matcher, query, expected_position
):
    result = matcher.match(query, CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.SUBSTITUTION
    assert result.error_position == expected_position
    # A substituted character still lands on a sentence character, so it
    # counts towards the base score. This is the appendix reading.
    assert result.matched_chars == len(query)


# --- insertion: the user typed one character too many ---------------------


@pytest.mark.parametrize(
    ("query", "expected_position"),
    [
        ("tthe cat", 2),
        ("the cbat", 6),   # not "the ccat"/"the caat": a doubled letter makes
        ("the caxt", 7),   # removing either copy equally valid -- a real tie,
                           # not a single expected position. See the test
                           # below for that case on its own terms.
    ],
)
def test_insertion_loses_one_character_from_the_base(
    matcher, query, expected_position
):
    result = matcher.match(query, CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.INSERTION
    assert result.error_position == expected_position
    # The extra character lands on nothing, so it must not be counted.
    assert result.matched_chars == len(query) - 1


def test_a_doubled_letter_ties_and_either_resolution_is_correct(matcher, scorer):
    """"the ccat" has an extra 'c' — removing either copy gives "the cat".

    Position 5 and position 6 are both at or past the penalty table's flat
    tail, so they score identically. Which one the matcher reports is not
    specified behaviour; only the score is.
    """
    result = matcher.match("the ccat", CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.INSERTION
    assert result.error_position in (5, 6)
    assert scorer.score(result) == 2 * 7 - 2  # matched_chars=7, tail penalty=2


def test_extra_character_past_the_end_of_the_sentence(matcher):
    """The walk can stop because the sentence ran out, not because of a typo.

    There is no character under the extra 't' to substitute for, so the only
    honest reading is an insertion.
    """
    result = matcher.match("the catx", "the cat")
    assert result is not None
    assert result.kind is MatchKind.INSERTION
    assert result.error_position == 8
    assert result.matched_chars == 7


# --- deletion: the user left one character out ----------------------------


@pytest.mark.parametrize(
    ("query", "expected_position"),
    [
        ("th cat", 3),
        ("the ct sat", 6),
        ("the cat sa on", 11),
    ],
)
def test_deletion_keeps_the_full_base(matcher, query, expected_position):
    """Note there is no case at position 1 or 2 here, and that is not an
    oversight: near the start a substitution only costs 5 while a missing
    character costs 8 or 10, so an early deletion is almost always outscored
    by some substitution reading elsewhere in the sentence. The two tests
    below pin that behaviour deliberately.
    """
    result = matcher.match(query, CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.DELETION
    assert result.error_position == expected_position
    # Every character the user typed still landed, so nothing is lost.
    assert result.matched_chars == len(query)


# --- which reading wins when a typo can be explained more than one way ----


def test_substitution_beats_deletion_at_the_same_position(matcher):
    """'the ct' can be read two ways, and substitution scores higher.

    Substituting the 't' for an 'a' gives 'the ca', which is in the sentence:
    base 12, penalty 1, total 11. Treating the 'a' as missing also works, but
    an insert/delete costs 2 at that position, so it only reaches 10.
    """
    result = matcher.match("the ct", CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.SUBSTITUTION
    assert result.error_position == 6


def test_a_missing_first_character_is_really_an_exact_match(matcher):
    """Dropping the leading character just moves the start of the match.

    'he cat' reads as a deletion, but it is literally a substring of the
    sentence one position later — so the free reading always wins, and a
    deletion at position 1 can never be the best alignment.
    """
    result = matcher.match("he cat", CAT_SENTENCE)
    assert result is not None
    assert result.kind is MatchKind.EXACT


def test_prefers_an_exact_match_over_a_nearby_typo(matcher):
    """'abc' matches at 4 exactly and at 0 with one substitution."""
    result = matcher.match("abc", "abd abc")
    assert result is not None
    assert result.kind is MatchKind.EXACT


def test_prefers_the_later_error_because_it_is_cheaper(matcher):
    """Position 1 costs 5, position 4 costs 2 — the matcher must take 4."""
    result = matcher.match("abcd", "xbcd abcz")
    assert result is not None
    assert result.kind is MatchKind.SUBSTITUTION
    assert result.error_position == 4


# --- things that are not matches ------------------------------------------


def test_two_edits_is_not_a_match(matcher):
    assert matcher.match("the dog", CAT_SENTENCE) is None


def test_empty_query_is_not_a_match(matcher):
    assert matcher.match("", CAT_SENTENCE) is None


def test_query_longer_than_the_sentence_is_not_a_match(matcher):
    assert matcher.match("the cat sat on the mat and more", "the cat") is None


def test_empty_sentence_is_not_a_match(matcher):
    assert matcher.match("the cat", "") is None
