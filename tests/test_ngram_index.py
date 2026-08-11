"""Tests for NGramIndex.

The index is a filter, so there is only one thing it must never do: lose a
sentence that really matches. Returning extra sentences is allowed and
expected — the matcher throws those out. Most of this file is about that
asymmetry.
"""

from __future__ import annotations

import pytest

from autocomplete.domain.matcher import OneEditMatcher
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from tests.doubles import AppendixScorer

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat and a dog are friends",
    "python decorators wrap functions",
]


@pytest.fixture
def index() -> NGramIndex:
    built = NGramIndex()
    for sentence_id, sentence in enumerate(CORPUS):
        built.add(sentence_id, sentence)
    return built


@pytest.fixture
def matcher() -> OneEditMatcher:
    return OneEditMatcher(AppendixScorer())


# --- the index actually narrows things down -------------------------------


def test_exact_query_keeps_the_right_sentence_and_drops_the_rest(index):
    candidates = set(index.candidates("sat on the mat"))
    assert 0 in candidates
    assert 2 not in candidates
    assert 3 not in candidates


def test_a_query_from_one_sentence_does_not_drag_in_unrelated_ones(index):
    candidates = set(index.candidates("decorators wrap"))
    assert candidates == {3}


# --- the pigeonhole split, from both sides --------------------------------


def test_typo_in_the_second_half_is_rescued_by_the_first(index):
    """'the mut' is nowhere in the corpus, so only 'sat on ' can find this."""
    assert 0 in set(index.candidates("sat on the mut"))


def test_typo_in_the_first_half_is_rescued_by_the_second(index):
    """'sut on ' is nowhere in the corpus, so only 'the mat' can find this."""
    assert 0 in set(index.candidates("sut on the mat"))


def test_a_half_that_is_nowhere_in_the_corpus_contributes_nothing(index):
    """Both halves unknown means nothing to offer — not a crash."""
    assert set(index.candidates("zzzzzzz qqqqqqq")) == set()


# --- the short-query fallback ---------------------------------------------


@pytest.mark.parametrize("query", ["cat", "cats", "cat s"])
def test_queries_too_short_to_split_fall_back_to_the_whole_corpus(
    index, query
):
    """Below 2*gram_size a half is shorter than one n-gram, so the index
    cannot be consulted at all and every sentence has to be verified.
    """
    assert set(index.candidates(query)) == set(range(len(CORPUS)))


def test_the_fallback_threshold_follows_the_gram_size():
    small = NGramIndex(gram_size=2)
    for sentence_id, sentence in enumerate(CORPUS):
        small.add(sentence_id, sentence)
    # With 2-grams the halves only need two characters each, so a 4-character
    # query is already indexable and must not trigger the fallback.
    assert set(small.candidates("cats")) != set(range(len(CORPUS)))


def test_empty_query_returns_nothing(index):
    assert set(index.candidates("")) == set()


# --- the property that actually matters -----------------------------------


ALPHABET = "aeioustx "


def one_edit_variants(text: str):
    """Every string one edit away from ``text``, over a small alphabet."""
    for i in range(len(text)):
        for character in ALPHABET:
            if character != text[i]:
                yield text[:i] + character + text[i + 1 :]
    for i in range(len(text)):
        yield text[:i] + text[i + 1 :]
    for i in range(len(text) + 1):
        for character in ALPHABET:
            yield text[:i] + character + text[i:]


def test_the_filter_never_loses_a_sentence_that_really_matches(index, matcher):
    """Cross-check the filter against the verifier, exhaustively.

    For every one-edit corruption of every substring, if the matcher says it
    matches the sentence it came from, the index must have offered that
    sentence. A failure here means real completions are silently missing.
    """
    checked = 0
    for sentence_id, sentence in enumerate(CORPUS):
        for start in range(0, len(sentence) - 8, 3):
            for query in one_edit_variants(sentence[start : start + 8]):
                if matcher.match(query, sentence) is None:
                    continue
                checked += 1
                assert sentence_id in set(index.candidates(query)), (
                    f"index lost sentence {sentence_id} for query {query!r}"
                )
    # Guard against the loop silently doing nothing.
    assert checked > 500
